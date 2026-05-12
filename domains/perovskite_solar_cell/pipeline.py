import json
import pickle
from copy import deepcopy

from utils import (
    get_name_from_id,
    remove_keys_recursive,
    remove_path_keys,
    remove_sections,
)

from nomad_llm_extraction.pipeline.schema_filling.llm_engine import LiteLLMEngine
from nomad_llm_extraction.transform.json_transformer import ProcessingPipeline
from nomad_llm_extraction.transform.utils import get_nomad_schema, resolve_schema

KEY_MAPPING = {
    'bandgap': 'band_gap',
    'PCE_at_the_start_of_the_experiment': 'PCE_at_start',
    'PCE_at_the_end_of_experiment': 'PCE_at_end',
    'a_ions': 'ions_a_site',
    'b_ions': 'ions_b_site',
    'x_ions': 'ions_x_site',
    # 'time': 'time',
}
REV_KEY_MAPPING = {value: key for key, value in KEY_MAPPING.items()}

SYSTEM_PROMPT = 'You are a world class AI that excels at extracting data about perovskite solar cells from papers. You only report single junction solar cells and no other types of solar cells. You never come up with data and only state data that have been measured and written in the paper and which you can confidently extract. It is better for you to skip than to report data you are uncertain in. Take care to separate devices. Do not extract data people took from other papers but only data reported for the first time in this paper. Do not convert units yourself and stick to the units reported in the paper. Be careful with decimal points. Do not try to come up with a value by doing maths or any inference. Stick to what is explicitly written. Be careful that the data you put together really belongs to the same device. Do not forget to get all the different cells/devices. There can be many. You can make a guess for dimensionality. Make sure to only use the allowed types and literal values provided in the schema. If there are options, choose one. The device stack has to be listed separately in the layers section of the schema with layer names as the names of the parts of the stack. Do not miss the stack/layers. Make sure to separate deposition steps like thermal annealing and spin coating, etc. Keep to the given schema.'
INSTRUCTION_TEXT = "Extract the data from the text of the paper. Only report data about devices for which you are certain that the extraction you provide is correct. Do not convert any value or unit. Do not forget to fill in the bandgap. Make sure it is correct for the cell to the best of your abilities. If you're not confident, skip it. Always fill the ions section and coefficients for the perovskite material. If it's not stated, you can infer it from the formula. For example, for MAPbI3 you get coefficients 1 for MA, 1 for Pb, and 3 for I."


def add_null_default_cond(section, state):
    return 'property' in state['type']


def add_null_default(jbobj, path, func_args):
    jbobj[path]['default'] = None
    return jbobj


def rename_cond(section, state):
    return state['name'] in REV_KEY_MAPPING


# Argument Builder Function
def rename_args(section, state):
    """Build the source and destination paths used for archive renaming using the target schema

    The pipeline stores the original archive path in `a_path` and the current
    field name in `name`. We compute the archive path that should exist after the
    rename using the parent archive path 'a_p_path' and return both the updated state and the original path.
    """
    return state, f'{state["p_path"]}.{REV_KEY_MAPPING[state["name"]]}'


# Transformation Function
def rename_section(jbobj, path, new_path):
    """Rename a field in the archive while preserving its value.

    A deep copy is used so the move stays explicit and safe even when the stored
    value is nested.
    """
    temp_value = deepcopy(jbobj[path])
    jbobj[new_path] = temp_value
    del jbobj[path]
    return jbobj


PERCENTAGE_KEYS = [
    'pce',
    'ff',
    'humidity',
    'PCE_at_the_start_of_the_experiment',
    'PCE_at_the_end_of_experiment',
    'PCE_after_1000_hours',
]


def pc_cond(section, state):
    return state['name'] in PERCENTAGE_KEYS


def update_unit_value_schema_p(jsobj, path, func_args):
    """Expand a schema field into a `value`/`unit` object schema.

    This temporarily converts a flat schema property with an unit associated with it into an object like:

    {
        "properties": {
            "value": <original field schema>,
            "unit": {"type": "string", "enum": [unit]}
        }
    }
    """
    value_schema = deepcopy(jsobj[path])

    jsobj[path] = {
        'properties': {
            'value': value_schema.copy(),
            'unit': {'type': 'string', 'const': '%'},
        }
    }

    for key in ['title', 'description']:
        if key in value_schema:
            jsobj[path][key] = value_schema[key]

    return jsobj


SPLIT_VALUE_UNIT = ['concentration_unit']


def split_value_unit_cond(section, state):
    """Return `True` for fields that should be split into value and unit."""
    return state['name'] in SPLIT_VALUE_UNIT


def split_value_unit(jbobj, path, func_args):
    """Split a `{value, unit}` entry into separate archive fields.

    Example:
        `concentration = {"value": 3, "unit": "mol/L"}`

    becomes:
        `concentration = 3`
        `concentration_unit = "mol/L"`
    """
    if jbobj[path] is None:
        return jbobj
    ppath = path.split('_unit')[0]
    value = jbobj[ppath]
    unit = jbobj[path]
    jbobj[ppath] = {'properties': {'value': value, 'unit': unit}}
    del jbobj[path]
    return jbobj


def unit_cond(section, state):
    return 'unit' in section


def unit_args(section, state):
    return state, section['unit']


def update_unit_value_schema(jsobj, path, unit):
    """Expand a schema field into a `value`/`unit` object schema.

    This temporarily converts a flat schema property with an unit associated with it into an object like:

    {
        "properties": {
            "value": <original field schema>,
            "unit": {"type": "string", "enum": [unit]}
        }
    }
    """
    value_schema = deepcopy(jsobj[path])
    del value_schema['unit']
    jsobj[path] = {
        'properties': {
            'value': value_schema.copy(),
            'unit': {'type': 'string', 'enum': [unit]},
        }
    }

    for key in ['title', 'description']:
        if key in value_schema:
            jsobj[path][key] = value_schema[key]

    return jsobj


exclude = [
    'nomad.datamodel.metainfo.basesections.v1.PublicationReference',
    'nomad.datamodel.data.EntryData',
    'perovskite_solar_cell_database.llm_extraction_schema.SectionRevision',
    'nomad.datamodel.data.ArchiveSection',
    'perovskite_solar_cell_database.llm_extraction_schema.LLMExtractedPerovskiteSolarCell.extraction_metadata',
    'perovskite_solar_cell_database.llm_extraction_schema.LLMExtractedPerovskiteSolarCell.layer_order',
    'perovskite_solar_cell_database.llm_extraction_schema.LLMExtractedPerovskiteSolarCell.DOI_number',
    'perovskite_solar_cell_database.llm_extraction_schema.LLMExtractedPerovskiteSolarCell.classic_entry',
    'perovskite_solar_cell_database.llm_extraction_schema.LLMExtractedPerovskiteSolarCell.reviewer_additional_notes',
    'nomad.datamodel.metainfo.basesections.v1.Component.mass',
    'nomad.datamodel.metainfo.basesections.v1.Component.mass_fraction',
    'nomad.datamodel.metainfo.basesections.v1.Component.name',
    # 'perovskite_solar_cell_database.composition.Impurity',
    'perovskite_solar_cell_database.composition.PerovskiteAIonComponent.system',
    'perovskite_solar_cell_database.composition.PerovskiteBIonComponent.system',
    'perovskite_solar_cell_database.composition.PerovskiteChemicalSection.cas_number',
    'perovskite_solar_cell_database.composition.PerovskiteChemicalSection.iupac_name',
    'perovskite_solar_cell_database.composition.PerovskiteChemicalSection.smiles',
    'perovskite_solar_cell_database.composition.PerovskiteCompositionSection.composition_estimate',
    'perovskite_solar_cell_database.composition.PerovskiteCompositionSection.long_form',
    'perovskite_solar_cell_database.composition.PerovskiteCompositionSection.short_form',
    'perovskite_solar_cell_database.composition.PerovskiteIonSection.source_compound_cas_number',
    'perovskite_solar_cell_database.composition.PerovskiteIonSection.source_compound_iupac_name',
    'perovskite_solar_cell_database.composition.PerovskiteIonSection.source_compound_molecular_formula',
    'perovskite_solar_cell_database.composition.PerovskiteIonSection.source_compound_smiles',
    'perovskite_solar_cell_database.composition.PerovskiteIonComponent.system',
    'perovskite_solar_cell_database.composition.PerovskiteXIonComponent.system',
    'nomad.datamodel.metainfo.basesections.v1.SystemComponent.system',
    # 'perovskite_solar_cell_database.llm_extraction_schema.LightSource.lamp',
    # 'perovskite_solar_cell_database.llm_extraction_schema.LightSource.type',
    # 'perovskite_solar_cell_database.llm_extraction_schema.Solute.concentration_unit'
]


def get_schema(multiple=False):
    schema = get_nomad_schema(
        'perovskite_solar_cell_database.llm_extraction_schema.LLMExtractedPerovskiteSolarCell'
    )
    schema = remove_sections(schema, exclude)
    schema = resolve_schema(schema, remove_defs=True, resolve_allOf=True)
    proc_pipeline = ProcessingPipeline(
        {
            'rename': [
                rename_section,  # Transformation Function
                rename_cond,  # Condition Function
                rename_args,  # Argument Builder Function
            ],
            'add_uv': [
                update_unit_value_schema,
                unit_cond,
                unit_args,
            ],
            'conc': [
                split_value_unit,
                split_value_unit_cond,
                None,
            ],
            'pc': [
                update_unit_value_schema_p,
                pc_cond,
                None,
            ],
            # 'default_null': [
            #     add_null_default,
            #     add_null_default_cond,
            #     None,
            # ],
        }
    )
    schema = proc_pipeline.apply(schema, proc_type='schema')
    schema = remove_keys_recursive(schema, ['$id', 'label'])
    schema = remove_path_keys(
        schema,
        [
            'properties.stability.properties.type',
            'properties.stability.properties.lamp',
        ],
    )
    if multiple:
        schema = {
            'type': 'object',
            'properties': {
                'cells': {
                    'type': 'array',
                    'items': schema,
                    'description': 'List of extracted solar cells',
                }
            },
        }
    return schema


def extract(text, multiple=False):
    llm_engine = LiteLLMEngine('claude-sonnet-4-20250514')
    schema = get_schema(multiple=multiple)
    prompt = f'{SYSTEM_PROMPT}\n{INSTRUCTION_TEXT}\n Here is the schema: {json.dumps(schema, indent=2)} \n\nHere is the text:\n{text}'
    return llm_engine.generate(prompt, schema)


if __name__ == '__main__':
    with open('text.txt') as f:
        text = f.read()
    # text = 'bandgaps 1.63 eV (I/Br ratio: 83:17), 1.68 eV (76:24), 1.74 eV (70:30), 1.80 eV (60:40), and 1.85 eV (55:45) '
    extracted = extract(text, multiple=True)
    with open('extracted_null_2.pkl', 'wb') as f:
        # json.dump(extracted, f, indent=2)
        pickle.dump(extracted, f)
    try:
        extraction = json.loads(extracted.choices[0].message.content)
        with open('extracted_null_2.json', 'w') as f:
            json.dump(extraction, f, indent=2)
    except Exception as e:
        print(f'Error parsing extraction as JSON: {e}')
