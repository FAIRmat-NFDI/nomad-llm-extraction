from copy import deepcopy

from nomad_llm_extraction.transform.json_transformer import ProcessingPipeline
from nomad_llm_extraction.transform.utils import remove_keys_recursive, remove_path_keys

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
KEY_MAPPING = {
    'bandgap': 'band_gap',
    'PCE_at_the_start_of_the_experiment': 'PCE_at_start',
    'PCE_at_the_end_of_experiment': 'PCE_at_end',
    'a_ions': 'ions_a_site',
    'b_ions': 'ions_b_site',
    'x_ions': 'ions_x_site',
    'time': 'time',
}
REV_KEY_MAPPING = {value: key for key, value in KEY_MAPPING.items()}


def pre_add_null_default_cond(section, state):
    return 'property' in state['type']


def pre_add_null_default(jbobj, path, func_args):
    jbobj[path]['default'] = None
    return jbobj


def pre_rename_cond(section, state):
    return state['name'] in REV_KEY_MAPPING


# Argument Builder Function
def pre_rename_args(section, state):
    """Build the source and destination paths used for archive renaming using the target schema

    The pipeline stores the original archive path in `a_path` and the current
    field name in `name`. We compute the archive path that should exist after the
    rename using the parent archive path 'a_p_path' and return both the updated state and the original path.
    """
    return state, f'{state["p_path"]}.{REV_KEY_MAPPING[state["name"]]}'


# Transformation Function
def pre_rename_section(jbobj, path, new_path):
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


def pre_pc_cond(section, state):
    return state['name'] in PERCENTAGE_KEYS


def pre_update_unit_value_schema_percentage(jsobj, path, func_args):
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


def pre_split_value_unit_cond(section, state):
    """Return `True` for fields that should be split into value and unit."""
    return state['name'] in SPLIT_VALUE_UNIT


def pre_split_value_unit(jbobj, path, func_args):
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


def pre_unit_cond(section, state):
    return 'unit' in section


def pre_unit_args(section, state):
    return state, section['unit']


def pre_update_unit_value_schema(jsobj, path, unit):
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


def get_schema(schema, multi_instance_field=None):
    proc_pipeline = ProcessingPipeline(
        {
            'rename': [
                pre_rename_section,  # Transformation Function
                pre_rename_cond,  # Condition Function
                pre_rename_args,  # Argument Builder Function
            ],
            'add_uv': [
                pre_update_unit_value_schema,
                pre_unit_cond,
                pre_unit_args,
            ],
            'conc': [
                pre_split_value_unit,
                pre_split_value_unit_cond,
                None,
            ],
            'pc': [
                pre_update_unit_value_schema_percentage,
                pre_pc_cond,
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
    path_keys_to_remove = [
        'properties.stability.properties.type',
        'properties.stability.properties.lamp',
    ]
    if multi_instance_field is not None:
        path_keys_to_remove = [
            f'properties.{multi_instance_field}.items.{key}'
            for key in path_keys_to_remove
        ]
    schema = remove_path_keys(
        schema,
        path_keys_to_remove,
    )
    return schema
