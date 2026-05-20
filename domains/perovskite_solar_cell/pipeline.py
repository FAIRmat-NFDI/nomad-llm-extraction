from post_proc_pipeline import build_pipeline
from pre_proc_schema import get_schema

from nomad_llm_extraction.pipeline import (
    ExtractionPipeline,
    NomadSchemaSource,
    PromptConfig,
)
from nomad_llm_extraction.pipeline.schema_filling.llm_engine import LiteLLMEngine

SYSTEM_PROMPT = 'You are a world class AI that excels at extracting data about perovskite solar cells from papers. You only report single junction solar cells and no other types of solar cells. You never come up with data and only state data that have been measured and written in the paper and which you can confidently extract. It is better for you to skip than to report data you are uncertain in. Take care to separate devices. Do not extract data people took from other papers but only data reported for the first time in this paper. Do not convert units yourself and stick to the units reported in the paper. Be careful with decimal points. Do not try to come up with a value by doing maths or any inference. Stick to what is explicitly written. Be careful that the data you put together really belongs to the same device. Do not forget to get all the different cells/devices. There can be many. You can make a guess for dimensionality. Make sure to only use the allowed types and literal values provided in the schema. If there are options, choose one. The device stack has to be listed separately in the layers section of the schema with layer names as the names of the parts of the stack. Do not miss the stack/layers. Make sure to separate deposition steps like thermal annealing and spin coating, etc. Keep to the given schema.'
INSTRUCTION_TEXT = "Extract the data from the text of the paper. Only report data about devices for which you are certain that the extraction you provide is correct. Do not convert any value or unit. Do not forget to fill in the bandgap. Make sure it is correct for the cell to the best of your abilities. If you're not confident, skip it. Always fill the ions section and coefficients for the perovskite material. If it's not stated, you can infer it from the formula. For example, for MAPbI3 you get coefficients 1 for MA, 1 for Pb, and 3 for I."

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

if __name__ == '__main__':
    with open('text.txt') as f:
        text = f.read()
    text = 'bandgaps 1.63 eV (I/Br ratio: 83:17), 1.68 eV (76:24), 1.74 eV (70:30), 1.80 eV (60:40), and 1.85 eV (55:45) '

    extraction_schema = NomadSchemaSource(
        'perovskite_solar_cell_database.llm_extraction_schema.LLMExtractedPerovskiteSolarCell',
        unit_value=True,
        remove_defs=True,
        resolve_allOf=True,
        exclude=exclude,
        optimizer=get_schema,
        multi_instance_field='cells',
    ).get_schema()

    postprocess_schema = NomadSchemaSource(
        'perovskite_solar_cell_database.llm_extraction_schema.LLMExtractedPerovskiteSolarCell',
        remove_defs=True,
    ).get_schema()

    proc = build_pipeline()

    def postprocessor(data, schema):
        cells = data.get('cells', [data]) if isinstance(data, dict) else data
        return proc.apply(cells, schema)

    engine = LiteLLMEngine(model_name='claude-4-sonnet-20250514')
    pipeline = ExtractionPipeline(
        engine=engine,
        extraction_schema=extraction_schema,
        postprocessing_schema=postprocess_schema,
        prompt_config=PromptConfig(
            system_prompt=SYSTEM_PROMPT,
            instruction_text=INSTRUCTION_TEXT,
        ),
        postprocessor=postprocessor,
    )

    result = pipeline.run(text)
    if result.success:
        processed_cells = result.postprocessed_data

    with open('extracted_null_4.pkl', 'wb') as f:
        # json.dump(extracted, f, indent=2)
        import pickle

        pickle.dump(result, f)
    try:
        with open('extracted_null_4.json', 'w') as f:
            import json

            json.dump(processed_cells, f, indent=2)
    except Exception as e:
        print(f'Error parsing extraction as JSON: {e}')
