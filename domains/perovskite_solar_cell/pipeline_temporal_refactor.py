import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from post_proc_pipeline import build_pipeline
from pre_proc_schema import get_schema
from temporalio.client import Client
from temporalio.worker import Worker
from validator import filter_unwanted

from nomad_llm_extraction.pipeline.input_sources.paper import PDFParser
from nomad_llm_extraction.pipeline.models import PromptConfig
from nomad_llm_extraction.pipeline.schema_filling.llm_engine import LiteLLMEngine
from nomad_llm_extraction.pipeline.schema_sources import NomadSchemaSource
from nomad_llm_extraction.pipeline.temporal_refactor_pipeline import (
    RegistryPipelineWorkflow,
    build_extraction_workflow_payload,
    run_registered_pipeline_stage,
)
from nomad_llm_extraction.pipeline.temporal_refactor_registry import (
    register_engine_factory,
    register_runtime_callable_factory,
)
from nomad_llm_extraction.utils.utils import get_safe_ctx

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


def process_to_nomad(data, doi, model_name):
    """Wraps the transformed data in the specific NOMAD schema envelope."""
    doi_url = 'https://www.doi.org/' + doi
    output_entries = []
    if data and 'cells' in data:
        for cell in data['cells']:
            entry = {
                'data': {
                    'm_def': 'perovskite_solar_cell_database.llm_extraction_schema.LLMExtractedPerovskiteSolarCell',
                    'DOI_number': doi_url,
                    'extraction_metadata': {
                        'model': model_name,
                        'model_version': model_name,
                    },
                    **cell,
                }
            }
            output_entries.append(entry)
    return output_entries


def build_perovskite_engine(config: dict[str, Any]) -> LiteLLMEngine:
    return LiteLLMEngine(model_name=config.get('model_name', 'gpt-4o-mini'))


def build_perovskite_filter(config: dict[str, Any]):
    _ = config
    return filter_unwanted


def build_perovskite_postprocessor(config: dict[str, Any]):
    _ = config
    proc = build_pipeline()

    def postprocessor(data, schema):
        cells = data.get('cells', [data]) if isinstance(data, dict) else data
        return {'cells': proc.apply(cells, schema)}

    return postprocessor


def register_perovskite_runtime_components() -> None:
    register_engine_factory('perovskite_litellm_engine', build_perovskite_engine)
    register_runtime_callable_factory('perovskite_filter', build_perovskite_filter)
    register_runtime_callable_factory(
        'perovskite_postprocessor', build_perovskite_postprocessor
    )


async def run_extraction(ctx_args: dict[str, Any]):
    client = await Client.connect('localhost:7233')
    register_perovskite_runtime_components()

    worker = Worker(
        client,
        task_queue='extraction_pipeline',
        workflows=[RegistryPipelineWorkflow],
        activities=[run_registered_pipeline_stage],
        activity_executor=ThreadPoolExecutor(max_workers=4),
    )

    worker_task = asyncio.create_task(worker.run())
    print('Worker started in the background...')

    print('Executing workflow...')
    pipeline_input = build_extraction_workflow_payload(
        text=ctx_args.get('text', ''),
        extraction_schema=ctx_args['extraction_schema'],
        postprocessing_schema=ctx_args.get('postprocessing_schema'),
        prompt_config=ctx_args['prompt_config'],
        engine_ref='perovskite_litellm_engine',
        engine_config=ctx_args.get('engine_config', {}),
        postprocessor_ref='perovskite_postprocessor',
        postprocessor_args=['filtered_data', 'postprocessing_schema'],
        filter_ref='perovskite_filter',
        filter_args=['extracted_data', 'text'],
    )

    result = await client.execute_workflow(
        RegistryPipelineWorkflow.run,
        pipeline_input,
        id=f'test-pipeline-workflow-{uuid.uuid4()}',
        task_queue='extraction_pipeline',
    )

    # print(f'Workflow finished! Result: {result}')
    worker_task.cancel()
    return result


def enforce_strict_schema(schema):
    """
    Recursively searches a JSON schema and adds 'additionalProperties': False
    to every nested object definition to satisfy OpenAI's strict mode.
    """
    if isinstance(schema, dict):
        # Identify if the current level is an object definition
        if schema.get('type') == 'object' or 'properties' in schema:
            schema['additionalProperties'] = False

        # Recursively traverse all child keys (e.g., 'properties', 'items')
        for key, value in schema.items():
            schema[key] = enforce_strict_schema(value)

    elif isinstance(schema, list):
        # Recursively traverse items in a list (e.g., inside 'anyOf' or 'allOf')
        for i in range(len(schema)):
            schema[i] = enforce_strict_schema(schema[i])

    return schema


if __name__ == '__main__':
    pdf_parser = PDFParser()
    PDF_PATH = 'downloads/10.1002--adfm.202517729.pdf'
    text = pdf_parser.parse_pdf(PDF_PATH)
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
    json.dump(extraction_schema, open('extraction_schema.json', 'w'), indent=2)
    extraction_schema = enforce_strict_schema(extraction_schema)
    postprocess_schema = NomadSchemaSource(
        'perovskite_solar_cell_database.llm_extraction_schema.LLMExtractedPerovskiteSolarCell',
        remove_defs=True,
    ).get_schema()
    json.dump(postprocess_schema, open('postprocess_schema.json', 'w'), indent=2)

    result = asyncio.run(
        run_extraction(
            {
                'text': text,
                'extraction_schema': extraction_schema,
                'postprocessing_schema': postprocess_schema,
                'prompt_config': PromptConfig(
                    system_prompt=SYSTEM_PROMPT,
                    instruction_text=INSTRUCTION_TEXT,
                ),
                'engine_config': {
                    # 'model_name': 'claude-4-sonnet-20250514',
                    'model_name': 'gpt-4o-mini',
                },
            }
        )
    )

    # print(result)
    # result.ctx = json.loads(get_safe_ctx(result.ctx))

    with open('extracted_null_5.pkl', 'wb') as pkl_file:
        import pickle

        pickle.dump(result, pkl_file)

    if result.success:
        processed_cells = result.postprocessed_data

    try:
        with open('extracted_null_5.json', 'w') as json_file:
            json.dump(processed_cells, json_file, indent=2)
    except Exception as e:
        print(f'Error parsing extraction as JSON: {e}')
