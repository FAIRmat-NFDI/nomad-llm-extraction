import os

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    import json

    import litellm
    from litellm import get_supported_openai_params, supports_response_schema
    from post_proc_pipeline import build_pipeline
    from pre_proc_schema import get_schema
    from validator import filter_unwanted

    from nomad_llm_extraction.pipeline.extraction_pipeline import ExtractionPipeline
    from nomad_llm_extraction.pipeline.input_sources.paper import PDFParser
    from nomad_llm_extraction.pipeline.models import PromptConfig
    from nomad_llm_extraction.pipeline.schema_filling.llm_engine import LiteLLMEngine
    from nomad_llm_extraction.pipeline.schema_sources import NomadSchemaSource
    from nomad_llm_extraction.pipeline.temporal_pipeline import (
        STAGES_REGISTRY,
        ExtractionPipelineConfig,
        PipelineWorkflow,
        PipelineWorkflowInput,
        TemporalExtractionPipeline,
        register_class_obj,
        register_processor_func,
        register_stage_func,
    )
    from nomad_llm_extraction.utils.export_to_nomad import (
        get_authentication_token,
        push_to_nomad,
    )
    from nomad_llm_extraction.utils.utils import get_safe_ctx, get_temporal_activities

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
supports_response_schema(model='gpt-5.4-mini-2026-03-17')
get_supported_openai_params(model='gpt-5.4-mini-2026-03-17')


def process_to_nomad(data, doi, model_name):
    """Wraps the transformed data in the specific NOMAD schema envelope."""
    doi_url = 'https://www.doi.org/' + doi
    # Run the transformation    output_entries = []

    # The input is usually a dict with "cells": [...]
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
                    **cell,  # Unpack the processed cell data here
                }
            }
            output_entries.append(entry)

    return output_entries


# ...
# ...
# async def main(activities):
#     client = await Client.connect('localhost:7233')
#     worker = Worker(
#         client,
#         task_queue='extraction_pipeline',
#         workflows=[PipelineWorkflow],
#         activities=activities,
#     )
#     await worker.run()
import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.worker import Worker


async def run_extraction(pipeline_config, ctx_args={}):
    # Connect to the local Temporal cluster (assuming it's running on default port)
    client = await Client.connect('localhost:7233')
    activities = list(get_temporal_activities(STAGES_REGISTRY).values())
    # Create the Worker
    worker = Worker(
        client,
        task_queue='extraction_pipeline',
        workflows=[PipelineWorkflow],
        activities=activities,
        activity_executor=ThreadPoolExecutor(
            max_workers=4
        ),  # Use threads for activities
    )

    # Start the Worker in the background so it doesn't block the script
    worker_task = asyncio.create_task(worker.run())
    print('Worker started in the background...')

    # Start and wait for the Workflow to complete
    print('Executing workflow...')
    pipeline_input = PipelineWorkflowInput(
        pipeline_config=pipeline_config, ctx_args=ctx_args
    )
    result = await client.execute_workflow(
        PipelineWorkflow.run,
        pipeline_input,
        id='test-pipeline-workflow',
        task_queue='extraction_pipeline',
    )

    print(f'Workflow finished! Result: {result}')

    # Optional: Cleanly shut down the worker once the workflow is done
    worker_task.cancel()
    return result


if __name__ == '__main__':
    # processed_cells = json.load(open('extracted_null_5.json'))
    # doi = '10.1002/adfm.202517729'
    # nomad_entries = process_to_nomad(
    #     processed_cells, doi, model_name='claude-4-sonnet-20250514'
    # )
    # with open('nomad_entries.json', 'w') as f:
    #     json.dump(nomad_entries, f, indent=2)
    # entry_name_format = doi.replace('/', '--') + '-cell-{index}'
    # push_to_nomad(
    #     entry_name_format,
    #     nomad_entries,
    #     get_authentication_token(),
    #     'dFoC11rYSaqGEjX7ZWCGNg',
    # )
    # p
    pdf_parser = PDFParser()
    PDF_PATH = 'downloads/10.1002--adfm.202517729.pdf'
    text = pdf_parser.parse_pdf(PDF_PATH)
    # with open('text.txt') as f:
    #     text = f.read()
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
    postprocess_schema = NomadSchemaSource(
        'perovskite_solar_cell_database.llm_extraction_schema.LLMExtractedPerovskiteSolarCell',
        remove_defs=True,
    ).get_schema()
    json.dump(postprocess_schema, open('postprocess_schema.json', 'w'), indent=2)
    proc = build_pipeline()

    def postprocessor(data, schema):
        cells = data.get('cells', [data]) if isinstance(data, dict) else data
        return {'cells': proc.apply(cells, schema)}

    # engine = LiteLLMEngine(model_name='claude-4-sonnet-20250514')
    engine = LiteLLMEngine(model_name='gpt-5.4-mini-2026-03-17')
    # register_class_obj('LiteLLMEngine', LiteLLMEngine)
    register_class_obj('LiteLLMEngine', engine)
    register_processor_func('postprocessor', postprocessor)
    register_processor_func('filter_unwanted', filter_unwanted)

    # pipeline = ExtractionPipeline(
    #     engine=engine,
    #     extraction_schema=extraction_schema,
    #     postprocessing_schema=postprocess_schema,
    #     prompt_config=PromptConfig(
    #         system_prompt=SYSTEM_PROMPT,
    #         instruction_text=INSTRUCTION_TEXT,
    #     ),
    #     postprocessor=postprocessor,
    #     postprocessor_args=['filtered_data', 'postprocessing_schema'],
    #     filter_func=filter_unwanted,
    #     filter_args=['extracted_data', 'text'],
    # )

    # result = pipeline.run(text)
    pipeline_config = ExtractionPipelineConfig(
        pipeline_type='temporal_extraction_pipeline',
        construct_args={
            'extraction_schema': extraction_schema,
            'postprocessing_schema': postprocess_schema,
            'prompt_config': PromptConfig(
                system_prompt=SYSTEM_PROMPT,
                instruction_text=INSTRUCTION_TEXT,
            ),
            'objclasses': {
                # 'engine': [
                #     'LiteLLMEngine',
                #     {
                #         'model_name': 'openai/gpt-5.4-mini-2026-03-17',
                #         'api_key': os.getenv('OPENAI_API_KEY'),
                #         # 'model_name': 'claude-4-sonnet-20250514',
                #         # 'api_key': os.getenv('ANTHROPIC_API_KEY'),
                #     },
                # ],
                'engine': 'LiteLLMEngine'
            },
            'processors': {
                'postprocessor': 'postprocessor',
                'filter_func': 'filter_unwanted',
            },
            'postprocessor_args': ['filtered_data', 'postprocessing_schema'],
            'filter_args': ['extracted_data', 'text'],
        },
    )
    ctx_args = {'text': text}
    result = asyncio.run(run_extraction(pipeline_config, ctx_args={'text': text}))
    print(result)
    result.ctx = json.loads(get_safe_ctx(result.ctx))
    with open('extracted_null_5.pkl', 'wb') as f:
        # json.dump(extracted, f, indent=2)
        import pickle

        pickle.dump(result, f)
    if result.success:
        processed_cells = result.postprocessed_data

    try:
        with open('extracted_null_5.json', 'w') as f:
            import json

            json.dump(processed_cells, f, indent=2)
    except Exception as e:
        print(f'Error parsing extraction as JSON: {e}')
