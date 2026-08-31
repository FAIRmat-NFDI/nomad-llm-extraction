from __future__ import annotations

import json

from loguru import logger
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    import json
    from dataclasses import dataclass
    from typing import Any

    from perla_activities import (
        PerlaPostProcessingWorkflow,
        PerlaPostProcessingWorkflowInput,
        filter_extraction,
        optimize_extraction_schema,
        postprocessor,
    )
    from pre_proc_schema import exclude

    from nomad_llm_extraction.pipeline.activities import (
        NomadSchemaConfig,
        build_prompt,
        get_nomad_schema,
        json_parse,
        llm_call,
        parse_text_from_pdf,
        validate_extraction_with_schema,
    )
    from nomad_llm_extraction.pipeline.workflows import (
        ExtractionWorkflow,
        ExtractionWorkflowInput,
        LLMCallWorkflow,
    )

SYSTEM_PROMPT = 'You are a world class AI that excels at extracting data about perovskite solar cells from papers. You only report single junction solar cells and no other types of solar cells. You never come up with data and only state data that have been measured and written in the paper and which you can confidently extract. It is better for you to skip than to report data you are uncertain in. Take care to separate devices. Do not extract data people took from other papers but only data reported for the first time in this paper. Do not convert units yourself and stick to the units reported in the paper. Be careful with decimal points. Do not try to come up with a value by doing maths or any inference. Stick to what is explicitly written. Be careful that the data you put together really belongs to the same device. Do not forget to get all the different cells/devices. There can be many. You can make a guess for dimensionality. Make sure to only use the allowed types and literal values provided in the schema. If there are options, choose one. The device stack has to be listed separately in the layers section of the schema with layer names as the names of the parts of the stack. Do not miss the stack/layers. Make sure to separate deposition steps like thermal annealing and spin coating, etc. Keep to the given schema.'
INSTRUCTION_TEXT = "Extract the data from the text of the paper. Only report data about devices for which you are certain that the extraction you provide is correct. Do not convert any value or unit. Do not forget to fill in the bandgap. Make sure it is correct for the cell to the best of your abilities. If you're not confident, skip it. Always fill the ions section and coefficients for the perovskite material. If it's not stated, you can infer it from the formula. For example, for MAPbI3 you get coefficients 1 for MA, 1 for Pb, and 3 for I."


import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from temporalio.client import Client
from temporalio.worker import Worker

all_activities = [
    build_prompt,
    llm_call,
    parse_text_from_pdf,
    validate_extraction_with_schema,
    optimize_extraction_schema,
    postprocessor,
    filter_extraction,
    json_parse,
    get_nomad_schema,
]


@dataclass
class PerlaWorkflowInput:
    pdf_path: str
    m_def: str
    model_name: str = 'claude-4-sonnet-20250514'


@workflow.defn
class PerlaCompleteWorkflow:
    @workflow.run
    async def run(self, inp: PerlaWorkflowInput) -> dict[str, Any]:
        extraction_schema_input = NomadSchemaConfig(
            m_def=inp.m_def,
            unit_value=True,
            remove_defs=True,
            resolve_allOf=True,
            exclude=exclude,
            multi_instance_field='cells',
        )
        extraction_schema = await workflow.execute_activity(
            get_nomad_schema,
            extraction_schema_input,
            start_to_close_timeout=timedelta(seconds=30),
        )
        extraction_schema = await workflow.execute_activity(
            optimize_extraction_schema,
            args=[extraction_schema, 'cells'],
            start_to_close_timeout=timedelta(seconds=30),
        )
        postprocess_schema = await workflow.execute_activity(
            get_nomad_schema,
            NomadSchemaConfig(
                m_def=inp.m_def,
                remove_defs=True,
            ),
            start_to_close_timeout=timedelta(seconds=30),
        )
        extraction_workflow_input = ExtractionWorkflowInput(
            pdf_path=inp.pdf_path,
            extraction_schema=extraction_schema,
            system_prompt=SYSTEM_PROMPT,
            instruction_text=INSTRUCTION_TEXT,
            llm_engine_config={'model_name': inp.model_name},
        )
        extraction_output = await workflow.execute_child_workflow(
            ExtractionWorkflow.run,
            extraction_workflow_input,
            id='test-pipeline-workflow',
            task_queue='extraction_pipeline',
        )
        if extraction_output.err_message:
            logger.error(f'LLM call failed with error: {extraction_output.err_message}')
            raise Exception(f'LLM call failed: {extraction_output.err_message}')
        extraction = extraction_output.extracted_data
        postprocess_input = PerlaPostProcessingWorkflowInput(
            data=extraction,
            schema=postprocess_schema,
            text=extraction_workflow_input.instruction_text,
        )
        postprocessed_data = await workflow.execute_child_workflow(
            PerlaPostProcessingWorkflow.run,
            postprocess_input,
            id='test-postprocess-workflow',
            task_queue='extraction_pipeline',
        )
        return postprocessed_data


async def run_extraction(
    pdf_path: str, m_def: str, model_name: str = 'claude-4-sonnet-20250514'
):
    client = await Client.connect('localhost:7233')
    worker = Worker(
        client,
        task_queue='extraction_pipeline',
        workflows=[
            LLMCallWorkflow,
            ExtractionWorkflow,
            PerlaPostProcessingWorkflow,
            PerlaCompleteWorkflow,
        ],
        activities=all_activities,
        activity_executor=ThreadPoolExecutor(
            max_workers=10
        ),  # Adjust max_workers as needed
    )
    worker_task = asyncio.create_task(worker.run())

    workflow_input = PerlaWorkflowInput(
        pdf_path=pdf_path, m_def=m_def, model_name=model_name
    )
    result = await client.execute_workflow(
        PerlaCompleteWorkflow.run,
        workflow_input,
        id='perla-complete-workflow',
        task_queue='extraction_pipeline',
    )
    worker_task.cancel()  # Cleanly shut down the worker after workflow completion
    return result


async def run_extraction_explicit(pdf_path: str, m_def: str):
    client = await Client.connect('localhost:7233')
    worker = Worker(
        client,
        task_queue='extraction_pipeline',
        workflows=[ExtractionWorkflow, PerlaPostProcessingWorkflow, LLMCallWorkflow],
        activities=all_activities,
        activity_executor=ThreadPoolExecutor(
            max_workers=10
        ),  # Adjust max_workers as needed
    )

    # Start the Worker in the background so it doesn't block the script
    worker_task = asyncio.create_task(worker.run())
    print('Worker started in the background...')

    extraction_schema_input = NomadSchemaConfig(
        m_def=m_def,
        unit_value=True,
        remove_defs=True,
        resolve_allOf=True,
        exclude=exclude,
        multi_instance_field='cells',
    )
    extraction_schema = await client.execute_activity(
        get_nomad_schema,
        extraction_schema_input,
        id='standalone-get-schema-activity',
        task_queue='extraction_pipeline',
        start_to_close_timeout=timedelta(seconds=30),
    )
    extraction_schema = await client.execute_activity(
        optimize_extraction_schema,
        args=[extraction_schema, 'cells'],
        id='standalone-optimize-schema-activity',
        task_queue='extraction_pipeline',
        start_to_close_timeout=timedelta(seconds=30),
    )
    postprocess_schema = await client.execute_activity(
        get_nomad_schema,
        NomadSchemaConfig(
            m_def=m_def,
            remove_defs=True,
        ),
        id='standalone-get-postprocess-schema-activity',
        task_queue='extraction_pipeline',
        start_to_close_timeout=timedelta(seconds=30),
    )
    json.dump(extraction_schema, open('extraction_schema.json', 'w'), indent=2)

    engine_config = {'model_name': 'claude-4-sonnet-20250514'}

    workflow_input = ExtractionWorkflowInput(
        pdf_path=pdf_path,
        extraction_schema=extraction_schema,
        system_prompt=SYSTEM_PROMPT,
        instruction_text=INSTRUCTION_TEXT,
        llm_engine_config=engine_config,
    )

    # Start and wait for the Workflow to complete
    print('Executing workflow...')
    extraction_output = await client.execute_workflow(
        ExtractionWorkflow.run,
        workflow_input,
        id='test-pipeline-workflow',
        task_queue='extraction_pipeline',
    )
    print(
        f'Extraction workflow finished! Raw extraction: {extraction_output.raw_output}'
    )

    postprocess_input = PerlaPostProcessingWorkflowInput(
        data=extraction_output.extracted_data,
        schema=postprocess_schema,
        text=workflow_input.instruction_text,
    )
    print('Post-processing...')
    postprocessed_data = await client.execute_workflow(
        PerlaPostProcessingWorkflow.run,
        postprocess_input,
        id='test-postprocess-workflow',
        task_queue='extraction_pipeline',
    )

    print(f'Workflow finished! Result: {postprocessed_data}')

    # Optional: Cleanly shut down the worker once the workflow is done
    worker_task.cancel()
    return postprocessed_data


def main():
    pdf_path = 'downloads/10.1002--adfm.202517729.pdf'
    m_def = 'perovskite_solar_cell_database.llm_extraction_schema.LLMExtractedPerovskiteSolarCell'

    out = asyncio.run(
        run_extraction(pdf_path, m_def, model_name='claude-4-sonnet-20250514')
    )
    json.dump(out, open('final_output.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
