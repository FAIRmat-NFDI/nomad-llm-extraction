from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    import json
    from dataclasses import dataclass
    from datetime import timedelta
    from typing import Any

    from schemas import BatteryData, ExperimentIdentifiers

    from nomad_llm_extraction.pipeline import BASE_ACTIVITIES, BASE_WORKFLOWS
    from nomad_llm_extraction.pipeline.activities import (
        InlineSchemaConfig,
        get_inline_schema,
        parse_text_from_pdf,
    )
    from nomad_llm_extraction.pipeline.workflows import (
        ExtractionWorkflow,
        ExtractionWorkflowInput,
    )

import asyncio
from datetime import timedelta

from temporalio.client import Client
from temporalio.worker import Worker

SYSTEM_PROMPT = (
    'You are an expert scientific data extractor. Extract structured information from research papers.\n'
    '1. Only extract explicitly mentioned text.\n'
    '2. Identify associated units for numbers.\n'
    '3. Set missing values to null.\n'
)

INSTRUCTION_TEXT = """                *** CRITICAL INSTRUCTION ***
                Extract data SPECIFICALLY AND ONLY for the cell/experiment identified as: "{cell_id}"

                - If the text mentions properties for other cells, IGNORE THEM COMPLETELY.
                - Only extract data explicitly linked to "{cell_id}".
                - Set cell_identifier to "{cell_id}" in the output."""

ID_SYSTEM_PROMPT = (
    'Analyze the following text and list the unique identifiers for every battery cell'
    'or experiment described by the authors.\n'
    'Rules:\n'
    '1. Look for sample names (e.g., "NMC-811", "Cell A", "Gen 2", "Pouch Cell #1").\n'
    '2. Ignore commercial references or competitor baselines unless they were tested by these authors.\n'
    '3. If only one cell is tested (e.g., "the proposed battery"), return ["default"].\n'
)


@dataclass
class BatteryLLMExtractionWorkflowInput:
    pdf_path: str
    model_name: str = 'claude-4-sonnet-20250514'


@workflow.defn
class BatteryLLMExtractionWorkflow:
    @workflow.run
    async def run(self, inp: BatteryLLMExtractionWorkflowInput) -> dict[str, Any]:
        extraction_results = {}
        battery_schema = BatteryData.model_json_schema()
        identifiers_schema = ExperimentIdentifiers.model_json_schema()
        engine_config = {'model_name': inp.model_name}

        text, doi = await workflow.execute_activity(
            parse_text_from_pdf,
            inp.pdf_path,
            start_to_close_timeout=timedelta(seconds=30),
        )
        identifier_workflow_input = ExtractionWorkflowInput(
            text=text,
            extraction_schema=identifiers_schema,
            system_prompt=ID_SYSTEM_PROMPT,
            instruction_text='',
            llm_engine_config=engine_config,
        )
        identification_result = await workflow.execute_child_workflow(
            ExtractionWorkflow.run,
            identifier_workflow_input,
            id='identification_workflow',
        )

        if identification_result.err_message:
            print(
                f'Error in identifier extraction: {identification_result.err_message}'
            )
            cell_ids = ['default']
        else:
            cell_ids = identification_result.extracted_data.get(
                'identifiers', ['default']
            ) or ['default']
        extraction_results['identifiers'] = cell_ids
        battery_extraction_schema = await workflow.execute_activity(
            get_inline_schema,
            InlineSchemaConfig(
                inline_schema=battery_schema,
                remove_defs=True,
                resolve_allOf=True,
                remove_null_anyof=True,
            ),
            start_to_close_timeout=timedelta(seconds=30),
        )
        for cell_id in cell_ids:
            instruction_text = INSTRUCTION_TEXT.format(cell_id=cell_id)
            extraction_workflow_input = ExtractionWorkflowInput(
                text=text,
                extraction_schema=battery_extraction_schema,
                system_prompt=SYSTEM_PROMPT,
                instruction_text=instruction_text,
                llm_engine_config=engine_config,
            )
            extraction_result = await workflow.execute_child_workflow(
                ExtractionWorkflow.run,
                extraction_workflow_input,
                id=f'extraction_workflow_{cell_id}',
            )
            if extraction_result.err_message:
                print(
                    f'Error in data extraction for cell {cell_id}: {extraction_result.err_message}'
                )
                extraction_results[cell_id] = extraction_result.err_message
            else:
                print(f'Extraction successful for cell {cell_id}')
                extraction_results[cell_id] = extraction_result.extracted_data
        return extraction_results


async def run_extraction(
    pdf_path: str, m_def: str, model_name: str = 'claude-4-sonnet-20250514'
):
    client = await Client.connect('localhost:7233')
    worker = Worker(
        client,
        task_queue='extraction_pipeline',
        workflows=[
            *BASE_WORKFLOWS,
            BatteryLLMExtractionWorkflow,
        ],
        activities=BASE_ACTIVITIES,
    )
    worker_task = asyncio.create_task(worker.run())

    workflow_input = BatteryLLMExtractionWorkflowInput(
        pdf_path=pdf_path, model_name=model_name
    )
    result = await client.execute_workflow(
        BatteryLLMExtractionWorkflow.run,
        workflow_input,
        id='battery-extraction-workflow',
        task_queue='extraction_pipeline',
    )
    worker_task.cancel()  # Cleanly shut down the worker after workflow completion
    return result


def main():
    pdf_path = 'domains/battery/battery_paper.pdf'  # Update with your PDF path
    model_name = 'claude-4-sonnet-20250514'  # Update with your desired model
    extraction_results = asyncio.run(run_extraction(pdf_path, model_name))
    print(json.dumps(extraction_results, indent=2))


if __name__ == '__main__':
    main()
