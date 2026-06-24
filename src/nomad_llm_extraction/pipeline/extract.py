import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from temporalio import workflow
from temporalio.client import Client
from temporalio.envconfig import ClientConfig
from temporalio.worker import Worker

from nomad_llm_extraction.config import DEFAULT_EXTRACTION_METADATA

with workflow.unsafe.imports_passed_through():
    from datetime import timedelta
    from typing import Any

    from nomad_llm_extraction.pipeline import BASE_ACTIVITIES, BASE_WORKFLOWS
    from nomad_llm_extraction.pipeline.activities import (
        UploadToNomadInput,
        upload_to_nomad,
    )
    from nomad_llm_extraction.pipeline.workflows import (
        GeneralExtractionWorkflow,
        GeneralExtractionWorkflowInput,
    )
    from nomad_llm_extraction.utils.utils import load_yaml_config


TEMPORAL_CONFIG_PATH = os.getenv('TEMPORAL_CONFIG_PATH', 'temporal.toml')


async def start_worker():
    temporal_client_config = ClientConfig.load_client_connect_config(
        config_file=TEMPORAL_CONFIG_PATH
    )
    client = await Client.connect(**temporal_client_config)
    worker = Worker(
        client,
        task_queue='general_extraction_pipeline',
        workflows=BASE_WORKFLOWS,
        activities=BASE_ACTIVITIES,
        activity_executor=ThreadPoolExecutor(
            max_workers=10
        ),  # Adjust max_workers as needed
    )
    worker_task = asyncio.create_task(worker.run())
    return client, worker, worker_task


async def run_extraction_workflow(config: GeneralExtractionWorkflowInput):
    client, worker, worker_task = await start_worker()
    result = await client.execute_workflow(
        GeneralExtractionWorkflow.run,
        config,
        id='general-extraction-workflow',
        task_queue='general_extraction_pipeline',
    )
    worker_task.cancel()  # Cleanly shut down the worker after workflow completion
    return result


def extract(config: str | dict[str, Any] | GeneralExtractionWorkflowInput):
    if isinstance(config, str):
        config = load_yaml_config(config)
    if isinstance(config, dict):
        # config = dacite.from_dict(
        #     data_class=GeneralExtractionWorkflowInput, data=config
        # )
        config = GeneralExtractionWorkflowInput(**config)
    print('Starting extraction workflow with config:')
    print(config)
    result = asyncio.run(run_extraction_workflow(config))
    if result.err_message is not None:
        print(f'Error during extraction: {result.err_message}')
    else:
        print('Extraction successful')
    return result


async def run_upload_to_nomad(nomad_upload: UploadToNomadInput):
    client, worker, worker_task = await start_worker()
    result = await client.execute_activity(
        upload_to_nomad,
        nomad_upload,
        id='upload-to-nomad',
        task_queue='general_extraction_pipeline',
        start_to_close_timeout=timedelta(seconds=30),
    )
    worker_task.cancel()  # Cleanly shut down the worker after workflow completion
    return result


def upload_extraction_to_nomad(
    extraction_result: dict[str, Any], nomad_upload_config: dict[str, Any]
):
    nomad_upload_input = UploadToNomadInput(
        m_def=nomad_upload_config['m_def'],
        data=extraction_result,
        entry_name=nomad_upload_config['entry_name'],
        doi=nomad_upload_config.get('doi'),
        extraction_metadata=nomad_upload_config.get(
            'extraction_metadata', DEFAULT_EXTRACTION_METADATA
        ),
        multi_instance_field=nomad_upload_config.get('multi_instance_field'),
        upload_id=nomad_upload_config.get('upload_id'),
    )
    result = asyncio.run(run_upload_to_nomad(nomad_upload_input))
    if result is not None:
        print('Upload to Nomad successful')
    else:
        print('Failed to upload to Nomad')
    return result
