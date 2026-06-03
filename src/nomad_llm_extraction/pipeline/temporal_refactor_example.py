from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from temporalio.client import Client
from temporalio.worker import Worker

from nomad_llm_extraction.pipeline.temporal_refactor_pipeline import (
    RegistryPipelineWorkflow,
    build_extraction_workflow_payload,
    run_registered_pipeline_stage,
)
from nomad_llm_extraction.pipeline.temporal_refactor_registry import (
    register_engine_factory,
    register_runtime_callable_factory,
)

TASK_QUEUE = 'registry-pipeline-task-queue'


class DemoEngine:
    """Simple deterministic engine used by the Temporal refactor example."""

    def __init__(self, material: str = 'Si') -> None:
        self.material = material

    def generate(
        self, prompt: str, json_schema: dict[str, Any], optional_params: dict = {}
    ) -> str:
        # In a real engine this would call an LLM. Here we emit valid JSON.
        _ = prompt
        _ = json_schema
        _ = optional_params
        return '{"material": "%s", "efficiency": 0.21}' % self.material


def build_demo_engine(config: dict[str, Any]) -> DemoEngine:
    return DemoEngine(material=config.get('material', 'Si'))


def build_demo_filter(config: dict[str, Any]):
    min_eff = float(config.get('min_efficiency', 0.0))

    def _filter(extracted_data: dict[str, Any] | None) -> dict[str, Any]:
        data = dict(extracted_data or {})
        if float(data.get('efficiency', 0.0)) < min_eff:
            data['below_threshold'] = True
        return data

    return _filter


def build_demo_postprocessor(config: dict[str, Any]):
    label = str(config.get('label', 'demo'))

    def _postprocess(
        filtered_data: dict[str, Any] | None,
        postprocessing_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        data = dict(filtered_data or {})
        data['label'] = label
        data['postprocessing_schema_seen'] = postprocessing_schema is not None
        return data

    return _postprocess


def register_demo_runtime_components() -> None:
    register_engine_factory('demo_engine', build_demo_engine)
    register_runtime_callable_factory('demo_filter', build_demo_filter)
    register_runtime_callable_factory('demo_postprocessor', build_demo_postprocessor)


async def run_worker(server_url: str) -> None:
    register_demo_runtime_components()
    client = await Client.connect(server_url)
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[RegistryPipelineWorkflow],
        activities=[run_registered_pipeline_stage],
        activity_executor=ThreadPoolExecutor(max_workers=4),
    )
    await worker.run()


async def run_once(server_url: str) -> None:
    client = await Client.connect(server_url)

    extraction_schema = {
        'type': 'object',
        'properties': {
            'material': {'type': 'string'},
            'efficiency': {'type': 'number'},
        },
        'required': ['material', 'efficiency'],
        'additionalProperties': True,
    }

    payload = build_extraction_workflow_payload(
        text='A demo paper reports a silicon solar cell.',
        extraction_schema=extraction_schema,
        engine_ref='demo_engine',
        engine_config={'material': 'Si'},
        filter_ref='demo_filter',
        filter_config={'min_efficiency': 0.2},
        filter_args=['extracted_data'],
        postprocessor_ref='demo_postprocessor',
        postprocessor_config={'label': 'temporal-refactor'},
        postprocessor_args=['filtered_data', 'postprocessing_schema'],
        postprocessing_schema={'type': 'object'},
    )

    result = await client.execute_workflow(
        RegistryPipelineWorkflow.run,
        payload,
        id='registry-pipeline-example-run',
        task_queue=TASK_QUEUE,
    )
    print(result.model_dump_json(indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Demo for registry-based Temporal pipeline workflow.'
    )
    parser.add_argument(
        'mode',
        choices=['worker', 'run'],
        help='Use worker to host activities/workflow, run to execute one workflow.',
    )
    parser.add_argument(
        '--server-url',
        default='localhost:7233',
        help='Temporal server URL.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == 'worker':
        asyncio.run(run_worker(args.server_url))
        return

    if args.mode == 'run':
        asyncio.run(run_once(args.server_url))
        return

    raise ValueError(f'Unsupported mode: {args.mode}')


if __name__ == '__main__':
    main()
