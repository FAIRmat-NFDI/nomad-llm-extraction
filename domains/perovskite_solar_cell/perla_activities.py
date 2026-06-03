from temporalio import activity, workflow

with workflow.unsafe.imports_passed_through():
    from dataclasses import dataclass
    from datetime import timedelta
    from typing import Any

    from post_proc_pipeline import build_pipeline
    from pre_proc_schema import get_schema
    from validator import filter_unwanted
proc = build_pipeline()


@activity.defn
async def optimize_extraction_schema(
    schema: dict[str, Any], multi_instance_field: str
) -> dict[str, Any]:
    optimized_schema = get_schema(schema, multi_instance_field=multi_instance_field)
    return optimized_schema


@dataclass
class PostProcessingInput:
    data: dict[str, Any]
    schema: dict[str, Any]


@activity.defn
async def postprocessor(inp: PostProcessingInput) -> dict[str, Any]:
    cells = (
        inp.data.get('cells', [inp.data]) if isinstance(inp.data, dict) else inp.data
    )
    return {'cells': proc.apply(cells, inp.schema)}


@dataclass
class FilteringInput:
    data: dict[str, Any]
    text: str


@activity.defn
async def filter_extraction(inp: FilteringInput) -> dict[str, Any]:
    filtered_data = filter_unwanted(inp.data, inp.text)
    return filtered_data


@dataclass
class PerlaPostProcessingWorkflowInput:
    data: dict[str, Any]
    schema: dict[str, Any]
    text: str


@workflow.defn
class PerlaPostProcessingWorkflow:
    @workflow.run
    async def run(self, inp: PerlaPostProcessingWorkflowInput) -> dict[str, Any]:
        filtered = await workflow.execute_activity(
            filter_extraction,
            FilteringInput(data=inp.data, text=inp.text),
            start_to_close_timeout=timedelta(seconds=30),
        )
        postprocessed = await workflow.execute_activity(
            postprocessor,
            PostProcessingInput(data=filtered, schema=inp.schema),
            start_to_close_timeout=timedelta(seconds=30),
        )

        return postprocessed
