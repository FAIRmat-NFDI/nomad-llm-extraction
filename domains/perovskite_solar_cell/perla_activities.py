from temporalio import activity, workflow

with workflow.unsafe.imports_passed_through():
    from dataclasses import dataclass
    from datetime import timedelta
    from typing import Any

    from post_proc_pipeline import build_pipeline
    from pre_proc_schema import get_schema
    from validator import filter_unwanted
proc = build_pipeline()

"""
This module defines Temporal activities and workflows for post-processing extracted data specifically for the PERLA project.
The main activities include:
1. optimize_extraction_schema: Optimizes the extraction schema for better performance during post-processing.
2. postprocessor: Applies a series of post-processing steps to the extracted data using the transformation pipeline defined in post_proc_pipeline.py.
3. filter_extraction: Filters out potentially hallucinated values from the extracted data by cross-referencing with the original PDF text using functions defined in validator.py.

For each activity, we define input dataclasses to structure the input data.

The PerlaPostProcessingWorkflow orchestrates the execution of the filtering and post-processing activities in sequence.
"""

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
