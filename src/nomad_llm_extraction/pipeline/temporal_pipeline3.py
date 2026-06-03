import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from temporalio import workflow

from nomad_llm_extraction.pipeline.models import (
    PipelineResult,
    StageContext,
    StageFunc,
    StageResult,
)

# from nomad_llm_extraction.pipeline.registry_funcs import (
#     get_registered_processor_func,
#     get_registered_stage_func,
# )


logger = logging.getLogger(__name__)
with workflow.unsafe.imports_passed_through():
    from nomad_llm_extraction.pipeline.registry_funcs import (
        get_registered_processor_func,
        get_registered_stage_func,
    )
    from nomad_llm_extraction.utils.utils import get_temporal_activities


# def get_stage_functions(stages: OrderedDict[str, str]) -> dict[str, StageFunc]:
#     stage_funcs = {
#         stage_name: get_registered_stage_func(stage_func_name)
#         for stage_name, stage_func_name in stages.items()
#     }
#     return stage_funcs


def get_stage_functions(stages: list[tuple[str, str]]) -> list[tuple[str, StageFunc]]:
    stage_funcs = [
        (stage_name, get_registered_stage_func(stage_func_name))
        for stage_name, stage_func_name in stages
    ]
    return stage_funcs


@dataclass
class PipelineWorkflowInput:
    stages: list[tuple[str, str]]
    ctx: Any = None
    result_postprocessor_ref: str = field(default='default_result_postprocessor')
    result_postprocessor_config: dict[str, Any] | None = None


@workflow.defn(name='pipeline_workflow')
class PipelineWorkflow:
    @workflow.run
    async def run(self, workflow_input: PipelineWorkflowInput) -> PipelineResult:
        """Run all stages in order, returning a list of :class:`StageResult`.

        Returns early after the first failing stage.  Before- and after-hooks
        for the failing stage are still fired.
        """
        print(f'Workflow input stages: {workflow_input.stages}')
        stages = get_stage_functions(workflow_input.stages)
        print(f'Resolved stage functions: {stages}')
        stages = get_temporal_activities(stages)
        print(f'Temporal activities for stages: {stages}')
        ctx = workflow_input.ctx
        print(type(ctx))
        stage_results = []
        failed = None
        result_postprocessor = get_registered_processor_func(
            workflow_input.result_postprocessor_ref
        )
        for stage_name, stage_func in stages:
            print(f'Running stage: {stage_name} with function: {stage_func}')
            out: tuple[StageResult, StageContext] = await workflow.execute_activity(
                stage_func,
                args=[ctx, stage_name],
                start_to_close_timeout=timedelta(seconds=30),
            )
            stage_result, ctx = out
            stage_results.append(stage_result)

            if not stage_result.success:
                logger.error(
                    f"Stage '{stage_name}' failed with error: {stage_result.error}"
                )
                failed = stage_result
                break
            logger.info(f"Stage '{stage_name}' succeeded")

        return result_postprocessor(ctx, stage_results, failed)
