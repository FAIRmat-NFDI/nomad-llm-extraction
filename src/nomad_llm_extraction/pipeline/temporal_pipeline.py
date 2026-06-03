# @runtime_checkable
# class StageFunc(Protocol):
#     def __call__(self, ctx, stage_name: str) -> StageResult: ...
from __future__ import annotations

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    import logging
    from collections.abc import Callable
    from dataclasses import asdict, dataclass, field
    from typing import Any

    from nomad_llm_extraction.pipeline.models import (
        DataclassInstance,
        DefaultStageContext,
        ExtractionPipelineResult,
        LLMEngine,
        PipelineResult,
        PromptConfig,
        StageContext,
        StageFunc,
        StageHookSpec,
        StageResult,
    )
    from nomad_llm_extraction.pipeline.stages import (
        build_prompt,
        filter_extraction,
        json_parse,
        llm_call,
        run_postprocessing,
        validate_extraction_with_schema,
    )
    from nomad_llm_extraction.utils.utils import (
        get_temporal_activities,
        verify_activity_signature,
    )

    logger = logging.getLogger(__name__)
    from datetime import timedelta


@dataclass
class PipelineWorkflowInput:
    pipeline: TemporalPipeline
    ctx_args: dict[str, Any] = field(default_factory=dict)


class TemporalPipeline:
    def __init__(
        self,
        stages: dict[str, StageFunc] | None = None,
        ctx_factory: type[DataclassInstance] | None = None,
    ):
        self.stages = stages or {}
        for stage_name, stage_func in self.stages.items():
            verify_activity_signature(
                stage_func, expected_params={'ctx': StageContext, 'stage_name': str}
            )
        self.ctx_factory = ctx_factory or DefaultStageContext

    def get_ctx(self, ctx_args: dict[str, Any]) -> Any:
        return self.ctx_factory(**ctx_args)

    def get_result(self, ctx, stage_results, failed) -> PipelineResult:
        if failed is not None:
            result = PipelineResult(
                success=False,
                stages=stage_results,
                error=failed.error,
                ctx=asdict(ctx),
            )
        else:
            result = PipelineResult(
                success=True,
                stages=stage_results,
                ctx=asdict(ctx),
            )
        return result

    def get_stage_activities(self) -> dict[str, Callable[..., Any]]:
        return get_temporal_activities(self.stages)


class TemporalExtractionPipeline(TemporalPipeline):
    def __init__(
        self,
        engine: LLMEngine,
        extraction_schema: dict[str, Any],
        postprocessing_schema: dict[str, Any] | None = None,
        prompt_config: PromptConfig | None = None,
        postprocessor: Callable[[Any, dict[str, Any] | None], Any] | None = None,
        postprocessor_args: list[str] | None = None,
        filter_func: Callable[..., Any] | None = None,
        filter_args: list[str] | None = None,
    ) -> None:
        self.engine = engine
        self.extraction_schema = extraction_schema
        self.postprocessing_schema = postprocessing_schema
        self.prompt_config = prompt_config or PromptConfig()
        self.postprocessor = postprocessor
        self.postprocessor_args = postprocessor_args
        self.filter_func = filter_func
        self.filter_args = filter_args
        self.stages = {
            'build_prompt': build_prompt,
            'llm_call': llm_call,
            'parse_response': json_parse,
            'validate_extraction_with_schema': validate_extraction_with_schema,
            'filtering': filter_extraction,
            'postprocessing': run_postprocessing,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_ctx(self, ctx_args: dict[str, Any]):
        """Create a stage context for the extraction pipeline.

        Args:
            text: The input text to process.

        Returns:
            A :class:`StageContext` instance.
        """
        return StageContext(
            text=ctx_args.get('text', ''),
            engine=self.engine,
            extraction_schema=self.extraction_schema,
            postprocessing_schema=self.postprocessing_schema,
            postprocessor=self.postprocessor,
            postprocessor_args=self.postprocessor_args,
            filter_func=self.filter_func,
            filter_args=self.filter_args,
            prompt_config=self.prompt_config,
        )

    def get_result(self, ctx, stage_results, failed) -> ExtractionPipelineResult:
        if failed is not None:
            result = ExtractionPipelineResult(
                success=False,
                raw_llm_output=ctx.raw_output,
                stages=stage_results,
                error=failed.error,
                ctx=asdict(ctx),
            )
        else:
            result = ExtractionPipelineResult(
                success=True,
                raw_llm_output=ctx.raw_output,
                extracted_data=ctx.extracted_data,
                postprocessed_data=ctx.postprocessed_data,
                archive_data=ctx.archive_data,
                stages=stage_results,
                ctx=asdict(ctx),
            )
        return result


@workflow.defn(name='pipeline_workflow')
class PipelineWorkflow:
    @workflow.run
    async def run(self, input: PipelineWorkflowInput) -> PipelineResult:
        """Run all stages in order, returning a list of :class:`StageResult`.

        Returns early after the first failing stage.  Before- and after-hooks
        for the failing stage are still fired.
        """
        pipeline = input.pipeline
        ctx_args = input.ctx_args
        stage_results = []
        failed = None
        stages = pipeline.get_stage_activities()
        ctx = pipeline.get_ctx(ctx_args)
        for stage_name, stage_func in stages.items():
            stage_result: StageResult = await workflow.execute_activity(
                stage_func,
                arg=ctx,
                start_to_close_timeout=timedelta(seconds=30),
            )
            stage_results.append(stage_result)

            if not stage_result.success:
                logger.error(
                    f"Stage '{stage_name}' failed with error: {stage_result.error}"
                )
                failed = stage_result
                break
            logger.info(f"Stage '{stage_name}' succeeded")
        return pipeline.get_result(ctx, stage_results, failed)
