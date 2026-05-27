from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict
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
    StageRunner,
    build_prompt,
    filter_extraction,
    json_parse,
    llm_call,
    run_postprocessing,
    validate_extraction_with_schema,
)
from nomad_llm_extraction.utils.utils import verify_activity_signature

logger = logging.getLogger(__name__)


# @runtime_checkable
# class StageFunc(Protocol):
#     def __call__(self, ctx, stage_name: str) -> StageResult: ...


class Pipeline:
    def __init__(
        self,
        stages: dict[str, StageFunc] | None = None,
        stage_hooks: list[StageHookSpec] | None = None,
        visualizers: list[Callable[[PipelineResult], None]] | None = None,
        ctx_factory: type[DataclassInstance] | None = None,
    ):
        self.stages = stages or {}
        for stage_name, stage_func in self.stages.items():
            verify_activity_signature(
                stage_func, expected_params={'ctx': StageContext, 'stage_name': str}
            )
        self.stage_hooks = stage_hooks or []
        self.ctx_factory = ctx_factory or DefaultStageContext
        self.visualizers = visualizers or []

    def _build_runner(self) -> StageRunner:
        runner = StageRunner()
        for stage_name, stage_func in self.stages.items():
            runner.add_stage(stage_name, stage_func)
        for stage_name, when, hook in self.stage_hooks:
            runner.add_hook(stage_name, when, hook)
        return runner

    def get_temporal_activities(self) -> dict[str, Callable[..., Any]]:
        from temporalio import activity

        temporal_activities = {}
        for name, func in self.stages.items():
            wrapped_activity = activity.defn(name=name)(func)
            temporal_activities[name] = wrapped_activity
        return temporal_activities

    def _run(self, ctx) -> tuple[list[StageResult], StageResult | None]:
        runner = self._build_runner()
        stage_results = runner.run(ctx)
        failed = next((r for r in stage_results if not r.success), None)
        return stage_results, failed

    def get_result(self, ctx, stage_results, failed) -> PipelineResult:
        result = PipelineResult(success=True, stages=stage_results, ctx=asdict(ctx))
        if failed is not None:
            result.success = False
            result.error = failed.error
        return result

    def run(self, ctx=None) -> PipelineResult:
        ctx = ctx or self.ctx_factory()
        stage_results, failed = self._run(ctx)
        result = self.get_result(ctx, stage_results, failed)
        self._run_visualizers(result)
        return result

    def _run_visualizers(self, result: PipelineResult) -> None:
        for visualizer in self.visualizers:
            try:
                visualizer(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning('Visualizer %r raised: %s', visualizer, exc)


class ExtractionPipeline(Pipeline):
    def __init__(
        self,
        engine: LLMEngine,
        extraction_schema: dict[str, Any],
        postprocessing_schema: dict[str, Any] | None = None,
        prompt_config: PromptConfig | None = None,
        visualizers: list[Any] | None = None,
        stage_hooks: list[StageHookSpec] | None = None,
        postprocessor: Callable[[Any, dict[str, Any] | None], Any] | None = None,
        postprocessor_args: list[str] | None = None,
        filter_func: Callable[..., Any] | None = None,
        filter_args: list[str] | None = None,
    ) -> None:
        self.engine = engine
        self.extraction_schema = extraction_schema
        self.postprocessing_schema = postprocessing_schema
        self.prompt_config = prompt_config or PromptConfig()
        self.visualizers = visualizers or []
        self.stage_hooks: list[StageHookSpec] = stage_hooks or []
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
    def run(self, text: str | None = None) -> ExtractionPipelineResult:
        """Run the full extraction pipeline on *text*.

        Returns a :class:`PipelineResult` regardless of whether individual
        stages succeed or fail — exceptions are captured and stored in the
        result rather than propagating to the caller.

        Visualizers are called after the pipeline completes regardless of
        success or failure.
        """
        ctx = StageContext(
            text=text,
            engine=self.engine,
            extraction_schema=self.extraction_schema,
            postprocessing_schema=self.postprocessing_schema,
            postprocessor=self.postprocessor,
            postprocessor_args=self.postprocessor_args,
            filter_func=self.filter_func,
            filter_args=self.filter_args,
            prompt_config=self.prompt_config,
        )
        stage_results, failed = self._run(ctx)
        result = self.get_result(ctx, stage_results, failed)
        self._run_visualizers(result)
        return result

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
