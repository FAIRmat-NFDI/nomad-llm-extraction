"""Public ExtractionPipeline class with dependency injection.

Usage::

    from nomad_llm_extraction.pipeline import ExtractionPipeline, PromptConfig

    pipeline = ExtractionPipeline(
        engine=my_engine,
        extraction_schema_source=my_extraction_schema_source,
        postprocessing_schema_source=my_postprocessing_schema_source,
        prompt_config=PromptConfig(system_prompt='...', instruction_text='...'),
    )
    result = pipeline.run(paper_text)
    if result.success:
        print(result.extracted_data)

Stage hooks can be registered at construction time::

    def log_schema(ctx):
        print('Extraction schema loaded:', ctx.extraction_schema)

    pipeline = ExtractionPipeline(
        engine=my_engine,
        extraction_schema_source=my_extraction_schema_source,
        postprocessing_schema_source=my_postprocessing_schema_source,
        stage_hooks=[('extraction_schema_load', 'after', log_schema)],
    )
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Literal, Protocol, runtime_checkable

from temporalio import workflow

from nomad_llm_extraction.pipeline.models import (
    DataclassInstance,
    DefaultStageContext,
    ExtractionPipelineResult,
    PipelineResult,
    PromptConfig,
    Stage,
    StageContext,
    StageResult,
)
from nomad_llm_extraction.pipeline.stages import (
    LLMCallStage,
    ParseResponseStage,
    PostprocessingStage,
    PromptBuildStage,
    StageHook,
    StageRunner,
    ValidationStage,
)

logger = logging.getLogger(__name__)

# Type alias for stage hook tuples supplied at construction time.
StageHookSpec = tuple[str, Literal['before', 'after'], StageHook]


@runtime_checkable
class SchemaSource(Protocol):
    """Provider that returns a JSON-schema dict used to constrain LLM output."""

    def get_schema(self) -> dict[str, Any]: ...


@runtime_checkable
class LLMEngine(Protocol):
    """Minimal interface for an LLM backend."""

    def generate(
        self, prompt: str, json_schema: dict[str, Any], optional_params: dict = {}
    ) -> str: ...


class ExtractionPipeline_old:
    """Pipeline that extracts structured data from plain text using an LLM.

    All heavy dependencies are injected so that the pipeline can be used
    with different backends, schema sources, validators, and visualizers
    without subclassing.

    Stages are executed by a :class:`~nomad_llm_extraction.pipeline.stages.StageRunner`
    in the following order:

    1. ``extraction_schema_load``    — loads extraction schema.
    2. ``postprocessing_schema_load`` — loads postprocessing schema.
    3. ``prompt_build``              — assembles the LLM prompt.
    4. ``llm_extraction``            — calls the LLM engine.
    5. ``json_parse``                — parses the raw JSON response.
    6. ``validation``                — runs validators (non-aborting; failures recorded).
    7. ``postprocessing``            — applies optional postprocessor to extracted data.
    8. ``archive_shaping``           — applies optional archive shaper to postprocessed data.

    Args:
        engine: LLM backend implementing ``generate(prompt, json_schema) -> str``.
        extraction_schema_source: Object with ``get_schema() -> dict`` for
            prompt construction and LLM extraction constraints.
        postprocessing_schema_source: Object with ``get_schema() -> dict`` passed
            to the postprocessor so postprocessing can use a separate schema.
        prompt_config: System prompt and instruction text.
        validators: Optional list of callables ``(extracted_data) -> None`` that
            may raise to signal validation failure.  Failures are recorded in
            the ``'validation'`` stage result and ``ctx.metadata``, but do not
            abort subsequent stages.
        visualizers: Optional list of callables ``(result: PipelineResult) ->
            None`` invoked after the pipeline completes regardless of success.
        stage_hooks: Optional list of ``(stage_name, when, hook)`` tuples.
            Each hook is registered with the internal :class:`StageRunner` so
            that it fires before or after the named stage.  This is the primary
            extension point for validation, logging, and visualisation hooks
            that are tied to a specific stage rather than the overall result.
        postprocessor: Optional callable
            ``(extracted_data, postprocessing_schema) -> postprocessed``
            applied to parsed data during the ``postprocessing`` stage.
        archive_shaper: Optional callable ``(postprocessed_data) -> archive``
            applied to postprocessed data during the ``archive_shaping`` stage.
    """

    def __init__(
        self,
        engine: LLMEngine,
        extraction_schema: dict[str, Any],
        postprocessing_schema: dict[str, Any] | None = None,
        prompt_config: PromptConfig | None = None,
        validators: list[Any] | None = None,
        visualizers: list[Any] | None = None,
        stage_hooks: list[StageHookSpec] | None = None,
        postprocessor: Callable[[Any, dict[str, Any] | None], Any] | None = None,
        archive_shaper: Callable[[Any], Any] | None = None,
    ) -> None:
        self.engine = engine
        self.extraction_schema = extraction_schema
        self.postprocessing_schema = postprocessing_schema
        self.prompt_config = prompt_config or PromptConfig()
        self.validators = validators or []
        self.visualizers = visualizers or []
        self._stage_hooks: list[StageHookSpec] = stage_hooks or []
        self.postprocessor = postprocessor
        self.archive_shaper = archive_shaper

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, text: str) -> ExtractionPipelineResult:
        """Run the full extraction pipeline on *text*.

        Returns a :class:`PipelineResult` regardless of whether individual
        stages succeed or fail — exceptions are captured and stored in the
        result rather than propagating to the caller.

        Visualizers are called after the pipeline completes regardless of
        success or failure.
        """
        ctx = StageContext(
            extraction_schema=self.extraction_schema,
            postprocessing_schema=self.postprocessing_schema,
            text=text,
        )
        runner = self._build_runner()
        stage_results = runner.run(ctx)

        # Determine the first failed stage (if any) for error reporting.
        failed = next((r for r in stage_results if not r.success), None)
        if failed is not None:
            result = ExtractionPipelineResult(
                success=False,
                raw_llm_output=ctx.raw_output,
                stages=stage_results,
                error=failed.error,
            )
            self._run_visualizers(result)
            return result

        result = ExtractionPipelineResult(
            success=True,
            raw_llm_output=ctx.raw_output,
            extracted_data=ctx.extracted_data,
            postprocessed_data=ctx.postprocessed_data,
            archive_data=ctx.archive_data,
            stages=stage_results,
        )
        self._run_visualizers(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_runner(self) -> StageRunner:
        """Construct a :class:`StageRunner` wired with all pipeline stages and hooks."""
        runner = StageRunner()
        # runner.add_stage(SchemaLoadStage(self.extraction_schema_source, 'extraction'))
        # runner.add_stage(SchemaLoadStage(self.postprocessing_schema_source, 'postprocessing'))
        runner.add_stage(PromptBuildStage(self.prompt_config))
        runner.add_stage(LLMCallStage(self.engine))
        runner.add_stage(ParseResponseStage())
        runner.add_stage(ValidationStage(self.validators))
        runner.add_stage(PostprocessingStage(self.postprocessor))
        # runner.add_stage(ExportToNOMADStage(self.exporter))
        for stage_name, when, hook in self._stage_hooks:
            runner.add_hook(stage_name, when, hook)
        return runner

    def _run_visualizers(self, result: ExtractionPipelineResult) -> None:
        for visualizer in self.visualizers:
            try:
                visualizer(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning('Visualizer %r raised: %s', visualizer, exc)


class Pipeline:
    def __init__(
        self,
        stages: list[Stage] | None = None,
        stage_hooks: list[StageHookSpec] | None = None,
        visualizers: list[Callable[[PipelineResult], None]] | None = None,
        ctx_factory: type[DataclassInstance] | None = None,
    ):
        self.stages = stages or []
        self.stage_hooks = stage_hooks or []
        self.ctx_factory = ctx_factory or DefaultStageContext
        self.visualizers = visualizers or []

    def _build_runner(self) -> StageRunner:
        runner = StageRunner()
        for stage in self.stages:
            runner.add_stage(stage)
        for stage_name, when, hook in self.stage_hooks:
            runner.add_hook(stage_name, when, hook)
        return runner

    def get_temporal_activities(self) -> dict[str, Callable[..., Any]]:
        from temporalio import activity

        temporal_activities = {}
        for stage in self.stages:
            for func in stage.__dict__.values():
                if hasattr(func, '__name__'):
                    wrapped_activity = activity.defn(name=func.__name__)(func)
                    temporal_activities[func.__name__] = wrapped_activity
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


@workflow.defn(name='pipeline_workflow')
class PipelineWorkflow:
    @workflow.run
    async def run(self, pipeline: Pipeline, ctx=None) -> PipelineResult:
        """Run all stages in order, returning a list of :class:`StageResult`.

        Returns early after the first failing stage.  Before- and after-hooks
        for the failing stage are still fired.
        """
        return pipeline.run(ctx)


class ExtractionPipeline(Pipeline):
    def __init__(
        self,
        engine: LLMEngine,
        extraction_schema: dict[str, Any],
        postprocessing_schema: dict[str, Any] | None = None,
        prompt_config: PromptConfig | None = None,
        validators: list[Any] | None = None,
        visualizers: list[Any] | None = None,
        stage_hooks: list[StageHookSpec] | None = None,
        postprocessor: Callable[[Any, dict[str, Any] | None], Any] | None = None,
        archive_shaper: Callable[[Any], Any] | None = None,
    ) -> None:
        self.engine = engine
        self.extraction_schema = extraction_schema
        self.postprocessing_schema = postprocessing_schema
        self.prompt_config = prompt_config or PromptConfig()
        self.validators = validators or []
        self.visualizers = visualizers or []
        self._stage_hooks: list[StageHookSpec] = stage_hooks or []
        self.postprocessor = postprocessor
        self.archive_shaper = archive_shaper

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
            extraction_schema=self.extraction_schema,
            postprocessing_schema=self.postprocessing_schema,
            text=text,
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
            )
        else:
            result = ExtractionPipelineResult(
                success=True,
                raw_llm_output=ctx.raw_output,
                extracted_data=ctx.extracted_data,
                postprocessed_data=ctx.postprocessed_data,
                archive_data=ctx.archive_data,
                stages=stage_results,
            )
        return result

    def _build_runner(self) -> StageRunner:
        """Construct a :class:`StageRunner` wired with all pipeline stages and hooks."""
        runner = StageRunner()
        runner.add_stage(PromptBuildStage(self.prompt_config))
        runner.add_stage(LLMCallStage(self.engine))
        runner.add_stage(ParseResponseStage())
        runner.add_stage(ValidationStage(self.validators))
        runner.add_stage(PostprocessingStage(self.postprocessor))
        for stage_name, when, hook in self._stage_hooks:
            runner.add_hook(stage_name, when, hook)
        return runner
