"""Public ExtractionPipeline class with dependency injection.

Usage::

    from nomad_llm_extraction.pipeline import ExtractionPipeline, PromptConfig

    pipeline = ExtractionPipeline(
        engine=my_engine,
        schema_source=my_schema_source,
        prompt_config=PromptConfig(system_prompt='...', instruction_text='...'),
    )
    result = pipeline.run(paper_text)
    if result.success:
        print(result.extracted_data)

Stage hooks can be registered at construction time::

    def log_schema(ctx):
        print('Schema loaded:', ctx.schema)

    pipeline = ExtractionPipeline(
        engine=my_engine,
        schema_source=my_schema_source,
        stage_hooks=[('schema_load', 'after', log_schema)],
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from nomad_llm_extraction.pipeline.models import (
    PipelineResult,
    PromptConfig,
    StageResult,
)
from nomad_llm_extraction.pipeline.stages import (
    ArchiveShapingStage,
    LLMCallStage,
    ParseResponseStage,
    PostprocessingStage,
    PromptBuildStage,
    SchemaLoadStage,
    SchemaResolveStage,
    StageContext,
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


class ExtractionPipeline:
    """Pipeline that extracts structured data from plain text using an LLM.

    All heavy dependencies are injected so that the pipeline can be used
    with different backends, schema sources, validators, and visualizers
    without subclassing.

    Stages are executed by a :class:`~nomad_llm_extraction.pipeline.stages.StageRunner`
    in the following order:

    1. ``schema_load``    — fetches the JSON schema from the schema source.
    2. ``schema_resolve`` — applies optional schema resolver/optimizer.
    3. ``prompt_build``   — assembles the LLM prompt.
    4. ``llm_extraction`` — calls the LLM engine.
    5. ``json_parse``     — parses the raw JSON response.
    6. ``validation``     — runs validators (non-aborting; failures recorded).
    7. ``postprocessing`` — applies optional postprocessor to extracted data.
    8. ``archive_shaping``— applies optional archive shaper to postprocessed data.

    Args:
        engine: LLM backend implementing ``generate(prompt, json_schema) -> str``.
        schema_source: Object with ``get_schema() -> dict`` that returns the
            JSON schema constraining the LLM output.
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
        schema_resolver: Optional callable ``(schema: dict) -> dict`` applied
            to the loaded schema during the ``schema_resolve`` stage.
        postprocessor: Optional callable ``(extracted_data) -> postprocessed``
            applied to parsed data during the ``postprocessing`` stage.
        archive_shaper: Optional callable ``(postprocessed_data) -> archive``
            applied to postprocessed data during the ``archive_shaping`` stage.
    """

    def __init__(
        self,
        engine: LLMEngine,
        schema_source: SchemaSource,
        prompt_config: PromptConfig | None = None,
        validators: list[Any] | None = None,
        visualizers: list[Any] | None = None,
        stage_hooks: list[StageHookSpec] | None = None,
        schema_resolver: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        postprocessor: Callable[[Any], Any] | None = None,
        archive_shaper: Callable[[Any], Any] | None = None,
    ) -> None:
        self.engine = engine
        self.schema_source = schema_source
        self.prompt_config = prompt_config or PromptConfig()
        self.validators = validators or []
        self.visualizers = visualizers or []
        self._stage_hooks: list[StageHookSpec] = stage_hooks or []
        self.schema_resolver = schema_resolver
        self.postprocessor = postprocessor
        self.archive_shaper = archive_shaper

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, text: str) -> PipelineResult:
        """Run the full extraction pipeline on *text*.

        Returns a :class:`PipelineResult` regardless of whether individual
        stages succeed or fail — exceptions are captured and stored in the
        result rather than propagating to the caller.

        Visualizers are called after the pipeline completes regardless of
        success or failure.
        """
        ctx = StageContext(text=text)
        runner = self._build_runner()
        stage_results = runner.run(ctx)

        # Determine the first failed stage (if any) for error reporting.
        failed = next((r for r in stage_results if not r.success), None)
        if failed is not None:
            result = PipelineResult(
                success=False,
                raw_llm_output=ctx.raw_output,
                stages=stage_results,
                error=failed.error,
            )
            self._run_visualizers(result)
            return result

        result = PipelineResult(
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
        runner.add_stage(SchemaLoadStage(self.schema_source))
        runner.add_stage(SchemaResolveStage(self.schema_resolver))
        runner.add_stage(PromptBuildStage(self.prompt_config))
        runner.add_stage(LLMCallStage(self.engine))
        runner.add_stage(ParseResponseStage())
        runner.add_stage(ValidationStage(self.validators))
        runner.add_stage(PostprocessingStage(self.postprocessor))
        runner.add_stage(ArchiveShapingStage(self.archive_shaper))
        for stage_name, when, hook in self._stage_hooks:
            runner.add_hook(stage_name, when, hook)
        return runner

    def _run_visualizers(self, result: PipelineResult) -> None:
        for visualizer in self.visualizers:
            try:
                visualizer(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning('Visualizer %r raised: %s', visualizer, exc)
