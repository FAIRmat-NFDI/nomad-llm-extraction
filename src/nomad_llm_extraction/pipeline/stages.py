"""Explicit, composable pipeline stages for the extraction pipeline.

Each stage encapsulates a single unit of work and communicates through a shared
:class:`StageContext`.  The :class:`StageRunner` executes stages in sequence,
fires registered before/after hooks around each stage, and short-circuits on
failure so that subsequent stages are skipped.

Typical usage (wired automatically by :class:`~nomad_llm_extraction.pipeline.extraction_pipeline.ExtractionPipeline`)::

    from nomad_llm_extraction.pipeline.stages import (
        StageContext,
        StageRunner,
        ExtractionSchemaLoadStage,
        PostprocessingSchemaLoadStage,
        PromptBuildStage,
        LLMCallStage,
        ParseResponseStage,
        ValidationStage,
        PostprocessingStage,
        ArchiveShapingStage,
    )

    ctx = StageContext(text='paper text')
    runner = StageRunner()
    runner.add_stage(ExtractionSchemaLoadStage(extraction_schema_source))
    runner.add_stage(PostprocessingSchemaLoadStage(postprocessing_schema_source))
    runner.add_stage(PromptBuildStage(my_prompt_config))
    runner.add_stage(LLMCallStage(my_engine))
    runner.add_stage(ParseResponseStage())
    runner.add_stage(ValidationStage(my_validators))
    runner.add_stage(PostprocessingStage(my_postprocessor))
    runner.add_stage(ArchiveShapingStage(my_shaper))
    runner.add_hook('llm_extraction', 'after', lambda ctx: print(ctx.raw_output))
    stage_results = runner.run(ctx)

Design notes
------------
* **Context as shared state**: stages mutate ``ctx`` rather than returning data;
  this keeps the stage interface minimal and lets hooks observe intermediate state.
* **Hook isolation**: hook exceptions are caught and logged so that a misbehaving
  hook never aborts the pipeline.
* **Short-circuit on failure**: once a stage returns ``success=False`` the runner
  stops and returns the results collected so far (after firing any after-hooks for
  the failing stage).
* **After-hooks fire even on failure** so that observers (e.g. loggers, visualizers)
  can react to both successful and failed stages.
* **Strict engine contract**: :class:`LLMCallStage` requires the engine to return a
  plain ``str``; no raw response-object normalisation is performed.
* **Non-aborting validation**: :class:`ValidationStage` always succeeds at the runner
  level so that postprocessing and archive shaping still run; validator failures are
  recorded in ``ctx.metadata['validation_errors']`` and ``StageResult.data``.
"""

from __future__ import annotations

import json
import logging
import traceback
from collections.abc import Callable
from typing import Any, Literal

from nomad_llm_extraction.pipeline.models import (
    PromptConfig,
    Stage,
    StageContext,
    StageResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for hook callables
# ---------------------------------------------------------------------------

StageHook = Callable[['StageContext'], None]


class StageRunner:
    """Executes a sequence of :class:`Stage` objects, firing hooks around each one.

    Hooks are registered per stage-name and timing (``'before'`` / ``'after'``).
    Multiple hooks may be attached to the same (stage, timing) pair; they are
    called in registration order.

    Execution stops after the first failing stage (i.e. ``StageResult.success
    is False``), but *after*-hooks for the failing stage are still invoked.
    """

    def __init__(self) -> None:
        self._stages: list[Stage] = []
        self._hooks: dict[str, dict[str, list[StageHook]]] = {}

    def add_stage(self, stage: Stage) -> None:
        """Append *stage* to the execution sequence."""
        self._stages.append(stage)

    def add_hook(
        self,
        stage_name: str,
        when: Literal['before', 'after'],
        hook: StageHook,
    ) -> None:
        """Register *hook* to be called before or after the named stage.

        If *stage_name* does not match any stage that will be run, the hook is
        silently ignored at runtime (no error at registration time).
        """
        self._hooks.setdefault(stage_name, {}).setdefault(when, []).append(hook)

    def run(self, ctx) -> list[StageResult]:
        """Run all stages in order, returning a list of :class:`StageResult`.

        Returns early after the first failing stage.  Before- and after-hooks
        for the failing stage are still fired.
        """
        results: list[StageResult] = []
        for stage in self._stages:
            self._fire_hooks(stage.name, 'before', ctx)
            try:
                result = stage.run(ctx)
            except Exception as exc:  # noqa: BLE001
                logger.error('Stage %r failed: %s', stage.name, exc)
                result = StageResult(name=stage.name, success=False, error=str(exc))
            results.append(result)
            self._fire_hooks(stage.name, 'after', ctx)
            if not result.success:
                break
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fire_hooks(
        self,
        stage_name: str,
        when: Literal['before', 'after'],
        ctx: StageContext,
    ) -> None:
        for hook in self._hooks.get(stage_name, {}).get(when, []):
            try:
                hook(ctx)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    'Stage hook %r (%s %r) raised: %s',
                    hook,
                    when,
                    stage_name,
                    exc,
                )


# ---------------------------------------------------------------------------
# Concrete stage implementations
# ---------------------------------------------------------------------------


class SchemaLoadStage:
    name: str = 'schema_load'

    def __init__(self, schema_source: Any, schema_type: str = 'extraction') -> None:
        self._source = schema_source
        self._schema_type = schema_type

    def run(self, ctx: StageContext) -> StageResult:
        try:
            schema = self._source.get_schema()
            if self._schema_type == 'extraction':
                ctx.extraction_schema = schema
            elif self._schema_type == 'postprocessing':
                ctx.postprocessing_schema = schema
            else:
                raise ValueError(f'Unknown schema type: {self._schema_type}')
            return StageResult(
                name=f'{self.name} for {self._schema_type}', success=True, data=schema
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            print(traceback.format_exc())
            logger.error('Schema load failed: %s', msg)
            return StageResult(
                name=f'{self.name} for {self._schema_type}', success=False, error=msg
            )


class ExtractionSchemaLoadStage:
    """Fetch the extraction JSON schema from a schema source and store it in context.

    On success, ``ctx.extraction_schema`` is populated.
    """

    name: str = 'extraction_schema_load'

    def __init__(self, schema_source: Any) -> None:
        self._source = schema_source

    def run(self, ctx: StageContext) -> StageResult:
        try:
            schema = self._source.get_schema()
            ctx.extraction_schema = schema
            return StageResult(name=self.name, success=True, data=schema)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            logger.error('Schema load failed: %s', msg)
            return StageResult(name=self.name, success=False, error=msg)


class PostprocessingSchemaLoadStage:
    """Fetch the postprocessing JSON schema from a schema source and store it in context.

    On success, ``ctx.postprocessing_schema`` is populated.
    """

    name: str = 'postprocessing_schema_load'

    def __init__(self, schema_source: Any) -> None:
        self._source = schema_source

    def run(self, ctx: StageContext) -> StageResult:
        try:
            schema = self._source.get_schema()
            ctx.postprocessing_schema = schema
            return StageResult(name=self.name, success=True, data=schema)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            logger.error('Postprocessing schema load failed: %s', msg)
            return StageResult(name=self.name, success=False, error=msg)


class PromptBuildStage:
    """Assemble the LLM prompt from text, schema, and prompt configuration.

    Requires ``ctx.extraction_schema`` to be set (i.e.
    :class:`ExtractionSchemaLoadStage` must have
    run successfully before this stage).  On success, ``ctx.prompt`` is populated.
    """

    name: str = 'prompt_build'

    def __init__(self, prompt_config: PromptConfig | None = None) -> None:
        self._prompt_config = prompt_config or PromptConfig()

    def run(self, ctx: StageContext) -> StageResult:
        if ctx.extraction_schema is None:
            return StageResult(
                name=self.name,
                success=False,
                error='Cannot build prompt: extraction schema not loaded',
            )
        try:
            prompt = self._build_prompt(ctx.text, ctx.extraction_schema)
            ctx.prompt = prompt
            return StageResult(name=self.name, data=prompt, success=True)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            logger.error('Prompt build failed: %s', msg)
            return StageResult(name=self.name, success=False, error=msg)

    def _build_prompt(self, text: str, schema: dict[str, Any]) -> str:
        parts: list[str] = []
        if self._prompt_config.system_prompt:
            parts.append(self._prompt_config.system_prompt)
        if self._prompt_config.instruction_text:
            parts.append(self._prompt_config.instruction_text)
        parts.append(f'Here is the schema: {json.dumps(schema, indent=2)}')
        parts.append(f'Here is the text:\n{text}')
        return '\n'.join(parts)


class LLMCallStage:
    """Invoke the LLM engine and store the raw JSON string in context.

    Requires both ``ctx.prompt`` and ``ctx.extraction_schema`` to be set.  On success,
    ``ctx.raw_output`` holds the JSON string returned by the engine.

    The engine's ``generate`` method **must** return a plain ``str``.  If it
    returns any other type the stage fails immediately without attempting
    normalisation.  This enforces the strict engine boundary: adapters and
    wrappers (e.g. :class:`~nomad_llm_extraction.pipeline.schema_filling.llm_engine.LiteLLMEngine`)
    are responsible for extracting the content string before returning.
    """

    name: str = 'llm_extraction'

    def __init__(
        self,
        engine: Any,
        optional_params: dict[str, Any] | None = None,
    ) -> None:
        self._engine = engine
        self._optional_params = optional_params or {}

    def run(self, ctx: StageContext) -> StageResult:
        if ctx.prompt is None:
            return StageResult(
                name=self.name,
                success=False,
                error='Cannot call LLM: prompt not built',
            )
        if ctx.extraction_schema is None:
            return StageResult(
                name=self.name,
                success=False,
                error='Cannot call LLM: extraction schema not loaded',
            )
        try:
            raw = self._engine.generate(
                ctx.prompt, ctx.extraction_schema, self._optional_params
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            logger.error('LLM generation failed: %s', msg)
            return StageResult(name=self.name, success=False, error=msg)

        if not isinstance(raw, str):
            msg = (
                f'Engine returned {type(raw).__name__!r} instead of str; '
                'the engine must extract the content string before returning'
            )
            logger.error(msg)
            return StageResult(name=self.name, success=False, error=msg)

        ctx.raw_output = raw
        return StageResult(name=self.name, data=raw, success=True)


class ParseResponseStage:
    """Parse the raw LLM JSON string and store the result in context.

    Requires ``ctx.raw_output`` to be a valid JSON string.  On success,
    ``ctx.extracted_data`` holds the parsed Python object.
    """

    name: str = 'json_parse'

    def run(self, ctx: StageContext) -> StageResult:
        try:
            extracted = json.loads(ctx.raw_output)
            ctx.extracted_data = extracted
            return StageResult(name=self.name, data=extracted, success=True)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            logger.error('JSON parse failed: %s', msg)
            return StageResult(name=self.name, success=False, error=msg)


class ValidationStage:
    """Run a list of validator callables as an explicit pipeline stage.

    Each validator receives ``ctx.extracted_data`` and is expected to raise if
    the data is invalid.  Validator failures are recorded in
    ``ctx.metadata['validation_errors']`` (a ``{name: error_message}`` dict) and
    in ``StageResult.data``.

    The stage itself always returns ``success=True`` so that subsequent stages
    (postprocessing, archive shaping) continue to run regardless of validation
    outcome.  Hooks registered on the ``'validation'`` stage can inspect
    ``ctx.metadata['validation_errors']`` to react to failures.
    """

    name: str = 'validation'

    def __init__(self, validators: list[Callable[[Any], None]] | None = None) -> None:
        self._validators = validators or []

    def run(self, ctx: StageContext) -> StageResult:
        errors: dict[str, str] = {}
        for idx, validator in enumerate(self._validators):
            vname = getattr(validator, '__name__', f'validator_{idx}')
            try:
                validator(ctx.extracted_data)
            except Exception as exc:  # noqa: BLE001
                errors[vname] = str(exc)
        if errors:
            ctx.metadata['validation_errors'] = errors
        return StageResult(
            name=self.name,
            success=True,
            data={'validation_errors': errors},
        )


class PostprocessingStage:
    """Apply an optional postprocessor callable to the extracted data.

    If no processor is provided the stage passes ``ctx.extracted_data`` through
    to ``ctx.postprocessed_data`` unchanged.  On success,
    ``ctx.postprocessed_data`` is populated.
    """

    name: str = 'postprocessing'

    def __init__(
        self, processor: Callable[[Any, dict[str, Any] | None], Any] | None = None
    ) -> None:
        self._processor = processor

    def run(self, ctx: StageContext) -> StageResult:
        if self._processor is None:
            ctx.postprocessed_data = ctx.extracted_data
            return StageResult(name=self.name, success=True)
        try:
            ctx.postprocessed_data = self._processor(
                ctx.extracted_data, ctx.postprocessing_schema
            )
            return StageResult(name=self.name, success=True)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            logger.error('Postprocessing failed: %s', msg)
            return StageResult(name=self.name, success=False, error=msg)


class ExportToNOMADStage:
    """Export the postprocessed data to NOMAD format using an optional exporter callable.

    If no exporter is provided the stage passes ``ctx.postprocessed_data`` through
    to ``ctx.archive_data`` unchanged.  On success, ``ctx.archive_data`` is
    populated.
    """

    name: str = 'export_to_nomad'

    def __init__(self, exporter: Callable[[Any], Any] | None = None) -> None:
        self._exporter = exporter

    def run(self, ctx: StageContext) -> StageResult:
        if self._exporter is None:
            ctx.archive_data = ctx.postprocessed_data
            return StageResult(name=self.name, success=True)
        try:
            ctx.archive_data = self._exporter(ctx.postprocessed_data)
            return StageResult(name=self.name, success=True)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            logger.error('Export to NOMAD failed: %s', msg)
            return StageResult(name=self.name, success=False, error=msg)


class ArchiveShapingStage:
    """Apply an optional archive shaper callable to the postprocessed data.

    If no shaper is provided the stage passes ``ctx.postprocessed_data`` through
    to ``ctx.archive_data`` unchanged.  On success, ``ctx.archive_data`` is
    populated.
    """

    name: str = 'archive_shaping'

    def __init__(self, shaper: Callable[[Any], Any] | None = None) -> None:
        self._shaper = shaper

    def run(self, ctx: StageContext) -> StageResult:
        if self._shaper is None:
            ctx.archive_data = ctx.postprocessed_data
            return StageResult(name=self.name, success=True)
        try:
            ctx.archive_data = self._shaper(ctx.postprocessed_data)
            return StageResult(name=self.name, success=True)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            logger.error('Archive shaping failed: %s', msg)
            return StageResult(name=self.name, success=False, error=msg)
