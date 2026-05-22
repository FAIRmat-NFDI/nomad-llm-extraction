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
from collections.abc import Callable
from typing import Any, Literal

from nomad_llm_extraction.pipeline.models import (
    PromptConfig,
    StageContext,
    StageFunc,
    StageHook,
    StageResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for hook callables
# ---------------------------------------------------------------------------


class StageRunner:
    """Executes a sequence of :class:`Stage` objects, firing hooks around each one.

    Hooks are registered per stage-name and timing (``'before'`` / ``'after'``).
    Multiple hooks may be attached to the same (stage, timing) pair; they are
    called in registration order.

    Execution stops after the first failing stage (i.e. ``StageResult.success
    is False``), but *after*-hooks for the failing stage are still invoked.
    """

    def __init__(self) -> None:
        self._stages: dict[str, StageFunc] = {}
        self._hooks: dict[str, dict[str, list[StageHook]]] = {}

    def add_stage(self, stage_name: str, stage: StageFunc) -> None:
        """Append *stage* to the execution sequence."""
        self._stages[stage_name] = stage

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
        for stage_name, stage_func in self._stages.items():
            self._fire_hooks(stage_name, 'before', ctx)
            try:
                result = stage_func(ctx, stage_name)
            except Exception as exc:  # noqa: BLE001
                logger.error('Stage %r failed: %s', stage_name, exc)
                result = StageResult(name=stage_name, success=False, error=str(exc))
            results.append(result)
            self._fire_hooks(stage_name, 'after', ctx)
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


def build_prompt(ctx: StageContext, stage_name: str = 'build_prompt') -> StageResult:
    if ctx.extraction_schema is None:
        return StageResult(
            name=stage_name,
            success=False,
            error='Cannot build prompt: extraction schema not loaded',
        )
    try:
        text = ctx.text
        schema = ctx.extraction_schema
        parts: list[str] = []
        if ctx.prompt_config.system_prompt:
            parts.append(ctx.prompt_config.system_prompt)
        if ctx.prompt_config.instruction_text:
            parts.append(ctx.prompt_config.instruction_text)
        parts.append(f'Here is the schema: {json.dumps(schema, indent=2)}')
        parts.append(f'Here is the text:\n{text}')
        ctx.prompt = '\n'.join(parts)
        return StageResult(name=stage_name, data=ctx.prompt, success=True)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        logger.error('Prompt build failed: %s', msg)
    return StageResult(name=stage_name, success=False, error=msg)


def llm_call(ctx: StageContext, stage_name: str = 'llm_call') -> StageResult:
    if ctx.prompt is None:
        return StageResult(
            name=stage_name,
            success=False,
            error='Cannot call LLM: prompt not built',
        )
    if ctx.extraction_schema is None:
        return StageResult(
            name=stage_name,
            success=False,
            error='Cannot call LLM: extraction schema not loaded',
        )
    try:
        raw = ctx.engine.generate(
            ctx.prompt, ctx.extraction_schema, ctx.optional_params
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        logger.error('LLM generation failed: %s', msg)
        return StageResult(name=stage_name, success=False, error=msg)

    if not isinstance(raw, str):
        msg = (
            f'Engine returned {type(raw).__name__!r} instead of str; '
            'the engine must extract the content string before returning'
        )
        logger.error(msg)
        return StageResult(name=stage_name, success=False, error=msg)

    ctx.raw_output = raw
    return StageResult(name=stage_name, data=raw, success=True)


def json_parse(ctx: StageContext, stage_name: str = 'json_parse') -> StageResult:
    try:
        extracted = json.loads(ctx.raw_output)
        ctx.extracted_data = extracted
        return StageResult(name=stage_name, data=extracted, success=True)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        logger.error('JSON parse failed: %s', msg)
        return StageResult(name=stage_name, success=False, error=msg)


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

def run_postprocessing(ctx: StageContext, stage_name: str = 'postprocessing') -> StageResult:
    if ctx.postprocessor is None:
        ctx.postprocessed_data = ctx.extracted_data
        return StageResult(name=stage_name, success=True)
    try:
        ctx.postprocessed_data = ctx.postprocessor(
            ctx.extracted_data, ctx.postprocessing_schema
        )
        return StageResult(name=stage_name, success=True)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        logger.error('Postprocessing failed: %s', msg)
        return StageResult(name=stage_name, success=False, error=msg)
