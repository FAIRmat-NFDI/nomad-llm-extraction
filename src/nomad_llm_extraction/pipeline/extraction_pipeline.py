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
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, runtime_checkable

from nomad_llm_extraction.pipeline.models import (
    PipelineResult,
    PromptConfig,
    StageResult,
)

logger = logging.getLogger(__name__)


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

    Args:
        engine: LLM backend implementing ``generate(prompt, json_schema) -> str``.
        schema_source: Object with ``get_schema() -> dict`` that returns the
            JSON schema constraining the LLM output.
        prompt_config: System prompt and instruction text.
        validators: Optional list of callables ``(extracted_data) -> None`` that
            may raise to signal validation failure.  Called after a successful
            JSON parse; failures are recorded but do not re-raise.
        visualizers: Optional list of callables ``(result: PipelineResult) ->
            None`` invoked after the pipeline completes regardless of success.
    """

    def __init__(
        self,
        engine: LLMEngine,
        schema_source: SchemaSource,
        prompt_config: PromptConfig | None = None,
        validators: list[Any] | None = None,
        visualizers: list[Any] | None = None,
    ) -> None:
        self.engine = engine
        self.schema_source = schema_source
        self.prompt_config = prompt_config or PromptConfig()
        self.validators = validators or []
        self.visualizers = visualizers or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, text: str) -> PipelineResult:
        """Run the full extraction pipeline on *text*.

        Returns a :class:`PipelineResult` regardless of whether individual
        stages succeed or fail — exceptions are captured and stored in the
        result rather than propagating to the caller.
        """
        stages: list[StageResult] = []

        # Stage 1: load schema
        schema, schema_stage = self._load_schema()
        stages.append(schema_stage)
        if not schema_stage.success:
            return PipelineResult(
                success=False,
                stages=stages,
                error=schema_stage.error,
            )

        # Stage 2: call LLM
        raw_output, llm_stage = self._call_llm(text, schema)
        stages.append(llm_stage)
        if not llm_stage.success:
            return PipelineResult(
                success=False,
                stages=stages,
                error=llm_stage.error,
            )

        # Stage 3: parse JSON
        extracted, parse_stage = self._parse_output(raw_output)
        stages.append(parse_stage)
        if not parse_stage.success:
            return PipelineResult(
                success=False,
                raw_llm_output=raw_output,
                stages=stages,
                error=parse_stage.error,
            )

        # Stage 4: run validators (best-effort; failures recorded, not raised)
        validation_stages = self._run_validators(extracted)
        stages.extend(validation_stages)

        result = PipelineResult(
            success=True,
            raw_llm_output=raw_output,
            extracted_data=extracted,
            stages=stages,
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_schema(self) -> tuple[dict[str, Any] | None, StageResult]:
        try:
            schema = self.schema_source.get_schema()
            return schema, StageResult(name='schema_load', success=True, data=schema)
        except Exception as exc:
            msg = str(exc)
            logger.error('Schema load failed: %s', msg)
            return None, StageResult(name='schema_load', success=False, error=msg)

    def _call_llm(
        self, text: str, schema: dict[str, Any]
    ) -> tuple[str | None, StageResult]:
        prompt = self._build_prompt(text, schema)
        try:
            raw = self.engine.generate(prompt, schema)
            return raw, StageResult(name='llm_extraction', success=True)
        except Exception as exc:
            msg = str(exc)
            logger.error('LLM generation failed: %s', msg)
            return None, StageResult(name='llm_extraction', success=False, error=msg)

    def _parse_output(
        self, raw_output: str | None
    ) -> tuple[Any | None, StageResult]:
        try:
            extracted = json.loads(raw_output)
            return extracted, StageResult(name='json_parse', success=True)
        except Exception as exc:
            msg = str(exc)
            logger.error('JSON parse failed: %s', msg)
            return None, StageResult(name='json_parse', success=False, error=msg)

    def _run_validators(self, extracted: Any) -> list[StageResult]:
        results = []
        for idx, validator in enumerate(self.validators):
            name = getattr(validator, '__name__', f'validator_{idx}')
            try:
                validator(extracted)
                results.append(StageResult(name=name, success=True))
            except Exception as exc:
                results.append(
                    StageResult(name=name, success=False, error=str(exc))
                )
        return results

    def _run_visualizers(self, result: PipelineResult) -> None:
        for visualizer in self.visualizers:
            try:
                visualizer(result)
            except Exception as exc:
                logger.warning('Visualizer %r raised: %s', visualizer, exc)

    def _build_prompt(self, text: str, schema: dict[str, Any]) -> str:
        parts = []
        if self.prompt_config.system_prompt:
            parts.append(self.prompt_config.system_prompt)
        if self.prompt_config.instruction_text:
            parts.append(self.prompt_config.instruction_text)
        parts.append(f'Here is the schema: {json.dumps(schema, indent=2)}')
        parts.append(f'Here is the text:\n{text}')
        return '\n'.join(parts)
