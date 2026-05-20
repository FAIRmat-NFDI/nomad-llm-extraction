"""Tests for the public pipeline API (Task 1).

These tests verify:
- Pydantic model instantiation and field validation
- ExtractionPipeline dependency-injection contract
- run(text) -> PipelineResult return type and structure
- Error propagation into PipelineResult (no uncaught exceptions)
"""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from nomad_llm_extraction.pipeline.extraction_pipeline import ExtractionPipeline
from nomad_llm_extraction.pipeline.models import (
    ModelConfig,
    PipelineResult,
    PromptConfig,
    SchemaSourceConfig,
    StageHookConfig,
    StageResult,
)

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _make_engine(return_value: Any = None):
    """Return a mock StructuredLLMEngine-compatible object."""
    engine = MagicMock()
    engine.generate.return_value = return_value
    return engine


def _make_schema_source(schema: dict | None = None):
    """Return a mock SchemaSource-compatible object."""
    source = MagicMock()
    source.get_schema.return_value = schema or {'type': 'object', 'properties': {}}
    return source


def _make_schema_sources(
    extraction_schema: dict | None = None,
    postprocessing_schema: dict | None = None,
):
    return (
        _make_schema_source(extraction_schema),
        _make_schema_source(postprocessing_schema),
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestPromptConfig:
    def test_defaults(self):
        cfg = PromptConfig()
        assert cfg.system_prompt == ''
        assert cfg.instruction_text == ''

    def test_custom_values(self):
        cfg = PromptConfig(system_prompt='You are an expert.', instruction_text='Extract data.')
        assert cfg.system_prompt == 'You are an expert.'
        assert cfg.instruction_text == 'Extract data.'


class TestModelConfig:
    def test_requires_model_name(self):
        with pytest.raises(Exception):
            ModelConfig()  # model_name is required

    def test_minimal(self):
        cfg = ModelConfig(model_name='gpt-4o')
        assert cfg.model_name == 'gpt-4o'
        assert cfg.api_url is None
        assert cfg.api_key == ''
        assert cfg.optional_params == {}

    def test_full(self):
        cfg = ModelConfig(
            model_name='gpt-4o',
            api_url='http://localhost:8000',
            api_key='sk-test',
            optional_params={'temperature': 0.2},
        )
        assert cfg.optional_params == {'temperature': 0.2}


class TestStageResult:
    def test_success(self):
        sr = StageResult(name='extraction', success=True, data={'key': 'value'})
        assert sr.success is True
        assert sr.error is None

    def test_failure(self):
        sr = StageResult(name='extraction', success=False, error='LLM timeout')
        assert sr.success is False
        assert sr.data is None


class TestPipelineResult:
    def test_success_result(self):
        pr = PipelineResult(success=True, extracted_data={'cells': []})
        assert pr.success is True
        assert pr.error is None
        assert pr.stages == []

    def test_failure_result(self):
        pr = PipelineResult(success=False, error='Schema not found')
        assert pr.success is False
        assert pr.extracted_data is None

    def test_with_stages(self):
        stages = [
            StageResult(name='schema_load', success=True),
            StageResult(name='extraction', success=True, data={'x': 1}),
        ]
        pr = PipelineResult(success=True, stages=stages)
        assert len(pr.stages) == 2


# ---------------------------------------------------------------------------
# ExtractionPipeline construction tests
# ---------------------------------------------------------------------------

class TestExtractionPipelineConstruction:
    def test_requires_engine_and_two_schema_sources(self):
        """Pipeline must accept engine plus extraction/postprocessing schema sources."""
        engine = _make_engine()
        extraction_schema_source, postprocessing_schema_source = _make_schema_sources()
        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
        )
        assert pipeline is not None

    def test_accepts_optional_prompt_config(self):
        engine = _make_engine()
        extraction_schema_source, postprocessing_schema_source = _make_schema_sources()
        prompt_cfg = PromptConfig(system_prompt='sys', instruction_text='instr')
        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
            prompt_config=prompt_cfg,
        )
        assert pipeline.prompt_config.system_prompt == 'sys'

    def test_default_prompt_config_if_omitted(self):
        engine = _make_engine()
        extraction_schema_source, postprocessing_schema_source = _make_schema_sources()
        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
        )
        assert isinstance(pipeline.prompt_config, PromptConfig)

    def test_accepts_validators_list(self):
        engine = _make_engine()
        extraction_schema_source, postprocessing_schema_source = _make_schema_sources()
        validator = MagicMock()
        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
            validators=[validator],
        )
        assert validator in pipeline.validators

    def test_accepts_visualizers_list(self):
        engine = _make_engine()
        extraction_schema_source, postprocessing_schema_source = _make_schema_sources()
        viz = MagicMock()
        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
            visualizers=[viz],
        )
        assert viz in pipeline.visualizers


# ---------------------------------------------------------------------------
# ExtractionPipeline.run() tests
# ---------------------------------------------------------------------------

class TestExtractionPipelineRun:
    def _simple_pipeline(self, llm_payload: Any = None):
        schema = {'type': 'object', 'properties': {'value': {'type': 'number'}}}
        extracted = {'value': 42}
        engine = _make_engine(return_value=llm_payload or json.dumps(extracted))
        extraction_schema_source, postprocessing_schema_source = _make_schema_sources(
            extraction_schema=schema,
            postprocessing_schema={'type': 'object', 'properties': {}},
        )
        return (
            ExtractionPipeline(
                engine=engine,
                extraction_schema_source=extraction_schema_source,
                postprocessing_schema_source=postprocessing_schema_source,
            ),
            extracted,
        )

    def test_run_returns_pipeline_result(self):
        pipeline, _ = self._simple_pipeline()
        result = pipeline.run('some paper text')
        assert isinstance(result, PipelineResult)

    def test_run_success_flag_set(self):
        pipeline, _ = self._simple_pipeline()
        result = pipeline.run('some paper text')
        assert result.success is True

    def test_run_raw_output_stored(self):
        extracted = {'value': 42}
        raw = json.dumps(extracted)
        pipeline, _ = self._simple_pipeline(llm_payload=raw)
        result = pipeline.run('some paper text')
        assert result.raw_llm_output == raw

    def test_run_extracted_data_parsed(self):
        pipeline, expected = self._simple_pipeline()
        result = pipeline.run('some paper text')
        assert result.extracted_data == expected

    def test_run_uses_both_schema_sources(self):
        schema = {'type': 'object', 'properties': {}}
        engine = _make_engine(return_value='{}')
        extraction_schema_source, postprocessing_schema_source = _make_schema_sources(
            extraction_schema=schema,
            postprocessing_schema=schema,
        )
        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
        )
        pipeline.run('text')
        extraction_schema_source.get_schema.assert_called_once()
        postprocessing_schema_source.get_schema.assert_called_once()

    def test_run_calls_engine_with_text(self):
        extraction_schema = {'type': 'object', 'title': 'extraction', 'properties': {}}
        postprocessing_schema = {
            'type': 'object',
            'title': 'postprocessing',
            'properties': {},
        }
        engine = _make_engine(return_value='{}')
        extraction_schema_source, postprocessing_schema_source = _make_schema_sources(
            extraction_schema=extraction_schema,
            postprocessing_schema=postprocessing_schema,
        )
        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
        )
        pipeline.run('my paper text')
        engine.generate.assert_called_once()
        call_args = engine.generate.call_args
        # The prompt (first positional arg) should contain the text
        prompt_arg = call_args[0][0] if call_args[0] else call_args[1].get('prompt', '')
        assert 'my paper text' in prompt_arg
        assert call_args[0][1] == extraction_schema

    def test_postprocessor_receives_postprocessing_schema(self):
        extraction_schema = {'type': 'object', 'title': 'llm-schema'}
        postprocessing_schema = {'type': 'object', 'title': 'pp-schema'}
        engine = _make_engine(return_value='{"value": 1}')
        extraction_schema_source, postprocessing_schema_source = _make_schema_sources(
            extraction_schema=extraction_schema,
            postprocessing_schema=postprocessing_schema,
        )

        seen: list[dict[str, Any]] = []

        def postprocessor(data, schema):
            seen.append(schema)
            return data

        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
            postprocessor=postprocessor,
        )
        result = pipeline.run('my paper text')
        assert result.success is True
        assert seen == [postprocessing_schema]

    def test_run_handles_engine_exception_gracefully(self):
        schema = {'type': 'object', 'properties': {}}
        engine = MagicMock()
        engine.generate.side_effect = RuntimeError('LLM unavailable')
        extraction_schema_source, postprocessing_schema_source = _make_schema_sources(
            extraction_schema=schema,
            postprocessing_schema=schema,
        )
        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
        )
        result = pipeline.run('text')
        assert result.success is False
        assert result.error is not None
        assert 'LLM unavailable' in result.error

    def test_run_handles_schema_source_exception_gracefully(self):
        engine = _make_engine(return_value='{}')
        extraction_schema_source = MagicMock()
        extraction_schema_source.get_schema.side_effect = FileNotFoundError(
            'schema.json not found'
        )
        postprocessing_schema_source = _make_schema_source()
        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
        )
        result = pipeline.run('text')
        assert result.success is False
        assert 'schema.json not found' in result.error

    def test_run_handles_invalid_json_output(self):
        engine = _make_engine(return_value='not valid json {{{')
        extraction_schema_source, postprocessing_schema_source = _make_schema_sources()
        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
        )
        result = pipeline.run('text')
        assert result.success is False
        assert result.extracted_data is None

    def test_prompt_config_injected_into_prompt(self):
        schema = {'type': 'object', 'properties': {}}
        engine = _make_engine(return_value='{}')
        extraction_schema_source, postprocessing_schema_source = _make_schema_sources(
            extraction_schema=schema,
            postprocessing_schema=schema,
        )
        prompt_cfg = PromptConfig(
            system_prompt='SYSTEM_MARKER', instruction_text='INSTRUCTION_MARKER'
        )
        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
            prompt_config=prompt_cfg,
        )
        pipeline.run('paper text')
        call_args = engine.generate.call_args
        prompt_arg = call_args[0][0] if call_args[0] else call_args[1].get('prompt', '')
        assert 'SYSTEM_MARKER' in prompt_arg
        assert 'INSTRUCTION_MARKER' in prompt_arg

    def test_visualizers_called_after_run(self):
        """run() must call each visualizer with the PipelineResult after completion."""
        pipeline, _ = self._simple_pipeline()
        viz = MagicMock()
        pipeline.visualizers = [viz]
        result = pipeline.run('paper text')
        viz.assert_called_once_with(result)

    def test_visualizers_called_even_on_failure(self):
        """Visualizers must be called even when the pipeline fails."""
        engine = MagicMock()
        engine.generate.side_effect = RuntimeError('boom')
        extraction_schema_source, postprocessing_schema_source = _make_schema_sources()
        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
        )
        viz = MagicMock()
        pipeline.visualizers = [viz]
        result = pipeline.run('paper text')
        viz.assert_called_once_with(result)

    def test_visualizer_exception_does_not_propagate(self):
        """A visualizer that raises must not abort the pipeline result."""
        pipeline, _ = self._simple_pipeline()
        bad_viz = MagicMock(side_effect=RuntimeError('viz error'))
        pipeline.visualizers = [bad_viz]
        result = pipeline.run('paper text')
        assert isinstance(result, PipelineResult)

    def test_multiple_visualizers_all_called(self):
        """All visualizers in the list must be invoked."""
        pipeline, _ = self._simple_pipeline()
        viz1, viz2 = MagicMock(), MagicMock()
        pipeline.visualizers = [viz1, viz2]
        result = pipeline.run('paper text')
        viz1.assert_called_once_with(result)
        viz2.assert_called_once_with(result)


# ---------------------------------------------------------------------------
# SchemaSourceConfig model tests
# ---------------------------------------------------------------------------

class TestSchemaSourceConfig:
    def test_all_fields_optional(self):
        cfg = SchemaSourceConfig()
        assert cfg.schema_path is None
        assert cfg.schema_url is None
        assert cfg.inline_schema is None

    def test_schema_path(self):
        cfg = SchemaSourceConfig(schema_path='/data/schema.json')
        assert cfg.schema_path == '/data/schema.json'

    def test_schema_url(self):
        cfg = SchemaSourceConfig(schema_url='https://example.com/schema.json')
        assert cfg.schema_url == 'https://example.com/schema.json'

    def test_inline_schema(self):
        schema = {'type': 'object', 'properties': {'x': {'type': 'number'}}}
        cfg = SchemaSourceConfig(inline_schema=schema)
        assert cfg.inline_schema == schema

    def test_multiple_sources_allowed(self):
        """Config may specify several sources; caller decides priority."""
        cfg = SchemaSourceConfig(
            schema_path='/data/schema.json',
            schema_url='https://example.com/schema.json',
        )
        assert cfg.schema_path is not None
        assert cfg.schema_url is not None


# ---------------------------------------------------------------------------
# StageHookConfig model tests
# ---------------------------------------------------------------------------

class TestStageHookConfig:
    def test_requires_stage_name(self):
        with pytest.raises(Exception):
            StageHookConfig()

    def test_defaults(self):
        cfg = StageHookConfig(stage_name='llm_extraction')
        assert cfg.stage_name == 'llm_extraction'
        assert cfg.when == 'after'
        assert cfg.enabled is True

    def test_before_hook(self):
        cfg = StageHookConfig(stage_name='schema_load', when='before')
        assert cfg.when == 'before'

    def test_disabled_hook(self):
        cfg = StageHookConfig(stage_name='json_parse', enabled=False)
        assert cfg.enabled is False

    def test_invalid_when_rejected(self):
        with pytest.raises(Exception):
            StageHookConfig(stage_name='x', when='during')


# ---------------------------------------------------------------------------
# Engine contract: LiteLLMEngine returns a response object, not a raw string
# ---------------------------------------------------------------------------

def _make_litellm_response(content: str) -> MagicMock:
    """Build a minimal mock that mirrors the litellm ModelResponse structure."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    # Ensure it is NOT treated as a string
    resp.__class__ = object  # not str
    return resp


class TestEngineOutputNormalization:
    """ExtractionPipeline enforces a strict str engine contract.

    Engines must return a plain ``str``; the pipeline does not normalise
    litellm ``ModelResponse`` objects.  Only string-returning engines are
    supported at the pipeline boundary.
    """

    def test_plain_string_still_works(self):
        """Plain-string engine output (e.g. from mocks) must continue to work."""
        payload = json.dumps({'y': 99})
        engine = _make_engine(return_value=payload)
        extraction_schema_source, postprocessing_schema_source = _make_schema_sources()
        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
        )
        result = pipeline.run('text')
        assert result.success is True
        assert result.extracted_data == {'y': 99}

    def test_non_str_engine_output_fails(self):
        """Pipeline must fail if engine returns a non-string (strict contract)."""
        engine = _make_engine(return_value=_make_litellm_response('{"x": 1}'))
        extraction_schema_source, postprocessing_schema_source = _make_schema_sources()
        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
        )
        result = pipeline.run('text')
        assert result.success is False
        assert result.extracted_data is None

    def test_invalid_content_in_plain_string_fails_gracefully(self):
        """Bad JSON string must result in a failed PipelineResult."""
        engine = _make_engine(return_value='not json {{{')
        extraction_schema_source, postprocessing_schema_source = _make_schema_sources()
        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
        )
        result = pipeline.run('text')
        assert result.success is False
        assert result.extracted_data is None


# ---------------------------------------------------------------------------
# Public __init__.py export tests
# ---------------------------------------------------------------------------

class TestPublicExports:
    def test_schema_source_config_exported(self):
        from nomad_llm_extraction.pipeline import SchemaSourceConfig as SSC
        assert SSC is SchemaSourceConfig

    def test_stage_hook_config_exported(self):
        from nomad_llm_extraction.pipeline import StageHookConfig as SHC
        assert SHC is StageHookConfig
