"""Tests for pipeline/stages.py (Task 3).

Covers:
- StageContext creation and field access
- StageRunner sequential execution, short-circuit on failure, before/after hooks
- Hook exception isolation (hooks must not abort the run)
- Concrete stages: SchemaLoadStage, PromptBuildStage, LLMCallStage, ParseResponseStage
- ExtractionPipeline integration with stage hooks
- LiteLLMEngine.generate() returns a plain str (tightened engine boundary)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from nomad_llm_extraction.pipeline.models import PromptConfig, StageResult
from nomad_llm_extraction.pipeline.schema_filling.llm_engine import LiteLLMEngine
from nomad_llm_extraction.pipeline.stages import (
    ExtractionSchemaLoadStage,
    LLMCallStage,
    ParseResponseStage,
    PostprocessingSchemaLoadStage,
    PromptBuildStage,
    Stage,
    StageContext,
    StageRunner,
)

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class _SuccessStage:
    """Minimal Stage that always succeeds and optionally appends to a log."""

    def __init__(self, name: str, log: list[str] | None = None) -> None:
        self.name = name
        self._log = log

    def run(self, ctx: StageContext) -> StageResult:
        if self._log is not None:
            self._log.append(self.name)
        return StageResult(name=self.name, success=True)


class _FailureStage:
    """Minimal Stage that always fails."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, ctx: StageContext) -> StageResult:
        return StageResult(name=self.name, success=False, error='intentional failure')


# ---------------------------------------------------------------------------
# StageContext
# ---------------------------------------------------------------------------


class TestStageContext:
    def test_required_text_field(self):
        ctx = StageContext(text='paper text')
        assert ctx.text == 'paper text'

    def test_optional_fields_default_to_none(self):
        ctx = StageContext(text='paper text')
        assert ctx.extraction_schema is None
        assert ctx.postprocessing_schema is None
        assert ctx.prompt is None
        assert ctx.raw_output is None
        assert ctx.extracted_data is None

    def test_metadata_defaults_to_empty_dict(self):
        ctx = StageContext(text='text')
        assert ctx.metadata == {}

    def test_fields_are_mutable(self):
        ctx = StageContext(text='text')
        ctx.extraction_schema = {'type': 'object'}
        ctx.postprocessing_schema = {'type': 'object'}
        ctx.prompt = 'my prompt'
        ctx.raw_output = '{"x": 1}'
        ctx.extracted_data = {'x': 1}
        assert ctx.extraction_schema == {'type': 'object'}
        assert ctx.postprocessing_schema == {'type': 'object'}
        assert ctx.prompt == 'my prompt'
        assert ctx.raw_output == '{"x": 1}'
        assert ctx.extracted_data == {'x': 1}

    def test_metadata_is_mutable(self):
        ctx = StageContext(text='text')
        ctx.metadata['key'] = 'value'
        assert ctx.metadata['key'] == 'value'


# ---------------------------------------------------------------------------
# Stage Protocol
# ---------------------------------------------------------------------------


class TestStageProtocol:
    def test_concrete_stage_satisfies_protocol(self):
        stage = _SuccessStage('test')
        assert isinstance(stage, Stage)

    def test_stage_without_name_does_not_satisfy_protocol(self):
        class NoName:
            def run(self, ctx: StageContext) -> StageResult:
                return StageResult(name='x', success=True)

        assert not isinstance(NoName(), Stage)

    def test_stage_without_run_does_not_satisfy_protocol(self):
        class NoRun:
            name = 'x'

        assert not isinstance(NoRun(), Stage)


# ---------------------------------------------------------------------------
# StageRunner
# ---------------------------------------------------------------------------


class TestStageRunnerBasic:
    def test_empty_runner_returns_empty_list(self):
        runner = StageRunner()
        ctx = StageContext(text='text')
        assert runner.run(ctx) == []

    def test_single_successful_stage(self):
        runner = StageRunner()
        runner.add_stage(_SuccessStage('s1'))
        ctx = StageContext(text='text')
        results = runner.run(ctx)
        assert len(results) == 1
        assert results[0].name == 's1'
        assert results[0].success is True

    def test_multiple_stages_all_succeed(self):
        log: list[str] = []
        runner = StageRunner()
        runner.add_stage(_SuccessStage('s1', log))
        runner.add_stage(_SuccessStage('s2', log))
        runner.add_stage(_SuccessStage('s3', log))
        ctx = StageContext(text='text')
        results = runner.run(ctx)
        assert len(results) == 3
        assert log == ['s1', 's2', 's3']

    def test_failure_short_circuits(self):
        log: list[str] = []
        runner = StageRunner()
        runner.add_stage(_SuccessStage('s1', log))
        runner.add_stage(_FailureStage('fail'))
        runner.add_stage(_SuccessStage('s3', log))
        ctx = StageContext(text='text')
        results = runner.run(ctx)
        assert len(results) == 2  # s1 + fail; s3 not reached
        assert log == ['s1']
        assert results[-1].success is False

    def test_results_ordered_by_execution(self):
        runner = StageRunner()
        runner.add_stage(_SuccessStage('first'))
        runner.add_stage(_SuccessStage('second'))
        ctx = StageContext(text='text')
        results = runner.run(ctx)
        assert [r.name for r in results] == ['first', 'second']


class TestStageRunnerHooks:
    def test_before_hook_called(self):
        log: list[str] = []
        runner = StageRunner()
        runner.add_stage(_SuccessStage('s1'))
        runner.add_hook('s1', 'before', lambda ctx: log.append('before'))
        runner.run(StageContext(text='text'))
        assert 'before' in log

    def test_after_hook_called(self):
        log: list[str] = []
        runner = StageRunner()
        runner.add_stage(_SuccessStage('s1'))
        runner.add_hook('s1', 'after', lambda ctx: log.append('after'))
        runner.run(StageContext(text='text'))
        assert 'after' in log

    def test_hooks_fire_in_correct_order(self):
        log: list[str] = []

        class RecordingStage:
            name = 's1'

            def run(self, ctx: StageContext) -> StageResult:
                log.append('stage')
                return StageResult(name='s1', success=True)

        runner = StageRunner()
        runner.add_stage(RecordingStage())
        runner.add_hook('s1', 'before', lambda ctx: log.append('before'))
        runner.add_hook('s1', 'after', lambda ctx: log.append('after'))
        runner.run(StageContext(text='text'))
        assert log == ['before', 'stage', 'after']

    def test_multiple_before_hooks_all_called(self):
        log: list[str] = []
        runner = StageRunner()
        runner.add_stage(_SuccessStage('s1'))
        runner.add_hook('s1', 'before', lambda ctx: log.append('h1'))
        runner.add_hook('s1', 'before', lambda ctx: log.append('h2'))
        runner.run(StageContext(text='text'))
        assert log == ['h1', 'h2']

    def test_multiple_after_hooks_all_called(self):
        log: list[str] = []
        runner = StageRunner()
        runner.add_stage(_SuccessStage('s1'))
        runner.add_hook('s1', 'after', lambda ctx: log.append('h1'))
        runner.add_hook('s1', 'after', lambda ctx: log.append('h2'))
        runner.run(StageContext(text='text'))
        assert log == ['h1', 'h2']

    def test_hook_receives_context(self):
        received: list[StageContext] = []
        runner = StageRunner()
        runner.add_stage(_SuccessStage('s1'))

        def capture(ctx: StageContext) -> None:
            received.append(ctx)

        runner.add_hook('s1', 'after', capture)
        ctx = StageContext(text='my text')
        runner.run(ctx)
        assert len(received) == 1
        assert received[0] is ctx

    def test_before_hook_exception_does_not_abort(self):
        """A raising before-hook must not prevent the stage from running."""
        log: list[str] = []
        runner = StageRunner()
        runner.add_stage(_SuccessStage('s1', log))
        runner.add_hook('s1', 'before', lambda ctx: (_ for _ in ()).throw(RuntimeError('boom')))
        results = runner.run(StageContext(text='text'))
        assert log == ['s1']
        assert results[0].success is True

    def test_after_hook_exception_does_not_abort(self):
        """A raising after-hook must not affect subsequent stages."""
        log: list[str] = []
        runner = StageRunner()
        runner.add_stage(_SuccessStage('s1'))
        runner.add_hook('s1', 'after', lambda ctx: (_ for _ in ()).throw(RuntimeError('boom')))
        runner.add_stage(_SuccessStage('s2', log))
        results = runner.run(StageContext(text='text'))
        assert log == ['s2']
        assert len(results) == 2

    def test_after_hook_called_even_on_stage_failure(self):
        """After-hooks fire even when the stage fails (for observation purposes)."""
        log: list[str] = []
        runner = StageRunner()
        runner.add_stage(_FailureStage('fail'))
        runner.add_hook('fail', 'after', lambda ctx: log.append('after'))
        runner.run(StageContext(text='text'))
        assert 'after' in log

    def test_hook_on_unknown_stage_does_not_error(self):
        """Hooks registered for a stage not in the runner must be silently ignored."""
        runner = StageRunner()
        runner.add_stage(_SuccessStage('s1'))
        runner.add_hook('nonexistent', 'before', lambda ctx: None)
        results = runner.run(StageContext(text='text'))
        assert len(results) == 1


# ---------------------------------------------------------------------------
# ExtractionSchemaLoadStage / PostprocessingSchemaLoadStage
# ---------------------------------------------------------------------------


class TestExtractionSchemaLoadStage:
    def test_loads_extraction_schema_into_context(self):
        source = MagicMock()
        source.get_schema.return_value = {'type': 'object', 'properties': {}}
        stage = ExtractionSchemaLoadStage(source)
        ctx = StageContext(text='text')
        result = stage.run(ctx)
        assert result.success is True
        assert ctx.extraction_schema == {'type': 'object', 'properties': {}}

    def test_result_contains_schema_data(self):
        schema = {'type': 'object'}
        source = MagicMock()
        source.get_schema.return_value = schema
        stage = ExtractionSchemaLoadStage(source)
        result = stage.run(StageContext(text='text'))
        assert result.data == schema

    def test_stage_name_is_extraction_schema_load(self):
        stage = ExtractionSchemaLoadStage(MagicMock())
        assert stage.name == 'extraction_schema_load'

    def test_failure_on_source_error(self):
        source = MagicMock()
        source.get_schema.side_effect = FileNotFoundError('schema not found')
        stage = ExtractionSchemaLoadStage(source)
        ctx = StageContext(text='text')
        result = stage.run(ctx)
        assert result.success is False
        assert 'schema not found' in result.error
        assert ctx.extraction_schema is None


class TestPostprocessingSchemaLoadStage:
    def test_loads_postprocessing_schema_into_context(self):
        source = MagicMock()
        source.get_schema.return_value = {'type': 'object', 'properties': {}}
        stage = PostprocessingSchemaLoadStage(source)
        ctx = StageContext(text='text')
        result = stage.run(ctx)
        assert result.success is True
        assert ctx.postprocessing_schema == {'type': 'object', 'properties': {}}

    def test_stage_name_is_postprocessing_schema_load(self):
        stage = PostprocessingSchemaLoadStage(MagicMock())
        assert stage.name == 'postprocessing_schema_load'


# ---------------------------------------------------------------------------
# PromptBuildStage
# ---------------------------------------------------------------------------


class TestPromptBuildStage:
    def test_builds_prompt_stored_in_context(self):
        stage = PromptBuildStage(PromptConfig())
        ctx = StageContext(text='paper text', extraction_schema={'type': 'object'})
        result = stage.run(ctx)
        assert result.success is True
        assert ctx.prompt is not None
        assert 'paper text' in ctx.prompt

    def test_system_prompt_included(self):
        stage = PromptBuildStage(PromptConfig(system_prompt='SYS_MARKER'))
        ctx = StageContext(text='text', extraction_schema={'type': 'object'})
        stage.run(ctx)
        assert 'SYS_MARKER' in ctx.prompt

    def test_instruction_text_included(self):
        stage = PromptBuildStage(PromptConfig(instruction_text='INSTR_MARKER'))
        ctx = StageContext(text='text', extraction_schema={'type': 'object'})
        stage.run(ctx)
        assert 'INSTR_MARKER' in ctx.prompt

    def test_schema_serialized_into_prompt(self):
        schema = {'type': 'object', 'properties': {'x': {'type': 'number'}}}
        stage = PromptBuildStage(PromptConfig())
        ctx = StageContext(text='text', extraction_schema=schema)
        stage.run(ctx)
        assert '"x"' in ctx.prompt

    def test_stage_name_is_prompt_build(self):
        stage = PromptBuildStage()
        assert stage.name == 'prompt_build'

    def test_fails_when_extraction_schema_missing(self):
        stage = PromptBuildStage(PromptConfig())
        ctx = StageContext(text='text')  # no extraction_schema
        result = stage.run(ctx)
        assert result.success is False
        assert ctx.prompt is None

    def test_default_prompt_config_when_omitted(self):
        stage = PromptBuildStage()
        ctx = StageContext(text='hi', extraction_schema={'type': 'object'})
        result = stage.run(ctx)
        assert result.success is True


# ---------------------------------------------------------------------------
# LLMCallStage
# ---------------------------------------------------------------------------


class TestLLMCallStage:
    def test_calls_engine_and_stores_output(self):
        engine = MagicMock()
        engine.generate.return_value = '{"answer": 42}'
        stage = LLMCallStage(engine)
        ctx = StageContext(
            text='text', extraction_schema={'type': 'object'}, prompt='my prompt'
        )
        result = stage.run(ctx)
        assert result.success is True
        assert ctx.raw_output == '{"answer": 42}'

    def test_engine_called_with_prompt_and_schema(self):
        engine = MagicMock()
        engine.generate.return_value = '{}'
        schema = {'type': 'object'}
        stage = LLMCallStage(engine)
        ctx = StageContext(text='text', extraction_schema=schema, prompt='the prompt')
        stage.run(ctx)
        engine.generate.assert_called_once()
        args = engine.generate.call_args[0]
        assert args[0] == 'the prompt'
        assert args[1] == schema

    def test_optional_params_forwarded(self):
        engine = MagicMock()
        engine.generate.return_value = '{}'
        params = {'temperature': 0.1}
        stage = LLMCallStage(engine, optional_params=params)
        ctx = StageContext(text='text', extraction_schema={}, prompt='p')
        stage.run(ctx)
        args = engine.generate.call_args[0]
        assert args[2] == params

    def test_stage_name_is_llm_extraction(self):
        stage = LLMCallStage(MagicMock())
        assert stage.name == 'llm_extraction'

    def test_failure_on_engine_error(self):
        engine = MagicMock()
        engine.generate.side_effect = RuntimeError('timeout')
        stage = LLMCallStage(engine)
        ctx = StageContext(text='text', extraction_schema={}, prompt='p')
        result = stage.run(ctx)
        assert result.success is False
        assert 'timeout' in result.error
        assert ctx.raw_output is None

    def test_fails_when_prompt_missing(self):
        stage = LLMCallStage(MagicMock())
        ctx = StageContext(text='text', extraction_schema={})  # no prompt
        result = stage.run(ctx)
        assert result.success is False

    def test_fails_when_extraction_schema_missing(self):
        stage = LLMCallStage(MagicMock())
        ctx = StageContext(text='text', prompt='p')  # no extraction schema
        result = stage.run(ctx)
        assert result.success is False

    def test_fails_if_engine_returns_non_str(self):
        """LLMCallStage must fail if engine returns a non-string (strict contract)."""
        engine = MagicMock()
        engine.generate.return_value = object()  # not a str

        stage = LLMCallStage(engine)
        ctx = StageContext(text='text', extraction_schema={}, prompt='p')
        result = stage.run(ctx)
        assert result.success is False
        assert ctx.raw_output is None


# ---------------------------------------------------------------------------
# ParseResponseStage
# ---------------------------------------------------------------------------


class TestParseResponseStage:
    def test_parses_valid_json(self):
        stage = ParseResponseStage()
        ctx = StageContext(text='text', raw_output='{"value": 42}')
        result = stage.run(ctx)
        assert result.success is True
        assert ctx.extracted_data == {'value': 42}

    def test_stage_name_is_json_parse(self):
        stage = ParseResponseStage()
        assert stage.name == 'json_parse'

    def test_failure_on_invalid_json(self):
        stage = ParseResponseStage()
        ctx = StageContext(text='text', raw_output='not valid json {{{')
        result = stage.run(ctx)
        assert result.success is False
        assert ctx.extracted_data is None

    def test_failure_when_raw_output_is_none(self):
        stage = ParseResponseStage()
        ctx = StageContext(text='text')  # raw_output is None
        result = stage.run(ctx)
        assert result.success is False

    def test_handles_nested_json(self):
        payload = {'cells': [{'efficiency': 0.25}, {'efficiency': 0.30}]}
        stage = ParseResponseStage()
        ctx = StageContext(text='text', raw_output=json.dumps(payload))
        result = stage.run(ctx)
        assert result.success is True
        assert ctx.extracted_data == payload


# ---------------------------------------------------------------------------
# ExtractionPipeline integration with stage hooks
# ---------------------------------------------------------------------------


class TestExtractionPipelineWithStageHooks:
    """Verify that ExtractionPipeline wires stage hooks through to the StageRunner."""

    def _make_pipeline(self, stage_hooks=None):
        from nomad_llm_extraction.pipeline.extraction_pipeline import ExtractionPipeline

        engine = MagicMock()
        engine.generate.return_value = json.dumps({'result': 'ok'})
        extraction_schema_source = MagicMock()
        extraction_schema_source.get_schema.return_value = {'type': 'object'}
        postprocessing_schema_source = MagicMock()
        postprocessing_schema_source.get_schema.return_value = {'type': 'object'}
        return ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
            stage_hooks=stage_hooks,
        )

    def test_pipeline_accepts_stage_hooks_param(self):
        log: list[str] = []
        hook = ('extraction_schema_load', 'after', lambda ctx: log.append('after_schema'))
        pipeline = self._make_pipeline(stage_hooks=[hook])
        result = pipeline.run('text')
        assert result.success is True
        assert 'after_schema' in log

    def test_before_hook_fires_before_stage(self):
        log: list[str] = []

        def before_hook(ctx: StageContext) -> None:
            # At this point extraction schema must NOT yet be loaded
            log.append('before' if ctx.extraction_schema is None else 'too_late')

        hook = ('extraction_schema_load', 'before', before_hook)
        pipeline = self._make_pipeline(stage_hooks=[hook])
        pipeline.run('text')
        assert 'before' in log
        assert 'too_late' not in log

    def test_after_hook_fires_after_stage(self):
        log: list[str] = []

        def after_hook(ctx: StageContext) -> None:
            log.append('after' if ctx.extraction_schema is not None else 'too_early')

        hook = ('extraction_schema_load', 'after', after_hook)
        pipeline = self._make_pipeline(stage_hooks=[hook])
        pipeline.run('text')
        assert 'after' in log
        assert 'too_early' not in log

    def test_multiple_hooks_all_fired(self):
        log: list[str] = []
        hooks = [
            ('extraction_schema_load', 'after', lambda ctx: log.append('h1')),
            ('llm_extraction', 'after', lambda ctx: log.append('h2')),
        ]
        pipeline = self._make_pipeline(stage_hooks=hooks)
        pipeline.run('text')
        assert 'h1' in log
        assert 'h2' in log

    def test_hook_context_has_expected_data_after_llm(self):
        """After the llm_extraction stage the context should have raw_output set."""
        received: list[Any] = []

        def capture(ctx: StageContext) -> None:
            received.append(ctx.raw_output)

        pipeline = self._make_pipeline(stage_hooks=[('llm_extraction', 'after', capture)])
        pipeline.run('text')
        assert len(received) == 1
        assert received[0] is not None

    def test_no_hooks_param_still_works(self):
        pipeline = self._make_pipeline(stage_hooks=None)
        result = pipeline.run('text')
        assert result.success is True


# ---------------------------------------------------------------------------
# LiteLLMEngine boundary: generate() must return str
# ---------------------------------------------------------------------------


class TestLiteLLMEngineReturnType:
    """LiteLLMEngine.generate() must return a plain str, not a ModelResponse."""

    def test_generate_returns_str(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"x": 1}'

        with (
            patch(
                'nomad_llm_extraction.pipeline.schema_filling.llm_engine.LiteLLMEngine.__init__',
                return_value=None,
            ),
            patch(
                'litellm.completion',
                return_value=mock_response,
            ),
        ):
            engine = LiteLLMEngine.__new__(LiteLLMEngine)
            engine.model_name = 'gpt-4o'
            engine.base_url = None
            engine.params = ['response_format']

            result = engine.generate('my prompt', {'type': 'object'})

        assert isinstance(result, str), (
            f'generate() should return str, got {type(result).__name__}'
        )
        assert result == '{"x": 1}'

    def test_generate_content_is_json_string(self):
        payload = {'cells': [1, 2]}
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(payload)

        with (
            patch(
                'nomad_llm_extraction.pipeline.schema_filling.llm_engine.LiteLLMEngine.__init__',
                return_value=None,
            ),
            patch(
                'litellm.completion',
                return_value=mock_response,
            ),
        ):
            engine = LiteLLMEngine.__new__(LiteLLMEngine)
            engine.model_name = 'gpt-4o'
            engine.base_url = None
            engine.params = ['response_format']

            raw = engine.generate('prompt', {'type': 'object'})

        parsed = json.loads(raw)
        assert parsed == payload


# ---------------------------------------------------------------------------
# PostprocessingSchemaLoadStage
# ---------------------------------------------------------------------------


class TestPostprocessingSchemaLoadStageRunner:
    def test_stage_loads_postprocessing_schema(self):
        source = MagicMock()
        source.get_schema.return_value = {'type': 'object', 'properties': {'x': {}}}
        stage = PostprocessingSchemaLoadStage(source)
        ctx = StageContext(text='text')
        result = stage.run(ctx)
        assert result.success is True
        assert ctx.postprocessing_schema == {'type': 'object', 'properties': {'x': {}}}


# ---------------------------------------------------------------------------
# ValidationStage
# ---------------------------------------------------------------------------


class TestValidationStage:
    def test_stage_name_is_validation(self):
        from nomad_llm_extraction.pipeline.stages import ValidationStage

        assert ValidationStage().name == 'validation'

    def test_empty_validators_succeeds(self):
        from nomad_llm_extraction.pipeline.stages import ValidationStage

        stage = ValidationStage()
        ctx = StageContext(text='text', extracted_data={'x': 1})
        result = stage.run(ctx)
        assert result.success is True

    def test_passing_validators_do_not_record_errors(self):
        from nomad_llm_extraction.pipeline.stages import ValidationStage

        stage = ValidationStage(validators=[lambda d: None, lambda d: None])
        ctx = StageContext(text='text', extracted_data={'x': 1})
        result = stage.run(ctx)
        assert result.success is True
        assert not ctx.metadata.get('validation_errors')

    def test_failing_validator_recorded_in_metadata(self):
        from nomad_llm_extraction.pipeline.stages import ValidationStage

        def bad_validator(data):
            raise ValueError('bad data')

        stage = ValidationStage(validators=[bad_validator])
        ctx = StageContext(text='text', extracted_data={})
        result = stage.run(ctx)
        # Stage itself still succeeds (non-aborting behaviour)
        assert result.success is True
        assert 'bad_validator' in ctx.metadata.get('validation_errors', {})

    def test_all_validators_called_even_after_failure(self):
        from nomad_llm_extraction.pipeline.stages import ValidationStage

        called: list[str] = []

        def v1(d):
            called.append('v1')
            raise ValueError('fail')

        def v2(d):
            called.append('v2')

        stage = ValidationStage(validators=[v1, v2])
        ctx = StageContext(text='text', extracted_data={})
        stage.run(ctx)
        assert called == ['v1', 'v2']

    def test_validation_errors_in_result_data(self):
        from nomad_llm_extraction.pipeline.stages import ValidationStage

        def failing(data):
            raise AssertionError('not valid')

        stage = ValidationStage(validators=[failing])
        ctx = StageContext(text='text', extracted_data={})
        result = stage.run(ctx)
        assert result.data is not None
        assert 'validation_errors' in result.data

    def test_hook_can_target_validation_stage(self):
        """A StageRunner hook registered on 'validation' must fire around the stage."""
        from nomad_llm_extraction.pipeline.stages import ValidationStage

        log: list[str] = []
        runner = StageRunner()
        runner.add_stage(ValidationStage())
        runner.add_hook('validation', 'after', lambda ctx: log.append('after_validation'))
        runner.run(StageContext(text='text', extracted_data={}))
        assert 'after_validation' in log


# ---------------------------------------------------------------------------
# PostprocessingStage
# ---------------------------------------------------------------------------


class TestPostprocessingStage:
    def test_stage_name_is_postprocessing(self):
        from nomad_llm_extraction.pipeline.stages import PostprocessingStage

        assert PostprocessingStage().name == 'postprocessing'

    def test_no_processor_passes_extracted_data_through(self):
        from nomad_llm_extraction.pipeline.stages import PostprocessingStage

        data = {'x': 1}
        stage = PostprocessingStage()
        ctx = StageContext(text='text', extracted_data=data)
        result = stage.run(ctx)
        assert result.success is True
        assert ctx.postprocessed_data is data

    def test_processor_callable_applied(self):
        from nomad_llm_extraction.pipeline.stages import PostprocessingStage

        stage = PostprocessingStage(processor=lambda d, s: {**d, 'added': s['name']})
        ctx = StageContext(
            text='text',
            extracted_data={'x': 1},
            postprocessing_schema={'name': True},
        )
        result = stage.run(ctx)
        assert result.success is True
        assert ctx.postprocessed_data == {'x': 1, 'added': True}

    def test_failure_on_processor_error(self):
        from nomad_llm_extraction.pipeline.stages import PostprocessingStage

        def bad(d, s):
            raise RuntimeError('crash')

        stage = PostprocessingStage(processor=bad)
        ctx = StageContext(text='text', extracted_data={})
        result = stage.run(ctx)
        assert result.success is False
        assert 'crash' in result.error

    def test_none_extracted_data_becomes_postprocessed_none(self):
        from nomad_llm_extraction.pipeline.stages import PostprocessingStage

        stage = PostprocessingStage()
        ctx = StageContext(text='text')  # extracted_data is None
        result = stage.run(ctx)
        assert result.success is True
        assert ctx.postprocessed_data is None


# ---------------------------------------------------------------------------
# ArchiveShapingStage
# ---------------------------------------------------------------------------


class TestArchiveShapingStage:
    def test_stage_name_is_archive_shaping(self):
        from nomad_llm_extraction.pipeline.stages import ArchiveShapingStage

        assert ArchiveShapingStage().name == 'archive_shaping'

    def test_no_shaper_passes_postprocessed_data_through(self):
        from nomad_llm_extraction.pipeline.stages import ArchiveShapingStage

        data = {'y': 2}
        stage = ArchiveShapingStage()
        ctx = StageContext(text='text', postprocessed_data=data)
        result = stage.run(ctx)
        assert result.success is True
        assert ctx.archive_data is data

    def test_shaper_callable_applied(self):
        from nomad_llm_extraction.pipeline.stages import ArchiveShapingStage

        stage = ArchiveShapingStage(shaper=lambda d: {'archive': d})
        ctx = StageContext(text='text', postprocessed_data={'y': 2})
        result = stage.run(ctx)
        assert result.success is True
        assert ctx.archive_data == {'archive': {'y': 2}}

    def test_failure_on_shaper_error(self):
        from nomad_llm_extraction.pipeline.stages import ArchiveShapingStage

        stage = ArchiveShapingStage(shaper=lambda d: (_ for _ in ()).throw(TypeError('bad')))
        ctx = StageContext(text='text', postprocessed_data={})
        result = stage.run(ctx)
        assert result.success is False
        assert 'bad' in result.error

    def test_none_postprocessed_data_becomes_archive_none(self):
        from nomad_llm_extraction.pipeline.stages import ArchiveShapingStage

        stage = ArchiveShapingStage()
        ctx = StageContext(text='text')  # postprocessed_data is None
        result = stage.run(ctx)
        assert result.success is True
        assert ctx.archive_data is None


# ---------------------------------------------------------------------------
# StageContext — new fields
# ---------------------------------------------------------------------------


class TestStageContextNewFields:
    def test_postprocessed_data_defaults_to_none(self):
        ctx = StageContext(text='text')
        assert ctx.postprocessed_data is None

    def test_archive_data_defaults_to_none(self):
        ctx = StageContext(text='text')
        assert ctx.archive_data is None

    def test_postprocessed_data_is_mutable(self):
        ctx = StageContext(text='text')
        ctx.postprocessed_data = {'p': 1}
        assert ctx.postprocessed_data == {'p': 1}

    def test_archive_data_is_mutable(self):
        ctx = StageContext(text='text')
        ctx.archive_data = {'a': 2}
        assert ctx.archive_data == {'a': 2}


# ---------------------------------------------------------------------------
# ExtractionPipeline — full stage ordering and validator integration
# ---------------------------------------------------------------------------


class TestExtractionPipelineFullStages:
    """Verify the complete ordered stage list and validator-as-stage behaviour."""

    def _make_pipeline(self, validators=None, postprocessor=None, archive_shaper=None,
                       stage_hooks=None):
        from nomad_llm_extraction.pipeline.extraction_pipeline import ExtractionPipeline

        engine = MagicMock()
        engine.generate.return_value = json.dumps({'result': 'ok'})
        extraction_schema_source = MagicMock()
        extraction_schema_source.get_schema.return_value = {'type': 'object'}
        postprocessing_schema_source = MagicMock()
        postprocessing_schema_source.get_schema.return_value = {
            'type': 'object',
            'properties': {'pp': {'type': 'string'}},
        }
        return ExtractionPipeline(
            engine=engine,
            extraction_schema_source=extraction_schema_source,
            postprocessing_schema_source=postprocessing_schema_source,
            validators=validators,
            postprocessor=postprocessor,
            archive_shaper=archive_shaper,
            stage_hooks=stage_hooks,
        )

    def test_stage_names_in_results(self):
        pipeline = self._make_pipeline()
        result = pipeline.run('text')
        assert result.success is True
        names = [s.name for s in result.stages]
        assert 'extraction_schema_load' in names
        assert 'postprocessing_schema_load' in names
        assert 'prompt_build' in names
        assert 'llm_extraction' in names
        assert 'json_parse' in names
        assert 'validation' in names
        assert 'postprocessing' in names
        assert 'archive_shaping' in names

    def test_stage_order(self):
        pipeline = self._make_pipeline()
        result = pipeline.run('text')
        names = [s.name for s in result.stages]
        expected_order = [
            'extraction_schema_load',
            'postprocessing_schema_load',
            'prompt_build',
            'llm_extraction', 'json_parse', 'validation',
            'postprocessing', 'archive_shaping',
        ]
        assert names == expected_order

    def test_validator_failure_recorded_in_stages(self):
        def bad_validator(data):
            raise ValueError('bad data')

        pipeline = self._make_pipeline(validators=[bad_validator])
        result = pipeline.run('text')
        assert result.success is True
        val_stage = next(s for s in result.stages if s.name == 'validation')
        assert val_stage.success is True  # non-aborting
        errors = val_stage.data.get('validation_errors', {})
        assert 'bad_validator' in errors

    def test_hook_on_validation_stage(self):
        log: list[str] = []
        pipeline = self._make_pipeline(
            stage_hooks=[('validation', 'after', lambda ctx: log.append('after_val'))]
        )
        pipeline.run('text')
        assert 'after_val' in log

    def test_hook_on_postprocessing_stage(self):
        log: list[str] = []
        pipeline = self._make_pipeline(
            stage_hooks=[('postprocessing', 'after', lambda ctx: log.append('after_post'))]
        )
        pipeline.run('text')
        assert 'after_post' in log

    def test_hook_on_archive_shaping_stage(self):
        log: list[str] = []
        pipeline = self._make_pipeline(
            stage_hooks=[('archive_shaping', 'after', lambda ctx: log.append('after_archive'))]
        )
        pipeline.run('text')
        assert 'after_archive' in log

    def test_hook_on_postprocessing_schema_load_stage(self):
        log: list[str] = []
        pipeline = self._make_pipeline(
            stage_hooks=[
                (
                    'postprocessing_schema_load',
                    'after',
                    lambda ctx: log.append('after_post_schema'),
                )
            ]
        )
        pipeline.run('text')
        assert 'after_post_schema' in log

    def test_postprocessor_applied(self):
        pipeline = self._make_pipeline(
            postprocessor=lambda d, s: {**d, 'processed': 'properties' in s}
        )
        captured: list[Any] = []
        pipeline._stage_hooks.append(
            ('postprocessing', 'after', lambda ctx: captured.append(ctx.postprocessed_data))
        )
        pipeline.run('text')
        assert captured and captured[0].get('processed') is True

    def test_archive_shaper_applied(self):
        pipeline = self._make_pipeline(
            archive_shaper=lambda d: {'shaped': d}
        )
        captured: list[Any] = []
        pipeline._stage_hooks.append(
            ('archive_shaping', 'after', lambda ctx: captured.append(ctx.archive_data))
        )
        pipeline.run('text')
        assert captured and 'shaped' in captured[0]

    def test_pipeline_result_has_archive_data(self):
        pipeline = self._make_pipeline()
        result = pipeline.run('text')
        assert result.success is True
        assert hasattr(result, 'archive_data')

    def test_pipeline_result_has_postprocessed_data(self):
        pipeline = self._make_pipeline()
        result = pipeline.run('text')
        assert result.success is True
        assert hasattr(result, 'postprocessed_data')
