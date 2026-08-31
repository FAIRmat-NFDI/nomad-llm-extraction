from types import SimpleNamespace

import pytest
from temporalio.exceptions import ApplicationError

from nomad_llm_extraction.pipeline import workflows
from nomad_llm_extraction.pipeline.models import (
    ExtractionValidationOutput,
    ExtractionWorkflowInput,
    InlineSchemaConfig,
    LLMCallInput,
    LLMCallOutput,
    LLMEngineConfig,
    NomadSchemaConfig,
)


@pytest.fixture
def engine_config():
    return LLMEngineConfig(model_name='test-model')


@pytest.fixture(autouse=True)
def mock_workflow_logger(monkeypatch):
    monkeypatch.setattr(
        workflows.workflow,
        'logger',
        SimpleNamespace(
            warning=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None
        ),
    )
    # workflow.info() only works inside a real workflow context; the child
    # workflow ids are prefixed with the parent's id
    monkeypatch.setattr(
        workflows.workflow,
        'info',
        lambda: SimpleNamespace(workflow_id='wf-test'),
    )


@pytest.fixture
def llm_input(engine_config):
    return LLMCallInput(
        prompt='extract this',
        extraction_schema={'type': 'object'},
        engine_config=engine_config,
    )


@pytest.mark.asyncio
async def test_llm_call_workflow_returns_validated_extraction(monkeypatch, llm_input):
    calls = []

    async def execute_activity(activity, value, **kwargs):
        calls.append((activity, value, kwargs))
        if activity is workflows.llm_call:
            return '{"bandgap": 1.5}'
        if activity is workflows.json_parse:
            return False, {'bandgap': 1.5}
        return ExtractionValidationOutput(validated=True)

    monkeypatch.setattr(workflows.workflow, 'execute_activity', execute_activity)

    result = await workflows.LLMCallWorkflow().run(llm_input)

    assert result == LLMCallOutput(
        extracted_data={'bandgap': 1.5},
        raw_output='{"bandgap": 1.5}',
    )
    assert [call[0] for call in calls] == [
        workflows.llm_call,
        workflows.json_parse,
        workflows.validate_extraction_with_schema,
    ]
    assert calls[2][1].extraction_schema == llm_input.extraction_schema
    assert calls[0][2]['retry_policy'] == workflows.DEFAULT_RETRY_POLICY


@pytest.mark.asyncio
async def test_llm_call_workflow_rejects_empty_llm_output(monkeypatch, llm_input):
    async def execute_activity(*args, **kwargs):
        return ''

    monkeypatch.setattr(workflows.workflow, 'execute_activity', execute_activity)

    with pytest.raises(Exception, match='LLM did not return any output'):
        await workflows.LLMCallWorkflow().run(llm_input)


@pytest.mark.asyncio
async def test_llm_call_workflow_raises_json_parser_error(monkeypatch, llm_input):
    async def execute_activity(activity, *args, **kwargs):
        if activity is workflows.llm_call:
            return 'not json'
        return True, {'error': 'invalid JSON'}

    monkeypatch.setattr(workflows.workflow, 'execute_activity', execute_activity)

    with pytest.raises(Exception, match='invalid JSON'):
        await workflows.LLMCallWorkflow().run(llm_input)


@pytest.mark.asyncio
async def test_llm_call_workflow_returns_validation_error(monkeypatch, llm_input):
    async def execute_activity(activity, *args, **kwargs):
        if activity is workflows.llm_call:
            return '{"wrong": true}'
        if activity is workflows.json_parse:
            return False, {'wrong': True}
        return ExtractionValidationOutput(validated=False, message='missing bandgap')

    monkeypatch.setattr(workflows.workflow, 'execute_activity', execute_activity)

    result = await workflows.LLMCallWorkflow().run(llm_input)

    assert result.extracted_data == {'wrong': True}
    assert result.err_message == 'ValidationError: missing bandgap'


@pytest.mark.asyncio
async def test_extraction_workflow_builds_pdf_prompt_and_retries(
    monkeypatch, engine_config
):
    activities = []
    child_inputs = []
    child_results = iter(
        [
            LLMCallOutput(
                extracted_data={'bad': True},
                raw_output='bad output',
                err_message='schema mismatch',
            ),
            LLMCallOutput(extracted_data={'bandgap': 1.5}, raw_output='good output'),
        ]
    )

    async def execute_activity(activity, value, **kwargs):
        activities.append((activity, value, kwargs))
        if activity is workflows.parse_text_from_pdf:
            return 'pdf text', '10.1234/example'
        assert activity is workflows.build_prompt
        return 'built prompt'

    async def execute_child_workflow(workflow, value, **kwargs):
        child_inputs.append((workflow, value, kwargs))
        return next(child_results)

    monkeypatch.setattr(workflows.workflow, 'execute_activity', execute_activity)
    monkeypatch.setattr(
        workflows.workflow, 'execute_child_workflow', execute_child_workflow
    )
    inp = ExtractionWorkflowInput(
        extraction_schema={'type': 'object'},
        pdf_path='paper.pdf',
        llm_engine_config=engine_config,
        max_retry_attempts=2,
    )

    result = await workflows.ExtractionWorkflow().run(inp)

    assert result.extracted_data == {'bandgap': 1.5}
    assert result.retries == 1
    assert 'Previous attempt output:\nbad output' in result.retry_prompt
    assert [call[0] for call in activities] == [
        workflows.parse_text_from_pdf,
        workflows.build_prompt,
    ]
    assert child_inputs[0][2]['id'] == 'wf-test_llm_call_attempt_0'
    assert child_inputs[1][1].prompt.startswith('built prompt')
    assert 'schema mismatch' in child_inputs[1][1].prompt


@pytest.mark.asyncio
async def test_extraction_workflow_returns_result_after_retry_limit(
    monkeypatch, engine_config
):
    child_inputs = []

    async def execute_child_workflow(workflow, value, **kwargs):
        child_inputs.append((workflow, value, kwargs))
        return LLMCallOutput(
            extracted_data={'bad': True}, raw_output='bad', err_message='invalid'
        )

    monkeypatch.setattr(
        workflows.workflow, 'execute_child_workflow', execute_child_workflow
    )
    inp = ExtractionWorkflowInput(
        extraction_schema={'type': 'object'},
        prompt='ready prompt',
        llm_engine_config=engine_config,
        max_retry_attempts=2,
    )

    result = await workflows.ExtractionWorkflow().run(inp)

    assert result.err_message == 'Max retry attempts reached.'
    assert result.retries == 2
    assert len(child_inputs) == 2
    assert child_inputs[1][1].prompt.startswith('ready prompt')
    assert 'Previous attempt output:\nbad' in result.retry_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('extraction_schema', 'text', 'pdf_path', 'message'),
    [
        (
            None,
            'text',
            None,
            'Either extraction_schema or schema_config must be provided',
        ),
        ({'type': 'object'}, None, 'empty.pdf', 'No text parsed from PDF:empty.pdf'),
    ],
)
async def test_extraction_workflow_wraps_input_failures(
    monkeypatch, engine_config, extraction_schema, text, pdf_path, message
):
    async def execute_activity(activity, *args, **kwargs):
        assert activity is workflows.parse_text_from_pdf
        return '', None

    monkeypatch.setattr(workflows.workflow, 'execute_activity', execute_activity)
    inp = ExtractionWorkflowInput(
        extraction_schema=extraction_schema,
        text=text,
        pdf_path=pdf_path,
        llm_engine_config=engine_config,
    )

    with pytest.raises(ApplicationError, match=message) as error:
        await workflows.ExtractionWorkflow().run(inp)

    assert error.value.non_retryable is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('schema_config', 'schema_activity'),
    [
        (NomadSchemaConfig(m_def='example.section'), workflows.get_nomad_schema),
        (
            InlineSchemaConfig(inline_schema={'type': 'object'}),
            workflows.get_inline_schema,
        ),
    ],
)
async def test_extraction_workflow_resolves_schema_config_before_llm_call(
    monkeypatch, engine_config, schema_config, schema_activity
):
    activities = []
    child_inputs = []

    async def execute_activity(activity, value, **kwargs):
        activities.append((activity, value))
        if activity is schema_activity:
            assert value is schema_config
            return {'type': 'object'}
        assert activity is workflows.build_prompt
        assert value.extraction_schema == {'type': 'object'}
        return 'built prompt'

    async def execute_child_workflow(workflow, value, **kwargs):
        child_inputs.append((workflow, value, kwargs))
        return LLMCallOutput(extracted_data={'value': 1}, raw_output='raw output')

    monkeypatch.setattr(workflows.workflow, 'execute_activity', execute_activity)
    monkeypatch.setattr(
        workflows.workflow, 'execute_child_workflow', execute_child_workflow
    )
    inp = ExtractionWorkflowInput(
        schema_config=schema_config,
        text='source text',
        llm_engine_config=engine_config,
    )

    result = await workflows.ExtractionWorkflow().run(inp)

    assert result.extracted_data == {'value': 1}
    assert [activity for activity, _ in activities] == [
        schema_activity,
        workflows.build_prompt,
    ]
    assert child_inputs[0][1].extraction_schema == {'type': 'object'}


@pytest.mark.asyncio
async def test_extraction_workflow_wraps_schema_resolution_error(
    monkeypatch, engine_config
):
    async def execute_activity(*args, **kwargs):
        raise RuntimeError('schema service unavailable')

    monkeypatch.setattr(workflows.workflow, 'execute_activity', execute_activity)
    inp = ExtractionWorkflowInput(
        schema_config=InlineSchemaConfig(inline_schema={'type': 'object'}),
        text='source text',
        llm_engine_config=engine_config,
    )

    with pytest.raises(ApplicationError, match='schema service unavailable') as error:
        await workflows.ExtractionWorkflow().run(inp)

    assert error.value.non_retryable is True
