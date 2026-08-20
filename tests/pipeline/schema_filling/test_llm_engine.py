from types import SimpleNamespace

import pytest

from nomad_llm_extraction.pipeline.schema_filling import llm_engine


def test_format_tool_call_schema_strips_metadata_and_sorts_required_fields():
    schema = {
        'title': 'Measurement',
        'description': 'A measured value.',
        'type': 'object',
        'properties': {'value': {'type': 'number'}},
        'required': ['value', 'name'],
    }

    assert llm_engine.format_tool_call_schema(schema) == {
        'name': 'Measurement',
        'description': 'A measured value.',
        'parameters': {
            'type': 'object',
            'properties': {'value': {'type': 'number'}},
            'required': ['name', 'value'],
        },
    }


def test_structured_llm_engine_requires_subclass_implementation():
    with pytest.raises(NotImplementedError, match='Subclasses must implement'):
        llm_engine.StructuredLLMEngine().generate('prompt', '{}')


def test_init_records_supported_params_and_sets_api_key(monkeypatch):
    monkeypatch.setattr(
        llm_engine,
        'get_supported_openai_params',
        lambda *, model: ['response_format', 'tools', 'tool_choice'],
    )
    monkeypatch.setattr(llm_engine, 'supports_response_schema', lambda *, model: True)

    engine = llm_engine.LiteLLMEngine('test-model', 'https://example.invalid', 'key')

    assert engine.model_name == 'test-model'
    assert engine.base_url == 'https://example.invalid'
    assert engine.params == ['response_format', 'tools', 'tool_choice']
    assert llm_engine.litellm.api_key == 'key'


def test_generate_dispatches_to_selected_generation_method(monkeypatch):
    engine = llm_engine.LiteLLMEngine('test-model')
    response_format = object()
    tool_call = object()
    monkeypatch.setattr(
        engine, 'generate_with_response_format', lambda *args: response_format
    )
    monkeypatch.setattr(engine, 'generate_with_tool_call', lambda *args: tool_call)

    assert (
        engine.generate('prompt', {}, {'temperature': 0}, 'response_format')
        is response_format
    )
    assert engine.generate('prompt', {}, {}, 'tool_call') is tool_call


def test_generate_rejects_unknown_method():
    engine = llm_engine.LiteLLMEngine('test-model')

    with pytest.raises(ValueError, match='Invalid method: unknown'):
        engine.generate('prompt', {}, method='unknown')


def test_response_format_sends_strict_schema_and_returns_content(monkeypatch):
    engine = llm_engine.LiteLLMEngine('test-model', 'https://example.invalid')
    completion_calls = []
    monkeypatch.setattr(engine, 'check_additional_params', lambda params: None)
    monkeypatch.setattr(
        llm_engine.litellm,
        'completion',
        lambda **kwargs: (
            completion_calls.append(kwargs)
            or SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"value": 1}'))
                ]
            )
        ),
    )

    result = engine.generate_with_response_format(
        'extract this', {'type': 'object'}, {'temperature': 0}
    )

    assert result == '{"value": 1}'
    assert completion_calls == [
        {
            'model': 'test-model',
            'api_base': 'https://example.invalid',
            'messages': [{'role': 'user', 'content': 'extract this'}],
            'response_format': {
                'type': 'json_schema',
                'json_schema': {
                    'schema': {'type': 'object'},
                    'strict': True,
                    'name': 'ResponseSchema',
                },
            },
            'drop_params': True,
            'temperature': 0,
        }
    ]


def test_response_format_reraises_malformed_completion_response(monkeypatch):
    engine = llm_engine.LiteLLMEngine('test-model')
    monkeypatch.setattr(engine, 'check_additional_params', lambda params: None)
    monkeypatch.setattr(
        llm_engine.litellm,
        'completion',
        lambda **kwargs: SimpleNamespace(choices=[]),
    )

    with pytest.raises(IndexError):
        engine.generate_with_response_format('prompt', {})


def test_tool_call_parses_schema_and_returns_function_arguments(monkeypatch):
    engine = llm_engine.LiteLLMEngine('test-model')
    completion_calls = []
    monkeypatch.setattr(engine, 'check_additional_params', lambda params: None)
    monkeypatch.setattr(
        llm_engine.litellm,
        'completion',
        lambda **kwargs: (
            completion_calls.append(kwargs)
            or SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(
                                        arguments='{"name": "sample"}'
                                    )
                                )
                            ]
                        )
                    )
                ]
            )
        ),
    )

    result = engine.generate_with_tool_call(
        'extract this',
        '{"title": "Sample", "type": "object", "required": ["name"]}',
        {'max_tokens': 10},
    )

    assert result == '{"name": "sample"}'
    assert completion_calls[0]['tools'] == [
        {
            'type': 'function',
            'function': {
                'name': 'Sample',
                'description': 'Extracted data from the document.',
                'parameters': {
                    'type': 'object',
                    'required': ['name'],
                },
            },
        }
    ]
    assert completion_calls[0]['tool_choice'] == {
        'type': 'function',
        'function': {'name': 'Sample'},
    }
    assert completion_calls[0]['reasoning_effort'] == 'none'
    assert completion_calls[0]['max_tokens'] == 10


def test_tool_call_rejects_invalid_json_schema_before_completion(monkeypatch):
    engine = llm_engine.LiteLLMEngine('test-model')
    monkeypatch.setattr(
        llm_engine.litellm,
        'completion',
        lambda **kwargs: pytest.fail('completion must not be called'),
    )

    with pytest.raises(ValueError, match='Expecting'):
        engine.generate_with_tool_call('prompt', '{not-json}')
