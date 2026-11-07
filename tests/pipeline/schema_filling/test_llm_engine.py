import copy
from types import SimpleNamespace

import pytest

from nomad_llm_extraction.pipeline.schema_filling import llm_engine

# what pydantic writes for `unresolved: list[Ambiguous | NotFound]` with
# `Field(discriminator='reason')`
UNION_SCHEMA = {
    '$defs': {
        'Ambiguous': {
            'type': 'object',
            'properties': {
                'reason': {'const': 'ambiguous', 'type': 'string'},
                'candidates': {'type': 'array', 'items': {'enum': ['C001', 'C002']}},
                'said': {'type': 'string'},
            },
            'required': ['reason', 'candidates', 'said'],
            'additionalProperties': False,
        },
        'NotFound': {
            'type': 'object',
            'properties': {
                'reason': {'const': 'not_found', 'type': 'string'},
                'said': {'type': 'string'},
            },
            'required': ['reason', 'said'],
            'additionalProperties': False,
        },
    },
    'type': 'object',
    'properties': {
        'unresolved': {
            'type': 'array',
            'items': {
                'oneOf': [{'$ref': '#/$defs/Ambiguous'}, {'$ref': '#/$defs/NotFound'}],
                'discriminator': {
                    'propertyName': 'reason',
                    'mapping': {
                        'ambiguous': '#/$defs/Ambiguous',
                        'not_found': '#/$defs/NotFound',
                    },
                },
            },
        }
    },
    'required': ['unresolved'],
}


def test_enforceable_schema_turns_oneof_into_anyof_and_drops_the_discriminator():
    given = copy.deepcopy(UNION_SCHEMA)

    out = llm_engine.enforceable_schema(given)

    items = out['properties']['unresolved']['items']
    assert items == {
        'anyOf': [{'$ref': '#/$defs/Ambiguous'}, {'$ref': '#/$defs/NotFound'}]
    }
    # the tag of each branch is an enum of one, the rest is as given
    assert out['$defs']['NotFound']['properties']['reason'] == {
        'enum': ['not_found'],
        'type': 'string',
    }
    assert (
        out['$defs']['Ambiguous']['properties']['candidates']
        == (UNION_SCHEMA['$defs']['Ambiguous']['properties']['candidates'])
    )
    assert given == UNION_SCHEMA  # the caller's schema is not touched


def test_enforceable_schema_rewrites_at_any_depth_and_keeps_an_existing_anyof():
    out = llm_engine.enforceable_schema(
        {
            'anyOf': [{'type': 'null'}],
            'oneOf': [{'type': 'string'}],
            'properties': {'a': {'items': [{'oneOf': [{'type': 'integer'}]}]}},
            'tag': {'const': 'x', 'enum': ['x', 'y']},
        }
    )

    assert out == {
        'anyOf': [{'type': 'null'}, {'type': 'string'}],
        'properties': {'a': {'items': [{'anyOf': [{'type': 'integer'}]}]}},
        'tag': {'enum': ['x', 'y']},
    }
    assert llm_engine.enforceable_schema('x') == 'x'


def test_the_union_survives_litellms_rewrite_for_gemini():
    """The failure this guards against: litellm's Gemini schema builder
    drops `oneOf`, leaving `items: {}`, while the answer is still validated
    against the full schema."""
    common_utils = pytest.importorskip('litellm.llms.vertex_ai.common_utils')
    build = getattr(common_utils, '_build_vertex_schema', None)
    if build is None:
        pytest.skip('litellm no longer has _build_vertex_schema')

    as_given = build(copy.deepcopy(UNION_SCHEMA))
    as_sent = build(llm_engine.enforceable_schema(UNION_SCHEMA))

    assert as_given['properties']['unresolved']['items'] == {}
    branches = as_sent['properties']['unresolved']['items']['anyOf']
    assert [b['properties']['reason'] for b in branches] == [
        {'enum': ['ambiguous'], 'type': 'string'},
        {'enum': ['not_found'], 'type': 'string'},
    ]


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


def _completion_recorder(completion_calls, content='{}'):
    return lambda **kwargs: (
        completion_calls.append(kwargs)
        or SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content,
                        tool_calls=[
                            SimpleNamespace(function=SimpleNamespace(arguments=content))
                        ],
                    )
                )
            ]
        )
    )


def test_response_format_sends_the_enforceable_schema(monkeypatch):
    engine = llm_engine.LiteLLMEngine('test-model')
    completion_calls = []
    monkeypatch.setattr(engine, 'check_additional_params', lambda params: None)
    monkeypatch.setattr(
        llm_engine.litellm, 'completion', _completion_recorder(completion_calls)
    )

    engine.generate_with_response_format('prompt', UNION_SCHEMA)

    sent = completion_calls[0]['response_format']['json_schema']['schema']
    assert sent == llm_engine.enforceable_schema(UNION_SCHEMA)
    assert 'oneOf' not in str(sent) and 'discriminator' not in str(sent)


def test_tool_call_sends_the_enforceable_schema(monkeypatch):
    engine = llm_engine.LiteLLMEngine('test-model')
    completion_calls = []
    monkeypatch.setattr(engine, 'check_additional_params', lambda params: None)
    monkeypatch.setattr(
        llm_engine.litellm, 'completion', _completion_recorder(completion_calls)
    )

    engine.generate_with_tool_call('prompt', UNION_SCHEMA)

    sent = completion_calls[0]['tools'][0]['function']['parameters']
    assert (
        sent['properties'] == llm_engine.enforceable_schema(UNION_SCHEMA)['properties']
    )
    assert 'oneOf' not in str(sent) and 'discriminator' not in str(sent)
