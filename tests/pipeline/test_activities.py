import json
import sys
from types import SimpleNamespace

import pytest

from nomad_llm_extraction.pipeline import activities
from nomad_llm_extraction.pipeline.models import (
    BuildPromptInput,
    ExtractionValidationInput,
    InlineSchemaConfig,
    LLMCallInput,
    LLMEngineConfig,
    NomadSchemaConfig,
    NomadUnitConversionInput,
    UploadToNomadInput,
)


class SchemaSourceStub:
    result = {'normalized': True}
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)

    def get_schema(self):
        return self.result


def test_get_inline_schema_uses_inline_schema_and_options(monkeypatch):
    monkeypatch.setattr(activities, 'InlineSchemaSource', SchemaSourceStub)
    config = InlineSchemaConfig(
        inline_schema={'type': 'object'},
        remove_defs=True,
        resolve_allOf=True,
        remove_null_anyof=True,
        exclude={'key': ['ignore']},
        multi_instance_field='items',
    )

    assert activities.get_inline_schema(config) == {'normalized': True}
    assert SchemaSourceStub.instances[-1].kwargs == {
        'schema': {'type': 'object'},
        'remove_defs': True,
        'resolve_allOf': True,
        'remove_null_anyof': True,
        'exclude': {'key': ['ignore']},
        'multi_instance_field': 'items',
    }


def test_get_inline_schema_loads_schema_file(tmp_path, monkeypatch):
    monkeypatch.setattr(activities, 'InlineSchemaSource', SchemaSourceStub)
    schema_path = tmp_path / 'schema.json'
    schema_path.write_text(json.dumps({'title': 'from file'}))

    result = activities.get_inline_schema(
        InlineSchemaConfig(schema_path=str(schema_path))
    )

    assert result == {'normalized': True}
    assert SchemaSourceStub.instances[-1].kwargs['schema'] == {'title': 'from file'}


def test_get_inline_schema_requires_a_schema():
    with pytest.raises(ValueError, match='Either schema or schema_path'):
        activities.get_inline_schema(InlineSchemaConfig())


def test_get_nomad_schema_passes_configuration_to_source(monkeypatch):
    monkeypatch.setattr(activities, 'NomadSchemaSource', SchemaSourceStub)
    config = NomadSchemaConfig(m_def='package.Section', unit_value=True)

    assert activities.get_nomad_schema(config) == {'normalized': True}
    assert SchemaSourceStub.instances[-1].kwargs['m_def'] == 'package.Section'
    assert SchemaSourceStub.instances[-1].kwargs['unit_value'] is True


def test_parse_text_from_pdf_disables_cache_and_returns_doi(monkeypatch):
    created_with = []

    class Parser:
        def __init__(self, **kwargs):
            created_with.append(kwargs)

        def parse_pdf(self, path):
            assert path == 'paper.pdf'
            return 'paper text'

    monkeypatch.setattr(activities, 'PDFParser', Parser)
    monkeypatch.setattr(activities, 'extract_doi_from_pdf', lambda path: '10.1/example')

    assert activities.parse_text_from_pdf('paper.pdf') == ('paper text', '10.1/example')
    assert created_with == [{'use_cache': False}]


@pytest.mark.parametrize(
    ('system_prompt', 'instruction_text', 'expected'),
    [
        (
            'system',
            'extract values',
            'system\nextract values\nHere is the schema: {\n  "type": "object"\n}\n'
            'Here is the text:\nsource',
        ),
        (
            '',
            '',
            'Here is the schema: {\n  "type": "object"\n}\nHere is the text:\nsource',
        ),
    ],
)
def test_build_prompt_omits_empty_optional_sections(
    system_prompt, instruction_text, expected
):
    prompt = activities.build_prompt(
        BuildPromptInput(
            text='source',
            extraction_schema={'type': 'object'},
            system_prompt=system_prompt,
            instruction_text=instruction_text,
        )
    )

    assert prompt == expected


def test_llm_call_constructs_engine_and_returns_raw_response(monkeypatch):
    constructed = []

    class Engine:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def generate(self, prompt, schema, optional_params):
            assert (prompt, schema, optional_params) == (
                'prompt',
                {'type': 'object'},
                {'temperature': 0},
            )
            return 'raw response'

    monkeypatch.setitem(
        sys.modules,
        'nomad_llm_extraction.pipeline.schema_filling.llm_engine',
        SimpleNamespace(LiteLLMEngine=Engine),
    )

    result = activities.llm_call(
        LLMCallInput(
            prompt='prompt',
            extraction_schema={'type': 'object'},
            engine_config=LLMEngineConfig(model_name='test-model'),
            optional_params={'temperature': 0},
        )
    )

    assert result == 'raw response'
    assert constructed == [
        {'model_name': 'test-model', 'api_key': None, 'api_url': None}
    ]


def test_json_parse_reports_invalid_json():
    assert activities.json_parse('{"number": 1}') == (False, {'number': 1})

    failed, payload = activities.json_parse('{not json}')

    assert failed is True
    assert payload['error'].startswith('Failed to parse JSON:')


def test_validate_extraction_with_schema_returns_validation_result(monkeypatch):
    monkeypatch.setattr(
        activities, 'validate_with_schema', lambda data, schema: (False, 'missing name')
    )

    result = activities.validate_extraction_with_schema(
        ExtractionValidationInput(
            extracted_data={'name': None}, extraction_schema={'required': ['name']}
        )
    )

    assert result.validated is False
    assert result.message == 'missing name'


def test_upload_to_nomad_forwards_all_upload_fields(monkeypatch):
    received = {}
    monkeypatch.setattr(
        activities,
        'upload_extraction_to_nomad',
        lambda **kwargs: received.update(kwargs),
    )
    upload = UploadToNomadInput(
        m_def='package.Section',
        data={'name': 'sample'},
        entry_name='entry',
        doi='10.1/example',
        extraction_metadata={'model_name': 'test'},
        multi_instance_field='samples',
        upload_id='upload-id',
    )

    assert activities.upload_to_nomad(upload) is None
    assert received == upload.model_dump()


def test_convert_nomad_units_applies_unit_conversion_pipeline(monkeypatch):
    received = {}

    class Pipeline:
        def __init__(self, steps):
            received['steps'] = steps

        def apply(self, data, schema):
            received['args'] = (data, schema)
            return {'converted': data}

    monkeypatch.setattr(activities, 'ProcessingPipeline', Pipeline)

    result = activities.convert_nomad_units(
        NomadUnitConversionInput(data={'mass': 1}, proc_schema={'mass': 'kg'})
    )

    assert result == {'converted': {'mass': 1}}
    assert received['steps'] == {
        'unit_conversion': [
            activities.convert_unit,
            activities.unit_cond,
            activities.unit_args,
        ]
    }
    assert received['args'] == ({'mass': 1}, {'mass': 'kg'})
