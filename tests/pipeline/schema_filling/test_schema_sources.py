import pytest

from nomad_llm_extraction.pipeline.schema_filling import schema_sources


def test_schema_source_requires_subclass_to_set_schema():
    with pytest.raises(NotImplementedError, match='Subclasses must set self._schema'):
        schema_sources.SchemaSource().get_schema()


def test_inline_source_prunes_resolves_and_optimizes_copy(monkeypatch):
    input_schema = {
        'title': 'Input',
        'properties': {'keep': {'type': 'string'}, 'remove': {'type': 'string'}},
    }
    calls = []

    def prune(schema, values, *, by):
        calls.append(('prune', values, by))
        schema['properties'].pop('remove')
        return schema

    def resolve(schema, **kwargs):
        calls.append(('resolve', kwargs))
        schema['resolved'] = True
        return schema

    def optimize(schema):
        calls.append(('optimize',))
        schema['optimized'] = True
        return schema

    monkeypatch.setattr(schema_sources, 'prune_schema', prune)
    monkeypatch.setattr(schema_sources, 'resolve_schema', resolve)
    source = schema_sources.InlineSchemaSource(
        input_schema,
        optimizer=optimize,
        remove_defs=True,
        resolve_allOf=True,
        remove_null_anyof=True,
        exclude={'property': ['remove']},
    )

    result = source.get_schema()

    assert result == {
        'title': 'Input',
        'properties': {'keep': {'type': 'string'}},
        'resolved': True,
        'optimized': True,
    }
    assert input_schema['properties']['remove'] == {'type': 'string'}
    assert calls == [
        ('prune', ['remove'], 'property'),
        (
            'resolve',
            {
                'remove_defs': True,
                'resolve_allOf': True,
                'remove_null_anyof': True,
            },
        ),
        ('optimize',),
    ]


def test_inline_source_wraps_single_instance_schema_after_processing(monkeypatch):
    monkeypatch.setattr(
        schema_sources, 'resolve_schema', lambda schema, **kwargs: schema
    )
    source = schema_sources.InlineSchemaSource(
        {'title': 'Sample', 'description': 'A sample.', 'type': 'object'},
        multi_instance_field='samples',
    )

    assert source.get_schema() == {
        'title': 'SampleInstances',
        'description': 'A sample.',
        'type': 'object',
        'properties': {
            'samples': {
                'type': 'array',
                'items': {
                    'title': 'Sample',
                    'description': 'A sample.',
                    'type': 'object',
                },
            }
        },
    }


def test_nomad_source_fetches_schema_with_requested_unit_value(monkeypatch):
    schema = {'title': 'Nomad', 'type': 'object'}
    calls = []
    monkeypatch.setattr(
        schema_sources,
        'get_nomad_schema',
        lambda m_def, *, unit_value: calls.append((m_def, unit_value)) or schema,
    )
    monkeypatch.setattr(schema_sources, 'resolve_schema', lambda value, **kwargs: value)

    result = schema_sources.NomadSchemaSource('EntryData', unit_value=True).get_schema()

    assert result == schema
    assert calls == [('EntryData', True)]
