"""Tests for pipeline/schema_sources.py (Task 2).

Covers:
- InlineSchemaSource: inline dict, $ref resolution, optimizer hooks
- NomadSchemaSource: m_def fetch, resolution, optimizer hooks, unit_value flag
- Protocol conformance for SchemaSource
- ExtractionPipeline integration using the new concrete sources
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nomad_llm_extraction.pipeline.extraction_pipeline import SchemaSource
from nomad_llm_extraction.pipeline.schema_sources import (
    InlineSchemaSource,
    NomadSchemaSource,
    SchemaOptimizer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'value': {'type': 'number'},
    },
}

SCHEMA_WITH_DEFS: dict[str, Any] = {
    '$defs': {
        'Measurement': {
            'type': 'object',
            'properties': {'amount': {'type': 'number'}},
        }
    },
    'type': 'object',
    'properties': {
        'result': {'$ref': '#/$defs/Measurement'},
    },
}


# ---------------------------------------------------------------------------
# InlineSchemaSource
# ---------------------------------------------------------------------------


class TestInlineSchemaSource:
    def test_returns_schema_dict(self):
        source = InlineSchemaSource(SIMPLE_SCHEMA)
        schema = source.get_schema()
        assert isinstance(schema, dict)

    def test_schema_content_preserved(self):
        source = InlineSchemaSource(SIMPLE_SCHEMA)
        schema = source.get_schema()
        assert schema['type'] == 'object'
        assert 'name' in schema['properties']

    def test_does_not_mutate_input(self):
        original = json.loads(json.dumps(SIMPLE_SCHEMA))
        source = InlineSchemaSource(SIMPLE_SCHEMA)
        source.get_schema()
        assert SIMPLE_SCHEMA == original

    def test_resolves_refs_by_default(self):
        """$ref definitions should be inlined by default."""
        source = InlineSchemaSource(SCHEMA_WITH_DEFS)
        schema = source.get_schema()
        result_prop = schema['properties']['result']
        # After ref resolution, the $ref is replaced with the actual schema
        assert '$ref' not in result_prop
        assert result_prop.get('type') == 'object' or 'properties' in result_prop

    def test_remove_defs_option(self):
        source = InlineSchemaSource(SCHEMA_WITH_DEFS, remove_defs=True)
        schema = source.get_schema()
        assert '$defs' not in schema

    def test_optimizer_applied_after_resolution(self):
        """Optimizer callable receives the resolved schema and its return value is used."""

        def add_title(s: dict) -> dict:
            s = dict(s)
            s['title'] = 'Optimized'
            return s

        source = InlineSchemaSource(SIMPLE_SCHEMA, optimizer=add_title)
        schema = source.get_schema()
        assert schema.get('title') == 'Optimized'

    def test_optimizer_can_drop_properties(self):
        """Optimizer may return a simplified schema."""

        def strip_properties(s: dict) -> dict:
            s = dict(s)
            s.pop('properties', None)
            return s

        source = InlineSchemaSource(SIMPLE_SCHEMA, optimizer=strip_properties)
        schema = source.get_schema()
        assert 'properties' not in schema

    def test_optimizer_receives_resolved_schema(self):
        """Optimizer must receive the resolved (ref-expanded) schema, not the raw input."""
        received: list[dict] = []

        def capture(s: dict) -> dict:
            received.append(s)
            return s

        source = InlineSchemaSource(SCHEMA_WITH_DEFS, optimizer=capture)
        source.get_schema()
        assert len(received) == 1
        # The captured schema should have refs resolved
        result_prop = received[0]['properties']['result']
        assert '$ref' not in result_prop

    def test_no_optimizer_by_default(self):
        source = InlineSchemaSource(SIMPLE_SCHEMA)
        schema = source.get_schema()
        # No extra fields injected
        assert 'title' not in schema

    def test_satisfies_schema_source_protocol(self):
        source = InlineSchemaSource(SIMPLE_SCHEMA)
        assert isinstance(source, SchemaSource)

    def test_callable_type_alias_accepted(self):
        """Ensure SchemaOptimizer type alias is importable and usable as annotation."""
        optimizer: SchemaOptimizer = lambda s: s  # noqa: E731
        source = InlineSchemaSource(SIMPLE_SCHEMA, optimizer=optimizer)
        schema = source.get_schema()
        assert schema is not None


# ---------------------------------------------------------------------------
# NomadSchemaSource
# ---------------------------------------------------------------------------

_FAKE_NOMAD_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'efficiency': {'type': 'number'},
        'bandgap': {'type': 'number'},
    },
}


class TestNomadSchemaSource:
    """Tests that mock the HTTP call inside get_nomad_schema."""

    def _patched_source(
        self,
        m_def: str = 'nomad.datamodel.SomeSection',
        *,
        unit_value: bool = False,
        optimizer: SchemaOptimizer | None = None,
        remove_defs: bool = False,
        resolve_allOf: bool = False,
        fake_schema: dict | None = None,
    ) -> NomadSchemaSource:
        return NomadSchemaSource(
            m_def,
            unit_value=unit_value,
            optimizer=optimizer,
            remove_defs=remove_defs,
            resolve_allOf=resolve_allOf,
        )

    @patch(
        'nomad_llm_extraction.pipeline.schema_sources.get_nomad_schema',
        return_value=_FAKE_NOMAD_SCHEMA,
    )
    def test_returns_schema_dict(self, mock_fetch):
        source = self._patched_source()
        schema = source.get_schema()
        assert isinstance(schema, dict)

    @patch(
        'nomad_llm_extraction.pipeline.schema_sources.get_nomad_schema',
        return_value=_FAKE_NOMAD_SCHEMA,
    )
    def test_schema_content_from_nomad(self, mock_fetch):
        source = self._patched_source()
        schema = source.get_schema()
        assert 'efficiency' in schema.get('properties', {})

    @patch(
        'nomad_llm_extraction.pipeline.schema_sources.get_nomad_schema',
        return_value=_FAKE_NOMAD_SCHEMA,
    )
    def test_calls_get_nomad_schema_with_m_def(self, mock_fetch):
        m_def = 'nomad.datamodel.SomeSection'
        source = self._patched_source(m_def)
        source.get_schema()
        mock_fetch.assert_called_once()
        call_kwargs = mock_fetch.call_args
        # m_def should be the first positional arg
        assert call_kwargs[0][0] == m_def

    @patch(
        'nomad_llm_extraction.pipeline.schema_sources.get_nomad_schema',
        return_value=_FAKE_NOMAD_SCHEMA,
    )
    def test_unit_value_forwarded(self, mock_fetch):
        source = self._patched_source(unit_value=True)
        source.get_schema()
        _, kwargs = mock_fetch.call_args
        assert kwargs.get('unit_value') is True

    @patch(
        'nomad_llm_extraction.pipeline.schema_sources.get_nomad_schema',
        return_value=_FAKE_NOMAD_SCHEMA,
    )
    def test_optimizer_applied(self, mock_fetch):
        def add_tag(s: dict) -> dict:
            s = dict(s)
            s['x-source'] = 'nomad'
            return s

        source = self._patched_source(optimizer=add_tag)
        schema = source.get_schema()
        assert schema.get('x-source') == 'nomad'

    @patch(
        'nomad_llm_extraction.pipeline.schema_sources.get_nomad_schema',
        side_effect=ValueError('404 Error fetching schema'),
    )
    def test_propagates_fetch_error(self, mock_fetch):
        source = self._patched_source()
        with pytest.raises(ValueError, match='404 Error fetching schema'):
            source.get_schema()

    @patch(
        'nomad_llm_extraction.pipeline.schema_sources.get_nomad_schema',
        return_value=_FAKE_NOMAD_SCHEMA,
    )
    def test_satisfies_schema_source_protocol(self, mock_fetch):
        source = self._patched_source()
        assert isinstance(source, SchemaSource)

    @patch(
        'nomad_llm_extraction.pipeline.schema_sources.get_nomad_schema',
        return_value=_FAKE_NOMAD_SCHEMA,
    )
    def test_resolve_allOf_flag_forwarded_to_resolve_schema(self, mock_fetch):
        """resolve_allOf=True should be forwarded to resolve_schema."""
        with patch(
            'nomad_llm_extraction.pipeline.schema_sources.resolve_schema',
            wraps=__import__(
                'nomad_llm_extraction.transform.utils', fromlist=['resolve_schema']
            ).resolve_schema,
        ) as mock_resolve:
            source = self._patched_source(resolve_allOf=True)
            source.get_schema()
            mock_resolve.assert_called_once()
            _, kwargs = mock_resolve.call_args
            assert kwargs.get('resolve_allOf') is True


# ---------------------------------------------------------------------------
# Integration: ExtractionPipeline + InlineSchemaSource
# ---------------------------------------------------------------------------


class TestExtractionPipelineWithSchemaSources:
    """End-to-end smoke tests combining ExtractionPipeline with concrete sources."""

    def test_pipeline_with_inline_source(self):
        from nomad_llm_extraction.pipeline.extraction_pipeline import ExtractionPipeline

        schema = {'type': 'object', 'properties': {'x': {'type': 'number'}}}
        extracted = {'x': 3.14}
        engine = MagicMock()
        engine.generate.return_value = json.dumps(extracted)

        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=InlineSchemaSource(schema),
            postprocessing_schema_source=InlineSchemaSource(schema),
        )
        result = pipeline.run('some text')
        assert result.success is True
        assert result.extracted_data == extracted

    def test_pipeline_with_inline_source_and_optimizer(self):
        from nomad_llm_extraction.pipeline.extraction_pipeline import ExtractionPipeline

        schema = {'type': 'object', 'properties': {'x': {'type': 'number'}}}

        call_log: list[dict] = []

        def recording_optimizer(s: dict) -> dict:
            call_log.append(s)
            return s

        engine = MagicMock()
        engine.generate.return_value = '{}'

        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=InlineSchemaSource(
                schema, optimizer=recording_optimizer
            ),
            postprocessing_schema_source=InlineSchemaSource(schema),
        )
        pipeline.run('text')
        assert len(call_log) == 1, 'Optimizer should have been called exactly once'

    @patch(
        'nomad_llm_extraction.pipeline.schema_sources.get_nomad_schema',
        return_value=_FAKE_NOMAD_SCHEMA,
    )
    def test_pipeline_with_nomad_source(self, mock_fetch):
        from nomad_llm_extraction.pipeline.extraction_pipeline import ExtractionPipeline

        engine = MagicMock()
        engine.generate.return_value = json.dumps({'efficiency': 0.25})

        pipeline = ExtractionPipeline(
            engine=engine,
            extraction_schema_source=NomadSchemaSource('nomad.datamodel.SomeSection'),
            postprocessing_schema_source=NomadSchemaSource(
                'nomad.datamodel.SomeSection'
            ),
        )
        result = pipeline.run('paper text')
        assert result.success is True
        assert result.extracted_data == {'efficiency': 0.25}


# ---------------------------------------------------------------------------
# Public exports from schema_sources module
# ---------------------------------------------------------------------------


class TestSchemaSourcesPublicAPI:
    def test_inline_schema_source_importable_from_pipeline(self):
        from nomad_llm_extraction.pipeline import InlineSchemaSource as ISS

        assert ISS is InlineSchemaSource

    def test_nomad_schema_source_importable_from_pipeline(self):
        from nomad_llm_extraction.pipeline import NomadSchemaSource as NSS

        assert NSS is NomadSchemaSource

    def test_schema_optimizer_importable_from_schema_sources(self):
        from nomad_llm_extraction.pipeline.schema_sources import SchemaOptimizer as SO

        assert SO is SchemaOptimizer

    def test_pipeline_import_does_not_eagerly_pull_scalpl(self):
        """Importing nomad_llm_extraction.pipeline must not load scalpl at import time.

        scalpl is part of the transform dependency chain; it should only be
        imported when a schema-source symbol is actually accessed, not when the
        pipeline package is first imported.
        """
        import os
        import subprocess
        import sys
        from pathlib import Path

        # Resolve the 'src' directory relative to this test file so the
        # subprocess uses the same source tree under test (not a stale install).
        src_dir = str(Path(__file__).parent.parent.parent / 'src')

        env = os.environ.copy()
        # Prepend src_dir so it shadows any installed copy of the package.
        python_path = src_dir
        if env.get('PYTHONPATH'):
            python_path = src_dir + os.pathsep + env['PYTHONPATH']
        env['PYTHONPATH'] = python_path

        code = (
            'import sys; '
            'import nomad_llm_extraction.pipeline; '
            "assert 'scalpl' not in sys.modules, "
            "'scalpl was eagerly imported by nomad_llm_extraction.pipeline'"
        )
        result = subprocess.run(
            [sys.executable, '-c', code],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, (
            f'scalpl was eagerly imported:\nstdout: {result.stdout}\nstderr: {result.stderr}'
        )
