"""Concrete SchemaSource implementations for the extraction pipeline.

Each class satisfies the :class:`~nomad_llm_extraction.pipeline.extraction_pipeline.SchemaSource`
protocol by exposing a ``get_schema() -> dict`` method.

All implementations:

1. Resolve JSON ``$ref`` pointers using :func:`~nomad_llm_extraction.transform.utils.resolve_schema`
   so that downstream code always receives a fully-dereferenced schema.
2. Accept an optional *optimizer* callable – a hook of type :data:`SchemaOptimizer` that
   receives the resolved schema and may return a modified version.  The hook is intentionally
   generic so that future agent-driven schema-improvement strategies can be plugged in without
   changing the pipeline API.

Example usage::

    from nomad_llm_extraction.pipeline.schema_sources import (
        InlineSchemaSource,
        NomadSchemaSource,
    )

    # From an inline dict
    source = InlineSchemaSource(my_schema_dict)

    # From NOMAD m_def
    source = NomadSchemaSource('nomad.datamodel.perovskite_solar_cell.PerovskiteSolarCell')

    # With an optimizer that strips unwanted properties
    def drop_descriptions(schema):
        schema = dict(schema)
        schema.pop('description', None)
        return schema

    source = InlineSchemaSource(my_schema_dict, optimizer=drop_descriptions)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nomad_llm_extraction.transform.utils import get_nomad_schema, resolve_schema, remove_sections

# Type alias for the optimizer hook.
# A SchemaOptimizer receives a resolved JSON-schema dict and returns a
# (possibly modified) JSON-schema dict.  Identity lambdas are acceptable.
SchemaOptimizer = Callable[[dict[str, Any]], dict[str, Any]]


class InlineSchemaSource:
    """Schema source backed by a JSON-schema dict supplied at construction time.

    The input schema is never mutated; :func:`resolve_schema` is called to
    dereference any ``$ref`` pointers before the optional *optimizer* runs.

    Args:
        schema: A JSON-schema dict.  May contain ``$ref`` and ``$defs``.
        optimizer: Optional callable ``(schema) -> schema`` applied after ref
            resolution.  Useful for pruning, annotating, or restructuring the
            schema before it is sent to the LLM.
        remove_defs: When ``True``, ``$defs`` are removed from the resolved
            schema (forwarded to :func:`resolve_schema`).
        resolve_allOf: When ``True``, ``allOf`` lists are merged into a single
            dict (forwarded to :func:`resolve_schema`).
    """

    def __init__(
        self,
        schema: dict[str, Any],
        *,
        optimizer: SchemaOptimizer | None = None,
        remove_defs: bool = False,
        resolve_allOf: bool = False,
        remove_null_anyof: bool = False,
    ) -> None:
        self._schema = schema
        self._optimizer = optimizer
        self._remove_defs = remove_defs
        self._resolve_allOf = resolve_allOf
        self._remove_null_anyof = remove_null_anyof

    def get_schema(self) -> dict[str, Any]:
        """Return the resolved (and optionally optimized) schema."""
        schema = resolve_schema(
            self._schema,
            remove_defs=self._remove_defs,
            resolve_allOf=self._resolve_allOf,
            remove_null_anyof=self._remove_null_anyof,
        )
        if self._optimizer is not None:
            schema = self._optimizer(schema)
        return schema


class NomadSchemaSource:
    """Schema source that fetches a JSON schema from the NOMAD API.

    Uses :func:`~nomad_llm_extraction.transform.utils.get_nomad_schema` to
    retrieve the schema for a given ``m_def`` path, then resolves ``$ref``
    pointers before returning.

    Args:
        m_def: NOMAD metainfo definition path (e.g.
            ``'nomad.datamodel.perovskite_solar_cell.PerovskiteSolarCell'``).
        unit_value: When ``True``, request unit-value formatted quantities from
            the NOMAD schema endpoint.
        optimizer: Optional callable ``(schema) -> schema`` applied after the
            schema has been fetched and resolved.
        remove_defs: When ``True``, ``$defs`` are stripped from the resolved
            schema.
        resolve_allOf: When ``True``, ``allOf`` lists are merged into a single
            dict.
    """

    def __init__(
        self,
        m_def: str,
        *,
        unit_value: bool = False,
        optimizer: SchemaOptimizer | None = None,
        remove_defs: bool = False,
        resolve_allOf: bool = False,
        remove_null_anyof: bool = False,
        exclude_fields: list[str] | None = None,
    ) -> None:
        self._m_def = m_def
        self._unit_value = unit_value
        self._optimizer = optimizer
        self._remove_defs = remove_defs
        self._resolve_allOf = resolve_allOf
        self._remove_null_anyof = remove_null_anyof
        self._exclude_fields = exclude_fields

    def get_schema(self) -> dict[str, Any]:
        """Fetch, resolve and optionally optimize the NOMAD schema."""
        schema = get_nomad_schema(self._m_def, unit_value=self._unit_value)
        schema = remove_sections(schema, sections_to_remove=self._exclude_fields)
        schema = resolve_schema(
            schema,
            remove_defs=self._remove_defs,
            resolve_allOf=self._resolve_allOf,
            remove_null_anyof=self._remove_null_anyof,
        )
        if self._optimizer is not None:
            schema = self._optimizer(schema)
        return schema
