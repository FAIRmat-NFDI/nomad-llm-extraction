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
from copy import deepcopy
from collections.abc import Callable
from typing import Any, Literal

from nomad_llm_extraction.transform.utils import (
    get_nomad_schema,
    resolve_schema,
    prune_schema,
)

# Type alias for the optimizer hook.
# A SchemaOptimizer receives a resolved JSON-schema dict and returns a
# (possibly modified) JSON-schema dict.  Identity lambdas are acceptable.
SchemaOptimizer = Callable[[dict[str, Any]], dict[str, Any]]


class SchemaSource:

    def __init__(self,
        optimizer: SchemaOptimizer | None = None,
        remove_defs: bool = False,
        resolve_allOf: bool = False,
        remove_null_anyof: bool = False,
        exclude: list[str] | None = None,
        multi_instance_field: str | None = None,
    ) -> None:
        self._schema: dict[str, Any] | None = None
        self._optimizer = optimizer
        self._remove_defs = remove_defs
        self._resolve_allOf = resolve_allOf
        self._remove_null_anyof = remove_null_anyof
        self._exclude = exclude
        self._multi_instance_field = multi_instance_field
    def get_schema(self) -> dict[str, Any]:
        if self._schema is None:
            raise NotImplementedError("Subclasses must set self._schema before calling get_schema()")
        schema = deepcopy(self._schema)
        if self._exclude is not None:
            schema = prune_schema(schema, self._exclude)
        schema = resolve_schema(
            schema,
            remove_defs=self._remove_defs,
            resolve_allOf=self._resolve_allOf,
            remove_null_anyof=self._remove_null_anyof,
        )
        if self._optimizer is not None:
            schema = self._optimizer(schema)

        if self._multi_instance_field is not None:
            # If multi_instance_field is set, we assume the schema describes a single instance
            # and we wrap it in an array under the multi_instance_field key.
            schema = {
                'type': 'object',
                'properties': {
                    self._multi_instance_field: {
                        'type': 'array',
                        'items': schema,
                    }
                },
            }
        return schema

class InlineSchemaSource(SchemaSource):
    def __init__(
        self,
        schema: dict[str, Any],
        optimizer: SchemaOptimizer | None = None,
        remove_defs: bool = False,
        resolve_allOf: bool = False,
        remove_null_anyof: bool = False,
            exclude: list[str] | None = None,
            multi_instance_field: str | None = None,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            remove_defs=remove_defs,
            resolve_allOf=resolve_allOf,
            remove_null_anyof=remove_null_anyof,
            exclude=exclude,
            multi_instance_field=multi_instance_field,
        )
        self._schema = schema

class NomadSchemaSource(SchemaSource):
    def __init__(
        self,
        m_def: str,
        unit_value: bool = False,
        optimizer: SchemaOptimizer | None = None,
        remove_defs: bool = False,
        resolve_allOf: bool = False,
        remove_null_anyof: bool = False,
        exclude: list[str] | None = None,
        multi_instance_field: str | None = None,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            remove_defs=remove_defs,
            resolve_allOf=resolve_allOf,
            remove_null_anyof=remove_null_anyof,
            exclude=exclude,
            multi_instance_field=multi_instance_field,
        )
        self._m_def = m_def
        self._unit_value = unit_value
        self._schema = get_nomad_schema(m_def, unit_value=unit_value)

