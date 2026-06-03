from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nomad_llm_extraction.pipeline.models import (
    PipelineResult,
    StageContext,
    StageResult,
)

EngineFactory = Callable[[dict[str, Any]], Any]
RuntimeCallableFactory = Callable[[dict[str, Any]], Callable[..., Any]]
ContextBuilder = Callable[[dict[str, Any], dict[str, Any]], StageContext]
ContextSerializer = Callable[[StageContext], dict[str, Any]]
ResultBuilder = Callable[
    [dict[str, Any], list[StageResult], StageResult | None], PipelineResult
]


@dataclass
class PipelineDefinition:
    """Runtime definition for a serializable Temporal pipeline type."""

    stage_order: tuple[str, ...]
    stages: dict[str, Callable[[StageContext, str], StageResult]]
    build_context: ContextBuilder
    serialize_context: ContextSerializer
    build_result: ResultBuilder


_PIPELINE_DEFINITIONS: dict[str, PipelineDefinition] = {}
_ENGINE_FACTORIES: dict[str, EngineFactory] = {}
_CALLABLE_FACTORIES: dict[str, RuntimeCallableFactory] = {}


def register_pipeline_definition(name: str, definition: PipelineDefinition) -> None:
    _PIPELINE_DEFINITIONS[name] = definition


def get_pipeline_definition(name: str) -> PipelineDefinition:
    if name not in _PIPELINE_DEFINITIONS:
        raise KeyError(
            f'Unknown pipeline_type={name!r}. Register it via register_pipeline_definition().'
        )
    return _PIPELINE_DEFINITIONS[name]


def register_engine_factory(name: str, factory: EngineFactory) -> None:
    _ENGINE_FACTORIES[name] = factory


def resolve_engine(name: str | None, config: dict[str, Any] | None = None) -> Any:
    if name is None:
        return None
    if name not in _ENGINE_FACTORIES:
        raise KeyError(
            f'Unknown engine_ref={name!r}. Register it via register_engine_factory().'
        )
    return _ENGINE_FACTORIES[name](config or {})


def register_runtime_callable_factory(
    name: str, factory: RuntimeCallableFactory
) -> None:
    _CALLABLE_FACTORIES[name] = factory


def resolve_runtime_callable(
    name: str | None, config: dict[str, Any] | None = None
) -> Callable[..., Any] | None:
    if name is None:
        return None
    if name not in _CALLABLE_FACTORIES:
        raise KeyError(
            f'Unknown callable_ref={name!r}. Register it via register_runtime_callable_factory().'
        )
    return _CALLABLE_FACTORIES[name](config or {})
