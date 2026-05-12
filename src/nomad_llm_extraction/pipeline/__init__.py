"""nomad_llm_extraction.pipeline – public API.

Quickstart::

    from nomad_llm_extraction.pipeline import ExtractionPipeline, PromptConfig, ModelConfig

    pipeline = ExtractionPipeline(
        engine=my_engine,
        schema_source=my_schema_source,
        prompt_config=PromptConfig(system_prompt='...', instruction_text='...'),
    )
    result = pipeline.run(paper_text)
"""

from nomad_llm_extraction.pipeline.extraction_pipeline import (
    ExtractionPipeline,
    LLMEngine,
    SchemaSource,
)
from nomad_llm_extraction.pipeline.models import (
    ModelConfig,
    PipelineResult,
    PromptConfig,
    SchemaSourceConfig,
    StageHookConfig,
    StageResult,
)

# InlineSchemaSource, NomadSchemaSource, and SchemaOptimizer are loaded lazily
# to avoid pulling in the transform dependency chain (scalpl etc.) at import
# time. They remain fully accessible via `from nomad_llm_extraction.pipeline
# import InlineSchemaSource` or attribute access.
_SCHEMA_SOURCES: dict[str, object] = {}
_SCHEMA_SOURCES_NAMES = frozenset({'InlineSchemaSource', 'NomadSchemaSource', 'SchemaOptimizer'})


def __getattr__(name: str) -> object:
    if name in _SCHEMA_SOURCES_NAMES:
        if not _SCHEMA_SOURCES:
            from nomad_llm_extraction.pipeline import schema_sources as _ss  # noqa: PLC0415

            _SCHEMA_SOURCES['InlineSchemaSource'] = _ss.InlineSchemaSource
            _SCHEMA_SOURCES['NomadSchemaSource'] = _ss.NomadSchemaSource
            _SCHEMA_SOURCES['SchemaOptimizer'] = _ss.SchemaOptimizer
        return _SCHEMA_SOURCES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'ExtractionPipeline',
    'InlineSchemaSource',
    'LLMEngine',
    'ModelConfig',
    'NomadSchemaSource',
    'PipelineResult',
    'PromptConfig',
    'SchemaOptimizer',
    'SchemaSource',
    'SchemaSourceConfig',
    'StageHookConfig',
    'StageResult',
]
