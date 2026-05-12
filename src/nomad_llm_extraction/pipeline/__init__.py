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

__all__ = [
    'ExtractionPipeline',
    'LLMEngine',
    'ModelConfig',
    'PipelineResult',
    'PromptConfig',
    'SchemaSource',
    'SchemaSourceConfig',
    'StageHookConfig',
    'StageResult',
]
