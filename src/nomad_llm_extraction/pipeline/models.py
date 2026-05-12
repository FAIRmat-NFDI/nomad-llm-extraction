"""Pydantic models for the public pipeline API.

These data classes define the configuration and result contracts used by
ExtractionPipeline and any downstream consumers.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    """Configuration for the LLM prompt."""

    system_prompt: str = ''
    instruction_text: str = ''


class ModelConfig(BaseModel):
    """Configuration for the LLM backend."""

    model_name: str
    api_url: str | None = None
    api_key: str = ''
    optional_params: dict[str, Any] = Field(default_factory=dict)


class SchemaSourceConfig(BaseModel):
    """Configuration describing where or how to obtain the JSON schema.

    All fields are optional; the consuming schema source implementation
    decides which combination of fields it supports.
    """

    schema_path: str | None = None
    schema_url: str | None = None
    inline_schema: dict[str, Any] | None = None


class StageHookConfig(BaseModel):
    """Configuration for a hook attached to a named pipeline stage."""

    stage_name: str
    when: Literal['before', 'after'] = 'after'
    enabled: bool = True


class StageResult(BaseModel):
    """Outcome of a single named pipeline stage."""

    name: str
    success: bool
    data: Any = None
    error: str | None = None


class PipelineResult(BaseModel):
    """Aggregated result returned by ExtractionPipeline.run()."""

    success: bool
    raw_llm_output: str | None = None
    extracted_data: Any = None
    postprocessed_data: Any = None
    archive_data: Any = None
    stages: list[StageResult] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
