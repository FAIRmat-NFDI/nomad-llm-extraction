"""Pydantic models for the public pipeline API.

These data classes define the configuration and result contracts used by
ExtractionPipeline and any downstream consumers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@runtime_checkable
class SchemaSource(Protocol):
    """Provider that returns a JSON-schema dict used to constrain LLM output."""

    def get_schema(self) -> dict[str, Any]: ...


@runtime_checkable
class LLMEngine(Protocol):
    """Minimal interface for an LLM backend."""

    def generate(
        self, prompt: str, json_schema: dict[str, Any], optional_params: dict = {}
    ) -> str: ...


@dataclass
class StageContext:
    """Mutable state shared between pipeline stages.

    Stages read inputs from *ctx* and write their outputs back into it so that
    subsequent stages and hooks can access the accumulated data.

    Attributes:
        text: The raw input text to extract from.
        extraction_schema: JSON schema loaded by
            :class:`ExtractionSchemaLoadStage` and used for prompt + LLM call.
        postprocessing_schema: JSON schema loaded by
            :class:`PostprocessingSchemaLoadStage` and passed to the postprocessor.
        prompt: Prompt string built by :class:`PromptBuildStage`.
        raw_output: Raw LLM output string (JSON) set by :class:`LLMCallStage`.
        extracted_data: Parsed Python object set by :class:`ParseResponseStage`.
        postprocessed_data: Data after optional postprocessing set by
            :class:`PostprocessingStage`.
        archive_data: Data after optional archive shaping set by
            :class:`ArchiveShapingStage`.
        metadata: Free-form dict for carrying auxiliary data (e.g. timing,
            token counts, validation errors) without polluting typed fields.
    """

    text: str
    prompt_config: PromptConfig | None = None
    # engine: LLMEngine | None = None
    engine_config: dict[str, Any] = field(default_factory=dict)
    optional_params: dict[str, Any] = field(default_factory=dict)
    extraction_schema: dict[str, Any] | None = None
    postprocessing_schema: dict[str, Any] | None = None
    # postprocessor_args: list[str] | None = None
    # postprocessor: Callable[[Any, dict[str, Any] | None], Any] | None = None
    postprocessed_data: Any = None
    # filter_func: Callable[..., Any] | None = None
    # filter_args: list[str] | None = None
    prompt: str | None = None
    raw_output: str | None = None
    extracted_data: Any = None
    archive_data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    filtered_data: Any = None


class DataclassInstance(Protocol):
    from dataclasses import Field

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


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


@runtime_checkable
class Stage(Protocol):
    """A single named unit of pipeline work.

    Implementations must expose a ``name`` attribute and a ``run`` method that
    accepts a :class:`StageContext` and returns a :class:`StageResult`.
    """

    name: str

    def run(self, ctx: StageContext) -> StageResult: ...


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


@dataclass
class DefaultStageContext:
    """Default context dataclass used by the pipeline if no custom context factory is provided."""

    pass


class PipelineResult(BaseModel):
    success: bool
    stages: list[StageResult] = Field(default_factory=list)
    ctx: dict = Field(default_factory=dict)
    error: str | None = None


class ExtractionPipelineResult(PipelineResult):
    """Aggregated result returned by ExtractionPipeline.run()."""

    success: bool
    raw_llm_output: str | None = None
    extracted_data: Any = None
    postprocessed_data: Any = None
    archive_data: Any = None
    stages: list[StageResult] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


StageHook = Callable[['StageContext'], None]
StageFunc = Callable[[Any, str], StageResult]
StageHookSpec = tuple[str, Literal['before', 'after'], StageHook]
