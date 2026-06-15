from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, SecretStr, field_serializer
from typing_extensions import TypedDict


class SchemaConfig(BaseModel):
    remove_defs: bool = Field(
        False, description='Whether to remove the #defs field from the schema.'
    )
    resolve_allOf: bool = Field(
        False, description='Whether to resolve allOf references in the schema.'
    )
    remove_null_anyof: bool = Field(
        False, description='Whether to remove null values from anyOf schemas.'
    )
    exclude: dict[str, list[str]] | None = Field(
        default=None,
        description=(
            'Dictionary of section ids, schema paths, keys to exclude during schema generation.'
            '{"$id": [...], "path": [...], "key": [...]}.'
        ),
    )
    multi_instance_field: str | None = Field(
        default=None,
        description='Field name used to indicate multi-instance handling in schema sources.',
    )


class NomadSchemaConfig(SchemaConfig):
    m_def: str = Field(..., description='NOMAD m_def identifier for schema generation.')
    unit_value: bool = Field(
        False,
        description=(
            'Whether to expand quantity fields with units into separate value and unit fields.'
        ),
    )


class InlineSchemaConfig(SchemaConfig):
    inline_schema: dict[str, Any] | None = Field(
        default=None,
        description='Inline schema object provided directly by the caller.',
    )
    schema_path: str | None = Field(
        default=None,
        description='Path to a JSON schema file used when schema is not provided inline.',
    )


class BuildPromptInput(BaseModel):
    """
    The prompt gets formatted as
    {system_prompt}\n\n
    {instruction_text}\n\n
    Here is the Schema:\n{schema}\n\n
    Here is the Text:\n{text}
    """

    text: str = Field(
        ...,
        description='Text extracted from the source document to be used for prompt construction.',
    )
    extraction_schema: dict[str, Any] = Field(
        ...,
        description='Schema used to guide the prompt construction for LLM extraction.',
    )
    system_prompt: str = Field(
        default='', description='System prompt to guide the LLM extraction.'
    )
    instruction_text: str = Field(
        default='', description='Additional instructions to guide the LLM extraction.'
    )


class LLMCallInput(BaseModel):
    prompt: str = Field(
        ..., description='The full prompt to be sent to the LLM for extraction.'
    )
    extraction_schema: dict[str, Any] = Field(
        ...,
        description='Schema used to validate the LLM output and guide the extraction process.',
    )
    engine_config: LLMEngineConfig = Field(
        ..., description='Configuration for the LLM engine.'
    )
    optional_params: dict[str, Any] = Field(default_factory=dict)


class ExtractionValidationInput(BaseModel):
    extracted_data: Any = Field(
        ..., description='Data extracted by the LLM that needs to be validated.'
    )
    extraction_schema: dict[str, Any] = Field(
        ..., description='Schema against which the extracted data should be validated.'
    )


class ExtractionValidationOutput(BaseModel):
    validated: bool
    message: str | None = Field(
        default=None,
        description='Validation message describing success details or failure reason.',
    )


class UploadToNomadInput(BaseModel):
    m_def: str = Field(..., description='NOMAD m_def identifier for schema generation.')
    data: dict[str, Any] = Field(..., description='Data to be uploaded to NOMAD.')
    entry_name: str = Field(
        ..., description='Name of the entry to be created in NOMAD.'
    )
    doi: str | None = Field(
        default=None,
        description='Optional DOI associated with the extracted paper.',
    )
    extraction_metadata: dict[str, Any] = Field(
        default_factory=lambda: {'model_name': 'LLM Extracted'}
    )
    multi_instance_field: str | None = Field(
        default=None,
        description='Field name used to map multiple extracted instances in NOMAD upload.',
    )
    upload_id: str | None = Field(
        default=None,
        description='Existing NOMAD upload identifier to append entries to.',
    )


class LLMCallOutput(BaseModel):
    extracted_data: dict[str, Any] = Field(default_factory=dict)
    raw_output: str = ''
    err_message: str | None = Field(
        default=None,
        description='Error message when the LLM call or downstream validation fails.',
    )


class NomadUnitConversionInput(BaseModel):
    data: dict
    proc_schema: dict


class LLMEngineConfig(BaseModel):
    model_name: str = Field(..., description='LLM model to be used for extraction.')
    api_key: SecretStr | None = Field(None, description='API token for LLM access.')
    api_url: str | None = Field(
        None, description='Custom API URL for the LLM endpoint, if applicable.'
    )

    @field_serializer('api_key', when_used='json')
    def dump_secret(self, v):
        return v.get_secret_value()


class ExtractionWorkflowInput(BaseModel):
    extraction_schema: dict[str, Any] | None = Field(
        default=None,
        description='Extraction schema used to guide the LLM structured output.',
    )
    prompt: str | None = Field(
        default=None,
        description='Prebuilt prompt; when set, prompt building from text is skipped.',
    )
    text: str | None = Field(
        default=None,
        description='Source paper text used to build prompt when prompt is not provided.',
    )
    pdf_path: str | None = Field(
        default=None,
        description='Path to the PDF file used to extract text when text is not provided.',
    )
    system_prompt: str = Field(
        default='', description='System prompt to guide the LLM extraction.'
    )
    instruction_text: str = Field(
        default='', description='Additional instructions to guide the LLM extraction.'
    )
    llm_engine_config: LLMEngineConfig = Field(
        ..., description='Configuration for the LLM engine.'
    )
    llm_engine_optional_params: dict[str, Any] = Field(default_factory=dict)
    max_retry_attempts: int = 3


class ExtractionWorkflowOutput(BaseModel):
    extracted_data: dict[str, Any] = Field(default_factory=dict)
    raw_output: str = Field('', description='Raw output returned from the LLM call.')
    err_message: str | None = Field(
        default=None,
        description='Error message if extraction did not complete successfully.',
    )
    retry_prompt: str = Field(
        default='',
        description=(
            'Prompt to be used for retrying the LLM call, typically containing error details and instructions for correction.'
        ),
    )
    retries: int = Field(
        default=0,
        description='Number of retry attempts made for the LLM call after initial failure.',
    )


class GeneralExtractionWorkflowInput(ExtractionWorkflowInput):
    schema_config: NomadSchemaConfig | InlineSchemaConfig
