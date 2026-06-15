from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_serializer

ModelName = Literal[
    'gpt-4o',
    # 'gpt-5',  # Uncomment when temperature support is correct in LiteLLM
    'claude-4-sonnet-20250514',
    #  'meta.llama3-70b-instruct-v1:0',  # Uncomment when someone can test it
]  # Restricted set of LLM model names supported.

from typing import Any

from nomad_llm_extraction.pipeline.models import GeneralExtractionWorkflowInput


class ActionFileHandlerInput(BaseModel):
    upload_id: str = Field(
        ...,
        description='Unique identifier for the project associated with the action.',
    )
    user_id: str = Field(
        ..., description='Unique identifier for the user who initiated the action.'
    )
    name: str | None = Field(
        default=None,
        description='Path to the PDF file to be processed./ Name prefix for the output files generated.',
    )
    data: list[dict[str, Any]] | None = Field(
        default=None,
        description='Data to save to upload',
    )


class ExtractActionWorkflowInput(GeneralExtractionWorkflowInput):
    upload_id: str = Field(
        ...,
        description='Unique identifier for the project associated with the action.',
    )
    user_id: str = Field(
        ..., description='Unique identifier for the user who initiated the action.'
    )
    extraction_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description='Metadata to be associated with the extracted entries in NOMAD.',
    )
    name: str | None = Field(
        default=None,
        description='Prefix for the output files generated.',
    )


class ExtractWorkflowInput(BaseModel):
    """
    Run this action to extract perovskite solar cells information from all PDFs in a
    project/upload.

    First, upload the research papers as PDF files to the project. Then, submit this action
    providing the project ID (upload ID) and api token for the chosen LLM. The action will
    find and process all PDF files in the project using the specified LLM, then create and
    process new entries for each detected solar cell and delete the source PDF files.
    """

    upload_id: str = Field(
        ...,
        description='Unique identifier for the project associated with the action.',
    )
    user_id: str = Field(
        ..., description='Unique identifier for the user who initiated the action.'
    )
    api_token: SecretStr = Field(..., description='API token for LLM access.')
    model: ModelName = Field(
        'claude-4-sonnet-20250514', description='LLM model to be used for extraction.'
    )
    config: str = Field(
        ...,
        description=('Path to the configuration file for the extraction pipeline.'),
    )

    @field_serializer('api_token', when_used='json')
    def dump_secret(self, v):
        return v.get_secret_value()


class SingleExtractionInput(BaseModel):
    """Data for extraction from a single pdf file."""

    upload_id: str = Field(
        ...,
        description='Unique identifier for the project associated with the action.',
    )
    user_id: str = Field(
        ..., description='Unique identifier for the user who initiated the action.'
    )
    pdf: str = Field(..., description='Path to the PDF file to be processed.')
    api_token: SecretStr = Field(..., description='API token for LLM access.')
    model: ModelName = Field(
        'claude-4-sonnet-20250514', description='LLM model to be used for extraction.'
    )

    @field_serializer('api_token', when_used='json')
    def dump_secret(self, v):
        return v.get_secret_value()


class ProcessNewFilesInput(BaseModel):
    """Data for processing new files activity."""

    upload_id: str = Field(
        ...,
        description='Unique identifier for the project associated with the action.',
    )
    user_id: str = Field(
        ..., description='Unique identifier for the user who initiated the action.'
    )
    results: dict[str, Any] = Field(
        ..., description='Paths to the new entries to be processed.'
    )


class CleanupInput(BaseModel):
    """Data for cleanup activity."""

    upload_id: str = Field(
        ...,
        description='Unique identifier for the project associated with the action.',
    )
    user_id: str = Field(
        ..., description='Unique identifier for the user who initiated the action.'
    )
    pdfs: list[str] = Field(..., description='Paths to the PDF files to be removed.')


class LLMEngineConfig(BaseModel):
    """Configuration for the LLM engine."""

    model_name: ModelName = Field(
        'claude-4-sonnet-20250514', description='LLM model to be used for extraction.'
    )
    api_key: SecretStr = Field(..., description='API token for LLM access.')
    api_url: str | None = Field(
        None, description='Custom API URL for the LLM endpoint, if applicable.'
    )

    @field_serializer('api_key', when_used='json')
    def dump_secret(self, v):
        return v.get_secret_value()


class SchemaConfig(BaseModel):
    remove_defs: bool = Field(
        False, description='Whether to remove definitions from the schema.'
    )


class GeneralExtractionWorkflowInput(BaseModel):
    """Data for the general extraction workflow."""

    upload_id: str = Field(
        ...,
        description='Unique identifier for the project associated with the action.',
    )
    user_id: str = Field(
        ..., description='Unique identifier for the user who initiated the action.'
    )
    text: str = Field(..., description='Text extracted from the PDF to be processed.')
    prompt: str = Field(..., description='System prompt to guide the LLM extraction.')
    instruction_text: str = Field(
        ..., description='Additional instructions to guide the LLM extraction.'
    )
    schema_config: dict = Field(
        ...,
        description=(
            'Configuration for generating the extraction schema, including m_def and other parameters.'
        ),
    )
    llm_engine_config: dict = Field(
        ...,
        description=(
            'Configuration for the LLM engine, including API endpoint and other parameters.'
        ),
    )
