from typing import Literal

from nomad.actions.assets.models import ActionAssetRef
from pydantic import BaseModel, Field, SecretStr, field_serializer

from nomad_llm_extraction.actions.llm_extractor.mdef_util import MDEF_LIST

DEFAULT_MODEL_NAME = 'Claude Sonnet 4.6'
ModelName = Literal[
    'Claude Sonnet 5',
    'GPT OSS 20b',
    'Llama 4 Scout',
    'LLama 4 Maverick',
    'LLama 3.3',
    'GPT 4o',
    'Claude Sonnet 4.6',
    'Claude Sonnet 4.v20250514',
    'Claude Fable 5',
    'Claude Opus 4.8',
    'GPT 5.6 Sol',
    'GPT 5.6 Terra',
    'GPT 5.6 Luna',
    'Gemini Pro Latest',
    'Gemini 3 Flash',
    'Gemini 3.6 Flash',
    'Gemini 3.5 Flash',
]  # Restricted set of LLM model names supported.
ModelAliases = {
    'Claude Sonnet 5': 'claude-sonnet-5',
    'GPT OSS 20b': 'gpt-oss-20b',
    'Llama 4 Scout': 'Llama-4-Scout-17B-16E-Instruct-FP8',
    'LLama 4 Maverick': 'Llama-4-Maverick-17B-128E-Instruct-FP8',
    'LLama 3.3': 'Llama-3.3-70B-Instruct',
    'GPT 4o': 'gpt-4o',
    'Claude Sonnet 4.6': 'claude-sonnet-4-6',
    'Claude Sonnet 4.v20250514': 'claude-4-sonnet-20250514',
    'Claude Fable 5': 'claude-fable-5',
    'Claude Opus 4.8': 'claude-opus-4-8',
    'GPT 5.6 Sol': 'gpt-5.6-sol',
    'GPT 5.6 Terra': 'gpt-5.6-terra',
    'GPT 5.6 Luna': 'gpt-5.6-luna',
    'Gemini Pro Latest': 'gemini/gemini-pro-latest',
    'Gemini 3 Flash': 'gemini/gemini-3-flash',
    'Gemini 3.6 Flash': 'gemini/gemini-3.6-flash',
    'Gemini 3.5 Flash': 'gemini/gemini-3.5-flash',
}
from typing import Any

from nomad_llm_extraction.pipeline.models import (
    ExtractionWorkflowInput as PipelineExtractionWorkflowInput,
)


class ActionId(BaseModel):
    action_instance_id: str | None = Field(
        default=None,
        description='Unique identifier for the action instance.',
    )


class ActionFileHandlerInput(ActionId):
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
    action_file_refs: list[ActionAssetRef] | None = Field(
        default=None,
        description='List of file references to be processed by the action.',
    )


class ExtractionWorkflowInput(PipelineExtractionWorkflowInput, ActionId):
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
    delete_source_pdfs: bool = Field(
        default=True,
        description='Whether to delete the source PDF files after processing.',
    )
    pdfs: list[ActionAssetRef] = Field(
        default_factory=lambda: [],
        description='List of PDF files to process. If not provided, the action will process all PDFs in the project.',
    )


class ExtractionActionInput(BaseModel):
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
    extraction_m_def: str = Field(
        ...,
        description='Nomad Section m_def to be used for extraction.',
        examples=MDEF_LIST,
    )
    model: str = Field(
        title=f'LLM Model [{DEFAULT_MODEL_NAME} (Default)]',
        default_factory=lambda: DEFAULT_MODEL_NAME,
        description='LLM model to be used for extraction.Select an LLM model for extraction or provide a custom model name.',
        examples=sorted(list(ModelAliases.keys())),
        json_schema_extra={
            'uiSchema': {
                'ui:allowClearTextInputs': True,
                'ui:placeholder': f'{DEFAULT_MODEL_NAME} (Default)',
            }
        },
    )
    api_base_url: str = Field(
        default_factory=lambda: '',
        title='API Base URL (Optional) for example, https://openrouter.ai/.',
        description="""
        Openai compatible Base URL for the LLM API.
        If you are from an academic institution, you can probably access open models via
        Blablador (API: https://api.blablador.fz-juelich.de/v1/ User guide: https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/).
        """,
    )

    text: str = Field(
        default_factory=lambda: '',
        description='Text to run the extraction on. If not provided, the action will extract text from all PDFs in the project.',
    )
    pdfs: list[ActionAssetRef] = Field(
        default_factory=lambda: [],
        title='PDF Files to Process',
        description='List of PDF files to process. If not provided, the action will process all PDFs in the project.',
        json_schema_extra={
            'x-nomad-widget': {'type': 'file-upload', 'accept': ['application/pdf']},
            'accept': ['application/pdf'],
            'uiSchema': {'ui:options': {'accept': '.pdf'}},
        },
    )
    delete_source_pdfs: bool = Field(
        default=True,
        description='Whether to delete the source PDF files after processing.',
        json_schema_extra={
            'uiSchema': {
                'ui:help': 'If checked, the source PDF files will be deleted after processing.',
            }
        },
    )
    extract_multiple_instances: bool = Field(
        default=True,
        description='Whether to extract multiple instances of the schema from the text.',
        json_schema_extra={
            'uiSchema': {
                'ui:help': 'If checked, the action will extract multiple instances of the schema from the text.',
            }
        },
    )

    @field_serializer('api_token', when_used='json')
    def dump_secret(self, v):
        return v.get_secret_value()

    @property
    def model_technical_name(self):
        if self.model == '':
            self.model = DEFAULT_MODEL_NAME
        if not self.model:
            raise ValueError(
                'No model specified and no model name provided. Please provide a model name or select a model from the list.'
            )
        model_name = (
            self.model if self.model not in ModelAliases else ModelAliases[self.model]
        )

        if model_name.startswith('openai/'):
            return model_name
        prefix = '' if self.api_base_url is None else 'openai/'  # for litellm
        return f'{prefix}{model_name}'


class ProcessNewFilesInput(ActionId):
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


class CleanupInput(ActionId):
    """Data for cleanup activity."""

    upload_id: str = Field(
        ...,
        description='Unique identifier for the project associated with the action.',
    )
    user_id: str = Field(
        ..., description='Unique identifier for the user who initiated the action.'
    )
    pdfs: list[str] = Field(..., description='Paths to the PDF files to be removed.')


class PostProcessingInput(ActionId):
    """Data for post-processing activity."""

    data: dict[str, Any] = Field(
        ..., description='extracted data to be post-processed.'
    )
    postprocessing_schema: dict[str, Any] = Field(
        ..., description='The schema to be used for post-processing.'
    )
