

# @dataclass
# class SchemaConfig:
#     remove_defs: bool = False
#     resolve_allOf: bool = False
#     remove_null_anyof: bool = False
#     exclude: list[str] | None = None
#     multi_instance_field: str | None = None


# @dataclass(kw_only=True)
# class NomadSchemaConfig(SchemaConfig):
#     m_def: str
#     unit_value: bool = False


# @dataclass(kw_only=True)
# class InlineSchemaConfig(SchemaConfig):
#     schema: dict[str, Any] | None = None
#     schema_path: str | None = None


# @dataclass
# class BuildPromptInput:
#     text: str
#     extraction_schema: dict[str, Any]
#     system_prompt: str = ''
#     instruction_text: str = ''


# @dataclass
# class LLMCallInput:
#     prompt: str
#     extraction_schema: dict[str, Any]
#     engine_config: dict[str, Any] = field(default_factory=dict)
#     optional_params: dict[str, Any] = field(default_factory=dict)


# @dataclass
# class ExtractionValidationInput:
#     extracted_data: Any
#     extraction_schema: dict[str, Any]


# @dataclass
# class ExtractionValidationOutput:
#     validated: bool
#     message: str | None = None



# @dataclass
# class UploadToNomadInput:
#     m_def: str
#     data: dict[str, Any]
#     entry_name: str
#     doi: str | None = None
#     extraction_metadata: dict[str, Any] = field(
#         default_factory=lambda: {'model_name': 'LLM Extracted'}
#     )
#     multi_instance_field: str | None = field(default=None)
#     upload_id: str | None = field(default=None)


# @dataclass
# class LLMCallOutput:
#     extracted_data: dict[str, Any] = field(default_factory=dict)
#     raw_output: str = ''
#     err_message: str | None = field(default=None)



# @dataclass
# class ExtractionWorkflowInput:
#     extraction_schema: dict[str, Any] | None = None
#     prompt: str | None = field(
#         default=None
#     )  # Allow prompt to be passed directly to skip prompt building
#     text: str | None = field(
#         default=None
#     )  # Allow text to be passed directly, which can be used if PDF parsing is not needed
#     pdf_path: str | None = field(
#         default=None
#     )  # Path to PDF, optional if text is provided directly
#     system_prompt: str = ''
#     instruction_text: str = ''
#     llm_engine_config: dict[str, Any] = field(default_factory=dict)
#     llm_engine_optional_params: dict[str, Any] = field(
#         default_factory=dict
#     )  # Add this field for optional LLM parameters
#     max_retry_attempts: int = 3


# @dataclass
# class ExtractionWorkflowOutput:
#     extracted_data: dict[str, Any] = field(default_factory=dict)
#     raw_output: str = ''
#     err_message: str | None = field(default=None)
#     retry_prompt: str = ''
#     retries: int = 0



# @dataclass(kw_only=True)
# class GeneralExtractionWorkflowInput(ExtractionWorkflowInput):
#     schema_config: NomadSchemaConfig | InlineSchemaConfig

