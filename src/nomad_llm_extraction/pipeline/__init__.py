from nomad_llm_extraction.pipeline.activities import (
    build_prompt,
    get_inline_schema,
    get_nomad_schema,
    json_parse,
    llm_call,
    parse_text_from_pdf,
    upload_to_nomad,
    validate_extraction_with_schema,
)
from nomad_llm_extraction.pipeline.workflows import ExtractionWorkflow, LLMCallWorkflow

BASE_ACTIVITIES = [
    build_prompt,
    json_parse,
    llm_call,
    validate_extraction_with_schema,
    parse_text_from_pdf,
    get_inline_schema,
    get_nomad_schema,
    upload_to_nomad,
]

BASE_WORKFLOWS = [
    LLMCallWorkflow,
    ExtractionWorkflow,
]
