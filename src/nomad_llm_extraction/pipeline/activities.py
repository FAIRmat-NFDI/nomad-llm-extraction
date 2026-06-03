from __future__ import annotations

from temporalio import activity, workflow

with workflow.unsafe.imports_passed_through():
    import json
    from dataclasses import dataclass, field
    from typing import Any

    from nomad_llm_extraction.pipeline.input_sources.paper import PDFParser
    from nomad_llm_extraction.pipeline.schema_sources import NomadSchemaSource
    from nomad_llm_extraction.utils.export_to_nomad import upload_extraction_to_nomad
    from nomad_llm_extraction.utils.utils import validate_with_schema

# ---------------------------------------------------------------------------
# parse_text_from_pdf
# ---------------------------------------------------------------------------


@dataclass
class NomadSchemaFetchInput:
    m_def: str
    unit_value: bool = False
    remove_defs: bool = False
    resolve_allOf: bool = False
    remove_null_anyof: bool = False
    exclude: list[str] | None = None
    multi_instance_field: str | None = None


@activity.defn
async def get_nomad_schema(inp: NomadSchemaFetchInput) -> dict[str, Any]:

    schema_source = NomadSchemaSource(
        m_def=inp.m_def,
        unit_value=inp.unit_value,
        remove_defs=inp.remove_defs,
        resolve_allOf=inp.resolve_allOf,
        remove_null_anyof=inp.remove_null_anyof,
        exclude=inp.exclude,
        multi_instance_field=inp.multi_instance_field,
    )
    return schema_source.get_schema()


@dataclass
class ParseTextInput:
    pdf_path: str
    method: str = 'pymupdf'


@activity.defn
async def parse_text_from_pdf(inp: ParseTextInput) -> str | None:
    parser = PDFParser(parse_method=inp.method, use_cache=False)
    text = parser.parse_pdf(inp.pdf_path)
    return text


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


@dataclass
class BuildPromptInput:
    text: str
    extraction_schema: dict[str, Any]
    system_prompt: str = ''
    instruction_text: str = ''


@activity.defn
async def build_prompt(inp: BuildPromptInput) -> str:
    parts: list[str] = []
    if inp.system_prompt:
        parts.append(inp.system_prompt)
    if inp.instruction_text:
        parts.append(inp.instruction_text)
    parts.append(f'Here is the schema: {json.dumps(inp.extraction_schema, indent=2)}')
    parts.append(f'Here is the text:\n{inp.text}')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# llm_call
# ---------------------------------------------------------------------------


@dataclass
class LLMCallInput:
    prompt: str
    extraction_schema: dict[str, Any]
    engine_config: dict[str, Any] = field(default_factory=dict)
    optional_params: dict[str, Any] = field(default_factory=dict)


@activity.defn
async def llm_call(inp: LLMCallInput) -> str:
    from nomad_llm_extraction.pipeline.schema_filling.llm_engine import LiteLLMEngine

    llm_engine = LiteLLMEngine(**inp.engine_config)
    raw = llm_engine.generate(inp.prompt, inp.extraction_schema, inp.optional_params)
    return raw


# ---------------------------------------------------------------------------
# json_parse
# ---------------------------------------------------------------------------


@activity.defn
async def json_parse(inp: str) -> dict[str, Any]:
    extracted = json.loads(inp)
    return extracted


# ---------------------------------------------------------------------------
# validate_extraction_with_schema
# ---------------------------------------------------------------------------


@dataclass
class ExtractionValidationInput:
    extracted_data: Any
    extraction_schema: dict[str, Any]


@dataclass
class ExtractionValidationOutput:
    validated: bool
    message: str | None = None


@activity.defn
async def validate_extraction_with_schema(
    inp: ExtractionValidationInput,
) -> ExtractionValidationOutput:
    validated, message = validate_with_schema(inp.extracted_data, inp.extraction_schema)
    return ExtractionValidationOutput(validated=validated, message=message)


@dataclass
class UploadToNomadInput:
    m_def: str
    data: dict[str, Any]
    doi: str
    model_name: str
    multi_instance_field: str | None = field(default=None)
    entry_name_format: str | None = field(default=None)
    upload_id: str | None = field(default=None)


@activity.defn
async def upload_to_nomad(inp: UploadToNomadInput) -> None:
    upload_extraction_to_nomad(
        m_def=inp.m_def,
        data=inp.data,
        doi=inp.doi,
        model_name=inp.model_name,
        multi_instance_field=inp.multi_instance_field,
        entry_name_format=inp.entry_name_format,
        upload_id=inp.upload_id,
    )
