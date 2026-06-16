from __future__ import annotations

from temporalio import activity, workflow

with workflow.unsafe.imports_passed_through():
    import json
    from typing import Any

    from nomad_llm_extraction.pipeline.input_sources.paper import PDFParser
    from nomad_llm_extraction.pipeline.models import (
        BuildPromptInput,
        ExtractionValidationInput,
        ExtractionValidationOutput,
        InlineSchemaConfig,
        LLMCallInput,
        NomadSchemaConfig,
        NomadUnitConversionInput,
        UploadToNomadInput,
    )
    from nomad_llm_extraction.pipeline.schema_filling.schema_sources import (
        InlineSchemaSource,
        NomadSchemaSource,
    )
    from nomad_llm_extraction.transform.common_transforms import (
        convert_unit,
        unit_args,
        unit_cond,
    )
    from nomad_llm_extraction.transform.json_transformer import ProcessingPipeline
    from nomad_llm_extraction.utils.export_to_nomad import upload_extraction_to_nomad
    from nomad_llm_extraction.utils.utils import (
        extract_doi_from_pdf,
        validate_with_schema,
    )

# ---------------------------------------------------------------------------
# parse_text_from_pdf
# ---------------------------------------------------------------------------


@activity.defn
async def get_inline_schema(inp: InlineSchemaConfig) -> dict[str, Any]:
    if inp.inline_schema is not None:
        schema = inp.inline_schema
    elif inp.schema_path is not None:
        with open(inp.schema_path) as f:
            schema = json.load(f)
    else:
        raise ValueError('Either schema or schema_path must be provided.')
    schema_source = InlineSchemaSource(
        schema=schema,
        remove_defs=inp.remove_defs,
        resolve_allOf=inp.resolve_allOf,
        remove_null_anyof=inp.remove_null_anyof,
        exclude=inp.exclude,
        multi_instance_field=inp.multi_instance_field,
    )
    return schema_source.get_schema()


@activity.defn
async def get_nomad_schema(inp: NomadSchemaConfig) -> dict[str, Any]:

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


@activity.defn
async def parse_text_from_pdf(pdf_path: str) -> tuple[str | None, str | None]:
    p
    parser = PDFParser(use_cache=False)
    text = parser.parse_pdf(pdf_path)
    doi = extract_doi_from_pdf(pdf_path)
    return text, doi


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


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


@activity.defn
async def llm_call(inp: LLMCallInput) -> str:
    from nomad_llm_extraction.pipeline.schema_filling.llm_engine import LiteLLMEngine

    llm_engine = LiteLLMEngine(**inp.engine_config.model_dump())
    raw = llm_engine.generate(inp.prompt, inp.extraction_schema, inp.optional_params)
    return raw


# ---------------------------------------------------------------------------
# json_parse
# ---------------------------------------------------------------------------


@activity.defn
async def json_parse(inp: str) -> tuple[bool, dict[str, Any]]:
    try:
        extracted = json.loads(inp)
        return False, extracted
    except json.JSONDecodeError as e:
        return True, {'error': f'Failed to parse JSON: {str(e)}'}


# ---------------------------------------------------------------------------
# validate_extraction_with_schema
# ---------------------------------------------------------------------------


@activity.defn
async def validate_extraction_with_schema(
    inp: ExtractionValidationInput,
) -> ExtractionValidationOutput:
    validated, message = validate_with_schema(inp.extracted_data, inp.extraction_schema)
    return ExtractionValidationOutput(validated=validated, message=message)


@activity.defn
async def upload_to_nomad(inp: UploadToNomadInput) -> None:
    upload_extraction_to_nomad(
        m_def=inp.m_def,
        data=inp.data,
        entry_name=inp.entry_name,
        doi=inp.doi,
        extraction_metadata=inp.extraction_metadata,
        multi_instance_field=inp.multi_instance_field,
        upload_id=inp.upload_id,
    )


@activity.defn
async def convert_nomad_units(inp: NomadUnitConversionInput) -> dict:
    return ProcessingPipeline(
        {'unit_conversion': [convert_unit, unit_cond, unit_args]}
    ).apply(inp.data, inp.proc_schema)
