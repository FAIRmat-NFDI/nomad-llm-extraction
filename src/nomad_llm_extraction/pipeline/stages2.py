from __future__ import annotations

import json
import logging
import traceback
from dataclasses import asdict
from typing import Any

from nomad_llm_extraction.pipeline.input_sources.paper import parse_text_from_pdf
from nomad_llm_extraction.pipeline.models import (
    PipelineResult,
    StageContext,
    StageResult,
)
from nomad_llm_extraction.utils.utils import validate_with_schema

logger = logging.getLogger(__name__)


def text_from_pdf(
    ctx: StageContext, stage_name: str = 'text_from_pdf'
) -> tuple[StageResult, StageContext]:
    if ctx.pdf_path is None:
        return StageResult(
            name=stage_name,
            success=False,
            error='Cannot extract text: no PDF path provided',
        ), ctx
    try:
        text = parse_text_from_pdf(ctx.pdf_path)
        ctx.text = text
        return StageResult(name=stage_name, data=text, success=True), ctx
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        logger.error('Text extraction failed: %s', msg)
        return StageResult(name=stage_name, success=False, error=msg), ctx


def build_prompt(
    ctx: StageContext, stage_name: str = 'build_prompt'
) -> tuple[StageResult, StageContext]:
    if ctx.extraction_schema is None:
        return StageResult(
            name=stage_name,
            success=False,
            error='Cannot build prompt: extraction schema not loaded',
        ), ctx
    try:
        text = ctx.text
        schema = ctx.extraction_schema
        parts: list[str] = []
        if ctx.prompt_config.system_prompt:
            parts.append(ctx.prompt_config.system_prompt)
        if ctx.prompt_config.instruction_text:
            parts.append(ctx.prompt_config.instruction_text)
        parts.append(f'Here is the schema: {json.dumps(schema, indent=2)}')
        parts.append(f'Here is the text:\n{text}')
        ctx.prompt = '\n'.join(parts)
        return StageResult(name=stage_name, data=ctx.prompt, success=True), ctx
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        logger.error('Prompt build failed: %s', msg)
    return StageResult(name=stage_name, success=False, error=msg), ctx


def llm_call(
    ctx: StageContext, stage_name: str = 'llm_call'
) -> tuple[StageResult, StageContext]:
    from nomad_llm_extraction.pipeline.schema_filling.llm_engine import LiteLLMEngine

    if ctx.prompt is None:
        return StageResult(
            name=stage_name,
            success=False,
            error='Cannot call LLM: prompt not built',
        ), ctx
    if ctx.extraction_schema is None:
        return StageResult(
            name=stage_name,
            success=False,
            error='Cannot call LLM: extraction schema not loaded',
        ), ctx
    llm_engine = LiteLLMEngine(**ctx.engine_config)
    try:
        raw = llm_engine.generate(
            ctx.prompt, ctx.extraction_schema, ctx.optional_params
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        logger.error('LLM generation failed: %s', msg)
        return StageResult(name=stage_name, success=False, error=msg), ctx

    if not isinstance(raw, str):
        msg = (
            f'Engine returned {type(raw).__name__!r} instead of str; '
            'the engine must extract the content string before returning'
        )
        logger.error(msg)
        return StageResult(name=stage_name, success=False, error=msg), ctx

    ctx.raw_output = raw
    return StageResult(name=stage_name, data=raw, success=True), ctx


def json_parse(
    ctx: StageContext, stage_name: str = 'json_parse'
) -> tuple[StageResult, StageContext]:
    try:
        extracted = json.loads(ctx.raw_output)
        ctx.extracted_data = extracted
        return StageResult(name=stage_name, data=extracted, success=True), ctx
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        logger.error('JSON parse failed: %s', msg)
        return StageResult(name=stage_name, success=False, error=msg), ctx


def validate_extraction_with_schema(
    ctx: StageContext, stage_name: str = 'jsonvalidation'
) -> tuple[StageResult, StageContext]:
    if ctx.extracted_data is None:
        return StageResult(
            name=stage_name,
            success=False,
            error='Cannot validate: no extracted data',
        ), ctx
    try:
        validated, message = validate_with_schema(
            ctx.extracted_data, ctx.extraction_schema
        )
        if not validated:
            return StageResult(name=stage_name, success=False, data=message), ctx
        return StageResult(name=stage_name, success=True), ctx
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        logger.error('Validation failed: %s', msg)
        return StageResult(name=stage_name, success=False, error=msg), ctx


def get_args(ctx, arg_names: list[str]) -> list[Any]:
    args = []
    for arg in arg_names:
        if not hasattr(ctx, arg):
            logger.warning('Context is missing expected attribute %r', arg)
        else:
            args.append(getattr(ctx, arg, None))
    return args


def get_result(self, ctx, stage_results, failed) -> PipelineResult:
    if failed is not None:
        result = PipelineResult(
            success=False,
            stages=stage_results,
            error=failed.error,
            ctx=asdict(ctx),
        )
    else:
        result = PipelineResult(
            success=True,
            stages=stage_results,
            ctx=asdict(ctx),
        )
    return result
