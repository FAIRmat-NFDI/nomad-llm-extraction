from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad_llm_extraction.pipeline.activities import (
        BuildPromptInput,
        ExtractionValidationInput,
        LLMCallInput,
        ParseTextInput,
        build_prompt,
        json_parse,
        llm_call,
        parse_text_from_pdf,
        validate_extraction_with_schema,
    )
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any


@dataclass
class ExtractionWorkflowInput:
    pdf_path: str
    extraction_schema: dict[str, Any]
    pdf_parser_method: str = 'pymupdf'  # or 'pymupdf'
    system_prompt: str = ''
    instruction_text: str = ''
    llm_engine_config: dict[str, Any] = field(default_factory=dict)
    llm_engine_optional_params: dict[str, Any] = field(
        default_factory=dict
    )  # Add this field for optional LLM parameters


@workflow.defn
class ExtractionWorkflow:
    @workflow.run
    async def run(self, inp: ExtractionWorkflowInput) -> dict[str, Any]:
        # Step 1: Extract text from PDF
        text = await workflow.execute_activity(
            parse_text_from_pdf,
            ParseTextInput(pdf_path=inp.pdf_path, method=inp.pdf_parser_method),
            start_to_close_timeout=timedelta(seconds=30),
        )

        if not text:
            raise ValueError(f'No text extracted from PDF at {inp.pdf_path}.')

        # Step 2: Build prompt for LLM
        prompt = await workflow.execute_activity(
            build_prompt,
            BuildPromptInput(
                text=text,
                extraction_schema=inp.extraction_schema,
                system_prompt=inp.system_prompt,
                instruction_text=inp.instruction_text,
            ),
            start_to_close_timeout=timedelta(seconds=30),
        )

        # Step 3: Call LLM to get raw output
        raw_output = await workflow.execute_activity(
            llm_call,
            LLMCallInput(
                prompt=prompt,
                extraction_schema=inp.extraction_schema,
                engine_config=inp.llm_engine_config,
                optional_params=inp.llm_engine_optional_params,
            ),
            start_to_close_timeout=timedelta(seconds=300),
        )

        if not raw_output:
            raise ValueError('LLM did not return any output.')

        extracted_data = await workflow.execute_activity(
            json_parse, raw_output, start_to_close_timeout=timedelta(seconds=30)
        )

        # Step 4: Validate extraction against schema
        validation_result = await workflow.execute_activity(
            validate_extraction_with_schema,
            ExtractionValidationInput(
                extracted_data=extracted_data,  # Use the parsed data for validation
                extraction_schema=inp.extraction_schema,
            ),
            start_to_close_timeout=timedelta(seconds=30),
        )

        if not validation_result.validated:
            raise ValueError(
                f'Extraction did not validate against schema: {validation_result.message}'
            )

        return extracted_data
