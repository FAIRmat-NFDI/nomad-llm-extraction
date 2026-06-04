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
class LLMCallOutput:
    extracted_data: dict[str, Any] = field(default_factory=dict)
    raw_output: str = ''
    err_message: str | None = field(default=None)


@workflow.defn
class LLMCallWorkflow:
    @workflow.run
    async def run(self, inp: LLMCallInput) -> LLMCallOutput:
        raw_output = await workflow.execute_activity(
            llm_call,
            inp,
            start_to_close_timeout=timedelta(seconds=300),
        )

        if not raw_output:
            return LLMCallOutput(
                extracted_data={},
                raw_output='',
                err_message='LLM did not return any output.',
            )

        error, extracted_data = await workflow.execute_activity(
            json_parse, raw_output, start_to_close_timeout=timedelta(seconds=30)
        )

        if error:
            return LLMCallOutput(
                extracted_data={},
                raw_output=raw_output,
                err_message=extracted_data.get(
                    'error', 'LoadingError: Failed to parse JSON output from LLM.'
                ),
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
            return LLMCallOutput(
                extracted_data=extracted_data,
                raw_output=raw_output,
                err_message=f'ValidationError: {validation_result.message}',
            )
        return LLMCallOutput(
            extracted_data=extracted_data, raw_output=raw_output, err_message=None
        )


@dataclass
class ExtractionWorkflowInput:
    extraction_schema: dict[str, Any]
    prompt: str | None = field(
        default=None
    )  # Allow prompt to be passed directly to skip prompt building
    text: str | None = field(
        default=None
    )  # Allow text to be passed directly, which can be used if PDF parsing is not needed
    pdf_path: str | None = field(
        default=None
    )  # Path to PDF, optional if text is provided directly
    pdf_parser_method: str = 'pymupdf'  # or 'pymupdf'
    system_prompt: str = ''
    instruction_text: str = ''
    llm_engine_config: dict[str, Any] = field(default_factory=dict)
    llm_engine_optional_params: dict[str, Any] = field(
        default_factory=dict
    )  # Add this field for optional LLM parameters
    max_retry_attempts: int = 3


@dataclass
class ExtractionWorkflowOutput:
    extracted_data: dict[str, Any] = field(default_factory=dict)
    raw_output: str = ''
    err_message: str | None = field(default=None)
    retry_prompt: str = ''
    retries: int = 0


@workflow.defn
class ExtractionWorkflow:
    @workflow.run
    async def run(self, inp: ExtractionWorkflowInput) -> ExtractionWorkflowOutput:
        prompt = inp.prompt
        if not prompt:
            text = inp.text
            if not text and inp.pdf_path:
                # Step 1: Extract text from PDF
                text = await workflow.execute_activity(
                    parse_text_from_pdf,
                    ParseTextInput(pdf_path=inp.pdf_path, method=inp.pdf_parser_method),
                    start_to_close_timeout=timedelta(seconds=30),
                )

            if not text:
                raise ValueError(f'No text extracted from PDF at {inp.pdf_path}.')
            # text = 'bandgaps 1.63 eV (I/Br ratio: 83:17), 1.68 eV (76:24), 1.74 eV (70:30), 1.80 eV (60:40), and 1.85 eV (55:45)'
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

        retry_count = 0
        retry_prompt = ''
        while retry_count < inp.max_retry_attempts:
            llm_call_output = await workflow.execute_child_workflow(
                LLMCallWorkflow.run,
                LLMCallInput(
                    prompt=prompt + retry_prompt,
                    extraction_schema=inp.extraction_schema,
                    engine_config=inp.llm_engine_config,
                    optional_params=inp.llm_engine_optional_params,
                ),
                id=f'llm_call_attempt_{retry_count}',
            )
            if llm_call_output.err_message is None:
                return ExtractionWorkflowOutput(
                    extracted_data=llm_call_output.extracted_data,
                    raw_output=llm_call_output.raw_output,
                    err_message=None,
                    retry_prompt=retry_prompt,
                    retries=retry_count,
                )
            else:
                retry_count += 1
                workflow.logger.warning(
                    f'LLM call attempt {retry_count} failed with error: {llm_call_output.err_message}. Retrying...'
                )
                if llm_call_output.raw_output:
                    retry_prompt += (
                        f'\n\nPrevious attempt output:\n{llm_call_output.raw_output}'
                    )
                retry_prompt += f'\n\nNote: The previous attempt resulted in an error: {llm_call_output.err_message}. Please try again and ensure the output adheres to the schema.'
        return ExtractionWorkflowOutput(
            extracted_data=llm_call_output.extracted_data,
            raw_output=llm_call_output.raw_output,
            err_message='Max retry attempts reached.',
            retry_prompt=retry_prompt,
            retries=retry_count,
        )
