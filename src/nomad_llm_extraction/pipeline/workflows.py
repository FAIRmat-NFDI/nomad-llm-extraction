from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from nomad_llm_extraction.pipeline.activities import (
        BuildPromptInput,
        ExtractionValidationInput,
        InlineSchemaConfig,
        LLMCallInput,
        NomadSchemaConfig,
        build_prompt,
        get_inline_schema,
        get_nomad_schema,
        json_parse,
        llm_call,
        parse_text_from_pdf,
        validate_extraction_with_schema,
    )
    from nomad_llm_extraction.pipeline.models import (
        ExtractionWorkflowInput,
        ExtractionWorkflowOutput,
        GeneralExtractionWorkflowInput,
        LLMCallOutput,
    )
from datetime import timedelta

DEFAULT_RETRY_POLICY = RetryPolicy(
    maximum_attempts=3,
)


@workflow.defn
class LLMCallWorkflow:
    @workflow.run
    async def run(self, inp: LLMCallInput) -> LLMCallOutput:
        error_msg = None
        extracted_data = {}
        raw_output = ''
        raw_output = await workflow.execute_activity(
            llm_call,
            inp,
            start_to_close_timeout=timedelta(seconds=300),
            retry_policy=DEFAULT_RETRY_POLICY,
        )

        if not raw_output:
            error_msg = 'LLM did not return any output.'
            raise Exception(error_msg)

        error, extracted_data = await workflow.execute_activity(
            json_parse,
            raw_output,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=DEFAULT_RETRY_POLICY,
        )

        if error:
            error_msg = extracted_data.get(
                'error', 'LoadingError: Failed to parse JSON output from LLM.'
            )
            raise Exception(error_msg)

        # Step 4: Validate extraction against schema
        validation_result = await workflow.execute_activity(
            validate_extraction_with_schema,
            ExtractionValidationInput(
                extracted_data=extracted_data,  # Use the parsed data for validation
                extraction_schema=inp.extraction_schema,
            ),
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=DEFAULT_RETRY_POLICY,
        )

        if not validation_result.validated:
            error_msg = f'ValidationError: {validation_result.message}'
            raise Exception(error_msg)
        return LLMCallOutput(
            extracted_data=extracted_data, raw_output=raw_output, err_message=None
        )


@workflow.defn
class ExtractionWorkflow:
    @workflow.run
    async def run(self, inp: ExtractionWorkflowInput) -> ExtractionWorkflowOutput:
        extracted_data = {}
        raw_output = ''
        error_msg = None
        try:
            if not inp.extraction_schema:
                error_msg = 'No extraction schema provided.'
                raise Exception(error_msg)
            prompt = inp.prompt
            if not prompt:
                text = inp.text
                if not text and inp.pdf_path:
                    # Step 1: Extract text from PDF
                    text, doi = await workflow.execute_activity(
                        parse_text_from_pdf,
                        inp.pdf_path,
                        start_to_close_timeout=timedelta(seconds=30),
                        retry_policy=DEFAULT_RETRY_POLICY,
                    )

                    if not text:
                        error_msg = f'No text parsed from PDF:{inp.pdf_path}'
                        raise Exception(error_msg)
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
                    retry_policy=DEFAULT_RETRY_POLICY,
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
                    retry_policy=DEFAULT_RETRY_POLICY,
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
                        retry_prompt += f'\n\nPrevious attempt output:\n{llm_call_output.raw_output}'
                    retry_prompt += f'\n\nNote: The previous attempt resulted in an error: {llm_call_output.err_message}. Please try again and ensure the output adheres to the schema.'
            return ExtractionWorkflowOutput(
                extracted_data=llm_call_output.extracted_data,
                raw_output=llm_call_output.raw_output,
                err_message='Max retry attempts reached.',
                retry_prompt=retry_prompt,
                retries=retry_count,
            )
        except Exception as e:
            workflow.logger.error(
                f'Extraction workflow failed with exception: {str(e)}'
            )
            raise ApplicationError(
                f'Extraction workflow failed with exception: {str(e)}',
                non_retryable=True,
            )
            # return ExtractionWorkflowOutput(
            #     extracted_data=extracted_data,
            #     raw_output=raw_output,
            #     err_message=f'Extraction workflow failed with exception: {str(e)}',
            # )


@workflow.defn
class GeneralExtractionWorkflow:
    @workflow.run
    async def run(
        self, inp: GeneralExtractionWorkflowInput
    ) -> ExtractionWorkflowOutput:
        try:
            if isinstance(inp.schema_config, NomadSchemaConfig):
                extraction_schema = await workflow.execute_activity(
                    get_nomad_schema,
                    inp.schema_config,
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=DEFAULT_RETRY_POLICY,
                )
            elif isinstance(inp.schema_config, InlineSchemaConfig):
                extraction_schema = await workflow.execute_activity(
                    get_inline_schema,
                    inp.schema_config,
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=DEFAULT_RETRY_POLICY,
                )
            extraction_workflow_input = ExtractionWorkflowInput(
                extraction_schema=extraction_schema,
                text=inp.text,
                pdf_path=inp.pdf_path,
                system_prompt=inp.system_prompt,
                instruction_text=inp.instruction_text,
                llm_engine_config=inp.llm_engine_config,
            )
            return await workflow.execute_child_workflow(
                ExtractionWorkflow.run,
                extraction_workflow_input,
                id='nomad_extraction_workflow',
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
        except Exception as e:
            workflow.logger.error(
                f'General extraction workflow failed with exception: {str(e)}'
            )
            raise ApplicationError(
                f'Extraction workflow failed with exception: {str(e)}',
                non_retryable=True,
            )
            # return ExtractionWorkflowOutput(
            #     extracted_data={},
            #     raw_output='',
            #     err_message=f'General extraction workflow failed with exception: {str(e)}',
            # )
