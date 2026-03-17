from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad_llm_extraction.actions.simple_action.activities import read_pdf_activity, extract_simple_data_activity, save_extracted_data_activity
    from nomad_llm_extraction.actions.simple_action.models import SimpleWorkflowInput

@workflow.defn
class SimpleExtractionWorkflow:
    @workflow.run
    async def run(self, data: SimpleWorkflowInput) -> str:
        retry_policy = RetryPolicy(maximum_attempts=3)

        workflow_config = {
            "domain_name": "battery",
            "system_prompt": "You are an expert materials scientist. Extract the requested experimental parameters.",
            "extraction_prompt_template": "Extract the experimental data from the following text:\n\n{text}",
            "extraction_schema_path": "nomad_llm_extraction.domains.battery.schemas.BatteryData"
        }

        workflow.logger.info("Starting automated PDF search and extraction...")

        paper_text: str = await workflow.execute_activity(
            read_pdf_activity,
            args=[data.upload_id], 
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=retry_policy,
        )

        extracted_json: str = await workflow.execute_activity(
            extract_simple_data_activity,
            args=[paper_text, workflow_config], 
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=retry_policy,
        )

        saved_path: str = await workflow.execute_activity(
            save_extracted_data_activity,
            args=[data.upload_id, extracted_json], 
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=retry_policy,
        )

        return f"Success! File saved to {saved_path}"