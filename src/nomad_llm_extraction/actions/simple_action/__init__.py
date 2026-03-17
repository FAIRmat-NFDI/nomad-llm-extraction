from nomad.actions import TaskQueue
from pydantic import Field
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad.config.models.plugins import ActionEntryPoint


class SimpleActionEntryPoint(ActionEntryPoint):
    task_queue: str = Field(
        default=TaskQueue.CPU, description='Determines the task queue for this action'
    )
    gemini_api_key: str = Field(
        default=None, description='API key for Gemini API.'
    )

    def load(self):
        from nomad.actions import Action

        from nomad_llm_extraction.actions.simple_action.activities import (
            extract_simple_data_activity,
            read_pdf_activity,
        )
        from nomad_llm_extraction.actions.simple_action.workflows import (
            SimpleExtractionWorkflow,
        )

        return Action(
            task_queue=self.task_queue,
            workflow=SimpleExtractionWorkflow,
            activities=[read_pdf_activity, extract_simple_data_activity],
        )


simple_action_entry_point = SimpleActionEntryPoint(
    name='SimpleExtractionAction',
    description='A simple action that extracts data from PDFs using an LLM.',
)
