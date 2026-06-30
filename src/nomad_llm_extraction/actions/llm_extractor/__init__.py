from nomad.actions import TaskQueue
from pydantic import Field
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad.config.models.plugins import ActionEntryPoint


class LLMExtractorActionEntryPoint(ActionEntryPoint):
    task_queue: str = Field(
        default=TaskQueue.CPU, description='Determines the task queue for this action'
    )

    def load(self):
        import inspect

        from nomad.actions import Action

        from nomad_llm_extraction.actions.llm_extractor import activities, workflows
        from nomad_llm_extraction.actions.llm_extractor.workflows import (
            ExtractionRouterWorkflow,
        )
        from nomad_llm_extraction.pipeline import BASE_ACTIVITIES, BASE_WORKFLOWS

        ACTION_ACTIVITIES = [
            obj
            for name, obj in inspect.getmembers(activities, inspect.isroutine)
            if hasattr(obj, '__temporal_activity_definition')
        ]
        ACTION_WORKFLOWS = [
            obj
            for name, obj in inspect.getmembers(workflows, inspect.isclass)
            if hasattr(obj, '__temporal_workflow_definition')
        ]

        return Action(
            task_queue=self.task_queue,
            workflow=ExtractionRouterWorkflow,
            activities=BASE_ACTIVITIES + ACTION_ACTIVITIES,
            child_workflows=BASE_WORKFLOWS + ACTION_WORKFLOWS,
        )


llm_extractor_action_entry_point = LLMExtractorActionEntryPoint(
    name='LLMDataExtractorAction',
    description='Extract data from research papers/text in the upload.',
)
