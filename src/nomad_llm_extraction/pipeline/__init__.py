import inspect

from nomad_llm_extraction.pipeline import activities, workflows

# We use list comprehension to find all Temporal activities in the module
BASE_ACTIVITIES = [
    obj
    for name, obj in inspect.getmembers(activities, inspect.isroutine)
    if hasattr(obj, '__temporal_activity_definition')
]
BASE_WORKFLOWS = [
    obj
    for name, obj in inspect.getmembers(workflows, inspect.isclass)
    if hasattr(obj, '__temporal_workflow_definition')
]
