import importlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]


def import_example(monkeypatch, directory: str, module: str):
    monkeypatch.syspath_prepend(str(ROOT / directory))
    sys.modules.pop(module, None)
    return importlib.import_module(module)


@pytest.mark.asyncio
async def test_perovskite_workflow_builds_current_extraction_input(monkeypatch):
    module = import_example(
        monkeypatch, 'domains/perovskite_solar_cell', 'perla_pipeline'
    )
    extraction_inputs = []
    workflow_input_kwargs = {}
    real_workflow_input = module.ExtractionWorkflowInput

    def capture_workflow_input(**kwargs):
        workflow_input_kwargs.update(kwargs)
        return real_workflow_input(**kwargs)

    async def execute_activity(activity, value=None, **kwargs):
        if activity is module.optimize_extraction_schema:
            return {'type': 'object'}
        return {'type': 'object'}

    async def execute_child_workflow(workflow_fn, value, **kwargs):
        if workflow_fn is module.ExtractionWorkflow.run:
            extraction_inputs.append(value)
            return SimpleNamespace(
                err_message=None,
                extracted_data={'cells': []},
                raw_output='{}',
            )
        return {'cells': []}

    monkeypatch.setattr(module.workflow, 'execute_activity', execute_activity)
    monkeypatch.setattr(
        module.workflow, 'execute_child_workflow', execute_child_workflow
    )
    monkeypatch.setattr(module, 'ExtractionWorkflowInput', capture_workflow_input)

    result = await module.PerlaCompleteWorkflow().run(
        module.PerlaWorkflowInput(
            pdf_path='paper.pdf',
            m_def='example.Perovskite',
            model_name='test-model',
        )
    )

    assert result == {'cells': []}
    assert extraction_inputs[0].pdf_path == 'paper.pdf'
    assert extraction_inputs[0].llm_engine_config.model_name == 'test-model'
    assert 'pdf_parser_method' not in workflow_input_kwargs


def test_battery_runner_accepts_pdf_path_and_model_name(monkeypatch):
    module = import_example(monkeypatch, 'domains/battery', 'battery_llm_pipeline')

    assert list(inspect.signature(module.run_extraction).parameters) == [
        'pdf_path',
        'model_name',
    ]
