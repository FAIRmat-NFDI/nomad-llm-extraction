import sys
import types
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import SecretStr
from temporalio.exceptions import ApplicationError

actions_models = types.ModuleType('nomad.actions.models')
actions_models.ActionStreamEventSeverity = SimpleNamespace(
    INFO='INFO', WARNING='WARNING', ERROR='ERROR', SUCCESS='SUCCESS'
)
actions_models.ActionStreamEventType = SimpleNamespace(MESSAGE='MESSAGE', STATE='STATE')
sys.modules.setdefault('nomad.actions.models', actions_models)

actions_streams = types.ModuleType('nomad.actions.streams')
actions_streams.ACTION_STREAM_TOPIC = 'action-stream-topic'


class _ActionStreamEvent:
    def __init__(self, **kwargs):
        self.payload = kwargs


actions_streams.ActionStreamEvent = _ActionStreamEvent
sys.modules.setdefault('nomad.actions.streams', actions_streams)

assets_models = types.ModuleType('nomad.actions.assets.models')
assets_models.ActionAssetRef = dict
sys.modules.setdefault('nomad.actions.assets.models', assets_models)

structlogging = types.ModuleType('nomad.utils.structlogging')
structlogging.get_logger = lambda *a, **k: SimpleNamespace(
    bind=lambda **_kwargs: SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        exception=lambda *args, **kwargs: None,
    )
)
sys.modules.setdefault('nomad.utils.structlogging', structlogging)

mdef_util = types.ModuleType('nomad_llm_extraction.actions.llm_extractor.mdef_util')
mdef_util.MDEF_LIST = ['EntryData (nomad.datamodel.data.EntryData)']
sys.modules.setdefault('nomad_llm_extraction.actions.llm_extractor.mdef_util', mdef_util)

from nomad_llm_extraction.actions.llm_extractor import workflows
from nomad_llm_extraction.actions.llm_extractor.error_handling import failed_result
from nomad_llm_extraction.actions.llm_extractor.models import ExtractionActionInput


def test_failed_result_normalizes_errors_to_string_list():
    result = failed_result(ValueError('boom'))
    assert result == {'success': False, 'errors': ['boom']}


@pytest.mark.asyncio
async def test_extraction_action_workflow_preserves_error_list_from_router(monkeypatch):
    async def execute_child_workflow(*args, **kwargs):
        return {'result': {}, 'success': False, 'errors': ['first', 'second']}

    async def execute_activity(*args, **kwargs):
        return None

    monkeypatch.setattr(
        workflows.workflow,
        'logger',
        SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None),
    )
    monkeypatch.setattr(workflows.workflow, 'execute_child_workflow', execute_child_workflow)
    monkeypatch.setattr(workflows.workflow, 'execute_activity', execute_activity)
    monkeypatch.setattr(
        workflows.workflow,
        'info',
        lambda: SimpleNamespace(
            workflow_id='wf-1', workflow_type='ExtractionActionWorkflow'
        ),
    )
    monkeypatch.setattr(
        workflows.workflow, 'now', lambda: datetime.now(tz=timezone.utc)
    )

    workflow_instance = object.__new__(workflows.ExtractionActionWorkflow)
    workflow_instance.events = SimpleNamespace(publish=lambda event: None)
    workflow_instance.stream = None

    action_input = ExtractionActionInput(
        upload_id='upload-id',
        user_id='user-id',
        api_token=SecretStr('token'),
        extraction_m_def='EntryData (nomad.datamodel.data.EntryData)',
        text='text to extract',
    )

    result = await workflows.ExtractionActionWorkflow.run(workflow_instance, action_input)

    assert result == {'success': False, 'errors': ['first', 'second']}


@pytest.mark.asyncio
async def test_extraction_action_workflow_raises_non_retryable_for_unexpected_exceptions(
    monkeypatch,
):
    async def execute_child_workflow(*args, **kwargs):
        raise RuntimeError('unexpected failure')

    async def execute_activity(*args, **kwargs):
        return None

    monkeypatch.setattr(
        workflows.workflow,
        'logger',
        SimpleNamespace(
            info=lambda *a, **k: None,
            error=lambda *a, **k: None,
            exception=lambda *a, **k: None,
        ),
    )
    monkeypatch.setattr(workflows.workflow, 'execute_child_workflow', execute_child_workflow)
    monkeypatch.setattr(workflows.workflow, 'execute_activity', execute_activity)
    monkeypatch.setattr(
        workflows.workflow,
        'info',
        lambda: SimpleNamespace(
            workflow_id='wf-2', workflow_type='ExtractionActionWorkflow'
        ),
    )
    monkeypatch.setattr(
        workflows.workflow, 'now', lambda: datetime.now(tz=timezone.utc)
    )

    workflow_instance = object.__new__(workflows.ExtractionActionWorkflow)
    workflow_instance.events = SimpleNamespace(publish=lambda event: None)
    workflow_instance.stream = None

    action_input = ExtractionActionInput(
        upload_id='upload-id',
        user_id='user-id',
        api_token=SecretStr('token'),
        extraction_m_def='EntryData (nomad.datamodel.data.EntryData)',
        text='text to extract',
    )

    with pytest.raises(ApplicationError, match='Unexpected exception') as error:
        await workflows.ExtractionActionWorkflow.run(workflow_instance, action_input)

    assert error.value.non_retryable is True
