import asyncio
import os
from collections.abc import Mapping, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import yaml
from pydantic import ValidationError
from temporalio import workflow
from temporalio.client import Client
from temporalio.envconfig import ClientConfig
from temporalio.worker import Worker

from nomad_llm_extraction.config import DEFAULT_EXTRACTION_METADATA

with workflow.unsafe.imports_passed_through():
    from datetime import timedelta
    from typing import Any

    from nomad_llm_extraction.pipeline import BASE_ACTIVITIES, BASE_WORKFLOWS
    from nomad_llm_extraction.pipeline.activities import (
        UploadToNomadInput,
        upload_to_nomad,
    )
    from nomad_llm_extraction.pipeline.workflows import (
        ExtractionWorkflow,
        ExtractionWorkflowInput,
    )
    from nomad_llm_extraction.utils.utils import load_yaml_config


TEMPORAL_CONFIG_PATH = os.getenv('TEMPORAL_CONFIG_PATH', 'temporal.toml')


class ConfigError(ValueError):
    """Raised when CLI extraction configuration cannot be composed."""


def load_yaml_mapping(path: str | Path, description: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f'{description} path does not exist: {config_path}')
    if not config_path.is_file():
        raise ConfigError(f'{description} path is not a file: {config_path}')
    try:
        with config_path.open(encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ConfigError(
            f'Malformed YAML in {description} {config_path}: {error}'
        ) from error
    except OSError as error:
        raise ConfigError(
            f'Could not read {description} {config_path}: {error}'
        ) from error
    if not isinstance(config, Mapping):
        raise ConfigError(f'{description} must contain a YAML mapping: {config_path}')
    return deepcopy(dict(config))


def merge_config(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def get_or_create_mapping(
    config: MutableMapping[str, Any], path: str
) -> MutableMapping[str, Any]:
    current: MutableMapping[str, Any] = config
    for key in path.split('.'):
        if key not in current:
            current[key] = {}
        value = current[key]
        if not isinstance(value, MutableMapping):
            raise ConfigError(f'Cannot set {path}: {key} is not a mapping.')
        current = value
    return current


def apply_dotted_setting(config: MutableMapping[str, Any], setting: str) -> None:
    if '=' not in setting:
        raise ConfigError(f'Invalid --set value {setting!r}; expected path=value.')
    path, raw_value = setting.split('=', 1)
    keys = path.split('.')
    if not path or any(not key for key in keys):
        raise ConfigError(
            f'Invalid --set path {path!r}; path segments cannot be empty.'
        )
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as error:
        raise ConfigError(f'Invalid YAML value for --set {path!r}: {error}') from error
    target = config
    for key in keys[:-1]:
        if key not in target:
            target[key] = {}
        if not isinstance(target[key], MutableMapping):
            raise ConfigError(f'Cannot set {path}: {key} is not a mapping.')
        target = target[key]
    target[keys[-1]] = value


def apply_common_flags(
    config: MutableMapping[str, Any],
    *,
    pdf_path: str | None = None,
    text: str | None = None,
    model_name: str | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    m_def: str | None = None,
    output_path: str | None = None,
    llm_extraction_method: str | None = None,
) -> None:
    for key, value in (
        ('pdf_path', pdf_path),
        ('text', text),
        ('output_path', output_path),
        ('llm_extraction_method', llm_extraction_method),
    ):
        if value is not None:
            config[key] = value

    engine_config = get_or_create_mapping(config, 'llm_engine_config')
    for key, value in (
        ('model_name', model_name),
        ('api_url', api_url),
        ('api_key', api_key),
    ):
        if value is not None:
            engine_config[key] = value

    if m_def is not None:
        config['m_def'] = m_def
        get_or_create_mapping(config, 'schema_config')['m_def'] = m_def
        if 'nomad_upload_config' in config:
            get_or_create_mapping(config, 'nomad_upload_config')['m_def'] = m_def


def build_effective_config(
    config_path: str | Path,
    *,
    override_file: str | Path | None = None,
    pdf_path: str | None = None,
    text: str | None = None,
    model_name: str | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    m_def: str | None = None,
    output_path: str | None = None,
    llm_extraction_method: str | None = None,
    set_values: str | list[str] | None = None,
) -> dict[str, Any]:
    config = load_yaml_mapping(config_path, 'base configuration')
    if override_file is not None:
        config = merge_config(
            config, load_yaml_mapping(override_file, 'override configuration')
        )
    apply_common_flags(
        config,
        pdf_path=pdf_path,
        text=text,
        model_name=model_name,
        api_url=api_url,
        api_key=api_key,
        m_def=m_def,
        output_path=output_path,
        llm_extraction_method=llm_extraction_method,
    )
    if set_values is None:
        return config
    settings = [set_values] if isinstance(set_values, str) else set_values
    for setting in settings:
        if not isinstance(setting, str):
            raise ConfigError(
                f'Invalid --set value {setting!r}; expected a path=value string.'
            )
        apply_dotted_setting(config, setting)
    return config


def validate_workflow_config(config: Mapping[str, Any]) -> ExtractionWorkflowInput:
    try:
        return ExtractionWorkflowInput.model_validate(config)
    except ValidationError as error:
        raise ConfigError(f'Invalid extraction configuration: {error}') from error


def write_yaml_config(config: Mapping[str, Any], path: str | Path) -> None:
    config_path = Path(path)
    try:
        with config_path.open('w', encoding='utf-8') as file:
            yaml.safe_dump(dict(config), file, sort_keys=False)
    except OSError as error:
        raise ConfigError(
            f'Could not write effective configuration {config_path}: {error}'
        ) from error


async def start_worker():
    temporal_client_config = ClientConfig.load_client_connect_config(
        config_file=TEMPORAL_CONFIG_PATH
    )
    client = await Client.connect(**temporal_client_config)
    worker = Worker(
        client,
        task_queue='general_extraction_pipeline',
        workflows=BASE_WORKFLOWS,
        activities=BASE_ACTIVITIES,
        activity_executor=ThreadPoolExecutor(
            max_workers=10
        ),  # Adjust max_workers as needed
    )
    worker_task = asyncio.create_task(worker.run())
    return client, worker, worker_task


async def run_extraction_workflow(config: ExtractionWorkflowInput):
    client, worker, worker_task = await start_worker()
    result = await client.execute_workflow(
        ExtractionWorkflow.run,
        config,
        id='general-extraction-workflow',
        task_queue='general_extraction_pipeline',
    )
    worker_task.cancel()  # Cleanly shut down the worker after workflow completion
    return result


def extract(config: str | dict[str, Any] | ExtractionWorkflowInput):
    if isinstance(config, str):
        config = load_yaml_config(config)
    if isinstance(config, dict):
        config = ExtractionWorkflowInput(**config)
    print('Starting extraction workflow with config:')
    print(config)
    result = asyncio.run(run_extraction_workflow(config))
    if result.err_message is not None:
        print(f'Error during extraction: {result.err_message}')
    else:
        print('Extraction successful')
    return result


async def run_upload_to_nomad(nomad_upload: UploadToNomadInput):
    client, worker, worker_task = await start_worker()
    result = await client.execute_activity(
        upload_to_nomad,
        nomad_upload,
        id='upload-to-nomad',
        task_queue='general_extraction_pipeline',
        start_to_close_timeout=timedelta(seconds=30),
    )
    worker_task.cancel()  # Cleanly shut down the worker after workflow completion
    return result


def upload_extraction_to_nomad(
    extraction_result: dict[str, Any], nomad_upload_config: dict[str, Any]
):
    nomad_upload_input = UploadToNomadInput(
        m_def=nomad_upload_config['m_def'],
        data=extraction_result,
        entry_name=nomad_upload_config['entry_name'],
        doi=nomad_upload_config.get('doi'),
        extraction_metadata=nomad_upload_config.get(
            'extraction_metadata', DEFAULT_EXTRACTION_METADATA
        ),
        multi_instance_field=nomad_upload_config.get('multi_instance_field'),
        upload_id=nomad_upload_config.get('upload_id'),
    )
    result = asyncio.run(run_upload_to_nomad(nomad_upload_input))
    if result is not None:
        print('Upload to Nomad successful')
    else:
        print('Failed to upload to Nomad')
    return result
