import hashlib
import inspect
import json
import urllib.parse
from collections.abc import Callable
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import pdf2doi
import requests
import yaml
from jsonschema import ValidationError, validate
from loguru import logger
from nomad.units import ureg as nomad_ureg


def convert_to_nomad_unit(value, from_unit, to_unit):
    quantity = nomad_ureg.Quantity(value, from_unit)
    converted_quantity = quantity.to(to_unit)
    return {'value': converted_quantity.magnitude, 'unit': to_unit}


def extract_doi_from_pdf(filepath) -> str:
    doi = 'NOT_FOUND'
    try:
        pdf2doi_results = pdf2doi.pdf2doi(filepath)
        if pdf2doi_results is None:
            return doi
        pdf2doi_results = (
            pdf2doi_results[0] if isinstance(pdf2doi_results, list) else pdf2doi_results
        )
        if pdf2doi_results.get('identifier_type') == 'DOI':
            doi = pdf2doi_results.get('identifier', doi)
    except Exception as e:
        print(f'Could not extract DOI from {filepath}: {e}')
    return doi


def validate_with_schema(data, schema) -> tuple[bool, str | None]:
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        return False, str(e)
    return True, None


def get_hash(filepath: Path | str, mode: str = 'md5') -> str:
    h = hashlib.new(mode)
    with open(filepath, 'rb') as file:
        data = file.read()
    h.update(data)
    digest = h.hexdigest()
    return digest


def safe_asdict(obj):
    allowed_keys = {
        f.name
        for f in fields(obj)
        if f.metadata.get('serialize', True)  # Defaults to True if no metadata exists
    }

    return {k: v for k, v in asdict(obj).items() if k in allowed_keys}


def safe_json_default(obj):
    if is_dataclass(obj):
        return asdict(obj)

    if obj is not None:
        return f'<unserializable: {type(obj).__name__} > {str(obj).split("at")[0]}>'
    return None


def safe_json_dumps(data):
    return json.dumps(data, default=safe_json_default, indent=2)


def get_safe_ctx(data):
    return json.loads(safe_json_dumps(data))


def load_yaml_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def verify_activity_signature(
    func: Callable[..., Any], expected_params: dict[str, type]
):
    """
    Checks that a function accepts the exact argument names and types expected.
    """
    sig = inspect.signature(func)

    # 1. Check if the parameter counts and names match
    for param_name, expected_type in expected_params.items():
        if param_name not in sig.parameters:
            raise TypeError(
                f"Validation Failed: Function '{func.__name__}' is missing "
                f"the required argument '{param_name}'."
            )

        param = sig.parameters[param_name]

        # 2. Check type annotations
        logger.info(f'{param_name}: expected {expected_type}, got {param.annotation}')
        if param.annotation != expected_type.__name__:
            raise TypeError(
                f"Validation Failed: Argument '{param_name}' in '{func.__name__}' "
                f'must be annotated as {expected_type.__name__}, got {param.annotation}.'
            )

    return True


def get_temporal_activities(stages) -> list[tuple[str, Callable[..., Any]]]:
    from temporalio import activity

    temporal_activities = []
    for name, func in stages:
        try:
            wrapped_activity = activity.defn(name=name)(func)
        except ValueError as e:
            if e.args[0] == 'Function already contains activity definition':
                wrapped_activity = func  # Already decorated, use as is
            else:
                raise
        temporal_activities.append((name, wrapped_activity))
    return temporal_activities


def get_nomad_schema(m_def, unit_value=False, exclude_fields=None):
    from nomad_llm_extraction.config import NOMAD_URL

    schema_url = NOMAD_URL + f'schemas/{m_def}?format=jsonschema'
    if unit_value:
        schema_url += '&unit_value=true'
    response = requests.get(schema_url)
    if response.status_code == 200:
        schema = response.json()
        return schema
    else:
        raise ValueError(
            f'{response.status_code} Error fetching schema for {m_def}: {response.text}'
        )
