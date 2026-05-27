import hashlib
from pathlib import Path

from jsonschema import ValidationError, validate
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


import inspect
from collections.abc import Callable
from typing import Any


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

        # param = sig.parameters[param_name]

        # # 2. Check type annotations
        # print(f'{param_name}: expected {expected_type}, got {param.annotation}')
        # if param.annotation != expected_type.__name__:
        #     raise TypeError(
        #         f"Validation Failed: Argument '{param_name}' in '{func.__name__}' "
        #         f'must be annotated as {expected_type.__name__}, got {param.annotation}.'
        #     )

    return True


def get_temporal_activities(stages) -> dict[str, Callable[..., Any]]:
    from temporalio import activity

    temporal_activities = {}
    for name, func in stages.items():
        wrapped_activity = activity.defn(name=name)(func)
        temporal_activities[name] = wrapped_activity
    return temporal_activities


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
