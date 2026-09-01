from collections.abc import Iterable
from typing import Any


def normalize_errors(errors: Any) -> list[str]:
    if errors is None:
        return []
    if isinstance(errors, BaseException):
        return [str(errors)]
    if isinstance(errors, str):
        return [errors]
    if isinstance(errors, Iterable):
        normalized = [str(error) for error in errors if error]
        return normalized
    return [str(errors)]


def failed_result(errors: Any, **payload: Any) -> dict[str, Any]:
    return {'success': False, 'errors': normalize_errors(errors), **payload}


def successful_result(**payload: Any) -> dict[str, Any]:
    return {'success': True, 'errors': [], **payload}

