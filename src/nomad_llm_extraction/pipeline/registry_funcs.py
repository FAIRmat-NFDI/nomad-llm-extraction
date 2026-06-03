import logging
from collections.abc import Callable
from typing import Any, Literal

from nomad_llm_extraction.pipeline.models import StageContext, StageFunc
from nomad_llm_extraction.pipeline.registry_config import (
    CLASS_OBJ_REGISTRY,
    FUNCTION_REGISTRY,
    STAGES_REGISTRY,
)
from nomad_llm_extraction.utils.utils import verify_activity_signature

logger = logging.getLogger(__name__)


def get_registered_stage_func(stage_func_name: str) -> StageFunc:
    # print(f'Registry Memory ID: {id(STAGES_REGISTRY)}, contents: {STAGES_REGISTRY}')
    if stage_func_name not in STAGES_REGISTRY:
        raise KeyError(
            f'Unknown stage function name: {stage_func_name}. '
            f'Available stages: {list(STAGES_REGISTRY.keys())}'
        )
    return STAGES_REGISTRY[stage_func_name]


def get_registered_processor_func(proc_func_name: str) -> Callable[..., Any]:
    if proc_func_name not in FUNCTION_REGISTRY:
        raise KeyError(
            f'Unknown processor function name: {proc_func_name}. '
            f'Available functions: {list(FUNCTION_REGISTRY.keys())}'
        )
    return FUNCTION_REGISTRY[proc_func_name]


def get_registered_object(
    self, objclass_name: str, constructor_args: dict[str, Any]
) -> Any:
    if objclass_name not in CLASS_OBJ_REGISTRY:
        raise KeyError(
            f'Unknown class/object name: {objclass_name}. '
            f'Available objects: {list(CLASS_OBJ_REGISTRY.keys())}'
        )
    return CLASS_OBJ_REGISTRY[objclass_name](**constructor_args)


def register_stage_func(
    stage_func_name: str, stage_func: StageFunc, overwrite: bool = False
):
    if stage_func_name in STAGES_REGISTRY:
        if not overwrite:
            raise KeyError(
                f'Stage function name {stage_func_name} is already registered'
            )
        else:
            logger.warning(
                f'Overwriting existing stage function registration for {stage_func_name}'
            )
    verify_activity_signature(
        stage_func, expected_params={'ctx': StageContext, 'stage_name': str}
    )
    STAGES_REGISTRY[stage_func_name] = stage_func
    print(f'Registered stage function: {stage_func_name} in {STAGES_REGISTRY}')


def register_processor_func(
    proc_func_name: str, proc_func: Callable[..., Any], overwrite: bool = False
):
    if proc_func_name in FUNCTION_REGISTRY:
        if not overwrite:
            raise KeyError(
                f'Processor function name {proc_func_name} is already registered'
            )
        else:
            logger.warning(
                f'Overwriting existing processor function registration for {proc_func_name}'
            )
    FUNCTION_REGISTRY[proc_func_name] = proc_func
    print(f'Registered processor function: {proc_func_name} in {FUNCTION_REGISTRY}')


def register_class_obj(obj_name: str, obj: Any, overwrite: bool = False):
    if obj_name in CLASS_OBJ_REGISTRY:
        if not overwrite:
            raise KeyError(f'Class/object name {obj_name} is already registered')
        else:
            logger.warning(
                f'Overwriting existing class/object registration for {obj_name}'
            )
    CLASS_OBJ_REGISTRY[obj_name] = obj
    print(f'Registered class/object: {obj_name} in {CLASS_OBJ_REGISTRY}')


registration_functions = {
    'stage': register_stage_func,
    'processor': register_processor_func,
    'class_obj': register_class_obj,
}


def register_part(
    name, type: Literal['stage', 'processor', 'class_obj'], overwrite=False
):
    def decorator(obj):
        try:
            registration_func = registration_functions[type]
        except KeyError:
            raise ValueError(
                f'Unknown registration type: {type}. '
                f'Valid types are: {list(registration_functions.keys())}'
            )
        registration_func(name, obj, overwrite=overwrite)

    return decorator
