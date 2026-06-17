"""
This module defines the post-processing pipeline for perovskite solar cell data extracted. It replicates the post processing steps defined in the original PERLA codebase,
but in a more modular and reusable way using the ProcessingPipeline class from nomad_llm_extraction.transform.json_transformer.
For a deeper understanding of the transformations applied please look at the perla_postproc_tutorial.ipynb notebook, where we go through the transformations. and how to write and use the transform pipeline."""

from nomad_llm_extraction.transform.common_transforms import (
    convert_unit,
    remove_none,
    remove_unit_value,
    rename_section,
    split_value_unit,
    unit_args,
    unit_cond,
)
from nomad_llm_extraction.transform.json_transformer import ProcessingPipeline
from nomad_llm_extraction.transform.utils import delete_section

KEY_MAPPING = {
    'bandgap': 'band_gap',
    'PCE_at_the_start_of_the_experiment': 'PCE_at_start',
    'PCE_at_the_end_of_experiment': 'PCE_at_end',
    'a_ions': 'ions_a_site',
    'b_ions': 'ions_b_site',
    'x_ions': 'ions_x_site',
    'time': 'time',
}

_REV_KEY_MAPPING = {value: key for key, value in KEY_MAPPING.items()}
_SPLIT_VALUE_UNIT = {'concentration'}
_SKIP_KEYS = {'additives'}


def _rename_cond(section, state):
    return state['name'] in _REV_KEY_MAPPING


def _rename_args(section, state):
    state['a_path'] = f'{state["a_p_path"]}.{_REV_KEY_MAPPING[state["name"]]}'
    return state, f'{state["a_p_path"]}.{state["name"]}'


def _delete_cond(section, state):
    return state['name'] in _SKIP_KEYS


def _split_value_unit_cond(section, state):
    return state['name'] in _SPLIT_VALUE_UNIT


def _layer_order_cond(section, state):
    return state['name'] == 'layers'


def _layer_order_args(section, state):
    return state, state['a_p_path']


def _get_layer_order(layers):
    if not layers or not isinstance(layers, list):
        return None
    names = [layer['name'] for layer in layers if layer.get('name')]
    return ','.join(names)


def _layer_order(c_data, path, func_args):
    c_data['layer_order'] = _get_layer_order(c_data[path])
    return c_data


def build_pipeline() -> ProcessingPipeline:
    return ProcessingPipeline(
        {
            'rename': [rename_section, _rename_cond, _rename_args],
            'unit_conversion': [convert_unit, unit_cond, unit_args],
            'split_unit_value': [split_value_unit, _split_value_unit_cond, None],
            'flatten_unit_value': [remove_unit_value, None, None],
            'delete_sections': [delete_section, _delete_cond, None],
            'layer_order': [_layer_order, _layer_order_cond, _layer_order_args],
            'remove_none': [remove_none, None, None],
        }
    )
