import json
from copy import deepcopy
from pathlib import Path

import pytest

# from benedict import benedict
from nomad.datamodel.metainfo.annotations import Rules
from scalpl import Cut

from nomad_llm_extraction.transform.common_transforms import (
    convert_unit,
    flatten_unit_value_schema,
    flatten_unit_value_schema_cond,
    remove_none,
    remove_unit_value,
    rename_section,
    split_value_unit,
    unit_args,
    unit_cond,
    update_unit_value_schema,
)
from nomad_llm_extraction.transform.inplace_transformer import InplaceTransformer
from nomad_llm_extraction.transform.json_transformer import (
    ProcessingPipeline,
    get_paths,
)
from nomad_llm_extraction.transform.utils import (
    check_path,
    delete_section,
    resolve_schema,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = ROOT / 'data' / '10.1002--aenm.202506634.json'
UPDATED_ARCHIVE_PATH = ROOT / 'data' / '10.1002--aenm.202506634_updated.archive.json'
PEROV_SCHEMA_PATH = ROOT / 'data' / 'llm_extraction_schema.json'
NOMAD_SCHEMA_PATH = ROOT / 'data' / 'nomad_llm_extraction_schema.json'
KEY_MAPPING = {
    'bandgap': 'band_gap',
    'PCE_at_the_start_of_the_experiment': 'PCE_at_start',
    'PCE_at_the_end_of_experiment': 'PCE_at_end',
    'a_ions': 'ions_a_site',
    'b_ions': 'ions_b_site',
    'x_ions': 'ions_x_site',
    'time': 'time',
}
REV_KEY_MAPPING = {value: key for key, value in KEY_MAPPING.items()}
SPLIT_VALUE_UNIT = ['concentration']
SKIP_KEYS = ['additives']


def rename_cond(section, state):
    return state['name'] in REV_KEY_MAPPING


def rename_args(section, state):
    state['a_path'] = f'{state["a_p_path"]}.{REV_KEY_MAPPING[state["name"]]}'
    return state, f'{state["a_p_path"]}.{state["name"]}'


def delete_cond(section, state):
    return state['name'] in SKIP_KEYS


def split_value_unit_cond(section, state):
    return state['name'] in SPLIT_VALUE_UNIT


def layer_order_cond(section, state):
    return state['name'] == 'layers'


def layer_order_args(section, state):
    return state, state['a_p_path']


def get_layer_order(layers):
    if not layers or not isinstance(layers, list):
        return None
    names = [layer['name'] for layer in layers if layer.get('name')]
    return ','.join(names)


def layer_order(c_data, path, func_args):
    c_data['layer_order'] = get_layer_order(c_data[path])
    return c_data


def clean_func(c_data, path, func_args):
    if 'additional_parameters' in path:
        return c_data
    if check_path(c_data, path):
        del c_data[path]
    return c_data


@pytest.fixture(scope='module')
def resolved_schema():
    with NOMAD_SCHEMA_PATH.open() as handle:
        return resolve_schema(json.load(handle), remove_defs=True, resolve_refs=True)


@pytest.fixture(scope='module')
def perov_resolved_schema():
    with PEROV_SCHEMA_PATH.open() as handle:
        return resolve_schema(json.load(handle), remove_null_anyof=True)


@pytest.fixture(scope='module')
def archive():
    with ARCHIVE_PATH.open() as handle:
        return json.load(handle)['cells']


@pytest.fixture(scope='module')
def expected_updated_archive():
    with UPDATED_ARCHIVE_PATH.open() as handle:
        return json.load(handle)


@pytest.fixture(scope='module')
def processing_pipeline():
    proc_pipeline = {
        'rename': [rename_section, rename_cond, rename_args],
        'unit_conversion': [convert_unit, unit_cond, unit_args],
        'split_unit_value': [split_value_unit, split_value_unit_cond, None],
        'flatten_unit_value': [remove_unit_value, None, None],
        'delete_sections': [delete_section, delete_cond, None],
        'layer_order': [layer_order, layer_order_cond, layer_order_args],
        'remove_none': [remove_none, None, None],
    }
    return ProcessingPipeline(proc_pipeline)


def test_processing_pipeline_matches_expected_archive(
    archive,
    expected_updated_archive,
    perov_resolved_schema,
    processing_pipeline,
    resolved_schema,
):
    updated_archive = processing_pipeline.apply(
        archive, resolved_schema, proc_type='archive'
    )
    cleaned_archive = processing_pipeline.clean(
        updated_archive, resolved_schema, clean_func=clean_func
    )

    assert cleaned_archive[0] == expected_updated_archive


def test_schema_pipeline_round_trips_unit_wrappers(resolved_schema):
    schema_pipeline = ProcessingPipeline(
        {'add_unit_value': [update_unit_value_schema, unit_cond, unit_args]}
    )
    reverse_schema_pipeline = ProcessingPipeline(
        {
            'remove_unit_value': [
                flatten_unit_value_schema,
                flatten_unit_value_schema_cond,
                None,
            ]
        }
    )

    updated_schema = schema_pipeline.apply(resolved_schema, proc_type='schema')
    unit_paths = get_paths(resolved_schema, unit_cond, unit_args, 'path')
    cut_schema = Cut(deepcopy(updated_schema))

    for path in unit_paths:
        assert path in cut_schema
        assert 'value' in cut_schema[path]['properties']
        assert 'unit' in cut_schema[path]['properties']

    flattened_schema = reverse_schema_pipeline.apply(updated_schema, proc_type='schema')
    flattened_cut_schema = Cut(deepcopy(flattened_schema))

    for path in unit_paths:
        assert path in flattened_cut_schema
        assert 'unit' in flattened_cut_schema[path]


def test_inplace_transformer_splits_concentration_units(archive):
    rules = {
        'split_concentration': Rules(
            name='split_concentration',
            rules={
                'map': {
                    'source': 'layers[n1].deposition[n2].solution.solutes[n3].concentration.value',
                    'target': 'layers[n1].deposition[n2].solution.solutes[n3].concentration',
                },
                'map2': {
                    'source': 'layers[n1].deposition[n2].solution.solutes[n3].concentration.unit',
                    'target': 'layers[n1].deposition[n2].solution.solutes[n3].concentration_unit',
                },
            },
        )
    }

    transformer = InplaceTransformer(rules)
    updated_archive = transformer.transform_inplace(archive, 'split_concentration')

    for item in range(len(archive)):
        transformed = Cut(deepcopy(updated_archive[item]))
        original = Cut(deepcopy(archive[item]))
        concentration_paths = get_paths(original, split_value_unit_cond, None, 'path')
        for path in concentration_paths:
            assert f'{path}.concentration_unit' in transformed
            assert f'{path}.concentration.value' not in transformed
