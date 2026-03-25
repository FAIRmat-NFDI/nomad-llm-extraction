import json
from copy import deepcopy
from pathlib import Path

import pytest
from benedict import benedict
from nomad.datamodel.metainfo.annotations import Rules
from perovskite_solar_cell_database.llm_extraction_schema import (
    LLMExtractedPerovskiteSolarCell,
)

from nomad_llm_extraction.transform.common_transforms import *
from nomad_llm_extraction.transform.inplace_transformer import InplaceTransformer
from nomad_llm_extraction.transform.json_transformer import (
    ProcessingPipeline,
    get_paths,
)
from nomad_llm_extraction.transform.utils import resolve_schema

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = ROOT / 'data' / '10.1002--aenm.202506634.json'
UPDATED_ARCHIVE_PATH = ROOT / 'data' / '10.1002--aenm.202506634_updated.archive.json'
SCHEMA_PATH = ROOT / 'data' / 'llm_extraction_schema.json'

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


def layer_order(b_data, path, func_args):
    b_data['layer_order'] = get_layer_order(b_data[path])
    return b_data


@pytest.fixture(scope='module')
def resolved_schema():
    return resolve_schema(
        LLMExtractedPerovskiteSolarCell.m_def.m_to_json_schema(), remove_defs=True
    )


@pytest.fixture(scope='module')
def perov_resolved_schema():
    with SCHEMA_PATH.open() as handle:
        return resolve_schema(json.load(handle))


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
        'rename': [rename_section, rename_cond, rename_args, 'archive'],
        'unit_conversion': [convert_unit, unit_cond, unit_args, 'archive'],
        'split_unit_value': [split_value_unit, split_value_unit_cond, None, 'archive'],
        'flatten_unit_value': [remove_unit_value, None, None, 'archive'],
        'delete_sections': [delete_section, delete_cond, None, 'archive'],
        'layer_order': [layer_order, layer_order_cond, layer_order_args, 'archive'],
        'remove_none': [remove_none, None, None, 'archive'],
    }
    return ProcessingPipeline(proc_pipeline)


def test_processing_pipeline_matches_expected_archive(
    archive,
    expected_updated_archive,
    perov_resolved_schema,
    processing_pipeline,
    resolved_schema,
):
    updated_archive = processing_pipeline.apply(archive, resolved_schema)
    cleaned_archive = processing_pipeline.clean(
        updated_archive, perov_resolved_schema, resolved_schema
    )

    assert cleaned_archive[0] == expected_updated_archive


def test_schema_pipeline_round_trips_unit_wrappers(resolved_schema):
    schema_pipeline = ProcessingPipeline(
        {'add_unit_value': [update_unit_value_schema, unit_cond, unit_args, 'schema']}
    )
    reverse_schema_pipeline = ProcessingPipeline(
        {
            'remove_unit_value': [
                flatten_unit_value_schema,
                flatten_unit_value_schema_cond,
                None,
                'schema',
            ]
        }
    )

    updated_schema = schema_pipeline.apply(resolved_schema)
    unit_paths = get_paths(resolved_schema, unit_cond, unit_args, 'path')
    benedict_schema = benedict(deepcopy(updated_schema))

    for path in unit_paths:
        assert path in benedict_schema
        assert 'value' in benedict_schema[path]['properties']
        assert 'unit' in benedict_schema[path]['properties']

    flattened_schema = reverse_schema_pipeline.apply(updated_schema)
    flattened_benedict = benedict(deepcopy(flattened_schema))

    for path in unit_paths:
        assert path in flattened_benedict
        assert 'unit' in flattened_benedict[path]


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
        transformed = benedict(deepcopy(updated_archive[item]))
        original = benedict(deepcopy(archive[item]))
        concentration_paths = [
            '.'.join(path.split('.')[:-1])
            for path in original.keypaths(indexes=True)
            if path.split('.')[-1] == 'concentration'
        ]
        for path in concentration_paths:
            assert f'{path}.concentration_unit' in transformed
            assert f'{path}.concentration.value' not in transformed
