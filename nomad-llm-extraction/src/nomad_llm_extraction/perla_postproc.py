# import jsonschema
import json
from copy import deepcopy

import jsonref
from benedict import benedict
from perovskite_solar_cell_database.llm_extraction_schema import (
    LLMExtractedPerovskiteSolarCell,
)

from nomad_llm_extraction.utils import get_path_b2, update_archive_b2

KEY_MAPPING = {
    'bandgap': 'band_gap',
    'PCE_at_the_start_of_the_experiment': 'PCE_at_start',
    'PCE_at_the_end_of_experiment': 'PCE_at_end',
    'a_ions': 'ions_a_site',
    'b_ions': 'ions_b_site',
    'x_ions': 'ions_x_site',
    'time': 'time',  # Keep name, but used for unit check context
}
REV_KEY_MAPPING = {v: k for k, v in KEY_MAPPING.items()}
SPLIT_VALUE_UNIT = ['concentration']


SKIP_KEYS = ['additives']


def unit_cond(section, state):
    return 'unit' in section


def unit_args(section, state):
    return state, section['unit']


def unit_cond2(section, state):
    return 'unit' in section and section['unit'] == 'mole / second'


def cond_re(section, state):
    return state['name'] in REV_KEY_MAPPING


def re_args(section, state):
    state['a_path'] = f'{state["a_p_path"]}.{REV_KEY_MAPPING[state["name"]]}'
    return state, f'{state["a_p_path"]}.{state["name"]}'


def rename(jbobj, path, new_path):
    temp_value = deepcopy(jbobj[path])
    jbobj[new_path] = temp_value
    del jbobj[path]
    return jbobj


def remove_uv(jbobj, path, func_args):
    if isinstance(jbobj[path], list):
        jbobj[path] = [
            i['value'] if (isinstance(i, dict) and 'value' in i) else i
            for i in jbobj[path]
        ]
    elif isinstance(jbobj[path], dict) and 'value' in jbobj[path]:
        jbobj[path] = jbobj[path]['value']
    return jbobj


def convert_unit(jbobj, path, unit):
    # print(j_path)
    if jbobj[path] is None:
        return jbobj
    items = jbobj[path]
    if isinstance(items, list):
        items = [i['value'] for i in items]
    else:
        items = items['value']
    jbobj[path] = items
    return jbobj


def convert_unit2(jbobj, path, unit):
    # print(j_path)
    if jbobj[path] is None:
        return jbobj
    items = jbobj[path]
    if isinstance(items, list):
        items = [i for i in items]
    else:
        items = items
    jbobj[path] = items
    return jbobj


def cond_del(section, state):
    return state['name'] in SKIP_KEYS


def delete(jbobj, path, func_args):
    del jbobj[path]
    return jbobj


def cond_split_value_unit(section, state):
    return state['name'] in SPLIT_VALUE_UNIT


def split_value_unit(jbobj, path, func_args):
    if jbobj[path] is None:
        return jbobj
    value = jbobj[path]['value']
    unit = jbobj[path]['unit']
    jbobj[path] = value
    jbobj[f'{path}_unit'] = unit
    return jbobj


def cond_layer(section, state):
    return state['name'] == 'layers'


def layer_order_args(section, state):
    return state, state['a_p_path']


def get_layer_order(layers):
    if not layers or not isinstance(layers, list):
        return None
    # Filter layers that have a name and join them
    names = [l['name'] for l in layers if l.get('name')]
    return ','.join(names)


def layer_order(jbobj, path, func_args):
    # if jbobj[path] is not None:
    jbobj['layer_order'] = get_layer_order(jbobj[path])
    return jbobj


def remove_none(jbobj, path, func_args):
    if jbobj[path] is None:
        del jbobj[path]
    return jbobj


perov_nomad_jschema = LLMExtractedPerovskiteSolarCell.m_def.m_to_json_schema()
resolved_schema = jsonref.replace_refs(perov_nomad_jschema, jsonschema=True)
perov_jschema = json.load(open('llm_extraction_schema.json'))
perov_resolved_schema = jsonref.replace_refs(perov_jschema, jsonschema=True)
archive = json.load(
    open('temp/extractions/claude-sonnet-4-20250514/10.1002--aenm.202506634.json')
)['cells']

proc_pipeline = {
    'rename': (cond_re, re_args, rename),
    'unit': (unit_cond, unit_args, convert_unit2),
    'split_uv': (cond_split_value_unit, None, split_value_unit),
    'flatten': (None, None, remove_uv),
    'delete': (cond_del, None, delete),
    'layer_order': (cond_layer, layer_order_args, layer_order),
    'remove_none': (None, None, remove_none),
}

updated_archive = [benedict(deepcopy(i)) for i in archive]
for i, (proc, (cond, get_func_args, func_apply)) in enumerate(proc_pipeline.items()):
    print(proc)
    paths = get_path_b2(resolved_schema, '', cond, get_func_args)
    updated_archive = update_archive_b2(deepcopy(updated_archive), paths, func_apply)

perov_paths = get_path_b2(perov_resolved_schema, '', None, None)
paths = get_path_b2(resolved_schema, '', None, None)
paths_c = [j[:-3] if j[-3:] == '[n]' else j for j in paths]
del_paths = {}
for i in perov_paths:
    if i not in paths_c:
        del_paths[i] = perov_paths[i]

updated_archive_del_perov = update_archive_b2(
    deepcopy(updated_archive), del_paths, delete
)

json.dump(
    updated_archive_del_perov[0],
    open('10.1002--aenm.202506634_updated.archive.json', 'w'),
    indent=2,
)
