from copy import deepcopy

from nomad_llm_extraction.utils.utils import convert_to_nomad_unit


def unit_cond(section, state):
    return 'unit' in section


def unit_args(section, state):
    return state, section['unit']


def convert_unit(b_data, path, unit):
    def convert_item(item, unit):
        if isinstance(item, dict) and 'value' in item and 'unit' in item:
            return convert_to_nomad_unit(item['value'], item['unit'], unit)
        return item

    if path.split('.')[-1] == 'value':
        return convert_unit(b_data, path.rsplit('.', 1)[0], unit)
    if b_data[path] is None:
        return b_data
    items = b_data[path]
    if isinstance(items, list):
        items = [convert_item(item, unit) for item in items]
    else:
        items = convert_item(items, unit)
    b_data[path] = items
    return b_data


def rename_section(b_data, path, new_path):
    b_data[new_path] = deepcopy(b_data[path])
    del b_data[path]
    return b_data


def remove_unit_value(b_data, path, func_args):
    if isinstance(b_data[path], list):
        b_data[path] = [
            item['value'] if isinstance(item, dict) and 'value' in item else item
            for item in b_data[path]
        ]
    elif isinstance(b_data[path], dict) and 'value' in b_data[path]:
        b_data[path] = b_data[path]['value']
    return b_data


def delete_section(b_data, path, func_args):
    del b_data[path]
    return b_data


def split_value_unit(b_data, path, func_args):
    if b_data[path] is None:
        return b_data
    value = b_data[path]['value']
    unit = b_data[path]['unit']
    b_data[path] = value
    b_data[f'{path}_unit'] = unit
    return b_data


def remove_none(b_data, path, func_args):
    if b_data[path] is None:
        del b_data[path]
    return b_data


def update_unit_value_schema(b_schema, path, unit):
    value_schema = deepcopy(b_schema[path])
    del value_schema['unit']
    b_schema[path] = {
        'properties': {
            'value': value_schema.copy(),
            'unit': {'type': 'string', 'enum': [unit]},
        }
    }
    for key in ['title', 'description']:
        if key in value_schema:
            b_schema[path][key] = value_schema[key]
    return b_schema


def flatten_unit_value_schema_cond(section, state):
    return (
        'properties' in section
        and 'value' in section['properties']
        and 'unit' in section['properties']
    )


def flatten_unit_value_schema(b_schema, path, unit=None):
    value_schema = deepcopy(b_schema[path])
    b_schema[path].update(value_schema['properties']['value'])
    b_schema[path]['unit'] = (
        unit if unit is not None else value_schema['properties']['unit']['enum'][0]
    )
    del b_schema[path]['properties']
    return b_schema
