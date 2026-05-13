import json
import os
import re
from copy import deepcopy
from typing import Any

import jsonref
import requests
from scalpl import Cut

NOMAD_URL = os.environ.get('NOMAD_URL', 'https://nomad-lab.eu/prod/v1/api/v1/')


def check_path(c_data, path):
    try:
        c_data[path]
        return True
    except Exception:
        return False


def check_path_keypaths(key_paths, path):
    return path in key_paths


def clean_path(p):
    return '.'.join([i for i in p.split('.') if i != ''])


def delete_section(c_data, path, func_args=None):
    if check_path(c_data, path):
        del c_data[path]
    return c_data


def clean_section(c_data, path, func_args=None):
    value = c_data[path]
    if (
        value is None
        or value == ''
        or (isinstance(value, (list, dict)) and len(value) == 0)
    ):
        del c_data[path]
    return c_data


def deref(data):
    """
    Convert a benedict object or a dictionary with references to a regular dictionary by serializing and deserializing it.
    """
    try:
        return json.loads(json.dumps(data))
    except Exception as e:
        if e.args[0] == 'Object of type Cut is not JSON serializable':
            return deref_cut(data)
        else:
            raise e


def deref_cut(c_data):
    """
    Convert a Cut object to a regular dictionary by serializing and deserializing it.
    """
    if isinstance(c_data, list):
        return [json.loads(json.dumps(i.data)) for i in c_data]
    return json.loads(json.dumps(c_data.data))


def get_cut_data(data):
    if isinstance(data, list):
        return [Cut(deepcopy(i)) for i in data]
    return Cut(deepcopy(data))


def get_array_regex(path):
    """
    Converts a path with array indexes given as [*]/[n] to a regex pattern that matches and captures paths with any index.
    """
    re_pattern = re.escape(path)
    re_pattern = (
        '^' + re.sub(r'\\\[((n|\\\*)\d*?)\\\]', r'\[(\\d+)\]', re_pattern) + '$'
    )
    return re.compile(re_pattern)


def get_all_paths(data, current_path=''):
    if isinstance(data, Cut):
        data = data.data
    paths = [current_path] if current_path else []
    if isinstance(data, dict):
        for k, v in data.items():
            new_path = f'{current_path}.{k}' if current_path else k
            paths.extend(get_all_paths(v, new_path))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_path = f'{current_path}[{i}]'
            paths.extend(get_all_paths(v, new_path))
    return set(paths)


def merge_all_of(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merges 'allOf' lists into a single dictionary.
    """
    if not isinstance(schema, dict):
        return schema

    if 'allOf' in schema:
        all_of_list = schema.pop('allOf')
        for subschema in all_of_list:
            # Recursively merge the subschema first
            merged_sub = merge_all_of(subschema)
            # Update the base schema with subschema properties
            for key, value in merged_sub.items():
                if key == 'properties':
                    schema_properties = schema.get('properties', {})
                    for ik, iv in value.items():
                        # skip for overriden values
                        if ik not in schema_properties:
                            schema_properties.update({ik: iv})
                    schema['properties'] = schema_properties
                elif key not in schema:
                    schema[key] = value
    if 'properties' in schema:
        for k, v in schema['properties'].items():
            schema['properties'][k] = merge_all_of(v)

    if 'items' in schema:
        schema['items'] = merge_all_of(schema['items'])

    return schema


def merge_all_of_lim(schema, mschema=None, limit_depth=False, proc_schemas=[]):
    """
    Recursively merges 'allOf' lists into a single dictionary.
    If limit depth is True prevents expansion of circular references.
    """
    print(schema.get('$id', schema.get('name', 'Missing')))
    proc_schemas = deepcopy(proc_schemas)
    if mschema is None:
        mschema = schema
    mschema = deepcopy(mschema)
    if '$id' in schema and limit_depth:
        if schema['$id'] not in proc_schemas:
            proc_schemas.append(schema['$id'])
            # print(proc_schemas[-1])
        else:
            s = {'$ref': schema['$id']}
            for rk in ['allOf', 'properties', 'anyOf', 'oneOf']:
                mschema.pop(rk, None)
            s.update(mschema)
            if s.get('$id', '') == schema['$id']:
                s.pop('$id')
            return s
    if not isinstance(schema, dict):
        return schema

    if 'allOf' in schema:
        all_of_list = schema.pop('allOf')
        mall_of_list = mschema.pop('allOf')
        for i in range(len(all_of_list)):
            subschema = all_of_list[i]
            msubschema = mall_of_list[i]
            # Recursively merge the subschema first
            merged_sub = merge_all_of_lim(
                subschema, msubschema, limit_depth, proc_schemas
            )
            # Update the base schema with subschema properties
            for key, value in merged_sub.items():
                if key == 'properties':
                    schema_properties = schema.get('properties', {})
                    for ik, iv in value.items():
                        # skip for overriden values
                        if ik not in schema_properties:
                            schema_properties.update({ik: iv})
                    schema['properties'] = schema_properties

    if 'properties' in schema:
        for k, v in schema['properties'].items():
            print(k)
            mv = mschema['properties'][k]
            schema['properties'][k] = merge_all_of_lim(v, mv, limit_depth, proc_schemas)

    if 'items' in schema:
        schema['items'] = merge_all_of_lim(
            schema['items'], mschema['items'], limit_depth, proc_schemas
        )
    return schema


def remove_null_anyof(schema):
    """Recursively removes {'type': 'null'} from anyOf lists"""
    if isinstance(schema, dict):
        if 'anyOf' in schema:
            anyOf = remove_null_anyof(
                [
                    i
                    for i in schema.pop('anyOf', [])
                    if i != {'type': 'null'} and i != {'type': None}
                ]
            )
            if len(anyOf) == 1:
                schema.update(anyOf[0])
            else:
                schema['anyOf'] = anyOf
        return {k: remove_null_anyof(v) for k, v in schema.items()}
    elif isinstance(schema, list):
        return [remove_null_anyof(i) for i in schema]
    return schema

def resolve_schema(schema, remove_defs=False, resolve_allOf=False, remove_null_anyof=False):
    """
    Resolves a JSON schema by replacing references and optionally merging 'allOf' lists and removing '$defs'.
    """
    if remove_null_anyof:
        schema = remove_null_anyof(schema)
    schema = dict(jsonref.replace_refs(schema, jsonschema=True, proxies=False))
    if remove_defs and '$defs' in schema:
        del schema['$defs']
    if resolve_allOf:
        schema = merge_all_of(deepcopy(schema))
    return json.loads(json.dumps(schema))

def get_nomad_schema(m_def, unit_value=False, exclude_fields=None):
    schema_url = f'{NOMAD_URL}schemas/{m_def}?format=jsonschema'
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

def get_name_from_id(section_id):
    return section_id.split('@')[0].split('/')[-1]

def remove_sections(data, sections_to_remove):
    """
    Recursively removes specific keys from a nested dictionary or list of dictionaries.
    """
    # Convert keys to a set for O(1) lookup performance
    if not isinstance(sections_to_remove, set):
        sections_to_remove = set(sections_to_remove)

    if isinstance(data, dict):
        for key, section in list(data.items()):
            if isinstance(section, dict) and '$id' in section:
                id_name = get_name_from_id(section['$id'])
                if id_name in sections_to_remove:
                    del data[key]
                    continue
            new_section = remove_sections(section, sections_to_remove)
            if new_section is None or (
                isinstance(new_section, (list, dict)) and len(new_section) == 0
            ):
                del data[key]
            else:
                data[key] = new_section

    elif isinstance(data, list):
        # If it's a list, check every item inside it
        new_data = []
        for i, item in enumerate(data):
            add = 1
            if isinstance(item, dict):
                for id_field in ['$ref', '$id']:
                    if id_field in item:
                        id_name = get_name_from_id(item[id_field])
                        if id_name in sections_to_remove:
                            add = 0
                            break
            if add:
                new_data.append(remove_sections(item, sections_to_remove))
        data = new_data
    return data