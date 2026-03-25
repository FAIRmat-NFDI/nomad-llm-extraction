import re
from copy import deepcopy

from loguru import logger

from nomad_llm_extraction.transform.utils import (
    check_path,
    clean_path,
    delete_section,
    deref,
    get_array_regex,
    get_b_data,
)

path_types = {'archive': 'a_path', 'schema': 'path'}


def default_cond(section, state):
    return True


def default_get_func_args(section, state):
    return state, None


def default_func_apply(b_data, path, func_args):
    return b_data


def get_section_type(section):
    """
    Infers the type of the section.
    """
    stype = section.get('type', '')
    if stype == '':
        stype = (
            'object'
            if 'allOf' in section or 'anyOf' in section or 'properties' in section
            else stype
        )
        stype = 'array' if 'items' in section else stype
    return stype


def update_state(state, name, p_name, stype=None, **kwargs):
    """
    Updates the state with the given name and path.
    The state contains:
    - name: the name of the current section
    - sname: the full path of the current section with section names in the schema
    - p_name: the name of the parent section
    - p_path: the path of the parent section in the schema
    - a_p_path: the path of the parent section in the json object
    - path: the path of the current section in the schema
    - a_path: the path of the current section in the json object
    - type: the type of the current section (property, array_property, section_property)
    """
    n_state = deepcopy(state)
    n_state['name'] = name
    n_state['p_name'] = p_name
    n_state['a_p_path'] = clean_path(f'{n_state["a_p_path"]}.{p_name}')
    n_state['p_path'] = clean_path(f'{n_state["p_path"]}.{p_name}')
    if stype == 'object':
        n_state['p_path'] = clean_path(
            f'{n_state["p_path"]}.{kwargs.get("ref_section", "allOf")}[{kwargs.get("idx", 0)}]'
        )
    elif stype == 'array':
        n_state['a_p_path'] = clean_path(f'{n_state["a_p_path"]}[n]')
    else:
        n_state['p_path'] = clean_path(f'{n_state["p_path"]}.properties')
    n_state['path'] = clean_path(f'{n_state["p_path"]}.{name}')
    n_state['a_path'] = clean_path(f'{n_state["a_p_path"]}.{name}')
    return n_state


def get_paths(
    section,
    cond=None,
    get_func_args=None,
    path_type='a_path',
    state={},
):
    """
    Traverses the json schema and returns the paths that satisfy the condition with arguments to be applied to the paths in the processing pipeline.

    Args:
        section: the json schema/section to traverse
        cond: Function that takes the section and state and returns a boolean indicating whether to include the path. If None, includes all paths.
        get_func_args: Function that takes the section and state and returns the arguments to be stored with the path. If None, stores None as the arguments.
        path_type: the type of path to return - 'a_path' for datapaths and 'path' for schema paths
        state: the state to be stored with the path, which can be updated with the get_func_args function.
    returns:
        A dictionary with the path as the key and the state, func_args, type as the value.
        func_args are the arguments for the function to be applied to the path
        The type can be 'property' or 'array' depending on the section, used to apply the functions in the pipeline.
    """
    # For jsonschemas from pydantic models
    if section == {'type': None}:
        return {}

    cond = default_cond if cond is None else cond
    get_func_args = default_get_func_args if get_func_args is None else get_func_args
    all_paths = {}
    prop_paths = {}
    arr_paths = {}

    if not state:
        state = {
            k: ''
            for k in [
                'sname',
                'name',
                'p_name',
                'p_path',
                'a_p_path',
                'path',
                'a_path',
                'type',
            ]
        }

    name = state['name']
    title = section.get('title', name)
    title = title if name == '' else name

    stype = get_section_type(section)

    if state['type'] == 'property' and cond(section, state):
        state, func_args = get_func_args(section, state)
        p = 'property'
        if stype == 'array':
            state['a_path'] = clean_path(f'{state["a_path"]}[n]')
            state['type'] = 'array_property'
            p = 'array'
        elif stype == 'object':
            state['type'] = 'section_property'
        prop_paths.update({title: [state, func_args, p]})

    if stype == 'object':
        for prop_name, prop_section in section.get('properties', {}).items():
            prop_state = update_state(state, prop_name, name)
            prop_state['type'] = 'property'
            prop_paths.update(
                get_paths(prop_section, cond, get_func_args, path_type, prop_state)
            )
        for i, v in prop_paths.items():
            v[0]['sname'] = clean_path(f'{title}.{i}')
        prop_paths = {v[0][path_type]: v for i, v in prop_paths.items()}

        sub_sections = [
            *[(i, 'allOf', s) for i, s in enumerate(section.get('allOf', []))],
            *[(i, 'anyOf', s) for i, s in enumerate(section.get('anyOf', []))],
        ]
        for idx, ref_section, sub_section in sub_sections:
            all_state = update_state(
                state, '', name, 'object', idx=idx, ref_section=ref_section
            )
            all_state['type'] = 'sub_section'
            all_paths.update(
                get_paths(sub_section, cond, get_func_args, path_type, all_state)
            )
        for i, v in all_paths.items():
            v[0]['sname'] = clean_path(f'{title}.{".".join(i.split(".")[:])}')
        all_paths = {v[0][path_type]: v for i, v in all_paths.items()}

    elif stype == 'array':
        arr_state = update_state(state, '', name, 'array')
        arr_state['type'] = 'array'
        arr_state['p_path'] = f'{arr_state["p_path"]}.items'
        arr_paths.update(
            get_paths(section['items'], cond, get_func_args, path_type, arr_state)
        )
        for i, v in arr_paths.items():
            v[0]['sname'] = clean_path(f'{title}.{i}')
        arr_paths = {v[0][path_type]: [*v[0:-1], 'array'] for i, v in arr_paths.items()}

    all_paths.update(arr_paths)
    all_paths.update(prop_paths)

    return all_paths


def get_array_paths(b_data, path):
    """
    Returns all the paths in the json object (benedict dict) that match the given path with array indexes.
    """
    re_pattern = get_array_regex(path)
    return [
        i for i in b_data.keypaths(indexes=True, sort=True) if re.match(re_pattern, i)
    ]


def update_data(b_data, paths, func_apply=None):
    """
    Applies the func_apply to the paths in the json object (benedict dict).
    If the path contains arrays, applies the func_apply to all the items in the arrays.
    """
    func_apply = default_func_apply if func_apply is None else func_apply

    if b_data is None:
        return None

    if isinstance(b_data, list):
        return [update_data(i, paths, func_apply) for i in b_data]

    for path, (state, func_args, stype) in paths.items():
        if stype == 'property' and check_path(b_data, path):
            b_data = func_apply(b_data, path, func_args)
        elif stype == 'array':
            if path[-3:] == '[n]':
                path = path[:-3]
            array_paths = get_array_paths(b_data, path)
            for array_path in array_paths:
                if check_path(b_data, array_path):
                    b_data = func_apply(b_data, array_path, func_args)
    return b_data


class ProcessingPipeline:
    def __init__(self, pipeline):
        """
        pipeline: dict of processing steps with the format:
        {
            'step_name': (func_apply, cond, get_func_args, proc_type)
        }
        func_apply: function that takes the json object, path, and func_args and returns the updated json object
        cond: function that takes the section and state and returns a boolean indicating whether to apply the func_apply
        get_func_args: function that takes the section and state and returns the arguments to be passed to func_apply
        proc_type: type of the object to be proccessed - 'archive' or 'schema'
        """
        self.pipeline = pipeline

    def apply(self, data, schema=None):
        """
        Applies the processing pipeline to the json object based on the schema.
        """
        if schema is None:
            schema = deepcopy(data)

        updated_b_data = get_b_data(data)
        for i, (proc_name, (func_apply, cond, get_func_args, proc_type)) in enumerate(
            self.pipeline.items()
        ):
            logger.info(f'Applying processing step {i}: {proc_name}')
            path_type = path_types[proc_type]
            paths = get_paths(schema, cond, get_func_args, path_type)
            updated_b_data = update_data(deepcopy(updated_b_data), paths, func_apply)
        return deref(updated_b_data)

    def clean(self, data, base_schema, target_schema, clean_func=None):
        """
        Cleans the json by deleting the paths that are in base_schema but not in target_schema.
        """
        clean_func = delete_section if clean_func is None else clean_func
        base_paths = get_paths(base_schema, None, None)
        target_paths = get_paths(target_schema, None, None)
        target_paths = [j[:-3] if j[-3:] == '[n]' else j for j in target_paths]
        del_paths = {}
        for i in base_paths:
            if i not in target_paths:
                del_paths[i] = base_paths[i]
        return deref(update_data(get_b_data(data), del_paths, clean_func))
