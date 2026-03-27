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
            if any(key in section for key in ['allOf', 'anyOf', 'properties'])
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
    n_state['p_name'] = p_name if p_name != '' else state['p_name']
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
    if section == {'type': None} or section == {'type': 'null'}:
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
                'sname',  # the full path of the current section with section names in the schema
                'name',  # the name of the current section
                'p_name',  # the name of the parent section
                'p_path',  # the path of the parent section in the schema
                'a_p_path',  # the path of the parent section in the json object
                'path',  # the path of the current section in the schema
                'a_path',  # the path of the current section in the json object
                'type',  # the type of the current section (property, array_property, section_property)
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

    def apply(self, data, schema=None, proc_type='archive'):
        """
        Applies the processing pipeline to the json object based on the schema.
        """
        if schema is None:
            schema = deepcopy(data)

        updated_b_data = get_b_data(data)
        for i, (proc_name, (func_apply, cond, get_func_args)) in enumerate(
            self.pipeline.items()
        ):
            logger.info(f'Applying processing step {i}: {proc_name}')
            path_type = path_types[proc_type]
            paths = get_paths(schema, cond, get_func_args, path_type)
            updated_b_data = update_data(deepcopy(updated_b_data), paths, func_apply)
        return deref(updated_b_data)

    def clean(self, data, target_schema=None, base_schema=None, clean_func=None):
        """
        Cleans the json by deleting the paths that are in base_schema but not in target_schema.
        """
        b_data = get_b_data(data)
        clean_func = delete_section if clean_func is None else clean_func
        if isinstance(b_data, list):
            return [
                self._clean(i, target_schema, base_schema, clean_func) for i in b_data
            ]
        else:
            return self._clean(b_data, target_schema, base_schema, clean_func)

    def _clean(self, b_data, target_schema=None, base_schema=None, clean_func=None):

        # If target_schema is None, delete none values and empty collections
        if target_schema is None:
            b_data.clean(strings=True, collections=True)
            return deref(b_data)
        # If base_schema is None, delete paths that are not in target_schema
        elif base_schema is None:
            data_key_paths = b_data.keypaths(indexes=True)
            target_paths = get_paths(target_schema, None, None, 'a_path')
            target_paths_arr = [i for i in target_paths if '[n]' in i]
            target_paths = [i[:-3] if i[-3:] == '[n]' else i for i in target_paths]
            target_paths_arr.extend(
                [i[:-3] for i in target_paths_arr if i[-3:] == '[n]']
            )
            del_paths = {
                i: [None, None, 'property']
                for i in data_key_paths
                if i not in target_paths
            }
            for i in target_paths_arr:
                arr_paths = get_array_paths(b_data, i)
                for arr_path in arr_paths:
                    if arr_path in del_paths:
                        del del_paths[arr_path]
            return deref(update_data(b_data, del_paths, clean_func))
        # If both target_schema and base_schema are provided, delete paths that are in base_schema but not in target_schema
        else:
            base_paths = get_paths(base_schema, None, None)
            target_paths = get_paths(target_schema, None, None)
            del_paths = {}
            for i in base_paths:
                if i not in target_paths:
                    del_paths[i] = base_paths[i]
            return deref(update_data(b_data, del_paths, clean_func))
