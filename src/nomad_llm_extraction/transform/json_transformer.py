import re

from copy import deepcopy


def default_cond(section, state):
    return True


def default_get_func_args(section, state):
    return state, None


def default_func_apply(jbobj, path, func_args):
    return jbobj


def clean_path(p):
    return '.'.join([i for i in p.split('.') if i != ''])


def check_path(jbobj, path):
    return path in jbobj


def update_state(state, name, p_name, stype=None, **kwargs):
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


def get_stype(section):
    stype = section.get('type', '')
    if stype == '':
        stype = (
            'object'
            if 'allOf' in section or 'anyOf' in section or 'properties' in section
            else stype
        )
        stype = 'array' if 'items' in section else stype
    return stype


def get_paths(
    section,
    state={},
    cond=None,
    get_func_args=lambda x, y: (y, None),
    path_type='a_path',
):

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

    stype = get_stype(section)

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
                get_paths(prop_section, prop_state, cond, get_func_args, path_type)
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
            all_paths.update(get_paths(sub_section, all_state, cond, get_func_args))
        for i, v in all_paths.items():
            v[0]['sname'] = clean_path(f'{title}.{".".join(i.split(".")[:])}')
        all_paths = {v[0][path_type]: v for i, v in all_paths.items()}

    elif stype == 'array':
        arr_state = update_state(state, '', name, 'array')
        arr_state['type'] = 'array'
        arr_state['p_path'] = f'{arr_state["p_path"]}.items'
        arr_paths.update(
            get_paths(section['items'], arr_state, cond, get_func_args, path_type)
        )
        for i, v in arr_paths.items():
            v[0]['sname'] = clean_path(f'{title}.{i}')
        arr_paths = {v[0][path_type]: [*v[0:-1], 'array'] for i, v in arr_paths.items()}

    all_paths.update(arr_paths)
    all_paths.update(prop_paths)

    return all_paths


def update_archive(jbobj, paths, func_apply=None):
    func_apply = default_func_apply if func_apply is None else func_apply

    if jbobj is None:
        return None
    if isinstance(jbobj, list):
        return [update_archive(i, paths, func_apply) for i in jbobj]
    for path, (state, func_args, stype) in paths.items():
        if stype == 'property' and check_path(jbobj, path):
            jbobj = func_apply(jbobj, path, func_args)
        elif stype == 'array':
            sub_paths = [i for i in path.split('[n]', maxsplit=1) if i != '']
            if check_path(jbobj, sub_paths[0]):
                if len(sub_paths) > 1:
                    n_stype = 'array' if '[n]' in sub_paths[1][1:] else 'property'
                    n_state = deepcopy(state)
                    n_state['a_path'] = sub_paths[1][1:]
                    n_paths = {n_state['a_path']: [n_state, func_args, n_stype]}
                    jbobj[sub_paths[0]] = update_archive(
                        jbobj[sub_paths[0]], n_paths, func_apply
                    )
                else:
                    jbobj = func_apply(jbobj, sub_paths[0], func_args)
    return jbobj


def get_array_paths(jbobj, path):
    re_pattern = re.escape(path)
    re_pattern = '^' + re_pattern.replace(r'\[n\]', r'\[(\d+)\]') + '$'
    return [
        i for i in jbobj.keypaths(indexes=True, sort=True) if re.match(re_pattern, i)
    ]


def update_archive2(jbobj, paths, func_apply=None):
    func_apply = default_func_apply if func_apply is None else func_apply

    if jbobj is None:
        return None
    if isinstance(jbobj, list):
        return [update_archive2(i, paths, func_apply) for i in jbobj]
    for path, (state, func_args, stype) in paths.items():
        if stype == 'property' and check_path(jbobj, path):
            jbobj = func_apply(jbobj, path, func_args)
        elif stype == 'array':
            if path[-3:] == '[n]':
                path = path[:-3]
            array_paths = get_array_paths(jbobj, path)
            for array_path in array_paths:
                if check_path(jbobj, array_path):
                    jbobj = func_apply(jbobj, array_path, func_args)
    return jbobj


def merge_all_of(schema):
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
                    schema['properties'] = {**schema.get('properties', {}), **value}
                else:
                    schema[key] = value

    if 'properties' in schema:
        for k, v in schema['properties'].items():
            schema['properties'][k] = merge_all_of(v)

    if 'items' in schema:
        schema['items'] = merge_all_of(schema['items'])

    return schema
