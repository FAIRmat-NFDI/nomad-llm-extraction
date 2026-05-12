def remove_keys_recursive(data, keys_to_remove):
    """
    Recursively removes specific keys from a nested dictionary or list of dictionaries.
    """
    # Convert keys to a set for O(1) lookup performance
    if not isinstance(keys_to_remove, set):
        keys_to_remove = set(keys_to_remove)

    if isinstance(data, dict):
        # We iterate over a list of keys because we can't mutate
        # a dictionary while iterating over it directly.
        for key in list(data.keys()):
            if key in keys_to_remove:
                del data[key]
            else:
                # Recurse into the value
                remove_keys_recursive(data[key], keys_to_remove)

    elif isinstance(data, list):
        # If it's a list, check every item inside it
        for item in data:
            remove_keys_recursive(item, keys_to_remove)

    return data


def remove_path_keys(data, keys_to_remove):
    if not isinstance(keys_to_remove, set):
        keys_to_remove = set(keys_to_remove)
    for k in keys_to_remove:
        path_parts = k.split('.')
        r = data
        try:
            for part in path_parts[:-1]:
                r = r[part]
            del r[path_parts[-1]]
        except Exception as e:
            print(f'Path {k} missing at {part}: {e}')
    return data


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
