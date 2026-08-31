from functools import cache

from nomad.cli.dev import _generate_metainfo
from nomad.schemas import get_schema


@cache
def get_all_validmdefs():
    from nomad.config import config
    from nomad.config.models.plugins import SchemaPackageEntryPoint
    from nomad.datamodel import Environment
    from nomad.metainfo import Package

    config.load_plugins()
    if config.plugins is not None:  # Added check
        for entry_point in config.plugins.entry_points.filtered_values():
            if (
                isinstance(entry_point, SchemaPackageEntryPoint)
                and entry_point.name != 'LLMExtractor'
            ):
                try:
                    entry_point.load()
                except Exception as e:
                    print(f'Error loading schema package {entry_point.name}: {e}')
                    continue
    export = Environment()

    # The registry dictionary will also contain all aliases. To not repeat
    # all of the aliases, we check that only unique values are added.
    unique_packages = set()
    for package in Package.registry.values():
        if package not in unique_packages:
            export.m_add_sub_section(Environment.packages, package)
            unique_packages.add(package)
    metainfo_json = _generate_metainfo(export)
    mdefs = []
    valid_mdefs = []
    for i in metainfo_json['packages']:
        mdef = i['name']
        if mdef.startswith('pynxtools'):
            continue
        for s in i['section_definitions']:
            if s.get('m_parent_sub_section') != 'inner_section_definitions':
                mdefs.append(f'{mdef}.{s["name"]}')
            for inner in s.get('inner_section_definitions', []):
                mdefs.append(f'{mdef}.{s["name"]}.{inner["name"]}')
    for mdef in mdefs:
        try:
            get_schema(mdef).m_to_json_schema()
            valid_mdefs.append(mdef)
        except Exception:
            continue
    return valid_mdefs


@cache
def _get_entry_data_section_names() -> list[str, ...]:
    """Return qualified names of EntryData sections from all loaded plugins."""
    from nomad.datamodel import EntryData, all_metainfo_packages
    from nomad.metainfo import Section

    environment = all_metainfo_packages()
    if environment is None:
        return ()

    entry_data_sections = {
        section
        for package in environment.packages
        for section in package.section_definitions
        if isinstance(section, Section)
        and section is not EntryData.m_def
        and section.m_follows(EntryData.m_def, self_as_definition=True)
    }
    print(f'Found {len(entry_data_sections)} EntryData sections in loaded plugins.')
    valid_mdefs = []
    for section in entry_data_sections:
        try:
            section.m_to_json_schema()
            valid_mdefs.append(section.qualified_name())
        except Exception:
            print(
                f'Failed to generate JSON schema for section {section.qualified_name()}.'
            )
            continue
    return sorted(valid_mdefs)


@cache
def _get_section_names() -> list[str, ...]:
    """Return qualified names of EntryData sections from all loaded plugins."""
    from nomad.datamodel import EntryData, all_metainfo_packages
    from nomad.metainfo import Section

    environment = all_metainfo_packages()
    if environment is None:
        return ()

    entry_data_sections = {
        section
        for package in environment.packages
        for section in package.section_definitions
        if isinstance(section, Section) and section is not EntryData.m_def
    }
    print(f'Found {len(entry_data_sections)} EntryData sections in loaded plugins.')
    valid_mdefs = []
    for section in entry_data_sections:
        try:
            section.m_to_json_schema()
            valid_mdefs.append(section.qualified_name())
        except Exception:
            print(
                f'Failed to generate JSON schema for section {section.qualified_name()}.'
            )
            continue
    return sorted(valid_mdefs)


# ENTRY_DATA_SECTION_NAMES = _get_entry_data_section_names()

# MDEF_LIST = sorted(get_all_validmdefs())
# MDEF_LIST = _get_entry_data_section_names()
MDEF_LIST = _get_section_names()
# MDEF_LIST = [f'{mdef.split(".")[-1]} ({mdef})' for mdef in MDEF_LIST]
