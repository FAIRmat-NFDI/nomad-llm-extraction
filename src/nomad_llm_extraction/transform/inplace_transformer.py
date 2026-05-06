import re
from copy import deepcopy
from typing import Any

from nomad.datamodel.metainfo.annotations import Rule, Rules
from nomad.utils.json_transformer import Transformer

from nomad_llm_extraction.transform.utils import (
    deref,
    get_all_paths,
    get_array_regex,
    get_cut_data,
)


def get_new_path(match, path, sections):
    """
    Constructs a new path by replacing the array index placeholders with actual index values from the matched path.
    """
    new_path = ''
    start = 0
    for ngroup, group in enumerate(match.groups()):
        new_path += path[start : sections[ngroup][0]] + f'[{group}]'
        start = sections[ngroup][1]
    new_path += path[start:]
    return new_path


def _resolve_rule(data, rule, name):
    """
    Resolves a single rule with array indexes of type [n<number>] and generates new rules for all matching paths in the source data.
    """
    c = 0
    resolved_rules = {}

    source_path = rule.source
    target_path = rule.target

    re_pattern = get_array_regex(source_path)
    capture_pattern = re.compile(r'\[(n\d+?)\]')

    source_index_positions = [x for x in capture_pattern.finditer(source_path)]
    target_index_positions = [x for x in capture_pattern.finditer(target_path)]
    if len(source_index_positions) != len(target_index_positions):
        raise ValueError(
            'Mismatch between source and target array index placeholders: '
            f"{len(source_index_positions)} in source path '{source_path}' vs "
            f"{len(target_index_positions)} in target path '{target_path}'."
        )
    source_path_sections = [i.span() for i in source_index_positions]
    target_path_sections = [i.span() for i in target_index_positions]
    data_paths = get_all_paths(data)
    for i in data_paths:
        match = re_pattern.match(i)
        if match:
            new_source_path = get_new_path(match, source_path, source_path_sections)
            new_target_path = get_new_path(match, target_path, target_path_sections)
            resolved_rules[f'{name}_resolved_{c}'] = Rule(
                source=new_source_path, target=new_target_path
            )
            c += 1
    return {name: Rules(name=name, rules=resolved_rules)}


def resolve_rules(data, rules):
    """
    Resolves rules with array indexes of type [n<number>] to apply the transformations to all matching paths in the source data.
    """
    resolved_rules = {}
    if isinstance(rules, dict):
        for rule_name, rule in rules.items():
            r_rules = {
                k: v
                for n, r in resolve_rules(data, rule).items()
                for k, v in r.rules.items()
            }
            resolved_rules[rule_name] = Rules(
                name=getattr(rule, 'name', rule_name), rules=r_rules
            )
    elif isinstance(rules, Rules):
        for rule_name, rule in rules.rules.items():
            resolved_rules.update(_resolve_rule(data, rule, rule_name))

    elif isinstance(rules, Rule):
        return _resolve_rule(data, rules, '')
    return resolved_rules


def delete_path(data, path):
    parts = Transformer.parse_path(path)
    current = data
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            try:
                del current[part]
            except Exception:
                print(f'{path} is not present in the data, skipping deletion')
        else:
            try:
                current = current[part]
            except Exception:
                print(f'{path} is not present in the data, skipping deletion')


def del_sources(data, rules):
    """
    Deletes the source paths present in the rules from the source data.
    """
    if isinstance(rules, dict):
        for rule_name, rule in rules.items():
            data = del_sources(data, rule)
    elif isinstance(rules, Rules):
        for rule_name, rule in rules.rules.items():
            try:
                delete_path(data, rule.source)
            except Exception as e:
                print(rule.source, e)
    elif isinstance(rules, Rule):
        try:
            delete_path(data, rule.source)
        except Exception as e:
            print(rule.source, e)
    return data


def update_source_data(transformed_data: dict[str, Any], rules=None):
    """
    Merges the transformed json into the base archive and deletes the source paths present in the rules.
    """
    transformed_data = deepcopy(transformed_data)
    if rules is not None:
        transformed_data = del_sources(transformed_data, rules)
    return transformed_data


class InplaceTransformer(Transformer):
    """
    Extends the base Transformer class to apply transformations in place on the source data, merging the transformed data back into the source and deleting the original paths based on the provided rules.
    Additionally, it resolves rules with array indexes of type [n<number>] to apply the transformations to all matching paths in the source data.
    Example array rule:
    {
        source: 'a[n1].b[n2].c.d[n3].e',
        target: 'f[n1].g.h[n2].i[n3].k.l'
    }
    The number of arrays (n1, n2, n3) in both paths should be the same.
    """

    def transform_inplace(
        self,
        source_data: dict[str, Any],
        mapping_name: str,
        remove_source: bool = True,
        resolve_array_rules: bool = True,
    ) -> dict[str, Any]:
        if isinstance(source_data, list):
            transformed_data = []
            for i, item in enumerate(source_data):
                transformed_data.append(
                    self.transform_inplace(
                        item, mapping_name, remove_source, resolve_array_rules
                    )
                )
            return transformed_data
        else:
            new_mapping_name = mapping_name
            if resolve_array_rules:
                new_mapping_name = f'{mapping_name}_resolved'
                resolved_rules = resolve_rules(source_data, self.mapping_dict)
                self.mapping_dict[new_mapping_name] = resolved_rules[mapping_name]
            transformed_data = self.transform(
                source_data, new_mapping_name, deepcopy(source_data)
            )
            if remove_source:
                transformed_data = update_source_data(
                    transformed_data=transformed_data,
                    rules=self.mapping_dict[f'{mapping_name}_resolved'],
                )
            return transformed_data
