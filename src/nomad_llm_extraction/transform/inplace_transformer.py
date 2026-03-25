import re
from copy import deepcopy
from typing import Any

from nomad.datamodel.metainfo.annotations import Rule, Rules
from nomad.utils.json_transformer import Transformer

from nomad_llm_extraction.transform.utils import (
    delete_section,
    deref,
    get_array_regex,
    get_b_data,
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


def _resolve_rule(b_data, rule, name):
    """
    Resolves a single rule with array indexes of type [n<number>] and generates new rules for all matching paths in the source data.
    """
    c = 0
    resolved_rules = {}

    source_path = rule.source
    target_path = rule.target

    re_pattern = get_array_regex(source_path)

    source_index_positions = [x for x in re.finditer(r'\[(n\d+?)\]', source_path)]
    target_index_positions = [x for x in re.finditer(r'\[(n\d+?)\]', target_path)]
    assert len(source_index_positions) == len(target_index_positions)
    source_path_sections = [i.span() for i in source_index_positions]
    target_path_sections = [i.span() for i in target_index_positions]

    for i in b_data.keypaths(indexes=True, sort=True):
        match = re.match(re_pattern, i)
        if match:
            new_source_path = get_new_path(match, source_path, source_path_sections)
            new_target_path = get_new_path(match, target_path, target_path_sections)
            resolved_rules[f'{name}_resolved_{c}'] = Rule(
                source=new_source_path, target=new_target_path
            )
            c += 1
    return {name: Rules(name=name, rules=resolved_rules)}


def resolve_rules(b_data, rules):
    """
    Resolves rules with array indexes of type [n<number>] to apply the transformations to all matching paths in the source data.
    """
    resolved_rules = {}
    if isinstance(rules, dict):
        for rule_name, rule in rules.items():
            r_rules = {
                k: v
                for n, r in resolve_rules(b_data, rule).items()
                for k, v in r.rules.items()
            }
            resolved_rules[rule_name] = Rules(
                name=getattr(rule, 'name', rule_name), rules=r_rules
            )
    elif isinstance(rules, Rules):
        for rule_name, rule in rules.rules.items():
            resolved_rules.update(_resolve_rule(b_data, rule, rule_name))

    elif isinstance(rules, Rule):
        return _resolve_rule(b_data, rules, '')
    return resolved_rules


def del_sources(b_source_data, rules):
    """
    Deletes the source paths present in the rules from the source data.
    """
    if isinstance(rules, dict):
        for rule_name, rule in rules.items():
            b_source_data = del_sources(b_source_data, rule)
    elif isinstance(rules, Rules):
        for rule_name, rule in rules.rules.items():
            b_source_data = delete_section(b_source_data, rule.source)
    elif isinstance(rules, Rule):
        b_source_data = delete_section(b_source_data, rule.source)
    return b_source_data


def update_source_data(
    source_data: dict[str, Any], transformed_data: dict[str, Any], rules=None
):
    """
    Merges the transformed json into the base archive and deletes the source paths present in the rules.
    """
    b_source_data = get_b_data(source_data)
    b_transformed_archive = get_b_data(transformed_data)
    b_source_data.merge(transformed_data)
    if rules is not None:
        b_source_data = del_sources(b_source_data, rules)
    return deref(b_source_data)


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
        self, source_data: dict[str, Any], mapping_name: str
    ) -> dict[str, Any]:
        if isinstance(source_data, list):
            transformed_data = []
            for i, item in enumerate(source_data):
                transformed_data.append(self.transform_inplace(item, mapping_name))
            return transformed_data
        else:
            b_source_data = get_b_data(source_data)
            resolved_rules = resolve_rules(b_source_data, self.mapping_dict)
            self.mapping_dict[f'{mapping_name}_resolved'] = resolved_rules[mapping_name]
            transformed_data = self.transform(
                source_data, f'{mapping_name}_resolved', deepcopy(source_data)
            )
            updated_json = update_source_data(
                source_data,
                transformed_data,
                self.mapping_dict[f'{mapping_name}_resolved'],
            )
            return updated_json
