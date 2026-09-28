import sys
import types

import pytest

from nomad_llm_extraction.actions.llm_extractor.activities import _get_upload_files


@pytest.mark.parametrize('module_name', ['nomad.uploads', 'nomad.actions.manager'])
def test_get_upload_files_is_found_in_either_module(monkeypatch, module_name):
    """nomad-lab moved get_upload_files from nomad.actions.manager to
    nomad.uploads; an empty `nomad.uploads` stands for the older nomad-lab."""
    for name in ('nomad.uploads', 'nomad.actions.manager'):
        module = types.ModuleType(name)
        if name == module_name:
            module.get_upload_files = lambda upload_id, user_id: (upload_id, user_id)
        monkeypatch.setitem(sys.modules, name, module)

    assert _get_upload_files('up', 'u') == ('up', 'u')
