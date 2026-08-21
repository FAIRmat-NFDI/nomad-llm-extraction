from subprocess import CalledProcessError

import pytest

from nomad_llm_extraction.config import get_repo_metadata, subprocess


@pytest.mark.parametrize(
    ('remote_url', 'expected_url'),
    [
        (
            'git@github.com:owner/repository.git',
            'https://github.com/owner/repository/tree/abc123',
        ),
        (
            'https://github.com/owner/repository.git',
            'https://github.com/owner/repository/tree/abc123',
        ),
        ('https://gitlab.com/owner/repository.git', None),
    ],
)
def test_get_repo_metadata_parses_github_remotes(monkeypatch, remote_url, expected_url):
    responses = iter(['abc123\n', f'{remote_url}\n'])
    monkeypatch.setattr(
        subprocess, 'check_output', lambda *args, **kwargs: next(responses)
    )

    assert get_repo_metadata() == ('abc123', expected_url)


@pytest.mark.parametrize(
    'error', [CalledProcessError(1, 'git', output='no remote'), FileNotFoundError()]
)
def test_get_repo_metadata_returns_none_when_git_is_unavailable(monkeypatch, error):
    def raise_error(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, 'check_output', raise_error)

    assert get_repo_metadata() == (None, None)
