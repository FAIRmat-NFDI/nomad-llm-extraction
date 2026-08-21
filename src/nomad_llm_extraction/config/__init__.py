import os
import pathlib
import subprocess
from importlib.resources import as_file, files
from urllib.parse import urlsplit

from loguru import logger

from nomad_llm_extraction.utils.utils import load_yaml_config

try:
    from nomad.config import config

    NOMAD_URL = config.client.url + '/v1/'

except Exception:
    NOMAD_URL = (
        os.environ.get('deployment_url')
        or os.environ.get('NOMAD_URL')
        or 'https://nomad-lab.eu/prod/v1/api/v1/'
    )


def get_repo_metadata():
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT, text=True
        ).strip()

        remote_url = subprocess.check_output(
            ['git', 'config', '--get', 'remote.origin.url'],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()

        if remote_url.startswith('git@github.com:'):
            repo_path = remote_url.removeprefix('git@github.com:')
        else:
            parsed = urlsplit(remote_url)
            if parsed.hostname != 'github.com':
                logger.warning('Origin is not a GitHub remote; omitting commit URL.')
                return commit_hash, None
            repo_path = parsed.path.lstrip('/')

        repo_path = repo_path.removesuffix('.git')
        return commit_hash, f'https://github.com/{repo_path}/tree/{commit_hash}'

    except subprocess.CalledProcessError as e:
        logger.error(
            "Error: Make sure you are in a git repository and 'origin' is set. message: %s",
            e.output,
        )
        return None, None
    except FileNotFoundError:
        logger.error('Git is not installed or not found in the system PATH.')
        return None, None


commit_hash, commit_url = get_repo_metadata()
DEFAULT_EXTRACTION_METADATA = {
    'model_name': 'LLM Extracted',
    'commit_hash': commit_hash,
    'commit_url': commit_url,
}


def _load_default_extraction_config() -> dict:
    # Prefer the packaged resource so wheel installs resolve the path reliably.
    resource = files('nomad_llm_extraction.config').joinpath(
        'action_default_config.yaml'
    )
    try:
        with as_file(resource) as resource_path:
            return load_yaml_config(resource_path.as_posix())
    except Exception:
        pass

    fallback_paths = [
        pathlib.Path.cwd() / 'config/action_default_config.yaml',
        pathlib.Path(__file__).resolve().parents[2]
        / 'config/action_default_config.yaml',
    ]
    for path in fallback_paths:
        if path.exists():
            return load_yaml_config(path.resolve().as_posix())

    raise FileNotFoundError(
        'Could not find action_default_config.yaml in packaged resources or fallback paths.'
    )


DEFAULT_EXTRACTION_ACTION_CONFIG = _load_default_extraction_config()
if DEFAULT_EXTRACTION_ACTION_CONFIG.get('extract_multiple_instances'):
    multi_instance_field = DEFAULT_EXTRACTION_ACTION_CONFIG['schema_config'].get(
        'multi_instance_field'
    )
    DEFAULT_EXTRACTION_ACTION_CONFIG['schema_config']['multi_instance_field'] = (
        multi_instance_field or 'extracted_instances'
    )
