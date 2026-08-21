import os
import pathlib
import urllib.parse
from importlib.resources import as_file, files

from loguru import logger

from nomad_llm_extraction.utils.utils import get_repo_metadata, load_yaml_config

try:
    from nomad.config import config

    NOMAD_URL = urllib.parse.urljoin(
        config.client.url,
        'v1/',
    )

except Exception:
    NOMAD_URL = (
        os.environ.get('deployment_url')
        or os.environ.get('NOMAD_URL')
        or 'https://nomad-lab.eu/prod/v1/api/v1/'
    )

logger.info(f'Using NOMAD_URL: {NOMAD_URL}')
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
