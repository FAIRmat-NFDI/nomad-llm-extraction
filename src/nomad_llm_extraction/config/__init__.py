import pathlib
from importlib.resources import as_file, files

from nomad_llm_extraction.utils.utils import get_repo_metadata, load_yaml_config

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
    DEFAULT_EXTRACTION_ACTION_CONFIG['schema_config']['multi_instance_field'] = (
        ('extracted_instances')
        if DEFAULT_EXTRACTION_ACTION_CONFIG['schema_config'].get('multi_instance_field')
        is not None
        else None
    )
