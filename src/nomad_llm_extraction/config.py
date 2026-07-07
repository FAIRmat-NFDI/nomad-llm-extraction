import pathlib

from nomad_llm_extraction.utils.utils import load_yaml_config

DEFAULT_EXTRACTION_METADATA = {
    'model_name': 'LLM Extracted',
}
try:
    DEFAULT_EXTRACTION_CONFIG = load_yaml_config(
        pathlib.Path('packages/nomad-llm-extraction/config/action_default_config.yaml')
        .resolve()
        .as_posix()
    )
except Exception:
    DEFAULT_EXTRACTION_CONFIG = load_yaml_config(
        pathlib.Path('config/action_default_config.yaml').resolve().as_posix()
    )
