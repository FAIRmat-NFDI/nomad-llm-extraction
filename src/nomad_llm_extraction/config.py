from nomad_llm_extraction.utils.utils import load_yaml_config
import pathlib
DEFAULT_EXTRACTION_METADATA = {
    'model_name': 'LLM Extracted',
}

DEFAULT_EXTRACTION_CONFIG = load_yaml_config(pathlib.Path('packages/nomad-llm-extraction/config/action_default_config.yaml').resolve().as_posix())