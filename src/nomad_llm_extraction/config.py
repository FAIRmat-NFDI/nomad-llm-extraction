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
if DEFAULT_EXTRACTION_CONFIG.get('extract_multiple_instances'):
    DEFAULT_EXTRACTION_CONFIG['schema_config']['multi_instance_field'] = (
        ('extracted_instances')
        if DEFAULT_EXTRACTION_CONFIG['schema_config'].get('multi_instance_field')
        is not None
        else None
    )
