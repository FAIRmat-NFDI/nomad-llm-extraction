import re
from copy import deepcopy
from typing import Any

from nomad_llm_extraction.actions.llm_extractor.models import ExtractionActionInput
from nomad_llm_extraction.config import (
    DEFAULT_EXTRACTION_ACTION_CONFIG,
    DEFAULT_EXTRACTION_METADATA,
)


def create_extraction_config(data: ExtractionActionInput):
    extraction_config = deepcopy(DEFAULT_EXTRACTION_ACTION_CONFIG)
    if data.extract_multiple_instances:
        multi_instance_field = 'extracted_instances'
    else:
        multi_instance_field = None
    extraction_config['schema_config'].update(
        {
            'm_def': data.extraction_m_def,
            'multi_instance_field': multi_instance_field,
        }
    )
    extraction_config['llm_engine_config'].update(
        {
            'api_key': data.api_token,
            'model_name': data.model_technical_name,
            'api_url': None if not data.api_base_url else data.api_base_url,
        }
    )
    extraction_config['extraction_metadata'].update({'model_name': data.model})
    if data.api_base_url:
        extraction_config['extraction_metadata']['api_url'] = data.api_base_url
    for key, value in DEFAULT_EXTRACTION_METADATA.items():
        extraction_config['extraction_metadata'].setdefault(key, value)
    return extraction_config


def create_extraction_metadata(
    model_name: str, extraction_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Create extraction metadata with the specified model name."""
    extraction_metadata = (
        extraction_metadata.copy()
    )  # Create a copy to avoid mutating the original
    for key, value in DEFAULT_EXTRACTION_METADATA.items():
        extraction_metadata.setdefault(key, value)
    extraction_metadata.update({'model_name': model_name})
    return extraction_metadata


def extract_doi(doi: str) -> str | None:
    """
    Extracts the DOI prefix and suffix (10.xxxx/xxxx) from a DOI string.
    Returns None if no valid DOI is found.
    """
    match = re.search(r'10\.\d{4,9}/[-._;()/:\w\[\]]+', doi, re.IGNORECASE)
    if match:
        return match.group(0)
    return None
