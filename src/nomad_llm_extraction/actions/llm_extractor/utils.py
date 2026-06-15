import re
from typing import Any


def create_extraction_metadata(
    model_name: str, extraction_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Create extraction metadata with the specified model name."""
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
