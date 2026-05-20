"""(Fetch Papers -> GROBID).

Example Usage:
    from nomad_llm_extraction.pipeline.input_sources.paper_to_text import pypdf_parse

    text = pypdf_parse("path/to/paper.pdf")

    text = grobid_parse("path/to/paper.pdf", api_url="http://localhost:8080/api")
"""

import logging
from pathlib import Path

import PyPDF2

logger = logging.getLogger(__name__)


def pypdf_parse(pdf_path: str | Path) -> str | None:
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        logger.error(f'PDF file not found at {path_obj}')
        return None

    try:
        text = ''
        with open(path_obj, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + '\n\n'

        if not text.strip():
            logger.warning(f'No text extracted from {path_obj}.')
            return None

        return text

    except Exception as e:
        logger.error(f'PDF extraction failed for {path_obj}: {e}')
        return None


# def grobid_parse(pdf_path: Union[str, Path], api_url: str = "http://localhost:8080/api") -> Optional[str]:
#     pass
