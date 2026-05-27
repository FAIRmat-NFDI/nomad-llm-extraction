"""(Fetch Papers -> GROBID).

Example Usage:
    from nomad_llm_extraction.pipeline.input_sources.paper_to_text import pypdf_parse

    text = pypdf_parse("path/to/paper.pdf")

    text = grobid_parse("path/to/paper.pdf", api_url="http://localhost:8080/api")
"""

import logging
from pathlib import Path

import pymupdf
import PyPDF2
from diskcache import Cache

from nomad_llm_extraction.utils.utils import get_hash

logger = logging.getLogger(__name__)


def pypdf_parse(pdf_path: str | Path) -> str | None:
    path_obj = Path(pdf_path)
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


def pymupdf_parse(pdf_path: str | Path) -> str:
    doc = pymupdf.open(pdf_path)
    text = ''
    for page in doc:
        text += page.get_text() + '\n\n'
    return text


parsers = {
    'pypdf': pypdf_parse,
    'pymupdf': pymupdf_parse,
}


class PDFParser:
    """light weight alternative for grobid"""

    def __init__(
        self,
        parse_method: str = 'pypdf',
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
    ):
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.cache = None
        if parse_method not in parsers:
            raise ValueError(
                f'Unsupported parse method: {parse_method}. Supported methods are: {list(parsers.keys())}'
            )
        self.parser = parsers[parse_method]
        if self.use_cache:
            self.cache = (
                Cache(self.cache_dir)
                if self.cache_dir
                else Cache(Path.home() / '.pdf_parser_cache')
            )
        logger.info(f'PDFParser initialized with method: {parse_method}')

    def parse_pdf(self, pdf_path: str | Path) -> str | None:
        if self.use_cache and self.cache is not None:
            filehash = get_hash(pdf_path)
            if cached := self.cache.get(str(filehash)):
                return cached
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            logger.error(f'PDF file not found at {path_obj}')
            return None

        try:
            text = self.parser(path_obj)
            if text is None or not text.strip():
                logger.warning(
                    f'No text extracted from {path_obj} using {self.parser.__name__}.'
                )
                return None
            if self.use_cache and self.cache is not None:
                self.cache.set(str(filehash), text)
            return text

        except Exception as e:
            logger.error(f'PDFParser failed for {path_obj}: {e}')
            return None


def parse_text_from_pdf(pdf_path: str | Path, method: str = 'pypdf') -> str | None:
    parser = PDFParser(parse_method=method)
    return parser.parse_pdf(pdf_path)
