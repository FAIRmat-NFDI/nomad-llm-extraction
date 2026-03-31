"""(Fetch Papers -> GROBID).

Example Usage:
    from nomad_llm_extraction.pipeline.input_sources.paper_to_text import PDFParser, extract_text_from_pdf
    
    # Method 1: Using the helper function directly
    text = extract_text_from_pdf("path/to/paper.pdf")
    
    # Method 2: Using the class
    parser = PDFParser()
    text = parser.parse(Path("path/to/paper.pdf"))
"""
import logging
import PyPDF2
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class PDFParser:
    """Generalized tool for extracting text from PDF files using PyPDF2."""
    
    def __init__(self):
        logger.info("PDFParser initialized.")

    def parse(self, pdf_path: Path) -> Optional[str]:
        if not pdf_path.exists():
            logger.error(f"PDF file not found at {pdf_path}")
            return None

        try:
            text = ""
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n\n"
            
            if not text.strip():
                logger.warning(f"No text extracted from {pdf_path}.")
                return None
                
            return text
            
        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            return None

def extract_text_from_pdf(pdf_path: str | Path) -> Optional[str]:
    """Helper function to quickly extract text from a physical PDF file."""
    parser = PDFParser()
    return parser.parse(Path(pdf_path))
