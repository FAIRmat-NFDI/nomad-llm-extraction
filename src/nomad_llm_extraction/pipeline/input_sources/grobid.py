import re
from typing import Any

import requests
from bs4 import BeautifulSoup

# from src.core.chunking import TextChunker
from loguru import logger


class GrobidClient:
    # It defaults to 8070, but can be overridden when initialized
    def __init__(self, base_url: str = 'http://localhost:8070'):
        self.base_url = base_url.rstrip('/')
        self.api_url = f'{self.base_url}/api/processFulltextDocument'
        self.health_url = f'{self.base_url}/api/isalive'
        logger.info(f'GROBID client configured for: {self.base_url}')

    def check_server_health(self) -> bool:
        try:
            response = requests.get(self.health_url, timeout=5)
            if response.status_code == 200:
                return True
            logger.warning(f'GROBID server check failed: {response.status_code}')
            return False
        except requests.exceptions.ConnectionError:
            logger.error('Could not connect to GROBID server.')
            return False

    def parse_pdf(self, paper_id: str, pdf_path: Path, output_dir: Path) -> str | None:
        if not pdf_path.exists():
            logger.error(f'PDF file not found for {paper_id} at {pdf_path}')
            return None

        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': (pdf_path.name, f, 'application/pdf')}
                params = {
                    'consolidateHeader': '0',
                    'consolidateCitations': '0',
                    'includeRawCitations': '0',
                    'includeRawAffiliations': '0',
                    'teiCoordinates': '0',
                }

                response = requests.post(
                    self.api_url, files=files, data=params, timeout=240
                )
                response.raise_for_status()

                xml_output = response.text
                output_dir.mkdir(parents=True, exist_ok=True)
                xml_save_path = output_dir / f'{paper_id}.grobid.xml'

                with open(xml_save_path, 'w', encoding='utf-8') as xf:
                    xf.write(xml_output)

                return xml_output

        except requests.exceptions.RequestException as e:
            logger.error(f'GROBID request failed for {paper_id}: {e}')
            return None


class GrobidXMLProcessor:
    """Parses TEI XML from GROBID and utilizes a TextChunker to structure the output."""

    def __init__(self, chunker: TextChunker):
        self.chunker = chunker

    def _clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    def process_xml(self, paper_id: str, grobid_xml: str) -> list[dict[str, Any]]:
        chunks = []
        try:
            soup = BeautifulSoup(grobid_xml, 'lxml-xml')

            # extract Header
            title = (
                self._clean_text(soup.find('titleStmt').find('title').get_text())
                if soup.find('titleStmt') and soup.find('titleStmt').find('title')
                else 'Unknown Title'
            )
            authors = [
                self._clean_text(auth.get_text()) for auth in soup.find_all('author')
            ]
            abstract_elem = soup.find('abstract')
            abstract = (
                self._clean_text(abstract_elem.get_text()) if abstract_elem else ''
            )

            header_text = f'Title: {title}\nAuthors: {"; ".join(authors)}\n\nAbstract:\n{abstract}'
            chunks.append(
                {
                    'chunk_id': f'{paper_id}_header',
                    'paper_id': paper_id,
                    'chunk_type': 'header',
                    'section': 'header',
                    'text': header_text,
                }
            )

            # extract Body using TextChunker
            body = soup.find('body')
            text_chunk_counter = 0

            if body:
                current_section = 'introduction'
                for div in body.find_all('div', recursive=False):
                    head = div.find('head')
                    if head:
                        raw_title = self._clean_text(head.get_text())
                        current_section = (
                            re.sub(r'^\d+(\.\d+)*\s*', '', raw_title).strip().lower()
                        )

                    section_text_parts = [
                        self._clean_text(p.get_text()) for p in div.find_all('p')
                    ]
                    section_text = '\n'.join(section_text_parts)

                    if section_text:
                        new_chunks, text_chunk_counter = self.chunker.chunk_text(
                            text=section_text,
                            doc_id=paper_id,
                            section=current_section,
                            chunk_type='body_text',
                            base_index=text_chunk_counter,
                        )
                        chunks.extend(new_chunks)

            # extract Tables and Figures
            fig_counter, table_counter = 0, 0
            for fig in soup.find_all('figure'):
                head = (
                    self._clean_text(fig.find('head').get_text())
                    if fig.find('head')
                    else ''
                )
                label = (
                    self._clean_text(fig.find('label').get_text())
                    if fig.find('label')
                    else ''
                )
                desc = (
                    self._clean_text(fig.find('figDesc').get_text())
                    if fig.find('figDesc')
                    else ''
                )

                if fig.get('type') == 'table':
                    table_elem = fig.find('table')
                    caption = f'{label}: {head}\n{desc}'.strip()

                    if table_elem:
                        table_rows = []
                        for row in table_elem.find_all('row'):
                            cells = [
                                self._clean_text(cell.get_text())
                                for cell in row.find_all('cell')
                            ]
                            if cells:
                                table_rows.append(' | '.join(cells))

                        table_content = (
                            f'{caption}\n\nTable Data:\n' + '\n'.join(table_rows)
                            if table_rows
                            else caption
                        )
                        if table_content.strip():
                            chunks.append(
                                {
                                    'chunk_id': f'{paper_id}_table_full_{table_counter}',
                                    'paper_id': paper_id,
                                    'chunk_type': 'table_full',
                                    'section': 'tables',
                                    'text': table_content,
                                }
                            )
                            table_counter += 1
                    elif caption:
                        chunks.append(
                            {
                                'chunk_id': f'{paper_id}_table_caption_{table_counter}',
                                'paper_id': paper_id,
                                'chunk_type': 'table_caption',
                                'section': 'tables',
                                'text': caption,
                            }
                        )
                        table_counter += 1
                else:
                    fig_text = f'{label}: {head}\n{desc}'.strip()
                    if fig_text:
                        chunks.append(
                            {
                                'chunk_id': f'{paper_id}_figure_{fig_counter}',
                                'paper_id': paper_id,
                                'chunk_type': 'figure_caption',
                                'section': 'figures',
                                'text': fig_text,
                            }
                        )
                        fig_counter += 1

            return chunks

        except Exception as e:
            logger.error(f'Failed to process GROBID XML for {paper_id}: {e}')
            return []
