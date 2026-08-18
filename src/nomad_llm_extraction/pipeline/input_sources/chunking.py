from typing import Any

import tiktoken
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from loguru import logger


class TextChunker:
    """
    A domain-agnostic text chunker utilizing parent-child recursive splitting.
    Useful for any long-form text (papers, transcripts, reports).
    """

    def __init__(
        self,
        parent_chunk_size=4000,
        parent_chunk_overlap=400,
        child_chunk_size=400,
        child_chunk_overlap=50,
    ):
        try:
            self.tokenizer = tiktoken.get_encoding('cl100k_base')
        except Exception:
            logger.warning('cl100k_base tokenizer not found, using p50k_base.')
            self.tokenizer = tiktoken.get_encoding('p50k_base')

        def length_function(text: str) -> int:
            return len(self.tokenizer.encode(text, allowed_special='all'))

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            length_function=length_function,
            separators=['\n\n', '\n', '. ', ' ', ''],
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            length_function=length_function,
            separators=['\n\n', '\n', '. ', ' ', ''],
        )

        logger.info(
            f'TextChunker initialized: parent={parent_chunk_size} tokens, child={child_chunk_size} tokens'
        )

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        section: str = 'unknown',
        chunk_type: str = 'body_text',
        base_index: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Takes a raw string of text and returns parent-child relationship chunks.
        Returns the chunks and the updated index counter.
        """
        chunks = []
        if not text or not text.strip():
            return chunks, base_index

        parent_texts = self.parent_splitter.split_text(text)
        current_index = base_index

        for p_index, parent_text in enumerate(parent_texts):
            parent_id = f'{doc_id}_text_{current_index}_p{p_index}'
            child_texts = self.child_splitter.split_text(parent_text)

            for c_index, child_text in enumerate(child_texts):
                child_id = f'{parent_id}_c{c_index}'
                chunks.append(
                    {
                        'chunk_id': child_id,
                        'paper_id': doc_id,
                        'chunk_type': chunk_type,
                        'section': section,
                        'text': child_text,
                        'metadata': {
                            'parent_id': parent_id,
                            'parent_text': parent_text,
                        },
                    }
                )
            current_index += 1

        return chunks, current_index
