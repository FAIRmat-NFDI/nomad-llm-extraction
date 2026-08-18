"""
Phase 3: Embed and Load (Smart Batching)
Separates massive chunks to process them singly at the end, preventing RAM explosions.
"""

import gc
import json
import logging
import os
import sqlite3
from pathlib import Path

import chromadb
import torch
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from src.paths import CHUNKS_DIR, DATA_DIR, DB_PATH, LOGS_DIR

NORMAL_BATCH_SIZE = 32
OVERSIZED_THRESHOLD = 3000
MAX_SEQ_LENGTH = 8192
SAFE_TOKEN_LIMIT = 2048
OVERLAP = 512

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOGS_DIR / 'embedding.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'

try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f'Using GPU: {gpu_name}')

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    embedding_model.max_seq_length = MAX_SEQ_LENGTH

    logger.info(f"Loaded model '{EMBEDDING_MODEL_NAME}' on {device}")
    EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

    class LocalEmbeddingFunction(embedding_functions.EmbeddingFunction):
        def __call__(self, texts):
            return embedding_model.encode(
                texts,
                batch_size=NORMAL_BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()

    local_embed_func = LocalEmbeddingFunction()

except Exception as e:
    logger.error(f'Failed to load model: {e}')
    exit(1)


class VectorDBLoader:
    def __init__(self, db_path, collection_name, embedding_dim, model_instance):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.model = model_instance

        # Creates data/chroma_db
        self.db_dir = self.db_path / 'chroma_db'
        self.db_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(path=str(self.db_dir))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=local_embed_func,
                metadata={'hnsw:space': 'cosine'},
            )
        except Exception as e:
            logger.error(f'Failed to initialize ChromaDB: {e}')
            exit(1)

    def load_chunks_to_db(self, chunk_file_path: Path):
        """
        Loads chunks. If a chunk is > 8192 tokens, it splits it into smaller parts.
        """
        try:
            with open(chunk_file_path, encoding='utf-8') as f:
                chunks = [json.loads(line) for line in f]

            if not chunks:
                return 0

            documents = []
            metadatas = []
            ids = []

            for chunk in chunks:
                text_content = chunk['text']
                base_id = f'{chunk["paper_id"]}_{chunk["chunk_id"]}'
                token_ids = self.model.tokenizer.encode(
                    text_content, add_special_tokens=False
                )
                total_tokens = len(token_ids)

                meta_payload = chunk.get('metadata', {})
                parent_text = meta_payload.get('parent_text', '')
                parent_id = meta_payload.get('parent_id', '')

                if total_tokens <= SAFE_TOKEN_LIMIT:
                    documents.append(text_content)
                    metadatas.append(
                        {
                            'paper_id': chunk['paper_id'],
                            'chunk_id': chunk['chunk_id'],
                            'section': chunk.get('section', 'unknown'),
                            'token_count': total_tokens,
                            'split_part': 0,
                            'parent_text': parent_text,
                            'parent_id': parent_id,
                        }
                    )
                    ids.append(base_id)

                else:
                    logger.info(
                        f'✂️ Splitting massive chunk {base_id} ({total_tokens} tokens) into pieces...'
                    )

                    part_counter = 0
                    start = 0
                    while start < total_tokens:
                        end = min(start + SAFE_TOKEN_LIMIT, total_tokens)
                        chunk_token_ids = token_ids[start:end]
                        sub_text = self.model.tokenizer.decode(
                            chunk_token_ids, skip_special_tokens=True
                        )
                        documents.append(sub_text)
                        metadatas.append(
                            {
                                'paper_id': chunk['paper_id'],
                                'chunk_id': chunk['chunk_id'],
                                'section': chunk.get('section', 'unknown'),
                                'token_count': len(chunk_token_ids),
                                'split_part': part_counter + 1,
                                'parent_text': parent_text,
                                'parent_id': parent_id,
                            }
                        )
                        ids.append(f'{base_id}_part{part_counter}')
                        start += SAFE_TOKEN_LIMIT - OVERLAP
                        part_counter += 1

            if documents:
                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

            count = len(documents)
            del documents, metadatas, ids, chunks
            import gc

            gc.collect()

            logger.info(
                f'Processed {chunk_file_path.name} -> Generated {count} vectors.'
            )
            return 1

        except Exception as e:
            logger.error(f'Error loading {chunk_file_path.name}: {e}')
            return 0


def main():
    logger.info('Starting Phase 3: Embed and Load (Smart Batching)...')

    papers_processed = 0
    COLLECTION_NAME = 'battery_papers'

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    vector_loader = VectorDBLoader(
        db_path=DATA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_dim=EMBEDDING_DIM,
        model_instance=embedding_model,
    )

    cursor.execute(
        "SELECT paper_id FROM papers WHERE parsing_status IN ('success_grobid', 'success_chunked', 'success_nougat')"
    )
    chunked_papers = [row[0] for row in cursor.fetchall()]

    try:
        result = vector_loader.collection.get(include=['metadatas'])

        if result and result.get('metadatas'):
            existing_paper_ids = set([m['paper_id'] for m in result['metadatas'] if m])
        else:
            existing_paper_ids = set()

        logger.info(f'Skipping {len(existing_paper_ids)} papers already in DB.')
    except Exception as e:
        logger.warning(f'Could not retrieve existing papers to skip: {e}')
        existing_paper_ids = set()

    count = 0
    for paper_id in chunked_papers:
        if paper_id in existing_paper_ids:
            continue

        chunk_file = CHUNKS_DIR / f'{paper_id}_chunks.jsonl'
        if chunk_file.exists():
            vector_loader.load_chunks_to_db(chunk_file)
            count += 1
            papers_processed += 1

    logger.info(f'Phase 3 Complete. Processed {count} new files.')
    conn.close()

    return {
        'embedding_model': EMBEDDING_MODEL_NAME,
        'papers_embedded': papers_processed,
    }


if __name__ == '__main__':
    main()
