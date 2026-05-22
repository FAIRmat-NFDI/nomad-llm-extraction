"""
Phase 5: Retrieval Pipeline (High Accuracy Upgrade)
"""

import logging
import sqlite3

import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from src.paths import DATA_DIR, DB_PATH, LOGS_DIR

EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'
CROSS_ENCODER_MODEL_NAME = 'BAAI/bge-reranker-v2-m3'

CHROMA_PATH = DATA_DIR / 'chroma_db'
COLLECTION_NAME = 'battery_papers'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'retrieval.log', mode='a'),  # Use LOGS_DIR
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(self):
        logger.info('Initializing Retrieval Pipeline (High Accuracy Mode)...')

        if not CHROMA_PATH.exists():
            raise FileNotFoundError(
                f'ChromaDB path not found: {CHROMA_PATH}. '
                'Did you run 03_embed_and_load.py?'
            )

        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info(f'Loaded query encoder model: {EMBEDDING_MODEL_NAME}')

            logger.info(
                f'Loading high-accuracy reranker: {CROSS_ENCODER_MODEL_NAME}...'
            )
            self.cross_encoder = CrossEncoder(
                CROSS_ENCODER_MODEL_NAME, max_length=512, trust_remote_code=True
            )
            logger.info(f'Loaded cross-encoder model: {CROSS_ENCODER_MODEL_NAME}')
        except Exception as e:
            logger.error(f'Failed to load models: {e}')
            raise

        class LocalEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __init__(self, model):
                self.model = model

            def __call__(self, texts):
                return self.model.encode(
                    texts, convert_to_numpy=True, show_progress_bar=False
                ).tolist()

        local_embed_func = LocalEmbeddingFunction(self.embedding_model)

        try:
            self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME, embedding_function=local_embed_func
            )
            logger.info(f'Connected to ChromaDB collection: {COLLECTION_NAME}')
        except Exception as e:
            logger.error(f'Failed to connect to ChromaDB: {e}')
            raise

        self.field_query_map = {
            'header': {
                'query_prompt': 'The title, authors, and abstract of the paper.',
                'filters': ['header'],
                'boost_sections': [],
            },
            'cell_info': {
                'query_prompt': (
                    "General experiment setup: cell identifier (e.g., 'Cell A', 'NMC811 Sample'), "
                    "cell type (e.g., 'coin cell', 'pouch cell', 'half-cell'), "
                    'N:P ratio or mass ratio of positive to negative electrodes.'
                ),
                'filters': [
                    'methods',
                    'experimental',
                    'materials',
                    'results',
                    'tables',
                    'experimental details',
                    'introduction',
                ],
                'boost_sections': ['methods', 'experimental', 'experimental details'],
            },
            'working_electrode': {
                'query_prompt': (
                    'Working electrode (cathode) properties: '
                    'active material (e.g., NMC811, LFP), synthesis, composition, '
                    'mass loading (mg/cm2), density, porosity, particle size, '
                    'electrode recipe (e.g. 8:1:1), binder, conductive additive (Super P).'
                ),
                'filters': [
                    'methods',
                    'experimental',
                    'materials',
                    'results',
                    'tables',
                    'experimental details',
                    'materials and methods',
                ],
                'boost_sections': [
                    'methods',
                    'experimental',
                    'experimental details',
                    'materials and methods',
                ],
            },
            'counter_electrode': {
                'query_prompt': (
                    'Counter electrode (anode) properties: '
                    'active material (e.g., graphite, silicon, Li metal), '
                    'lithium foil, mass loading, density, porosity.'
                ),
                'filters': [
                    'methods',
                    'experimental',
                    'materials',
                    'results',
                    'tables',
                    'experimental details',
                    'materials and methods',
                ],
                'boost_sections': [
                    'methods',
                    'experimental',
                    'experimental details',
                    'materials and methods',
                ],
            },
            'electrolyte': {
                'query_prompt': (
                    'Electrolyte composition: salt (e.g., LiPF6), '
                    'concentration (e.g., 1M, 1 mol/L), solvents (e.g., EC:DMC), '
                    'and additives (e.g., VC, FEC).'
                ),
                'filters': [
                    'methods',
                    'experimental',
                    'materials',
                    'experimental details',
                ],
                'boost_sections': ['methods', 'experimental', 'experimental details'],
            },
            'separator': {
                'query_prompt': (
                    'Separator properties: material (e.g., Celgard 2400, polypropylene), '
                    'thickness (µm), and porosity.'
                ),
                'filters': [
                    'methods',
                    'experimental',
                    'materials',
                    'experimental details',
                ],
                'boost_sections': ['methods', 'experimental', 'experimental details'],
            },
            'test_conditions': {
                'query_prompt': (
                    'Battery test conditions: temperature (°C or K), '
                    'voltage window (V, e.g. 0.01-3.0 V), C-rate (e.g., 0.1C, 1C), '
                    'current density (mA/g or A/g or mA/cm²), '
                    'formation protocol, and cycling protocol.'
                ),
                'filters': [
                    'methods',
                    'experimental',
                    'results',
                    'figure_caption',
                    'experimental details',
                ],
                'boost_sections': ['methods', 'experimental', 'experimental details'],
            },
            'performance_metrics': {
                'query_prompt': (
                    'Battery performance metrics: specific capacity (mAh/g), '
                    'areal capacity (mAh/cm²), cycle life (% retention over cycles), '
                    'coulombic efficiency (%), rate capability, impedance (Ohm).'
                ),
                'filters': [
                    'results',
                    'discussion',
                    'abstract',
                    'conclusion',
                    'figure_caption',
                    'table_caption',
                ],
                'boost_sections': ['results', 'discussion'],
            },
            'characterization': {
                'query_prompt': (
                    'Physical characterization data: '
                    'XRD peak positions or phase identification, '
                    'post-mortem analysis, SEM, XPS, TEM results.'
                ),
                'filters': [
                    'results',
                    'discussion',
                    'characterization',
                    'methods',
                    'experimental details',
                ],
                'boost_sections': ['results', 'discussion'],
            },
            'fabrication': {
                'query_prompt': (
                    'Electrode fabrication details: '
                    "drying conditions (e.g., '80 °C for 12 h'), "
                    'calendering pressure or final porosity/density.'
                ),
                'filters': [
                    'methods',
                    'experimental',
                    'materials',
                    'fabrication',
                    'experimental details',
                ],
                'boost_sections': [
                    'methods',
                    'experimental',
                    'experimental details',
                    'fabrication',
                ],
            },
        }

    def _apply_rrf(
        self, list_a: list[str], list_b: list[str], k: int = 60
    ) -> list[str]:
        """
        calculate reciprocal rank fusion (RRF) scores to merge two ranked lists.
        Returns a single list of unique chunks sorted by RRF score (highest first).
        """
        rrf_map = {}

        for rank, item in enumerate(list_a):
            if item not in rrf_map:
                rrf_map[item] = 0
            rrf_map[item] += 1 / (k + rank)
        for rank, item in enumerate(list_b):
            if item not in rrf_map:
                rrf_map[item] = 0
            rrf_map[item] += 1 / (k + rank)
        sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items]

    def retrieve_chunks_for_paper(
        self, paper_id: str, top_k: int = 20
    ) -> dict[str, list[str]]:
        logger.info(f'Retrieving chunks for paper_id: {paper_id}')
        retrieved_data: dict[str, list[str]] = {}

        try:
            paper_chunks_result = self.collection.get(
                where={'paper_id': paper_id}, include=['documents', 'metadatas']
            )
            all_paper_chunks = paper_chunks_result['documents']
            all_metadatas = paper_chunks_result['metadatas']

            if not all_paper_chunks:
                logger.warning(f'No chunks found for paper {paper_id}')
                return {}

            chunk_to_metadata = {
                chunk: meta for chunk, meta in zip(all_paper_chunks, all_metadatas)
            }

            tokenized_corpus = [chunk.split(' ') for chunk in all_paper_chunks]
            bm25 = BM25Okapi(tokenized_corpus)

        except Exception as e:
            logger.error(f'Error preparing BM25 for {paper_id}: {e}')
            return {}

        for field_name, config in self.field_query_map.items():
            query_prompt = config['query_prompt']
            boost_sections = config.get('boost_sections', [])
            candidate_k = top_k * 5

            where_filter = {'paper_id': {'$eq': paper_id}}

            try:
                vector_results = self.collection.query(
                    query_texts=[query_prompt],
                    n_results=candidate_k,
                    where=where_filter,
                    include=['documents', 'metadatas'],
                )

                vector_chunks = (
                    vector_results['documents'][0]
                    if vector_results['documents']
                    else []
                )
                vector_metas = (
                    vector_results['metadatas'][0]
                    if vector_results['metadatas']
                    else []
                )

                if not vector_chunks:
                    retrieved_data[field_name] = []
                    continue

                tokenized_query = query_prompt.split(' ')
                bm25_top_n = bm25.get_top_n(
                    tokenized_query, all_paper_chunks, n=candidate_k
                )

                unique_candidates = self._apply_rrf(vector_chunks, bm25_top_n)

                if not unique_candidates:
                    retrieved_data[field_name] = []
                    continue

                pairs = [[query_prompt, doc] for doc in unique_candidates]
                scores = self.cross_encoder.predict(pairs)

                boosted_scores = []
                for i, (chunk, score) in enumerate(zip(unique_candidates, scores)):
                    metadata = chunk_to_metadata.get(chunk, {})
                    chunk_section = metadata.get('section', '').lower()

                    if boost_sections and chunk_section in [
                        s.lower() for s in boost_sections
                    ]:
                        boosted_score = score * 1.2
                    else:
                        boosted_score = score

                    boosted_scores.append(boosted_score)

                sorted_indices = np.argsort(boosted_scores)[::-1]
                top_indices = sorted_indices[:top_k]

                parent_ids_seen = set()
                final_parent_texts = []

                for idx in top_indices:
                    child_text = unique_candidates[idx]
                    child_meta = chunk_to_metadata.get(child_text, {})

                    parent_id = child_meta.get('parent_id', '')
                    parent_text = child_meta.get('parent_text', '')

                    if parent_id and parent_text:
                        if parent_id not in parent_ids_seen:
                            final_parent_texts.append(parent_text)
                            parent_ids_seen.add(parent_id)
                    else:
                        final_parent_texts.append(child_text)

                    if len(final_parent_texts) >= top_k:
                        break

                retrieved_data[field_name] = final_parent_texts
                logger.info(
                    f"Field '{field_name}': Retrieved {len(final_parent_texts)} parent chunks (from {len(top_indices)} children)"
                )

            except Exception as e:
                logger.error(f"Error querying/ranking for field '{field_name}': {e}")
                retrieved_data[field_name] = []

        return retrieved_data

    @staticmethod
    def format_context_for_llm(retrieved_data: dict[str, list[str]]) -> str:
        context_str = ''
        for field_name, chunks in retrieved_data.items():
            context_str += f'--- Context for {field_name.replace("_", " ")} ---\n'
            if not chunks:
                context_str += '[No relevant chunks found]\n\n'
                continue

            unique_chunks = list(dict.fromkeys(chunks))

            if len(unique_chunks) == 1:
                context_str += f'[Chunk 1]: {unique_chunks[0]}\n'
            elif len(unique_chunks) == 2:
                context_str += f'[Chunk 1]: {unique_chunks[0]}\n'
                context_str += f'[Chunk 2]: {unique_chunks[1]}\n'
            else:
                context_str += f'[Chunk 1]: {unique_chunks[0]}\n'
                for i in range(2, len(unique_chunks)):
                    context_str += f'[Chunk {i}]: {unique_chunks[i - 1]}\n'
                context_str += f'[Chunk {len(unique_chunks)}]: {unique_chunks[-1]}\n'

            context_str += '\n'
        return context_str


def main():
    logger.info('--- Starting Phase 5 Retrieval Pipeline Test ---')
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT paper_id, title FROM papers 
            WHERE parsing_status = 'success_grobid' OR 
                  parsing_status = 'success_chunked' OR
                  parsing_status = 'success_nougat'
            LIMIT 5
        """)
        papers_to_process = cursor.fetchall()
        conn.close()
    except sqlite3.Error as e:
        logger.error(f'Failed to read from SQLite DB: {e}')
        return

    pipeline = RetrievalPipeline()
    for paper_id, title in papers_to_process:
        print('\n' + '=' * 80)
        logger.info(f'Processing Paper: {paper_id} ({title[:60]}...)')
        retrieved_data = pipeline.retrieve_chunks_for_paper(paper_id, top_k=3)
        llm_context_string = pipeline.format_context_for_llm(retrieved_data)
        print(llm_context_string)


if __name__ == '__main__':
    main()
