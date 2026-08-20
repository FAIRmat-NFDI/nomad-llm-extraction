"""
===================
Compares core extraction engines side-by-side using pure Pydantic and pure JSON.

Engines Tested:
  1. outlines_json       : Core Outlines + simple_schema.json
  2. outlines_pydantic   : Core Outlines + simple_schema.py (BatteryData)
  3. instructor_pydantic : Core Instructor + simple_schema.py (BatteryData)

Usage:
    python scripts/benchmark_recall.py [--no-docker]
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

sys.set_int_max_str_digits(100000)  # sometimes outlines run in infinite loop
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))



NUM_PAPERS = 5
TOP_K_CHUNKS = 25
TEMPERATURE = 0.1




def load_json_schema(schema_name: str) -> dict:
    path = SCHEMA_DIR / schema_name
    if not path.exists():
        raise FileNotFoundError(f'Schema not found: {path}')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


SYSTEM_PROMPT = (
    'You are an expert scientific data extractor. Extract structured information from research papers.\n'
    '1. Only extract explicitly mentioned text.\n'
    '2. Identify associated units for numbers.\n'
    '3. Set missing values to null.\n'
)
INSTRUCTION_TEXT = 'Extract each experiment.'


def build_prompt(text_context: str) -> str:
    return f"""{SYSTEM_PROMPT}\n{INSTRUCTION_TEXT}\nTEXT:\n{text_context}"""


def run_extraction(
    engine,
    # retrieval: RetrievalPipeline,
    paper_id: str,
    battery_schema: dict | type[BaseModel],
    identifiers_schema: dict | type[BaseModel],
) -> dict | None:

    # retrieved = retrieval.retrieve_chunks_for_paper(paper_id, top_k=TOP_K_CHUNKS)
    # all_chunks = [c for chunks in retrieved.values() for c in chunks]
    if not all_chunks:
        return None
    full_context = '\n\n---\n\n'.join(list(dict.fromkeys(all_chunks)))

    id_prompt = f'List unique identifiers for every battery cell described.\nTEXT:\n{full_context[:25000]}'
    try:
        id_result = engine.generate(
            id_prompt, identifiers_schema, temperature=TEMPERATURE
        )
        cell_ids = id_result.get('identifiers', ['default']) or ['default']
    except Exception:
        cell_ids = ['default']

    all_experiments, confidence_scores = [], []
    for cell_id in cell_ids:
        ctx = f'Extract ONLY for: "{cell_id}"\n---\n{full_context}'
        try:
            result = engine.generate(
                build_prompt(ctx), battery_schema, temperature=TEMPERATURE
            )
            exps = result.get('experiments', [])
            for exp in exps:
                exp['cell_identifier'] = cell_id
            all_experiments.extend(exps)
            confidence_scores.append(result.get('confidence_score', 0.5))
        except Exception as e:
            logger.warning(f"  Extract failed for '{cell_id}': {e}")

    if not all_experiments:
        return None
    return {
        'experiments': all_experiments,
        'confidence_score': sum(confidence_scores) / len(confidence_scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-docker', action='store_true', help='Skip Docker start')
    args = parser.parse_args()

    if args.no_docker and not server_is_up():
        logger.error('Server not reachable. Start it first.')
        sys.exit(1)
    elif not args.no_docker:
        if not server_is_up():
            logger.info(f'🚀 Starting fresh vLLM instance for {MODEL_NAME}...')
            if not start_vllm_container():
                sys.exit(1)
        else:
            logger.info('Re-initializing container to ensure correct model...')
            if not start_vllm_container():
                sys.exit(1)

    battery_schema_json = load_json_schema('simple_schema.json')
    identifiers_schema_json = load_json_schema('identifiers.json')

    logger.info('🔧  Initialising Core Engines…')
    outlines_engine = OutlinesEngine(api_url=VLLM_API_URL, model_name=MODEL_NAME)
    instructor_engine = InstructorEngine(api_url=VLLM_API_URL, model_name=MODEL_NAME)
    retrieval = RetrievalPipeline()

    result = run_extraction(
        engine, retrieval, paper_id, target_schema, target_id_schema
    )
    elapsed = time.time() - t0

    with open(
        REPORT_DIR / f'{config_name}_{paper_id}.json', 'w', encoding='utf-8'
    ) as f:
        json.dump(result or {}, f, indent=2)

    fill_rate, filled, total = field_fill_rate(result)
    recall, matched, gold_total = gold_standard_recall(result, gold_data or {})

    row = results_by_paper[paper_id]
    row[f'{config_name}_fill_rate'] = fill_rate
    row[f'{config_name}_recall'] = recall
    logger.info(f'     fill={fill_rate:.1%} recall={recall:.1%} time={elapsed:.0f}s')

    csv_rows = list(results_by_paper.values())
    fieldnames = list(csv_rows[0].keys())
    with open(
        REPORT_DIR / 'recall_comparison.csv', 'w', newline='', encoding='utf-8'
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print('\n╔══════════════════════════════════════════════════════════════════╗')
    print('║                        OVERALL SUMMARY                          ║')
    print('╠══════════════════════════════════════════════════════════════════╣')
    for cfg_name, _, _, _ in pipelines_to_test:
        avg_fill = sum(r[f'{cfg_name}_fill_rate'] for r in csv_rows) / len(csv_rows)
        avg_rec = sum(r[f'{cfg_name}_recall'] for r in csv_rows) / len(csv_rows)
        print(
            f'║  {cfg_name:<20} fill={avg_fill:.1%}  recall={avg_rec:.1%}             ║'
        )
    print('╚══════════════════════════════════════════════════════════════════╝')


if __name__ == '__main__':
    main()
