import json
import pickle
from dataclasses import dataclass
from traceback import format_exc
from typing import Any

from schemas import BatteryData, ExperimentIdentifiers

from nomad_llm_extraction.pipeline import InlineSchemaSource, Pipeline, PromptConfig
from nomad_llm_extraction.pipeline.models import StageContext
from nomad_llm_extraction.pipeline.schema_filling.llm_engine import LiteLLMEngine
from nomad_llm_extraction.pipeline.stages import (
    LLMCallStage,
    ParseResponseStage,
    PromptBuildStage,
)

SYSTEM_PROMPT = (
    'You are an expert scientific data extractor. Extract structured information from research papers.\n'
    '1. Only extract explicitly mentioned text.\n'
    '2. Identify associated units for numbers.\n'
    '3. Set missing values to null.\n'
)

INSTRUCTION_TEXT = 'Extract each experiment.'


def build_prompt(text_context: str) -> str:
    return f"""{SYSTEM_PROMPT}\nExtract each experiment.\nTEXT:\n{text_context}"""


@dataclass
class IndentifiersStageContext:
    text: str
    extraction_schema: dict[str, Any] | None = None
    postprocessing_schema: dict[str, Any] | None = None
    prompt: str | None = None
    raw_output: str | None = None
    extracted_data: Any = None


if __name__ == '__main__':
    try:
        all_results = []
        with open('text.txt') as f:
            text = f.read()
        text = 'Cell 1 - active material: LiNi0.8Co0.1Mn0.1O2, mass: 10 mg, thickness: 50 um; Cell 2 - active material: LiFePO4, mass: 5 mg, thickness: 30 um. \n Cell 3 - active material: LiCoO2, mass: 8 mg, thickness: 40 um.'
        battery_schema = BatteryData.model_json_schema()
        identifiers_schema = ExperimentIdentifiers.model_json_schema()
        extraction_schema = InlineSchemaSource(
            battery_schema,
            remove_defs=True,
            resolve_allOf=True,
            remove_null_anyof=True,
        ).get_schema()
        engine = LiteLLMEngine(model_name='claude-4-sonnet-20250514')
        id_prompt = (
            f'List unique identifiers for every battery cell described.\nTEXT:\n{text}'
        )
        extraction_prompt_config = PromptConfig(
            system_prompt=SYSTEM_PROMPT,
            instruction_text=INSTRUCTION_TEXT,
        )
        identifier_prompt_config = PromptConfig(
            system_prompt='List unique identifiers for every battery cell described.',
            instruction_text='',
        )
        stages = [
            PromptBuildStage(identifier_prompt_config),
            LLMCallStage(engine),
            ParseResponseStage(),
        ]
        identifier_pipeline = Pipeline(
            stages=stages, ctx_factory=IndentifiersStageContext
        )
        id_result = identifier_pipeline.run(
            IndentifiersStageContext(text=text, extraction_schema=identifiers_schema)
        )
        all_results.append(id_result)
        cell_ids = id_result.ctx['extracted_data'].get('identifiers', ['default']) or [
            'default'
        ]
        extraction_stages = [
            PromptBuildStage(extraction_prompt_config),
            LLMCallStage(engine),
            ParseResponseStage(),
        ]
        extraction_pipeline = Pipeline(stages=extraction_stages)
        all_experiments = {'identifiers': cell_ids}
        for cell_id in cell_ids:
            text_context = f'Extract ONLY for: "{cell_id}"\n---\n{text}'
            ctx = StageContext(text=text_context, extraction_schema=extraction_schema)
            result = extraction_pipeline.run(ctx)
            all_results.append(result)
            extracted = result.ctx['extracted_data']
            all_experiments[cell_id] = extracted
    except Exception as e:
        print(f'Error during extraction: {e}')
        print(format_exc())
    else:
        print('Extraction successful')
        with open('battery_extracted.json', 'w') as f:
            json.dump(all_experiments, f, indent=2)
    finally:
        with open('battery_extraction_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
