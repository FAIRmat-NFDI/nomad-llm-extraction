"""(Outlines/Instructor Constrained Decoding).

Example Usage:

    from nomad_llm_extraction.pipeline.schema_filling.llm_engine import OutlinesEngine, InstructorEngine

    # For Local vLLM
    # active_engine = OutlinesEngine(model_name="Qwen/Qwen2.5-72B", api_url="http://localhost:8000/v1")

    # For Local Ollama
    # active_engine = InstructorEngine(model_name="llama3.1", api_url="http://localhost:11434/v1")

    # For Cloud ChatGPT
    # active_engine = InstructorEngine(model_name="gpt-4o", api_key="sk-...")

    # json schema
    needed to use json schema for outlines
"""

import logging
import os
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

try:
    import litellm

    # Disable litellm error output
    litellm.suppress_debug_info = True
    litellm.set_verbose = False

    logging.getLogger('LiteLLM').setLevel(logging.CRITICAL)
    from litellm.caching.caching import Cache

    litellm.cache = Cache(
        type='redis',
        host=os.environ.get('REDIS_HOST', '127.0.0.1'),
        port=int(os.environ.get('REDIS_PORT', '6379')),
        ttl=int(os.environ.get('REDIS_TTL', '1000000')),
        password=os.environ.get('REDIS_PASSWORD'),
        namespace='litellm',
    )
except Exception:
    print(
        "Error importing LiteLLM. Make sure it's installed and configured properly if you intend to use it."
    )
    pass
import litellm
from litellm import get_supported_openai_params, supports_response_schema

logger = logging.getLogger(__name__)


class StructuredLLMEngine:
    """Base class for all structured extraction. its entire job is to define a strict contract or blueprint."""

    def generate(
        self, prompt: str, json_schema: str, optional_params: dict = {}
    ) -> str:
        raise NotImplementedError('Subclasses must implement the generate method.')


class LiteLLMEngine(StructuredLLMEngine):
    def __init__(self, model_name: str, api_url: str | None = None, api_key: str = ''):

        self.model_name = model_name
        params = get_supported_openai_params(model=model_name)
        if params is None:
            raise ValueError(f'Model {model_name} is not supported by LiteLLM.')
        assert 'response_format' in params, (
            f'Model {model_name} does not support response_format parameter required for structured output.'
        )
        assert supports_response_schema(model=model_name), (
            f'Model {model_name} does not support json schema response for structured output.'
        )

        self.params = params
        self.base_url = api_url

        if api_key:
            litellm.api_key = api_key

    def generate(
        self, prompt: str, json_schema: str, optional_params: dict = {}
    ) -> str:
        from litellm import completion

        response_format = {
            'type': 'json_schema',
            'json_schema': {
                'schema': json_schema,
                'strict': False,
                'name': 'ResponseSchema',
            },
            # 'strict': True,
        }
        params_to_use = {**optional_params}
        for param in optional_params:
            if param not in self.params:
                logger.warning(
                    f'Parameter {param} is not supported by model {self.model_name} and will be ignored.'
                )
                del params_to_use[param]
        try:
            resp = completion(
                model=self.model_name,
                base_url=self.base_url,
                messages=[{'role': 'user', 'content': prompt}],
                response_format=response_format,
                **params_to_use,
            )
        except Exception as e:
            logger.error(f'LiteLLM generation failed: {e}')
            raise
        return resp.choices[0].message.content


class OutlinesEngine(StructuredLLMEngine):
    def __init__(
        self, model_name: str, api_url: str | None = None, api_key: str = 'EMPTY'
    ):
        import outlines

        self.model_name = model_name

        if api_url:
            self.client = OpenAI(base_url=api_url, api_key=api_key)
        else:
            self.client = OpenAI(api_key=api_key if api_key != 'EMPTY' else None)

        # Using user's original logic
        self.model = outlines.from_vllm(self.client, model_name)
        logger.info(f'Initialized OutlinesEngine for {model_name}')

    def generate(
        self, prompt: str, response_model: type[T], temperature: float = 0.1
    ) -> T:
        try:
            # tell outlines to use the passed pydantic model
            result_json = self.model(
                prompt, output_type=response_model, temperature=temperature
            )
            return response_model.model_validate_json(result_json)
        except Exception as e:
            logger.error(f'Outlines generation failed: {e}')
            raise


class InstructorEngine(StructuredLLMEngine):
    def __init__(
        self, model_name: str, api_url: str | None = None, api_key: str = 'EMPTY'
    ):
        import instructor

        self.model_name = model_name

        if api_url:
            base_client = OpenAI(base_url=api_url, api_key=api_key)
        else:
            base_client = OpenAI(api_key=api_key if api_key != 'EMPTY' else None)

        self.client = instructor.from_openai(base_client)
        logger.info(f'Initialized InstructorEngine for {model_name}')

    def generate(
        self, prompt: str, response_model: type[T], temperature: float = 0.1
    ) -> T:
        try:
            return self.client.chat.completions.create(
                model=self.model_name,
                response_model=response_model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f'Instructor generation failed: {e}')
            raise
