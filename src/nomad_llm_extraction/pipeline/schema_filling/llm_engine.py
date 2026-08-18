import logging
import os
from typing import Any, TypeVar

import litellm
from litellm import get_supported_openai_params, supports_response_schema
from loguru import logger
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

try:
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


def format_tool_call_schema(schema: dict) -> dict:
    # adapted from https://github.com/567-labs/instructor/blob/47fdb2ca07119d389a3c0e8bc28b9930b814f294/instructor/v2/providers/openai/schema.py
    parameters = {k: v for k, v in schema.items() if k not in ('title', 'description')}
    parameters['required'] = sorted(schema.get('required', []))

    return {
        'name': schema.get('title', ''),
        'description': schema.get('description', ''),
        'parameters': parameters,
    }


class StructuredLLMEngine:
    """Base class for all structured extraction. its entire job is to define a strict contract or blueprint."""

    def generate(
        self, prompt: str, json_schema: str, optional_params: dict = {}
    ) -> str:
        raise NotImplementedError('Subclasses must implement the generate method.')


class LiteLLMEngine(StructuredLLMEngine):
    def __init__(self, model_name: str, api_url: str | None = None, api_key: str = ''):
        self.model_name = model_name
        params = []
        try:
            params = get_supported_openai_params(model=model_name)

            if params is None:
                raise ValueError(f'Model {model_name} is not supported by LiteLLM.')
            msgs = []
            for param in ['response_format', 'tools', 'tool_choice']:
                if param not in params:
                    msgs.append(
                        f'Model {model_name} does not support {param} parameter.\n'
                    )

            if not supports_response_schema(model=model_name):
                msgs.append(
                    f'Model {model_name} does not support json schema response for structured output.\n'
                )
            if msgs:
                logger.warning(
                    f'LiteLLM model {model_name} may not support all features:\n'
                    + '\n'.join(msgs)
                    + '\nProceeding with caution.'
                )
        except Exception as e:
            logger.warning(
                f'Error checking model support in LiteLLM: {e} \n Proceeding with caution.'
            )

        self.params = params or []
        self.base_url = api_url

        if api_key:
            litellm.api_key = (
                api_key if type(api_key) is str else api_key.get_secret_value()
            )

    def check_additional_params(self, optional_params: dict):
        if not self.params:
            logger.warning(
                f'No supported parameters found for model {self.model_name}. Cannot check optional parameters.'
            )
            return
        for param in optional_params:
            if param not in self.params:
                logger.warning(
                    f'Parameter: {param} is not supported by model {self.model_name}'
                )

    def generate(
        self, prompt: str, json_schema: str | dict[str, Any], optional_params: dict = {}
    ) -> str:
        from litellm import completion

        response_format = {
            'type': 'json_schema',
            'json_schema': {
                'schema': json_schema,
                'strict': True,
                'name': 'ResponseSchema',
            },
            # 'strict': True,
        }
        self.check_additional_params(optional_params)
        params_to_use = {**optional_params}
        try:
            resp = completion(
                model=self.model_name,
                api_base=self.base_url,
                messages=[{'role': 'user', 'content': prompt}],
                response_format=response_format,
                drop_params=True,
                **params_to_use,
            )
            message_content = resp.choices[0].message.content
        except (AttributeError, IndexError, TypeError, ValueError) as e:
            logger.error(
                f'LiteLLM generation failed due to unexpected response structure: {e}'
            )
            raise
        except Exception as e:
            logger.error(f'LiteLLM generation failed: {e}')
            raise
        return message_content

    def generate_with_tool_call(
        self, prompt: str, schema: dict, optional_params: dict = {}
    ) -> str:
        from litellm import completion

        params_to_use = {**optional_params}
        self.check_additional_params(optional_params)
        formatted_schema = format_tool_call_schema(schema)
        tools = [{'type': 'function', 'function': formatted_schema}]
        tool_choice = {
            'type': 'function',
            'function': {'name': formatted_schema['name']},
        }
        params_to_use.setdefault('reasoning_effort', None)
        try:
            resp = completion(
                model=self.model_name,
                api_base=self.base_url,
                messages=[{'role': 'user', 'content': prompt}],
                tools=tools,
                tool_choice=tool_choice,
                drop_params=True,
                **params_to_use,
            )
            message_content = resp.choices[0].message.tool_calls[0].function.arguments
        except (AttributeError, IndexError, TypeError, ValueError) as e:
            logger.error(
                f'LiteLLM generation with tool call failed due to unexpected response structure: {e}'
            )
            raise
        except Exception as e:
            logger.error(f'LiteLLM generation with tool call failed: {e}')
            raise
        return message_content
