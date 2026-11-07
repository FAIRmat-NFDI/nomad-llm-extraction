from typing import Any, Literal

import litellm
from litellm import get_supported_openai_params, supports_response_schema
from loguru import logger

litellm.suppress_debug_info = True
litellm.set_verbose = False
import logging

logging.getLogger('LiteLLM').setLevel(logging.CRITICAL)


def enforceable_schema(schema: Any) -> Any:
    """The schema as the providers can enforce it: `oneOf` becomes `anyOf`,
    `const` becomes a one-value `enum`, `discriminator` goes.

    Pydantic writes a discriminated union (`A | B` told apart by a field)
    as `oneOf` plus `discriminator`. Structured output at Gemini and
    OpenAI accepts `anyOf` only, and litellm's rewrite for Gemini drops
    `oneOf` altogether: the model was then free to answer anything in
    that place, while the answer was still validated against the full
    schema, so every answer failed and the retries could not help. For
    branches that cannot both match (a discriminated union) `anyOf` says
    the same thing; where they could, it is looser for the model only:
    the answer is validated against the caller's schema as before. The
    same rewrite drops `const` (pydantic's one-value Literal, the tag of
    such a union); `enum` is kept everywhere. Applied for every provider,
    so the model sees the same schema whatever the engine. The caller's
    dict is left as it is."""
    if isinstance(schema, list):
        return [enforceable_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == 'discriminator':
            continue
        if key in ('oneOf', 'anyOf'):
            out['anyOf'] = out.get('anyOf', []) + enforceable_schema(value)
        elif key == 'const':
            out.setdefault('enum', [value])
        elif key == 'enum':
            out['enum'] = list(value)
        else:
            out[key] = enforceable_schema(value)
    return out


def _schema_dict(json_schema: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(json_schema, str):
        import json

        try:
            return json.loads(json_schema)
        except json.JSONDecodeError as e:
            logger.error(f'Invalid JSON schema string: {e}')
            raise
    return json_schema


def format_tool_call_schema(schema: dict[str, Any]) -> dict:
    # adapted from https://github.com/567-labs/instructor/blob/47fdb2ca07119d389a3c0e8bc28b9930b814f294/instructor/v2/providers/openai/schema.py
    parameters = {k: v for k, v in schema.items() if k not in ('title', 'description')}
    parameters['required'] = sorted(schema.get('required', []))
    return {
        'name': schema.get('title', 'ExtractedData'),
        'description': schema.get('description', 'Extracted data from the document.'),
        'parameters': parameters,
    }


Method = Literal['response_format', 'tool_call']


class StructuredLLMEngine:
    """Base class for all structured extraction. its entire job is to define a strict contract or blueprint."""

    def generate(
        self,
        prompt: str,
        json_schema: str,
        optional_params: dict = {},
        method: Method = 'tool_call',
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
        self,
        prompt: str,
        json_schema: str | dict[str, Any],
        optional_params: dict = {},
        method: Method = 'tool_call',
    ) -> str:
        if method == 'response_format':
            return self.generate_with_response_format(
                prompt, json_schema, optional_params
            )
        elif method == 'tool_call':
            return self.generate_with_tool_call(prompt, json_schema, optional_params)
        else:
            raise ValueError(
                f'Invalid method: {method}. Supported methods are "response_format" and "tool_call".'
            )

    def generate_with_response_format(
        self, prompt: str, json_schema: str | dict[str, Any], optional_params: dict = {}
    ) -> str:
        from litellm import completion

        response_format = {
            'type': 'json_schema',
            'json_schema': {
                'schema': enforceable_schema(_schema_dict(json_schema)),
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
        self, prompt: str, json_schema: str | dict[str, Any], optional_params: dict = {}
    ) -> str:
        from litellm import completion

        json_schema = enforceable_schema(_schema_dict(json_schema))
        params_to_use = {**optional_params}
        self.check_additional_params(optional_params)
        formatted_schema = format_tool_call_schema(json_schema)
        params_to_use['tools'] = [{'type': 'function', 'function': formatted_schema}]
        params_to_use['tool_choice'] = {
            'type': 'function',
            'function': {'name': formatted_schema['name']},
        }
        params_to_use.setdefault('reasoning_effort', 'none')
        try:
            resp = completion(
                model=self.model_name,
                api_base=self.base_url,
                messages=[{'role': 'user', 'content': prompt}],
                drop_params=True,
                **params_to_use,
            )
            tool_calls = resp.choices[0].message.tool_calls
            if not tool_calls:
                logger.error(
                    f'No tool calls found in response: {resp} falling back to resp.choices[0].message.content'
                )
                return resp.choices[0].message.content
            else:
                message_content = tool_calls[0]
                for attr in ['function', 'arguments']:
                    if not hasattr(message_content, attr):
                        logger.error(
                            f'Missing attribute {attr} in tool call: {message_content}. Falling back to resp.choices[0].message.content'
                        )
                        return resp.choices[0].message.content
                    message_content = getattr(message_content, attr)
        except (AttributeError, IndexError, TypeError, ValueError) as e:
            logger.error(
                f'LiteLLM generation with tool call failed due to unexpected response structure: {e} returned response: {resp}'
            )
            raise
        except Exception as e:
            logger.error(
                f'LiteLLM generation with tool call failed: {e} returned response: {resp}'
            )
            raise
        return message_content
