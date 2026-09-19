"""LLMEngineConfig must serialize with and without an api_key: the key is
optional (the engine falls back to the provider env var), and workflow
inputs are JSON-serialized by Temporal on every hop."""

from nomad_llm_extraction.pipeline.models import LLMEngineConfig


def test_engine_config_serializes_without_api_key():
    config = LLMEngineConfig(model_name='gemini/gemini-2.5-flash')
    assert '"api_key":null' in config.model_dump_json()


def test_engine_config_serializes_the_api_key_when_given():
    config = LLMEngineConfig(model_name='gemini/gemini-2.5-flash', api_key='k')
    assert '"api_key":"k"' in config.model_dump_json()
