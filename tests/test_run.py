from nomad_llm_extraction.pipeline.extract import build_effective_config


def test_build_effective_config_applies_llm_extraction_method(tmp_path):
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(
        'llm_engine_config:\n'
        '  model_name: test-model\n'
        'extraction_schema:\n'
        '  type: object\n'
        'prompt: extract this\n'
    )

    config = build_effective_config(
        config_path, llm_extraction_method='response_format'
    )

    assert config['llm_extraction_method'] == 'response_format'
