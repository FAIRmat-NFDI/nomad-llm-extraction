from nomad_llm_extraction.transform.json_transformer import ProcessingPipeline


def test_perovskite_domain_module_exports_reference_api():
    from nomad_llm_extraction.domains.perovskite_solar_cell.pipeline import (
        INSTRUCTION_TEXT,
        KEY_MAPPING,
        SYSTEM_PROMPT,
        build_pipeline,
    )

    pipeline = build_pipeline()

    assert isinstance(SYSTEM_PROMPT, str)
    assert SYSTEM_PROMPT.strip()
    assert isinstance(INSTRUCTION_TEXT, str)
    assert INSTRUCTION_TEXT.strip()
    assert isinstance(KEY_MAPPING, dict)
    assert KEY_MAPPING
    assert isinstance(pipeline, ProcessingPipeline)
