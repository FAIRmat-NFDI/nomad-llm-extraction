# Tutorial

This tutorial walks through two concrete examples of the extraction pipeline.

> **Prerequisite:** Both examples use `LiteLLMEngine`, which requires `litellm`.
> Install it before running the snippets:
> ```sh
> pip install litellm
> ```

---

## Example 1 — Extract data with any JSON schema

This example requires no NOMAD installation.  It uses an inline JSON schema and a
cloud LLM.

```python
from nomad_llm_extraction.pipeline import ExtractionPipeline, PromptConfig
from nomad_llm_extraction.pipeline import InlineSchemaSource
from nomad_llm_extraction.pipeline.schema_filling.llm_engine import LiteLLMEngine

# 1. Define the output schema.
schema = {
    "type": "object",
    "properties": {
        "material": {
            "type": "string",
            "description": "Primary material studied in the paper",
        },
        "efficiency": {
            "type": "number",
            "description": "Power conversion efficiency (%)",
        },
        "bandgap": {
            "type": "number",
            "description": "Band gap energy (eV)",
        },
    },
}

# 2. Configure the LLM engine.
engine = LiteLLMEngine(model_name="gpt-4o", api_key="sk-...")

# 3. Build the pipeline.
pipeline = ExtractionPipeline(
    engine=engine,
    schema_source=InlineSchemaSource(schema, remove_defs=True),
    prompt_config=PromptConfig(
        system_prompt="You are a materials science expert.",
        instruction_text="Extract material properties from the abstract.",
    ),
)

# 4. Run.
paper_abstract = """
We report a perovskite solar cell with a band gap of 1.55 eV and a certified
power conversion efficiency of 25.1% based on a MAPbI3 absorber layer.
"""

result = pipeline.run(paper_abstract)

if result.success:
    print(result.extracted_data)
    # e.g. {'material': 'MAPbI3', 'efficiency': 25.1, 'bandgap': 1.55}
else:
    print("Extraction failed:", result.error)
    for stage in result.stages:
        if not stage.success:
            print(f"  Stage '{stage.name}' failed: {stage.error}")
```

The `PipelineResult` contains:

- `result.extracted_data` — the parsed JSON object
- `result.postprocessed_data` — same as `extracted_data` when no postprocessor is set
- `result.raw_llm_output` — the raw JSON string returned by the LLM
- `result.stages` — per-stage success flags and error messages

---

## Example 2 — NOMAD-oriented extraction with the perovskite domain module

This example fetches the schema directly from the NOMAD metainfo API and applies the
built-in perovskite postprocessor to normalize field names and units.

```python
from nomad_llm_extraction.pipeline import ExtractionPipeline, PromptConfig
from nomad_llm_extraction.pipeline import NomadSchemaSource
from nomad_llm_extraction.pipeline.schema_filling.llm_engine import LiteLLMEngine
from nomad_llm_extraction.domains.perovskite_solar_cell.pipeline import (
    build_pipeline,
    SYSTEM_PROMPT,
    INSTRUCTION_TEXT,
)

# 1. Schema source: fetch from NOMAD m_def.
#    unit_value=True requests {"value": ..., "unit": "..."} formatted quantities.
#    remove_defs=True strips $defs from the resolved schema to reduce prompt size.
schema_source = NomadSchemaSource(
    "nomad.datamodel.perovskite_solar_cell.PerovskiteSolarCell",
    unit_value=True,
    remove_defs=True,
)

# 2. Domain postprocessor: renames LLM fields to NOMAD fields, converts units,
#    flattens value/unit dicts, and removes null values.
#    ProcessingPipeline.apply(data, schema) drives the transforms; wrap it in
#    a plain callable to satisfy the postprocessor interface.
proc_pipeline = build_pipeline()

def postprocessor(data):
    # Extract the cells list from the LLM output dict if present.
    cells = data.get("cells", [data]) if isinstance(data, dict) else data
    return proc_pipeline.apply(cells)

# 3. LLM engine.
engine = LiteLLMEngine(model_name="gpt-4o", api_key="sk-...")

# 4. Build the pipeline.
pipeline = ExtractionPipeline(
    engine=engine,
    schema_source=schema_source,
    prompt_config=PromptConfig(
        system_prompt=SYSTEM_PROMPT,
        instruction_text=INSTRUCTION_TEXT,
    ),
    postprocessor=postprocessor,
)

# 5. Run.
paper_text = (
    "We report a perovskite solar cell based on a MAPbI3 absorber with a band gap "
    "of 1.55 eV and a certified power conversion efficiency of 25.1%."
)
result = pipeline.run(paper_text)

if result.success:
    # extracted_data: raw LLM output (LLM field names, value/unit dicts)
    llm_output = result.extracted_data

    # postprocessed_data: the processed cells list returned by proc_pipeline.apply().
    # Field names are renamed to NOMAD names and units are converted, but this is
    # NOT a full NOMAD archive dict.  Add an archive_shaper to produce one.
    processed_cells = result.postprocessed_data
    print(processed_cells)
```

### What the postprocessor does

The `build_pipeline()` postprocessor from the perovskite domain module applies these
transforms in order:

1. **rename** — maps LLM field names to NOMAD field names using `KEY_MAPPING`
   (e.g. `bandgap` → `band_gap`).
2. **unit_conversion** — converts quantity values to SI units where applicable.
3. **split_unit_value** — splits `{"value": v, "unit": u}` dicts into separate keys
   for fields listed in `SPLIT_VALUE_UNIT`.
4. **flatten_unit_value** — flattens remaining `{"value": v, "unit": u}` dicts.
5. **delete_sections** — removes fields listed in `SKIP_KEYS` (e.g. `additives`).
6. **layer_order** — derives a `layer_order` string from the extracted layers list.
7. **remove_none** — strips `None`-valued keys from the result.

---

## Adding stage hooks

Stage hooks let you inspect or log intermediate pipeline state without modifying the
pipeline logic.

```python
def log_after_llm(ctx):
    """Print how many characters the LLM returned."""
    raw = ctx.raw_output or ""
    print(f"LLM returned {len(raw)} characters")

def log_validation(ctx):
    """Warn if any validators reported errors."""
    errors = ctx.metadata.get("validation_errors", {})
    if errors:
        print("Validation issues:", errors)

pipeline = ExtractionPipeline(
    engine=engine,
    schema_source=schema_source,
    stage_hooks=[
        ("llm_extraction", "after", log_after_llm),
        ("validation", "after", log_validation),
    ],
)
```

Hook exceptions are caught and logged; they never abort the pipeline.

---

## Future extension points

The following capabilities have dedicated hooks but fuller workflows are not yet
implemented:

- **Validation** (`validators` parameter): currently accepts a list of callables; a
  schema-driven or provenance-aware validation layer is planned.
- **Visualization** (`visualizers` parameter): currently accepts a list of callables
  called after each run; richer dashboards or NOMAD upload previews are planned.
- **Agentic schema optimization** (`optimizer` on schema sources): the `optimizer`
  hook is the intended entry point for future agent-driven schema pruning and
  prompt-improvement strategies.
