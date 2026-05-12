# How to Use This Plugin

`nomad-llm-extraction` provides a generalized LLM extraction pipeline.  It can be used
standalone (with any JSON schema) or inside a NOMAD Oasis (with NOMAD `m_def` schemas).

## Installation

```sh
pip install nomad-llm-extraction
# or, for development:
uv pip install -e '.[dev]'
```

To add the plugin to a NOMAD Oasis, follow the
[NOMAD plugin documentation](https://nomad-lab.eu/prod/v1/staging/docs/plugins/plugins.html#add-a-plugin-to-your-nomad).

---

## Core pipeline API

### `ExtractionPipeline`

The main entry point.  Constructed by injecting dependencies; no subclassing needed.

```python
from nomad_llm_extraction.pipeline import ExtractionPipeline, PromptConfig

pipeline = ExtractionPipeline(
    engine=engine,                  # LLMEngine – required
    schema_source=schema_source,    # SchemaSource – required
    prompt_config=PromptConfig(     # optional
        system_prompt="...",
        instruction_text="...",
    ),
    validators=[...],               # optional – see Extension points
    visualizers=[...],              # optional
    stage_hooks=[...],              # optional
    schema_resolver=None,           # optional callable (schema) -> schema
    postprocessor=None,             # optional callable (data) -> data
    archive_shaper=None,            # optional callable (data) -> data
)

result = pipeline.run(paper_text)   # PipelineResult
```

`pipeline.run(text)` always returns a `PipelineResult`; exceptions are captured, not
raised.

### `PipelineResult`

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | `True` if every stage completed without error |
| `raw_llm_output` | `str \| None` | Raw JSON string from the LLM |
| `extracted_data` | `Any` | Parsed Python object from `json_parse` stage |
| `postprocessed_data` | `Any` | Output of the `postprocessing` stage |
| `archive_data` | `Any` | Output of the `archive_shaping` stage |
| `stages` | `list[StageResult]` | Per-stage name, success flag, and error |
| `error` | `str \| None` | Error message of the first failing stage |

### `PromptConfig`

```python
from nomad_llm_extraction.pipeline import PromptConfig

cfg = PromptConfig(
    system_prompt="You are a materials science expert.",
    instruction_text="Extract all experimental parameters.",
)
```

Both fields default to `''`; the pipeline concatenates them with the schema and input
text to build the LLM prompt.

---

## Schema sources

Schema sources satisfy the `SchemaSource` protocol: a single `get_schema() -> dict`
method.  Two concrete implementations are provided.

### `InlineSchemaSource` — use any JSON schema dict

```python
from nomad_llm_extraction.pipeline import InlineSchemaSource

schema = {
    "type": "object",
    "properties": {
        "material": {"type": "string"},
        "efficiency": {"type": "number"},
    },
}
source = InlineSchemaSource(schema)
```

Optional keyword arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `optimizer` | `None` | `(schema) -> schema` applied after ref resolution |
| `remove_defs` | `False` | Strip `$defs` from the resolved schema |
| `resolve_allOf` | `False` | Merge `allOf` arrays into a single dict |

### `NomadSchemaSource` — fetch from a NOMAD `m_def`

```python
from nomad_llm_extraction.pipeline import NomadSchemaSource

source = NomadSchemaSource(
    "nomad.datamodel.perovskite_solar_cell.PerovskiteSolarCell",
    unit_value=True,   # request unit-value formatted quantities
    remove_defs=True,
)
```

Uses `get_nomad_schema()` internally; requires network access to the NOMAD API.

---

## LLM engine

> **Prerequisite:** `litellm` is not included in the standard installation.
> Install it before using `LiteLLMEngine`:
> ```sh
> pip install litellm
> ```

`LiteLLMEngine` wraps [LiteLLM](https://docs.litellm.ai/) and supports any model that
provides structured JSON output via `response_format`.

```python
from nomad_llm_extraction.pipeline.schema_filling.llm_engine import LiteLLMEngine

engine = LiteLLMEngine(
    model_name="gpt-4o",     # any LiteLLM model string
    api_url=None,            # set for self-hosted endpoints (vLLM, Ollama, …)
    api_key="sk-...",        # leave empty if set via environment variable
)
```

Redis caching is configured automatically from env vars `REDIS_HOST`, `REDIS_PORT`,
`REDIS_TTL`, and `REDIS_PASSWORD` (defaults: `127.0.0.1:6379`).

The engine contract is strict: `generate(prompt, json_schema, optional_params) -> str`.
Any engine-like object satisfying this interface works.

---

## Extension points

All hooks and callables are injected at `ExtractionPipeline` construction time.

### `postprocessor` — domain-specific field mapping

A callable `(extracted_data: Any) -> Any`.  Applied during the `postprocessing` stage.
Result lands in `result.postprocessed_data`.

The perovskite domain module exposes a `build_pipeline()` factory that returns a
`ProcessingPipeline`.  Wrap `apply()` in a callable to satisfy the postprocessor
interface:

```python
from nomad_llm_extraction.domains.perovskite_solar_cell.pipeline import build_pipeline

proc = build_pipeline()

def postprocessor(data):
    cells = data.get("cells", [data]) if isinstance(data, dict) else data
    return proc.apply(cells)

pipeline = ExtractionPipeline(
    ...,
    postprocessor=postprocessor,
)
```

`proc.apply(cells)` returns the **processed cells list** (field names renamed,
units converted, `None` values stripped).  It is *not* a full NOMAD archive dict.
To produce a full archive object, add an `archive_shaper` that wraps the cells list
into the desired archive shape (e.g. `{"data": {"cells": processed_cells}}`).
```

### `archive_shaper` — reshape to a target archive format

A callable `(postprocessed_data: Any) -> Any`.  Applied during `archive_shaping`;
result lands in `result.archive_data`.  Use this to produce a NOMAD-shaped dict from
the postprocessed data.

### `validators` — non-aborting data validation

A list of callables `(extracted_data: Any) -> None` that raise on invalid data.
Validation failures are recorded in `result.stages` (the `validation` stage result's
`data['validation_errors']`); they do **not** abort subsequent stages.

```python
def check_efficiency(data):
    if data.get("efficiency", 0) < 0:
        raise ValueError("efficiency cannot be negative")

pipeline = ExtractionPipeline(..., validators=[check_efficiency])
```

Fuller validation workflows (schema-level, cross-field, provenance checks) are
**future work**; the `validators` list is the designed extension point.

### `visualizers` — inspect results after every run

A list of callables `(result: PipelineResult) -> None`.  Called after every run,
success or failure.  Exceptions inside visualizers are logged and swallowed.

```python
def print_summary(result):
    print("success:", result.success)
    if result.extracted_data:
        print("keys:", list(result.extracted_data.keys()))

pipeline = ExtractionPipeline(..., visualizers=[print_summary])
```

More capable visualization workflows (dashboards, NOMAD upload previews) are
**future work**; the `visualizers` list is the designed extension point.

### `stage_hooks` — intercept any named stage

A list of `(stage_name, when, hook)` tuples where `when` is `'before'` or `'after'`
and `hook` is a `(ctx: StageContext) -> None` callable.

```python
def log_raw_output(ctx):
    print("LLM raw output:", ctx.raw_output)

pipeline = ExtractionPipeline(
    ...,
    stage_hooks=[("llm_extraction", "after", log_raw_output)],
)
```

Stage names: `schema_load`, `schema_resolve`, `prompt_build`, `llm_extraction`,
`json_parse`, `validation`, `postprocessing`, `archive_shaping`.

### `schema_resolver` — transform the loaded schema

A callable `(schema: dict) -> dict` applied during the `schema_resolve` stage.
Distinct from the per-source `optimizer`: the resolver runs on the already-loaded
schema and is controlled at pipeline level.

### `optimizer` on schema sources — prune or annotate before the LLM call

Both `InlineSchemaSource` and `NomadSchemaSource` accept an `optimizer` keyword
argument `(schema: dict) -> dict`.  Useful for stripping irrelevant fields to reduce
token usage.  Agentic schema-improvement strategies are **future work** built on this
hook.

---

## Adding a new domain

1. Create `src/nomad_llm_extraction/domains/<your_domain>/pipeline.py`.
2. Define `SYSTEM_PROMPT`, `INSTRUCTION_TEXT`, and `KEY_MAPPING` (LLM field → target
   field names).
3. Build a `ProcessingPipeline` with domain-specific condition/transform triplets,
   following `domains/perovskite_solar_cell/pipeline.py` as the reference.
4. Expose a `build_pipeline()` factory and pass its return value as `postprocessor`
   when constructing `ExtractionPipeline`.
