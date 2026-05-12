# Copilot Instructions

## Project Overview
`nomad-llm-extraction` is a **NOMAD plugin** that uses LLMs to extract structured scientific data from research papers into [NOMAD](https://nomad-lab.eu/) archive schemas. It is **not** a standalone app — it lives inside the NOMAD ecosystem and depends on `nomad-lab`.

## Architecture & Data Flow

```
Paper text → LiteLLMEngine (structured JSON via LLM) → LLM archive JSON
    → ProcessingPipeline (schema-driven transforms) → NOMAD archive JSON
    → InplaceTransformer (rule-based field mapping) → final NOMAD archive
```

- **`pipeline/schema_filling/llm_engine.py`** — wraps LiteLLM for structured JSON generation. `LiteLLMEngine` is the production engine; uses Redis for caching (`REDIS_HOST`, `REDIS_PORT`, `REDIS_TTL`, `REDIS_PASSWORD` env vars). Always validates that the chosen model supports `response_format` and JSON schema before calling.
- **`transform/json_transformer.py`** — schema-driven recursive traversal of JSON objects. `ProcessingPipeline` chains multiple `(cond, get_func_args, func)` triplets applied during schema traversal. The `state` dict tracks `name`, `path` (schema path), `a_path` (archive/data path), `p_path`, `a_p_path`, `type`.
- **`transform/inplace_transformer.py`** — uses NOMAD's `Rules`/`Rule` annotations for direct field-to-field mapping within an archive. Handles `[n]`-style array index placeholders in paths.
- **`transform/common_transforms.py`** — reusable condition/args/transform functions (unit conversion, `remove_unit_value`, `split_value_unit`, `rename_section`, `remove_none`).
- **`actions/`** — Temporal.io workflow/activity scaffolding (currently a simple example; extend for async LLM pipeline orchestration).
- **`domains/perovskite_solar_cell/`** — reference domain implementation showing how to wire `ProcessingPipeline` for a real-world schema. Study this when adding a new domain.

## Key Patterns

### ProcessingPipeline transform triplet
Each transform step is a `(cond, get_func_args, func_apply)` triplet registered on the pipeline:
```python
pipeline.add_transform(
    cond=unit_cond,          # (section, state) -> bool
    get_func_args=unit_args, # (section, state) -> (state, extra_args)
    func_apply=convert_unit  # (scalpl_cut_obj, path, extra_args) -> scalpl_cut_obj
)
```
Data objects are wrapped with `scalpl.Cut` for dot-notation path access (e.g., `data["foo.bar[0].baz"]`).

### Schema resolution
Always call `resolve_schema(schema)` (from `transform/utils.py`) to dereference `$ref` before passing schemas to pipeline traversal. Use `get_nomad_schema(archive)` to obtain the NOMAD metainfo schema for an archive entry.

### Unit values
LLM output stores measurements as `{"value": ..., "unit": "..."}` dicts. Use `remove_unit_value`, `convert_unit`, or `split_value_unit` from `common_transforms.py` to normalize before writing to NOMAD.

### Adding a new domain
1. Create `domains/<domain>/pipeline.py` mirroring `domains/perovskite_solar_cell/pipeline.py`.
2. Define `SYSTEM_PROMPT`, `INSTRUCTION_TEXT`, `KEY_MAPPING` (LLM field name → NOMAD field name).
3. Build a `ProcessingPipeline` with domain-specific condition/transform triplets.
4. Use test data in `tests/data/` as fixtures; add a corresponding test in `tests/transform/`.

## Developer Workflows

```sh
# Install (use uv for speed; nomad-lab comes from MPCDF GitLab)
uv pip install -e '.[dev]'          # development
uv pip install -e '.[dev,tutorial]' # with Jupyter tutorial deps

# Test
python -m pytest -sv tests

# Lint / format
ruff check .
ruff format . --check
```

- The `nomad-lab` package source is `https://gitlab.mpcdf.mpg.de/nomad-lab/nomad-FAIR.git` (configured in `pyproject.toml` via `[tool.uv.sources]`).
- Test fixtures live in `tests/data/` (real archive JSONs + schemas). Use `ARCHIVE_PATH`, `PEROV_SCHEMA_PATH`, `NOMAD_SCHEMA_PATH` constants from `tests/transform/test_perla_postproc.py` as the reference pattern.
- `build/` mirrors `src/` — do not edit files there; they are build artefacts.
