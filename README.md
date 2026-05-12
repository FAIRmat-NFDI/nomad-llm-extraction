# nomad-llm-extraction

A generalized LLM extraction pipeline for structured scientific data.

`nomad-llm-extraction` turns unstructured research-paper text into structured JSON by
combining an LLM backend with a user-supplied JSON schema.  It can be used standalone
with any JSON schema, or integrated with [NOMAD](https://nomad-lab.eu/) where NOMAD
`m_def` schemas serve as the schema source and domain-specific postprocessors map the
LLM output to NOMAD archive shapes.

The **perovskite solar cell** domain module
(`src/nomad_llm_extraction/domains/perovskite_solar_cell/`) is the reference
implementation.  Copy it as a starting point for new domains.

## Quick start

### Minimal example — any JSON schema, direct text input

```python
from nomad_llm_extraction.pipeline import ExtractionPipeline, PromptConfig
from nomad_llm_extraction.pipeline import InlineSchemaSource
from nomad_llm_extraction.pipeline.schema_filling.llm_engine import LiteLLMEngine

schema = {
    "type": "object",
    "properties": {
        "material": {"type": "string"},
        "efficiency": {"type": "number"},
    },
}

engine = LiteLLMEngine(model_name="gpt-4o", api_key="sk-...")
pipeline = ExtractionPipeline(
    engine=engine,
    schema_source=InlineSchemaSource(schema),
    prompt_config=PromptConfig(
        system_prompt="You are a materials science expert.",
        instruction_text="Extract material properties from the paper.",
    ),
)

result = pipeline.run(paper_text)
if result.success:
    print(result.extracted_data)
```

### NOMAD example — `m_def` schema + domain postprocessor

```python
from nomad_llm_extraction.pipeline import ExtractionPipeline, PromptConfig
from nomad_llm_extraction.pipeline import NomadSchemaSource
from nomad_llm_extraction.pipeline.schema_filling.llm_engine import LiteLLMEngine
from nomad_llm_extraction.domains.perovskite_solar_cell.pipeline import (
    build_pipeline,
    SYSTEM_PROMPT,
    INSTRUCTION_TEXT,
)

schema_source = NomadSchemaSource(
    "nomad.datamodel.perovskite_solar_cell.PerovskiteSolarCell",
    unit_value=True,
    remove_defs=True,
)

# ProcessingPipeline.apply(data, schema) drives the field-mapping transforms.
# Wrap it in a plain callable to satisfy the postprocessor interface.
proc = build_pipeline()

def postprocessor(data):
    cells = data.get("cells", [data]) if isinstance(data, dict) else data
    return proc.apply(cells)

engine = LiteLLMEngine(model_name="gpt-4o", api_key="sk-...")
pipeline = ExtractionPipeline(
    engine=engine,
    schema_source=schema_source,
    prompt_config=PromptConfig(
        system_prompt=SYSTEM_PROMPT,
        instruction_text=INSTRUCTION_TEXT,
    ),
    postprocessor=postprocessor,
)

result = pipeline.run(paper_text)
if result.success:
    nomad_archive = result.postprocessed_data
```

## Pipeline stages

The pipeline runs these stages in order; each populates the shared `StageContext`:

| # | Stage name | What it does |
|---|------------|--------------|
| 1 | `schema_load` | Fetches the JSON schema from the schema source |
| 2 | `schema_resolve` | Applies the optional pipeline-level schema resolver |
| 3 | `prompt_build` | Assembles the LLM prompt from text + schema + config |
| 4 | `llm_extraction` | Calls the LLM engine |
| 5 | `json_parse` | Parses the raw JSON response |
| 6 | `validation` | Runs validators (non-aborting; failures are recorded) |
| 7 | `postprocessing` | Applies optional domain postprocessor |
| 8 | `archive_shaping` | Applies optional archive shaper |

## Extension points

All extension points are injected at construction time — no subclassing required:

| Parameter | Type | Purpose |
|-----------|------|---------|
| `postprocessor` | `(data) -> data` | Domain-specific field mapping and cleanup |
| `archive_shaper` | `(data) -> data` | Reshape to target archive format |
| `validators` | `list[(data) -> None]` | Non-aborting data validation; errors land in `result.stages` |
| `visualizers` | `list[(result) -> None]` | Called after every run, success or failure |
| `stage_hooks` | `list[(name, when, hook)]` | `before`/`after` hooks on any named stage |
| `schema_resolver` | `(schema) -> schema` | Transform schema during `schema_resolve` stage |
| `optimizer` on schema sources | `(schema) -> schema` | Prune or annotate schema before LLM call |

Fuller workflows for automated validation, result visualization, and agentic
prompt/schema optimization are **future work**; the hooks and callables above are the
designed extension points.

This `nomad` plugin was generated with `Cookiecutter` along with `@nomad`'s [`cookiecutter-nomad-plugin`](https://github.com/FAIRmat-NFDI/cookiecutter-nomad-plugin) template.

## Development

If you want to develop locally this plugin, clone the project and in the plugin folder, create a virtual environment (you can use Python 3.10, 3.11 or 3.12):
```sh
git clone https://github.com/FAIRmat-NFDI/nomad-llm-extraction.git
cd nomad-llm-extraction
python3.11 -m venv .pyenv
. .pyenv/bin/activate
```

Make sure to have `pip` upgraded:
```sh
pip install --upgrade pip
```

We recommend installing `uv` for fast pip installation of the packages:
```sh
pip install uv
```

Install the `nomad-lab` package:
```sh
uv pip install -e '.[dev]'
```

### Run the tests

You can run locally the tests:
```sh
python -m pytest -sv tests
```

where the `-s` and `-v` options toggle the output verbosity.

Our CI/CD pipeline produces a more comprehensive test report using the `pytest-cov` package. You can generate a local coverage report:
```sh
uv pip install pytest-cov
python -m pytest --cov=src tests
```

### Run linting and auto-formatting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting the code. Ruff auto-formatting is also a part of the GitHub workflow actions. You can run locally:
```sh
ruff check .
ruff format . --check
```

### Debugging

For interactive debugging of the tests, use `pytest` with the `--pdb` flag. We recommend using an IDE for debugging, e.g., _VSCode_. If that is the case, add the following snippet to your `.vscode/launch.json`:
```json
{
  "configurations": [
      {
        "name": "<descriptive tag>",
        "type": "debugpy",
        "request": "launch",
        "cwd": "${workspaceFolder}",
        "program": "${workspaceFolder}/.pyenv/bin/pytest",
        "justMyCode": true,
        "env": {
            "_PYTEST_RAISE": "1"
        },
        "args": [
            "-sv",
            "--pdb",
            "<path-to-plugin-tests>",
        ]
    }
  ]
}
```

where `<path-to-plugin-tests>` must be changed to the local path to the test module to be debugged.

The settings configuration file `.vscode/settings.json` automatically applies the linting and formatting upon saving the modified file.

### Documentation on Github pages

To view the documentation locally, install the related packages using:
```sh
uv pip install -r requirements_docs.txt
```

Run the documentation server:
```sh
mkdocs serve
```

## Adding this plugin to NOMAD

Currently, NOMAD has two distinct flavors that are relevant depending on your role as an user:
1. [A NOMAD Oasis](#adding-this-plugin-in-your-nomad-oasis): any user with a NOMAD Oasis instance.
2. [Local NOMAD installation and the source code of NOMAD](#adding-this-plugin-in-your-local-nomad-installation-and-the-source-code-of-nomad): internal developers.

### Adding this plugin in your NOMAD Oasis

Read the [NOMAD plugin documentation](https://nomad-lab.eu/prod/v1/staging/docs/howto/oasis/plugins_install.html) for all details on how to deploy the plugin on your NOMAD instance.

### Adding this plugin in your local NOMAD installation and the source code of NOMAD

We now recommend using the dedicated [`nomad-distro-dev`](https://github.com/FAIRmat-NFDI/nomad-distro-dev) repository to simplify the process. Please refer to that repository for detailed instructions.

### Template update

We use [`cruft`](https://github.com/cruft/cruft) to update the project based on template changes. To run the check for updates locally, run `cruft update` in the root of the project. More details see the instructions on [`cruft` website](https://cruft.github.io/cruft/#updating-a-project).

## Main contributors
| Name | E-mail     |
|------|------------|
| Sharat Patil | [patilsha@physik.hu-berlin.de](mailto:patilsha@physik.hu-berlin.de)
