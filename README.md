# nomad-llm-extraction

A generalized LLM extraction pipeline for structured scientific data.

`nomad-llm-extraction` turns unstructured research-paper text into structured JSON by
combining an LLM backend with a user-supplied JSON schema.  It can be used standalone
with any JSON schema, or integrated with [NOMAD](https://nomad-lab.eu/) where NOMAD
`m_def` schemas serve as the schema source and domain-specific postprocessors map the
LLM output to NOMAD archive shapes.

## Command-line extraction

Install the published package:

```sh
pip install nomad-llm-extraction
```

The installed `nomad-extract` command starts a local Temporal worker and runs the
extraction workflow. It reads `temporal.toml` by default; set
`TEMPORAL_CONFIG_PATH` to use a different Temporal client configuration.

Start with a YAML configuration such as:

```yaml
llm_engine_config:
  model_name: claude-sonnet-4-6
  api_key: YOUR_API_KEY
  api_url: null

schema_config:
  inline_schema:
    type: object
    properties:
      material:
        type: string

text: "The sample is a perovskite thin film."
system_prompt: ""
instruction_text: ""
max_retry_attempts: 3
output_path: extraction_result.json
```

`llm_engine_config.model_name` is required; `api_key` and `api_url` configure the
chosen LLM endpoint. Provide either `text`, `pdf_path`, or a prebuilt `prompt`.
Provide `extraction_schema` directly, or set `schema_config` to either an inline/schema
file configuration (`inline_schema` or `schema_path`) or a NOMAD schema configuration
(`m_def`). Schema options include `remove_defs`, `resolve_allOf`, `remove_null_anyof`,
`exclude`, and `multi_instance_field`; NOMAD schemas also accept `unit_value`.
`system_prompt`, `instruction_text`, `llm_engine_optional_params`, and
`max_retry_attempts` control the request and retries.

Run the configuration:

```sh
nomad-extract extract config.yaml
```

Compose a base configuration with an override file:

```sh
nomad-extract extract config.yaml --override_file production.yaml
```

Override individual YAML values with repeatable dotted paths. Values are parsed as YAML,
so quote strings where needed:

```sh
nomad-extract extract config.yaml \
  --set llm_engine_config.model_name=gpt-4o \
  --set max_retry_attempts=5 \
  --set 'schema_config.remove_defs=true'
```

Inspect the merged configuration without changing the source files by writing it before
the workflow runs:

```sh
nomad-extract extract config.yaml \
  --override_file production.yaml \
  --write_config effective-config.yaml
```

Common direct overrides are `--pdf_path`, `--text`, `--model_name`, `--api_url`,
`--api_key`, `--m_def`, and `--output_path`. The resulting extracted data is written as
indented JSON to `output_path` (or `extraction_result.json` when it is absent).

To also create NOMAD entries, add `--nomad` and configure `nomad_upload_config` with
at least `m_def` and `entry_name`; it may also contain `doi`, `extraction_metadata`,
`multi_instance_field`, and `upload_id`. The JSON output is written before the NOMAD
upload is attempted.

### Python API

Pass a YAML path, mapping, or `ExtractionWorkflowInput` to `extract`:

```python
from nomad_llm_extraction.pipeline.extract import extract

result = extract('config.yaml')
if result.err_message is None:
    print(result.extracted_data)
```

The returned `ExtractionWorkflowOutput` includes `extracted_data`, the LLM
`raw_output`, `err_message`, `retry_prompt`, and `retries`.

For NOMAD action installation and UI use, see the [action README](src/nomad_llm_extraction/actions/llm_extractor/README.md).
Developers extending workflow behavior should read the
[pipeline workflow guide](docs/how_to/extending_pipeline_workflows.md) and
[action workflow guide](docs/how_to/extending_action_workflows.md).

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
