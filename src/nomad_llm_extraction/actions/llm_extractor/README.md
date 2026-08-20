# LLM data extractor action

This NOMAD plugin provides the `LLMDataExtractorAction` action. Its entry point is
`nomad_llm_extraction.actions:llm_extractor_action_entry_point`, declared in the
`nomad.plugin` entry-point group. The action runs on the CPU task queue by default.

For standalone extraction, installation, and command-line configuration, see the
[root README](../../../../README.md). Workflow extension details are in the
[pipeline guide](../../../../docs/how_to/extending_pipeline_workflows.md) and the
[action guide](../../../../docs/how_to/extending_action_workflows.md).

## For NOMAD administrators

Install the package in the NOMAD environment and configure NOMAD to load its plugin
entry points. The package exposes both the action entry point above and the
`llm_extractor_schema` schema-package entry point. After NOMAD loads the plugin, the
action receives the base pipeline activities and workflows plus the action-specific
activities and workflows.

The action is implemented by `ExtractionActionWorkflow`; it is registered through the
entry point rather than by manually registering an individual workflow. The default
task queue can be changed through the entry point's `task_queue` setting.

## For users

Run the **LLMDataExtractorAction** in a NOMAD upload. Supply:

- an LLM API token;
- an extraction `m_def` identifying the NOMAD section to populate;
- a model selected from the action's supported models, or an optional free-text model
  name; and
- optionally, an API base URL.

The remaining inputs are optional:

- **Text**: when supplied, extraction runs on that text. When it is absent, the action
  discovers PDF files in the upload and processes each PDF.
- **Delete source PDFs**: enabled by default. When enabled in PDF mode, source PDFs are
  removed after processing.
- **Extract multiple instances**: enabled by default. It configures the extraction
  schema to use the `extracted_instances` multi-instance field.

In PDF mode, the action finds PDF files recursively in the upload, extracts text and a
DOI from each readable PDF, and runs one text extraction per PDF. PDFs that cannot be
read are skipped and reported as warnings; failures for individual PDFs are collected
in the action log. In text mode, no upload PDFs are inspected.

Successful extracted instances are written as archive JSON, processed into the upload's
`results` area, and returned as entry references. The action also creates an
`extraction_output.archive.json` archive containing the references and the action
inputs, excluding the upload ID, user ID, and API token. Progress and terminal success
or failure are published to NOMAD's action status stream.
