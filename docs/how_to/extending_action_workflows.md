# Extend action workflows

The NOMAD action layer wraps the reusable extraction pipeline for uploads. Its public
entry point is `llm_extractor_action_entry_point`, which creates an
`LLMExtractorActionEntryPoint`. Calling `load()` returns a NOMAD `Action` with
`ExtractionActionWorkflow` as the root workflow, all pipeline and action activities,
and all pipeline and action child workflows.

## Workflow structure

`ExtractionActionWorkflow` receives `ExtractionActionInput`, applies the action default
configuration, and starts `ExtractionRouterWorkflow`.

The router selects one of two paths:

- **PDF path:** `ExtractPDFWorkflow` finds PDFs in the upload, extracts text and DOI
  values, then starts `ExtractTextWorkflow` once per readable PDF.
- **Text or prompt path:** `ExtractTextWorkflow` runs directly, without searching the
  upload for PDFs.

`ExtractTextWorkflow` adapts the action input to the pipeline's
`ExtractionWorkflowInput`, starts `ExtractionWorkflow`, and maps valid data to NOMAD
archives. `ProcessExtractionsWorkflow` writes the archive JSON, processes it into the
upload, and returns entry references. The root workflow saves a separate extraction
output archive containing those references.

## Activities, status, and failures

Action activities encapsulate NOMAD file and processing operations:

- `get_list_of_pdfs` and `get_text_from_pdf` access upload files;
- `dump_extractions`, `process_new_files`, and `save_extraction_output` create and
  process generated archives;
- `remove_source_pdfs` performs optional cleanup; and
- `log_message` records collected PDF failures.

Workflows use Temporal activities with explicit timeouts and generally a three-attempt
retry policy. Keep NOMAD API calls, file access, and blocking work inside activities;
workflows should orchestrate typed inputs and results only.

Action workflow and activity result contracts should stay consistent: expected
operational failures are returned as dictionaries with
`success: false` and `errors: list[str]` so downstream workflows and the NOMAD GUI
can propagate those messages directly. Do not collapse nested error lists into a
single formatted string.

For unexpected workflow exceptions, log a full traceback and raise a
non-retryable `ApplicationError` so the workflow fails explicitly instead of
returning success-shaped fallbacks. Retryable infrastructure failures (for
example transient activity-level API or storage issues) should keep propagating
through Temporal activity boundaries so `RetryPolicy` can retry them.

The root, router, PDF, and processing workflows publish NOMAD `ActionStreamEvent`
objects on `ACTION_STREAM_TOPIC`. Preserve a terminal state event for both completion
and failure. Use non-terminal message events for progress, warnings for partial PDF
failures, and error severity for terminal failures so the NOMAD action UI receives an
accurate status stream.

## Extension points

To add an action input, extend `ExtractionActionInput` and carry it through the action
configuration or workflow input deliberately. To support another upload source, add a
child workflow selected by `ExtractionRouterWorkflow`; do not overload the PDF branch.
To alter generated entries, extend the processing boundary after
`ExtractionWorkflow` returns valid data and before `dump_extractions` writes archives.

When adding an activity or workflow, place it in the action `activities.py` or
`workflows.py`. The entry point discovers Temporal-decorated definitions from both
modules and combines them with the base pipeline definitions. Retain the entry point
and action metadata so NOMAD continues to load the plugin through its declared
`nomad.plugin` entry point.

## Tests

Test the router's PDF/text selection, no-PDF result, unreadable-PDF warning,
per-PDF partial failures, cleanup setting, processing results, and terminal status
events. Mock NOMAD file operations as activities and assert child-workflow inputs and
returned references. Pipeline extraction semantics are covered separately in
[Extend pipeline workflows](extending_pipeline_workflows.md).

For installation and UI inputs, read the
[LLM data extractor action README](https://github.com/FAIRmat-NFDI/nomad-llm-extraction/blob/main/src/nomad_llm_extraction/actions/llm_extractor/README.md).
