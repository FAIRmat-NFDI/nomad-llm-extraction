# Extend pipeline workflows

The reusable pipeline has two Temporal workflows:

- `LLMCallWorkflow` calls the LLM, parses its JSON response, and validates that response
  against the extraction schema.
- `ExtractionWorkflow` obtains a schema when necessary, obtains or builds a prompt, and
  invokes `LLMCallWorkflow` until validation succeeds or `max_retry_attempts` is
  reached.

The workflow interfaces are defined in
`nomad_llm_extraction.pipeline.models`. `LLMCallWorkflow` accepts `LLMCallInput` and
returns `LLMCallOutput`. `ExtractionWorkflow` accepts `ExtractionWorkflowInput` and
returns `ExtractionWorkflowOutput`. Keep additions to the inputs and outputs
serializable Pydantic fields so Temporal can persist workflow state.

## Activities and schema selection

`ExtractionWorkflow` uses activities rather than direct I/O:

1. When `extraction_schema` is supplied, it uses that schema unchanged.
2. Otherwise it accepts `NomadSchemaConfig` and calls `get_nomad_schema`, or accepts
   `InlineSchemaConfig` and calls `get_inline_schema`.
3. If neither an explicit schema nor a supported `schema_config` is present, the
   workflow fails.
4. If `prompt` is absent, it uses `text`; if text is absent and `pdf_path` is set, it
   calls `parse_text_from_pdf`, then calls `build_prompt`.
5. The child LLM workflow calls `llm_call`, `json_parse`, and
   `validate_extraction_with_schema`.

The schema source classes apply the common schema options (`remove_defs`,
`resolve_allOf`, `remove_null_anyof`, `exclude`, and `multi_instance_field`).
`NomadSchemaConfig` additionally supports `unit_value`. Add a schema source by adding
a Pydantic configuration type, an activity that returns a JSON schema, and an explicit
branch in `ExtractionWorkflow`; do not make workflows read schemas or files directly.

## Safe extension boundaries

Extend prompt construction, schema generation, parsing, validation, PDF parsing, or
LLM transport as activities in `pipeline.activities`. Preserve the existing workflow
sequence and pass typed activity inputs. Activities use a three-attempt retry policy;
the LLM call has a 300-second timeout and parsing, validation, schema, and prompt
activities have 30-second timeouts.

To change retry behavior, distinguish activity retries from extraction retries:
activities retry through `DEFAULT_RETRY_POLICY`, while `ExtractionWorkflow` repeats the
child workflow and appends the previous output and validation error to `retry_prompt`.
Validation errors are returned by `LLMCallWorkflow`; transport or JSON parsing failures
raise and fail the workflow.

Register new Temporal definitions in the module that owns them. The package builds
`BASE_ACTIVITIES` and `BASE_WORKFLOWS` by inspecting
`nomad_llm_extraction.pipeline.activities` and `.workflows`, so definitions in those
modules are automatically available to the local worker and NOMAD action entry point.

## Tests

Add workflow tests in `tests/pipeline/test_workflows.py` and activity/schema tests next
to their existing pipeline tests. Exercise the schema fallback branches (explicit,
inline, NOMAD, and missing), prompt versus text versus PDF inputs, validation retry
behavior, and terminal errors. Mock activities and child workflows at their Temporal
boundaries; tests should assert the typed workflow outputs rather than implementation
logging.

For action-specific orchestration, see
[Extend action workflows](extending_action_workflows.md). User-facing CLI and API
examples are in the
[root README](https://github.com/FAIRmat-NFDI/nomad-llm-extraction/blob/main/README.md).
