from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import activity, workflow

with workflow.unsafe.imports_passed_through():
    from nomad_llm_extraction.pipeline.models import (
        ExtractionPipelineResult,
        PipelineResult,
        PromptConfig,
        StageContext,
        StageResult,
    )
    from nomad_llm_extraction.pipeline.stages import (
        build_prompt,
        filter_extraction,
        json_parse,
        llm_call,
        run_postprocessing,
        validate_extraction_with_schema,
    )
    from nomad_llm_extraction.pipeline.temporal_refactor_models import (
        DEFAULT_EXTRACTION_STAGE_ORDER,
        StageActivityPayload,
        StageActivityResult,
        TemporalWorkflowPayload,
    )
    from nomad_llm_extraction.pipeline.temporal_refactor_registry import (
        PipelineDefinition,
        get_pipeline_definition,
        register_pipeline_definition,
        resolve_engine,
        resolve_runtime_callable,
    )


EXTRACTION_PIPELINE_TYPE = 'extraction'
RUN_REGISTERED_STAGE_ACTIVITY = 'run_registered_pipeline_stage'


def _build_extraction_context(
    state: dict[str, Any], pipeline_config: dict[str, Any]
) -> StageContext:
    prompt_cfg_data = dict(pipeline_config.get('prompt_config', {}))
    prompt_config = PromptConfig(**prompt_cfg_data)

    engine = resolve_engine(
        pipeline_config.get('engine_ref'),
        dict(pipeline_config.get('engine_config', {})),
    )
    postprocessor = resolve_runtime_callable(
        pipeline_config.get('postprocessor_ref'),
        dict(pipeline_config.get('postprocessor_config', {})),
    )
    filter_func = resolve_runtime_callable(
        pipeline_config.get('filter_ref'),
        dict(pipeline_config.get('filter_config', {})),
    )

    return StageContext(
        text=state.get('text', ''),
        prompt_config=prompt_config,
        engine=engine,
        optional_params=dict(state.get('optional_params', {})),
        extraction_schema=state.get('extraction_schema'),
        postprocessing_schema=state.get('postprocessing_schema'),
        postprocessor_args=state.get('postprocessor_args'),
        postprocessor=postprocessor,
        postprocessed_data=state.get('postprocessed_data'),
        filter_func=filter_func,
        filter_args=state.get('filter_args'),
        prompt=state.get('prompt'),
        raw_output=state.get('raw_output'),
        extracted_data=state.get('extracted_data'),
        archive_data=state.get('archive_data'),
        metadata=dict(state.get('metadata', {})),
        filtered_data=state.get('filtered_data'),
    )


def _serialize_extraction_context(ctx: StageContext) -> dict[str, Any]:
    # Keep only JSON-safe state fields. Runtime objects (engine/callables)
    # are reconstructed per activity invocation from registry refs.
    return {
        'text': ctx.text,
        'optional_params': dict(ctx.optional_params),
        'extraction_schema': ctx.extraction_schema,
        'postprocessing_schema': ctx.postprocessing_schema,
        'postprocessor_args': ctx.postprocessor_args,
        'filter_args': ctx.filter_args,
        'postprocessed_data': ctx.postprocessed_data,
        'prompt': ctx.prompt,
        'raw_output': ctx.raw_output,
        'extracted_data': ctx.extracted_data,
        'archive_data': ctx.archive_data,
        'metadata': dict(ctx.metadata),
        'filtered_data': ctx.filtered_data,
    }


def _build_extraction_result(
    state: dict[str, Any],
    stage_results: list[StageResult],
    failed: StageResult | None,
) -> ExtractionPipelineResult:
    if failed is not None:
        return ExtractionPipelineResult(
            success=False,
            raw_llm_output=state.get('raw_output'),
            stages=stage_results,
            error=failed.error,
            ctx=state,
        )

    return ExtractionPipelineResult(
        success=True,
        raw_llm_output=state.get('raw_output'),
        extracted_data=state.get('extracted_data'),
        postprocessed_data=state.get('postprocessed_data'),
        archive_data=state.get('archive_data'),
        stages=stage_results,
        ctx=state,
    )


def register_default_temporal_refactor_pipelines() -> None:
    """Register built-in serializable pipeline definitions."""

    register_pipeline_definition(
        EXTRACTION_PIPELINE_TYPE,
        PipelineDefinition(
            stage_order=DEFAULT_EXTRACTION_STAGE_ORDER,
            stages={
                'build_prompt': build_prompt,
                'llm_call': llm_call,
                'parse_response': json_parse,
                'validate_extraction_with_schema': validate_extraction_with_schema,
                'filtering': filter_extraction,
                'postprocessing': run_postprocessing,
            },
            build_context=_build_extraction_context,
            serialize_context=_serialize_extraction_context,
            build_result=_build_extraction_result,
        ),
    )


def build_extraction_workflow_payload(
    *,
    text: str,
    extraction_schema: dict[str, Any],
    engine_ref: str,
    postprocessing_schema: dict[str, Any] | None = None,
    prompt_config: PromptConfig | None = None,
    postprocessor_ref: str | None = None,
    postprocessor_args: list[str] | None = None,
    filter_ref: str | None = None,
    filter_args: list[str] | None = None,
    engine_config: dict[str, Any] | None = None,
    postprocessor_config: dict[str, Any] | None = None,
    filter_config: dict[str, Any] | None = None,
    optional_params: dict[str, Any] | None = None,
    stage_order: list[str] | None = None,
    stage_timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Build a fully-serializable payload for RegistryPipelineWorkflow."""

    payload = TemporalWorkflowPayload(
        pipeline_type=EXTRACTION_PIPELINE_TYPE,
        pipeline_config={
            'engine_ref': engine_ref,
            'engine_config': engine_config or {},
            'prompt_config': (
                prompt_config.model_dump() if prompt_config is not None else {}
            ),
            'postprocessor_ref': postprocessor_ref,
            'postprocessor_config': postprocessor_config or {},
            'filter_ref': filter_ref,
            'filter_config': filter_config or {},
        },
        ctx_args={
            'text': text,
            'extraction_schema': extraction_schema,
            'postprocessing_schema': postprocessing_schema,
            'postprocessor_args': postprocessor_args,
            'filter_args': filter_args,
            'optional_params': optional_params or {},
        },
        stage_order=stage_order or list(DEFAULT_EXTRACTION_STAGE_ORDER),
        stage_timeout_seconds=stage_timeout_seconds,
    )
    return payload.to_payload()


@activity.defn(name=RUN_REGISTERED_STAGE_ACTIVITY)
def run_registered_pipeline_stage(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one stage for a registered pipeline type using serializable state."""

    stage_payload = StageActivityPayload.from_payload(payload)
    definition = get_pipeline_definition(stage_payload.pipeline_type)

    if stage_payload.stage_name not in definition.stages:
        raise KeyError(
            f'Unknown stage={stage_payload.stage_name!r} for '
            f'pipeline_type={stage_payload.pipeline_type!r}.'
        )

    ctx = definition.build_context(stage_payload.state, stage_payload.pipeline_config)
    stage_func = definition.stages[stage_payload.stage_name]
    stage_result = stage_func(ctx, stage_payload.stage_name)
    next_state = definition.serialize_context(ctx)

    result = StageActivityResult(
        state=next_state,
        stage_result=stage_result.model_dump(mode='json'),
    )
    return result.to_payload()


@workflow.defn(name='registry_pipeline_workflow')
class RegistryPipelineWorkflow:
    @workflow.run
    async def run(self, payload: dict[str, Any]) -> PipelineResult:
        """Workflow that supports many pipeline types via a runtime registry."""

        workflow_payload = TemporalWorkflowPayload.from_payload(payload)
        definition = get_pipeline_definition(workflow_payload.pipeline_type)

        state = dict(workflow_payload.ctx_args)
        stage_results: list[StageResult] = []
        failed: StageResult | None = None

        stage_order = (
            workflow_payload.stage_order
            if workflow_payload.stage_order
            else list(definition.stage_order)
        )

        for stage_name in stage_order:
            activity_payload = StageActivityPayload(
                pipeline_type=workflow_payload.pipeline_type,
                pipeline_config=workflow_payload.pipeline_config,
                stage_name=stage_name,
                state=state,
            )
            activity_result_payload = await workflow.execute_activity(
                RUN_REGISTERED_STAGE_ACTIVITY,
                arg=activity_payload.to_payload(),
                start_to_close_timeout=timedelta(
                    seconds=workflow_payload.stage_timeout_seconds
                ),
            )
            activity_result = StageActivityResult.from_payload(activity_result_payload)
            state = activity_result.state
            stage_result = StageResult.model_validate(activity_result.stage_result)
            stage_results.append(stage_result)

            if not stage_result.success:
                failed = stage_result
                break

        return definition.build_result(state, stage_results, failed)


register_default_temporal_refactor_pipelines()
