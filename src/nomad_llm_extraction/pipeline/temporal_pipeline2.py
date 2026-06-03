# @runtime_checkable
# class StageFunc(Protocol):
#     def __call__(self, ctx, stage_name: str) -> StageResult: ...
from __future__ import annotations

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    import logging
    from collections.abc import Callable
    from dataclasses import asdict, dataclass, field
    from typing import Any

    from nomad_llm_extraction.pipeline.models import (
        DataclassInstance,
        DefaultStageContext,
        ExtractionPipelineResult,
        LLMEngine,
        PipelineResult,
        PromptConfig,
        StageContext,
        StageFunc,
        StageHookSpec,
        StageResult,
    )
    from nomad_llm_extraction.pipeline.registry_config import (
        CLASS_OBJ_REGISTRY,
        FUNCTION_REGISTRY,
        STAGES_REGISTRY,
    )
    from nomad_llm_extraction.pipeline.stages import (
        build_prompt,
        filter_extraction,
        json_parse,
        llm_call,
        run_postprocessing,
        validate_extraction_with_schema,
    )
    from nomad_llm_extraction.utils.utils import (
        get_temporal_activities,
        verify_activity_signature,
    )

    logger = logging.getLogger(__name__)
    from datetime import timedelta


@dataclass
class ExtractionPipelineConfig:
    pipeline_type: str
    construct_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineWorkflowInput:
    pipeline_config: ExtractionPipelineConfig
    ctx_args: dict[str, Any] = field(default_factory=dict)


def get_registered_stage_func(stage_func_name: str) -> StageFunc:
    if stage_func_name not in STAGES_REGISTRY:
        raise KeyError(
            f'Unknown stage function name: {stage_func_name}. '
            f'Available stages: {list(STAGES_REGISTRY.keys())}'
        )
    return STAGES_REGISTRY[stage_func_name]


def get_registered_processor_func(proc_func_name: str) -> Callable[..., Any]:
    if proc_func_name not in FUNCTION_REGISTRY:
        raise KeyError(
            f'Unknown processor function name: {proc_func_name}. '
            f'Available functions: {list(FUNCTION_REGISTRY.keys())}'
        )
    return FUNCTION_REGISTRY[proc_func_name]


def register_pipeline_definition(
    pipeline_type: str, pipeline_cls: type[TemporalPipeline]
):
    if pipeline_type in PIPELINE_REGISTRY:
        raise KeyError(f'Pipeline type {pipeline_type} is already registered')
    PIPELINE_REGISTRY[pipeline_type] = pipeline_cls


def register_stage_func(stage_func_name: str, stage_func: StageFunc):
    if stage_func_name in STAGES_REGISTRY:
        raise KeyError(f'Stage function name {stage_func_name} is already registered')
    verify_activity_signature(
        stage_func, expected_params={'ctx': StageContext, 'stage_name': str}
    )
    STAGES_REGISTRY[stage_func_name] = stage_func


def register_processor_func(proc_func_name: str, proc_func: Callable[..., Any]):
    if proc_func_name in FUNCTION_REGISTRY:
        raise KeyError(
            f'Processor function name {proc_func_name} is already registered'
        )
    FUNCTION_REGISTRY[proc_func_name] = proc_func
    print(f'Registered processor function: {proc_func_name} in {FUNCTION_REGISTRY}')


def register_class_obj(obj_name: str, obj: Any):
    if obj_name in CLASS_OBJ_REGISTRY:
        raise KeyError(f'Class/object name {obj_name} is already registered')
    CLASS_OBJ_REGISTRY[obj_name] = obj
    print(f'Registered class/object: {obj_name} in {CLASS_OBJ_REGISTRY}')


class TemporalPipeline:
    def __init__(
        self,
        objclasses: dict[str, Any] | None = None,
        stages: dict[str, str] | None = None,
        processors: dict[str, str] | None = None,
        ctx_factory: type[DataclassInstance] | None = None,
    ):
        stages = stages or {}
        processors = processors or {}
        objclasses = objclasses or {}
        self.stages = self.get_stage_functions(stages)
        self.processors = self.get_processors(processors)
        for proc_name, proc_func in self.processors.items():
            setattr(self, proc_name, proc_func)
        self.objects = self.get_objects(objclasses)
        for obj_name, obj in self.objects.items():
            setattr(self, obj_name, obj)
        self.ctx_factory = ctx_factory or DefaultStageContext

    # def get_objects(self, objclasses: dict[str, Any]) -> dict[str, Any]:
    #     objects = {}
    #     for obj_name, [obj_class_name, init_args] in objclasses.items():
    #         if obj_class_name in CLASS_OBJ_REGISTRY:
    #             objects[obj_name] = CLASS_OBJ_REGISTRY[obj_class_name](**init_args)
    #     return objects

    def get_objects(self, objclasses: dict[str, str]) -> dict[str, Any]:
        objects = {}
        for obj_name, obj_class_name in objclasses.items():
            if obj_class_name in CLASS_OBJ_REGISTRY:
                objects[obj_name] = CLASS_OBJ_REGISTRY[obj_class_name]
        return objects

    def get_stage_functions(self, stages: dict[str, str]) -> dict[str, StageFunc]:
        stage_funcs = {
            stage_name: get_registered_stage_func(stage_func_name)
            for stage_name, stage_func_name in stages.items()
        }

        for stage_name, stage_func in stage_funcs.items():
            verify_activity_signature(
                stage_func, expected_params={'ctx': StageContext, 'stage_name': str}
            )
        return stage_funcs

    def get_processors(
        self, processors: dict[str, str]
    ) -> dict[str, Callable[..., Any]]:
        proc_funcs = {
            proc_name: get_registered_processor_func(proc_func_name)
            for proc_name, proc_func_name in processors.items()
        }
        return proc_funcs

    def get_ctx(self, ctx_args: dict[str, Any]) -> Any:
        return self.ctx_factory(**ctx_args)

    def get_result(self, ctx, stage_results, failed) -> PipelineResult:
        if failed is not None:
            result = PipelineResult(
                success=False,
                stages=stage_results,
                error=failed.error,
                ctx=asdict(ctx),
            )
        else:
            result = PipelineResult(
                success=True,
                stages=stage_results,
                ctx=asdict(ctx),
            )
        return result

    def get_stage_activities(self) -> dict[str, Callable[..., Any]]:
        return get_temporal_activities(self.stages)


class TemporalExtractionPipeline(TemporalPipeline):
    def __init__(
        self,
        extraction_schema: dict[str, Any],
        objclasses: dict[str, Any] = {'engine': ['LLMEngine', {}]},
        postprocessing_schema: dict[str, Any] | None = None,
        prompt_config: PromptConfig | None = None,
        postprocessor_args: list[str] | None = None,
        filter_args: list[str] | None = None,
        processors: dict[str, str] | None = None,
    ) -> None:
        self.extraction_schema = extraction_schema
        self.postprocessing_schema = postprocessing_schema
        self.prompt_config = prompt_config or PromptConfig()
        self.postprocessor_args = postprocessor_args
        self.filter_args = filter_args
        stages = {
            'build_prompt': 'build_prompt',
            'llm_call': 'llm_call',
            'parse_response': 'json_parse',
            'validate_extraction_with_schema': 'validate_extraction_with_schema',
            'filtering': 'filter_extraction',
            'postprocessing': 'run_postprocessing',
        }
        super().__init__(
            objclasses=objclasses,
            stages=stages,
            processors=processors,
        )
        # self.stages = self.get_stage_functions(stages)
        # self.processors = self.get_processors(processors or {})
        # self.postprocessor = self.processors.get('postprocessor')
        # self.filter_func = self.processors.get('filter_func')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_ctx(self, ctx_args: dict[str, Any]):
        """Create a stage context for the extraction pipeline.

        Args:
            text: The input text to process.

        Returns:
            A :class:`StageContext` instance.
        """
        return StageContext(
            text=ctx_args.get('text', ''),
            engine=self.engine,
            extraction_schema=self.extraction_schema,
            postprocessing_schema=self.postprocessing_schema,
            postprocessor=self.postprocessor,
            postprocessor_args=self.postprocessor_args,
            filter_func=self.filter_func,
            filter_args=self.filter_args,
            prompt_config=self.prompt_config,
        )

    def get_result(self, ctx, stage_results, failed) -> ExtractionPipelineResult:
        if failed is not None:
            result = ExtractionPipelineResult(
                success=False,
                raw_llm_output=ctx.raw_output,
                stages=stage_results,
                error=failed.error,
                ctx=asdict(ctx),
            )
        else:
            result = ExtractionPipelineResult(
                success=True,
                raw_llm_output=ctx.raw_output,
                extracted_data=ctx.extracted_data,
                postprocessed_data=ctx.postprocessed_data,
                archive_data=ctx.archive_data,
                stages=stage_results,
                ctx=asdict(ctx),
            )
        return result


PIPELINE_REGISTRY: dict[str, type[TemporalPipeline]] = {
    'temporal_extraction_pipeline': TemporalExtractionPipeline
}


def build_extraction_pipeline(pipeline_config: ExtractionPipelineConfig):
    pipeline_type = pipeline_config.pipeline_type
    if pipeline_type not in PIPELINE_REGISTRY:
        raise ValueError(f'Unknown pipeline type: {pipeline_type}')
    return PIPELINE_REGISTRY[pipeline_type](**pipeline_config.construct_args)


@workflow.defn(name='pipeline_workflow')
class PipelineWorkflow:
    @workflow.run
    async def run(self, input: PipelineWorkflowInput) -> PipelineResult:
        """Run all stages in order, returning a list of :class:`StageResult`.

        Returns early after the first failing stage.  Before- and after-hooks
        for the failing stage are still fired.
        """
        pipeline_config = input.pipeline_config
        pipeline = build_extraction_pipeline(pipeline_config)
        ctx_args = input.ctx_args
        stage_results = []
        failed = None
        stages = pipeline.get_stage_activities()
        ctx = pipeline.get_ctx(ctx_args)
        for stage_name, stage_func in stages.items():
            stage_result: StageResult = await workflow.execute_activity(
                stage_func,
                arg=ctx,
                start_to_close_timeout=timedelta(seconds=30),
            )
            stage_results.append(stage_result)

            if not stage_result.success:
                logger.error(
                    f"Stage '{stage_name}' failed with error: {stage_result.error}"
                )
                failed = stage_result
                break
            logger.info(f"Stage '{stage_name}' succeeded")
        return pipeline.get_result(ctx, stage_results, failed)
