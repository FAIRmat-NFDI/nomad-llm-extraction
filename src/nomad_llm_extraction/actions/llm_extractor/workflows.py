from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from copy import deepcopy
    from dataclasses import dataclass

    from nomad.actions.models import ActionStreamEventSeverity, ActionStreamEventType
    from nomad.actions.streams import ACTION_STREAM_TOPIC, ActionStreamEvent
    from nomad.utils import get_logger
    from temporalio.contrib.workflow_streams import WorkflowStream

    from nomad_llm_extraction.actions.llm_extractor.activities import (
        dump_extractions,
        get_list_of_pdfs,
        get_text_from_pdf,
        log_message,
        process_new_files,
        remove_source_pdfs,
        save_extraction_output,
    )
    from nomad_llm_extraction.actions.llm_extractor.models import (
        ActionFileHandlerInput,
        CleanupInput,
        ExtractionActionInput,
        ExtractionWorkflowInput,
        ProcessNewFilesInput,
    )
    from nomad_llm_extraction.actions.llm_extractor.utils import (
        create_extraction_metadata,
    )
    from nomad_llm_extraction.config import (
        DEFAULT_EXTRACTION_ACTION_CONFIG as DEFAULT_CONFIG,
    )
    from nomad_llm_extraction.pipeline.workflows import ExtractionWorkflow
    from nomad_llm_extraction.pipeline.workflows import (
        ExtractionWorkflowInput as PipelineExtractionWorkflowInput,
    )
    from nomad_llm_extraction.utils.export_to_nomad import process_to_nomad

testing_result = {
    'result': {
        'catalyst': [
            {
                'data': {
                    'm_def': 'nomad.datamodel.results.Catalyst',
                    'catalyst_name': 'Pt-Ru/Al2O3-X5',
                    'preparation_method': 'wet impregnation',
                    'catalyst_type': ['supported metal'],
                    'support': 'gamma-alumina',
                    'characterization_methods': [
                        'X-ray diffraction (XRD)',
                        'scanning electron microscopy (SEM)',
                        'Brunauer-Emmett-Teller (BET)',
                    ],
                    'surface_area': 145.5,
                }
            }
        ]
    }
}

logger = get_logger(__name__)


@dataclass
class StatusEvent:
    state: str
    progress: int = 0
    detail: str = ''


@workflow.defn
class ExtractionActionWorkflow:
    @workflow.init
    def __init__(self, data: ExtractionActionInput):
        self.stream = WorkflowStream()
        self.events = self.stream.topic(ACTION_STREAM_TOPIC, type=ActionStreamEvent)

    @workflow.run
    async def run(self, data: ExtractionActionInput) -> dict:
        """
        Run this action to extract perovskite solar cells information from all PDFs in a
        project/upload.

        It first finds and processes all PDF files in the project using the specified LLM,
        then creates and processes new entries for each detected solar cell and deletes the
        source PDF files.
        """
        self.events.publish(
            ActionStreamEvent(
                type=ActionStreamEventType.MESSAGE,
                name='workflow_started',
                message='workflow started.',
                severity=ActionStreamEventSeverity.INFO,
                timestamp=workflow.now(),
            )
        )
        workflow.logger.info(
            f'Running LLM extraction action workflow for upload {data.upload_id}'
        )
        extraction_config = deepcopy(DEFAULT_CONFIG)
        if data.extract_multiple_instances:
            multi_instance_field = 'extracted_instances'
        else:
            multi_instance_field = None
        extraction_config['schema_config'].update(
            {
                'm_def': data.extraction_m_def,
                'multi_instance_field': multi_instance_field,
            }
        )
        extraction_config['llm_engine_config'].update(
            {
                'api_key': data.api_token,
                'model_name': data.model_technical_name,
                'api_url': data.api_base_url,
            }
        )
        extraction_config['extraction_metadata'].update(
            {'model_name': data.model_name or data.model}
        )
        extraction_workflow_input = ExtractionWorkflowInput(
            upload_id=data.upload_id,
            user_id=data.user_id,
            text=data.text,
            delete_source_pdfs=data.delete_source_pdfs,
            **extraction_config,
        )
        extraction_result = await workflow.execute_child_workflow(
            ExtractionRouterWorkflow.run,
            extraction_workflow_input,
            id=f'extraction-router-workflow::{workflow.info().workflow_id}',
        )
        if extraction_result['success'] is False:
            error_msg = (
                f'Extraction workflow failed with errors: {extraction_result["errors"]}'
            )
            result = {'success': False, 'errors': error_msg}
        else:
            self.events.publish(
                ActionStreamEvent(
                    type=ActionStreamEventType.MESSAGE,
                    name='extraction_completed',
                    message='Extractions completed.',
                    severity=ActionStreamEventSeverity.INFO,
                    timestamp=workflow.now(),
                )
            )
            extraction_output_archive = {
                'data': {
                    'm_def': 'nomad_llm_extraction.schema_packages.llm_extractor.LLMExtractionOutput',
                    'extracted_data': extraction_result['refs'],
                    'action_id': workflow.info().workflow_id,
                    'input_data': data.model_dump(),
                }
            }
            for i in ['upload_id', 'user_id', 'api_token']:
                extraction_output_archive['data']['input_data'].pop(i, None)
            save_result = await workflow.execute_activity(
                save_extraction_output,
                ActionFileHandlerInput(
                    upload_id=data.upload_id,
                    user_id=data.user_id,
                    name=f'{workflow.info().workflow_id.replace(".actions:llm_extractor_action_entry_point", "")}',
                    data=[extraction_output_archive],
                ),
                start_to_close_timeout=timedelta(seconds=60),
            )
            if save_result['success'] is False:
                error_msg = f'Failed to save extraction output with errors: {save_result["errors"]}'
                result = {'success': False, 'errors': error_msg}
            result = {'success': True, 'extraction_entries': extraction_result['refs']}
        if not result['success'] and result.get('errors'):
            self.events.publish(
                ActionStreamEvent(
                    type=ActionStreamEventType.STATE,
                    name='extraction_failed',
                    message=f'Extraction workflow failed with errors: {result["errors"]}',
                    severity=ActionStreamEventSeverity.ERROR,
                    timestamp=workflow.now(),
                    terminal=True,
                )
            )
            await workflow.execute_activity(
                log_message,
                str(result['errors']),
                start_to_close_timeout=timedelta(seconds=60),
            )
            return result
        workflow.logger.info(
            f'LLM Extraction action workflow completed with result: {result}'
        )
        self.events.publish(
            ActionStreamEvent(
                type=ActionStreamEventType.STATE,
                name='extraction_completed',
                message='Extraction workflow completed successfully.',
                severity=ActionStreamEventSeverity.SUCCESS,
                timestamp=workflow.now(),
                terminal=True,
            )
        )
        return result


@workflow.defn
class ExtractionRouterWorkflow:
    @workflow.init
    def __init__(self, data: ExtractionWorkflowInput):
        self.stream = WorkflowStream()
        self.events = self.stream.topic(ACTION_STREAM_TOPIC, type=ActionStreamEvent)

    @workflow.run
    async def run(self, data: ExtractionWorkflowInput) -> dict:
        """
        Router workflow to run the whole extraction process from PDFs in a project/upload.

        It first finds all PDF files in the upload, then runs the extraction workflow for each
        of them, and finally processes the extracted data to create new entries in NOMAD.
        """
        parent_workflow_id = workflow.info().workflow_id
        if not data.text and not data.prompt:
            self.events.publish(
                ActionStreamEvent(
                    type=ActionStreamEventType.MESSAGE,
                    name='finding_pdfs',
                    message='Finding PDF files in the upload.',
                    severity=ActionStreamEventSeverity.INFO,
                    timestamp=workflow.now(),
                )
            )
            result = await workflow.execute_child_workflow(
                ExtractPDFWorkflow.run,
                data,
                id=f'extract-pdf-workflow::{parent_workflow_id}',
            )
        else:
            self.events.publish(
                ActionStreamEvent(
                    type=ActionStreamEventType.MESSAGE,
                    name='extracting_text',
                    message='Extracting from provided text.',
                    severity=ActionStreamEventSeverity.INFO,
                    timestamp=workflow.now(),
                )
            )
            result = await workflow.execute_child_workflow(
                ExtractTextWorkflow.run,
                data,
                id=f'extract-text-workflow::{parent_workflow_id}',
            )
        if result['success'] is False:
            if result['result'] == {}:
                return {'refs': [], 'success': False, 'errors': result['errors']}
            else:
                self.events.publish(
                    ActionStreamEvent(
                        type=ActionStreamEventType.MESSAGE,
                        name='extraction_partial_success',
                        message='Extraction failed for some PDFs with errors (check logs for details).',
                        severity=ActionStreamEventSeverity.WARNING,
                        timestamp=workflow.now(),
                    )
                )

        processing_input = ProcessNewFilesInput(
            upload_id=data.upload_id,
            user_id=data.user_id,
            results=result['result'],
        )
        proccesing_result = await workflow.execute_child_workflow(
            ProcessExtractionsWorkflow.run,
            processing_input,
            id=f'process-extractions-workflow::{parent_workflow_id}',
        )
        return proccesing_result


@workflow.defn
class ProcessExtractionsWorkflow:
    @workflow.init
    def __init__(self, data: ProcessNewFilesInput):
        self.stream = WorkflowStream()
        self.events = self.stream.topic(ACTION_STREAM_TOPIC, type=ActionStreamEvent)

    @workflow.run
    async def run(self, data: ProcessNewFilesInput) -> dict:
        """
        Workflow to process the extracted data and create new entries in NOMAD.
        """
        errors = []
        retry_policy = RetryPolicy(
            maximum_attempts=3,
        )
        result_paths = []
        try:
            for name, extraction in data.results.items():
                workflow.logger.info(f'Processing extraction for {name}: {extraction}')
                save_paths = await workflow.execute_activity(
                    dump_extractions,
                    ActionFileHandlerInput(
                        upload_id=data.upload_id,
                        user_id=data.user_id,
                        name=f'{name}_extracted',
                        data=extraction,
                    ),
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=retry_policy,
                )
                workflow.logger.info(
                    f'Extraction results for {name} saved to: {save_paths}'
                )
            result_paths.extend(save_paths)
            input_for_processing = ProcessNewFilesInput(
                upload_id=data.upload_id,
                user_id=data.user_id,
                results={'paths': result_paths},
            )
            processing_result = await workflow.execute_activity(
                process_new_files,
                input_for_processing,
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=retry_policy,
            )
            result_entry_refs = processing_result['refs']
            if not processing_result['success']:
                errors.extend(processing_result['errors'])
        except ActivityError as e:
            error_msg = f'Extraction activity failed: {e}'
            if len(error_msg) > 10000:
                error_msg = error_msg[:10000] + '... [truncated]'
            errors.append(error_msg)
            return {'refs': [], 'success': False, 'errors': errors}

        return {'refs': result_entry_refs, 'success': errors == [], 'errors': errors}


@workflow.defn
class ExtractPDFWorkflow:
    @workflow.init
    def __init__(self, data: ExtractionWorkflowInput):
        self.stream = WorkflowStream()
        self.events = self.stream.topic(ACTION_STREAM_TOPIC, type=ActionStreamEvent)

    @workflow.run
    async def run(self, data: ExtractionWorkflowInput) -> dict:
        parent_workflow_id = workflow.info().workflow_id
        errors = []
        extractions = {}
        retry_policy = RetryPolicy(
            maximum_attempts=3,
        )
        action_metadata = ActionFileHandlerInput(
            upload_id=data.upload_id, user_id=data.user_id
        )
        list_of_pdfs = await workflow.execute_activity(
            get_list_of_pdfs,
            action_metadata,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=retry_policy,
        )
        if not list_of_pdfs['pdfs']:
            error_msg = 'No PDF files found in the upload.'
            errors.append(error_msg)
            return {'result': {}, 'success': False, 'errors': errors}
        pdf_errors = []
        try:
            for pdf in list_of_pdfs['pdfs']:
                parsed_pdf = await workflow.execute_activity(
                    get_text_from_pdf,
                    ActionFileHandlerInput(
                        upload_id=data.upload_id, user_id=data.user_id, name=pdf
                    ),
                    start_to_close_timeout=timedelta(seconds=120),
                    retry_policy=retry_policy,
                )
                pdf_text, doi = parsed_pdf
                if not pdf_text:
                    self.events.publish(
                        ActionStreamEvent(
                            type=ActionStreamEventType.MESSAGE,
                            name='pdf_text_extraction_failed',
                            message=f'Failed to read text from PDF: {pdf}',
                            severity=ActionStreamEventSeverity.WARNING,
                            timestamp=workflow.now(),
                        )
                    )
                    continue
                single_extraction_input = data.model_copy(
                    update={
                        'name': pdf.rsplit('.', 1)[0],
                        'text': pdf_text,
                        'extraction_metadata': {'doi': doi},
                    },
                    deep=True,
                )
                extraction_result = await workflow.execute_child_workflow(
                    ExtractTextWorkflow.run,
                    single_extraction_input,
                    id=f'extraction-workflow-{pdf}::{parent_workflow_id}',
                )
                if extraction_result['success'] is False:
                    error_msg = f'LLM extraction workflow failed for PDF {pdf} with errors: {extraction_result["errors"]}'
                    pdf_errors.append(error_msg)
                    continue
                else:
                    self.events.publish(
                        ActionStreamEvent(
                            type=ActionStreamEventType.MESSAGE,
                            name='pdf_extraction_success',
                            message=f'Successfully extracted data from PDF: {pdf}',
                            severity=ActionStreamEventSeverity.INFO,
                            timestamp=workflow.now(),
                        )
                    )
                extractions.update(extraction_result['result'])

        except ActivityError as e:
            error_msg = f'Extraction activity failed: {e}'
            if len(error_msg) > 10000:
                error_msg = error_msg[:10000] + '... [truncated]'
            errors.append(error_msg)

        finally:
            if data.delete_source_pdfs and list_of_pdfs['pdfs']:
                await workflow.execute_activity(
                    remove_source_pdfs,
                    CleanupInput(
                        upload_id=data.upload_id,
                        user_id=data.user_id,
                        pdfs=list_of_pdfs['pdfs'],
                    ),
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=retry_policy,
                )
            if pdf_errors:
                await workflow.execute_activity(
                    log_message,
                    '\n'.join(pdf_errors),
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=retry_policy,
                )
        success = errors == [] and not pdf_errors
        return {'result': extractions, 'success': success, 'errors': errors}


@workflow.defn
class ExtractTextWorkflow:
    @workflow.run
    async def run(self, data: ExtractionWorkflowInput) -> dict:
        parent_workflow_id = workflow.info().workflow_id
        name = data.name or data.upload_id
        workflow.logger.info(f'Running LLM extraction workflow for {name}')
        errors = []
        retry_policy = RetryPolicy(
            maximum_attempts=3,
        )
        if not data.text and not data.prompt:
            error_msg = 'No text or prompt provided.'
            workflow.logger.error(error_msg)
            errors.append(error_msg)
            return {'result': {}, 'success': False, 'errors': errors}
        extraction_workflow_input = PipelineExtractionWorkflowInput(**data.model_dump())
        if data.llm_engine_config.api_key is not None:
            extraction_workflow_input.llm_engine_config.api_key = (
                data.llm_engine_config.api_key
            )
        extraction_result = await workflow.execute_child_workflow(
            ExtractionWorkflow.run,
            extraction_workflow_input,
            id=f'extraction-workflow::{parent_workflow_id}',
            retry_policy=retry_policy,
        )
        if extraction_result.err_message:
            error_msg = extraction_result.err_message
            workflow.logger.error(error_msg)
            errors.append(error_msg)
            return {'result': {}, 'success': False, 'errors': errors}
        workflow.logger.info(
            f'LLM Extraction workflow completed successfully: {extraction_result.extracted_data}'
        )
        extraction_metadata = create_extraction_metadata(
            data.llm_engine_config.model_name, data.extraction_metadata
        )
        processed_extractions = process_to_nomad(
            m_def=extraction_workflow_input.schema_config.m_def,
            data=extraction_result.extracted_data,
            doi=extraction_metadata.get('doi'),
            multi_instance_field=extraction_workflow_input.schema_config.multi_instance_field,
            extraction_metadata=extraction_metadata,
        )
        workflow.logger.info(
            f'Processed extractions ready for NOMAD upload: {processed_extractions}'
        )
        return {
            'result': {name: processed_extractions},
            'success': errors == [],
            'errors': errors,
        }
