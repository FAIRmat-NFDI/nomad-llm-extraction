from datetime import timedelta

import dacite
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    import json
    from typing import Any

    from nomad_llm_extraction.actions.llm_extractor.activities import (
        dump_extractions,
        get_list_of_pdfs,
        get_text_from_pdf,
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
        DEFAULT_EXTRACTION_CONFIG,
        DEFAULT_EXTRACTION_METADATA,
    )
    from nomad_llm_extraction.pipeline.workflows import (
        GeneralExtractionWorkflow,
        GeneralExtractionWorkflowInput,
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


@workflow.defn
class ExtractionActionWorkflow:
    @workflow.run
    async def run(self, data: ExtractionActionInput) -> dict:
        """
        Run this action to extract perovskite solar cells information from all PDFs in a
        project/upload.

        It first finds and processes all PDF files in the project using the specified LLM,
        then creates and processes new entries for each detected solar cell and deletes the
        source PDF files.
        """
        extraction_config = DEFAULT_EXTRACTION_CONFIG
        extraction_config['schema_config'].update({'m_def': data.extraction_m_def})
        extraction_config['llm_engine_config'].update(
            {'api_key': data.api_token, 'model': data.model}
        )
        extraction_config['extraction_metadata'].update({'model_name': data.model})
        # extraction_config.update(**data.model_dump())
        extraction_workflow_input = ExtractionWorkflowInput(
            upload_id=data.upload_id,
            user_id=data.user_id,
            text=data.text,
            delete_source_pdfs=data.delete_source_pdfs,
            **extraction_config,
        )
        # extraction_result = await workflow.execute_child_workflow(
        #     ExtractionRouterWorkflow.run,
        #     extraction_workflow_input,
        #     id=f'extraction-router-workflow::{workflow.info().workflow_id}',
        # )
        # if extraction_result['success'] is False:
        #     error_msg = (
        #         f'Extraction workflow failed with errors: {extraction_result["errors"]}'
        #     )
        #     workflow.logger.error(error_msg)
        #     return {'refs': [], 'success': False, 'errors': extraction_result['errors']}
        extraction_output_archive = {
            'data': {
                'm_def': 'nomad_llm_extraction.schema_packages.llm_extractor.LLMExtractionOutput',
                # 'extracted_data': extraction_result['refs'],
                'extracted_data': [
                    '../uploads/R8xVJJhXR3mFy4c3qiCC-g/archive/3IuiEl1dqCpBhrtWtXF10OK5FuNB#/data'
                ],
                'action_id': workflow.info().workflow_id,
            }
        }
        save_result = await workflow.execute_activity(
            save_extraction_output,
            ActionFileHandlerInput(
                upload_id=data.upload_id,
                user_id=data.user_id,
                name='extraction_output',
                data=[extraction_output_archive],
            ),
            start_to_close_timeout=timedelta(seconds=60),
        )
        if save_result['success'] is False:
            error_msg = (
                f'Failed to save extraction output with errors: {save_result["errors"]}'
            )
            workflow.logger.error(error_msg)
            return {'success': False, 'errors': save_result['errors']}
        return {'success': True, 'errors': []}


@workflow.defn
class ExtractionRouterWorkflow:
    @workflow.run
    async def run(self, data: ExtractionWorkflowInput) -> dict:
        """
        Router workflow to run the whole extraction process from PDFs in a project/upload.

        It first finds all PDF files in the upload, then runs the extraction workflow for each
        of them, and finally processes the extracted data to create new entries in NOMAD.
        """
        parent_workflow_id = workflow.info().workflow_id
        if not data.text and not data.prompt:
            result = await workflow.execute_child_workflow(
                ExtractPDFWorkflow.run,
                data,
                id=f'extract-pdf-workflow::{parent_workflow_id}',
            )
        else:
            print(
                'No PDF files to process, running extraction workflow directly with provided text or prompt.'
            )
            result = await workflow.execute_child_workflow(
                ExtractTextWorkflow.run,
                data,
                id=f'extract-text-workflow::{parent_workflow_id}',
            )
        if result['success'] is False:
            error_msg = f'Extraction workflow failed with errors: {result["errors"]}'
            workflow.logger.error(error_msg)
            return {'refs': [], 'success': False, 'errors': result['errors']}

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
                print(f'Processing extraction for {name}:', extraction)
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
                print(f'Extraction results for {name} saved to: {save_paths}')
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
            workflow.logger.error(error_msg)
            if len(error_msg) > 10000:
                error_msg = error_msg[:10000] + '... [truncated]'
            errors.append(error_msg)
            return {'refs': [], 'success': False, 'errors': errors}

        return {'refs': result_entry_refs, 'success': errors == [], 'errors': errors}


@workflow.defn
class ExtractPDFWorkflow:
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
            workflow.logger.error(error_msg)
            errors.append(error_msg)
            return {'result': {}, 'success': False, 'errors': errors}

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
                    error_msg = f'Failed to extract text from PDF: {pdf}'
                    workflow.logger.error(error_msg)
                    errors.append(error_msg)
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
                    workflow.logger.error(error_msg)
                    errors.append(error_msg)
                    continue

                extractions.update(extraction_result['result'])
        except ActivityError as e:
            error_msg = f'Extraction activity failed: {e}'
            workflow.logger.error(error_msg)
            if len(error_msg) > 10000:
                error_msg = error_msg[:10000] + '... [truncated]'
            errors.append(error_msg)
            return {'result': {}, 'success': False, 'errors': errors}

        finally:
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

        return {'result': extractions, 'success': errors == [], 'errors': errors}


@workflow.defn
class ExtractTextWorkflow:
    @workflow.run
    async def run(self, data: ExtractionWorkflowInput) -> dict:
        parent_workflow_id = workflow.info().workflow_id
        name = data.name or data.upload_id
        print(f'Running LLM extraction workflow for {name}')
        errors = []
        retry_policy = RetryPolicy(
            maximum_attempts=3,
        )
        if not data.text and not data.prompt:
            error_msg = 'No text or prompt provided.'
            workflow.logger.error(error_msg)
            errors.append(error_msg)
            return {'result': {}, 'success': False, 'errors': errors}
        extraction_workflow_input = GeneralExtractionWorkflowInput(**data.model_dump())
        if data.llm_engine_config.api_key is not None:
            extraction_workflow_input.llm_engine_config.api_key = (
                data.llm_engine_config.api_key
            )
        extraction_result = await workflow.execute_child_workflow(
            GeneralExtractionWorkflow.run,
            extraction_workflow_input,
            id=f'extraction-workflow::{parent_workflow_id}',
            retry_policy=retry_policy,
        )
        if extraction_result.err_message:
            error_msg = f'LLM extraction workflow failed with error: {extraction_result.err_message}'
            workflow.logger.error(error_msg)
            errors.append(error_msg)
            return {'result': {}, 'success': False, 'errors': errors}
        print(
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
        print(f'Processed extractions ready for NOMAD upload: {processed_extractions}')
        return {
            'result': {name: processed_extractions},
            'success': errors == [],
            'errors': errors,
        }
