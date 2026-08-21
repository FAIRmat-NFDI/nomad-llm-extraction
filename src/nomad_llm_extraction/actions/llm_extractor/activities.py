import json
import time

from nomad.utils.structlogging import get_logger
from temporalio import activity

from nomad_llm_extraction.actions.llm_extractor.models import (
    ActionFileHandlerInput,
    CleanupInput,
    ProcessNewFilesInput,
)

MAX_ATTEMPT_NUM = 100  # attempts to reprocess upload with new entries
ACTION_NAME = 'nomad_llm_extraction_action'


@activity.defn(name=f'{ACTION_NAME}.logging')
def log_message(message: str) -> None:
    """
    Log a message to the workflow logger.
    """
    logger = get_logger(__name__)
    logger = logger.bind(workflow=activity.info().workflow_type)
    logger.info(message)


@activity.defn(name=f'{ACTION_NAME}.get_list_of_pdfs')
def get_list_of_pdfs(input_data: ActionFileHandlerInput) -> dict:
    """
    Find all PDF files in the upload if authorized user has access to the upload.
    """
    from nomad.actions.manager import get_upload_files

    logger = get_logger(__name__).bind(
        workflow=activity.info().workflow_type, activity=activity.info().activity_type
    )
    pdfs = []
    upload_files = get_upload_files(
        input_data.upload_id,
        input_data.user_id,
    )
    if upload_files is not None:
        raw_files = upload_files.raw_directory_list(
            path='',
            recursive=True,
            files_only=True,
        )
        for file_info in raw_files:
            if file_info.path.lower().endswith('.pdf'):
                pdfs.append(file_info.path)
    if len(pdfs) == 0:
        logger.error(
            f'No PDF files found in the upload with ID: {input_data.upload_id}'
        )
    else:
        logger.info(
            f'Found {len(pdfs)} PDF files in the upload with ID: {input_data.upload_id}'
        )
    return {
        'pdfs': pdfs,
    }


@activity.defn(name=f'{ACTION_NAME}.get_config')
def get_config(input_data):
    from nomad.actions.manager import get_upload_files

    from nomad_llm_extraction.utils.utils import load_yaml_config

    upload_files = get_upload_files(
        input_data.upload_id,
        input_data.user_id,
    )
    if upload_files is not None:
        raw_files = upload_files.raw_directory_list(
            path='',
            recursive=True,
            files_only=True,
        )
        for file_info in raw_files:
            if file_info.path.lower() == input_data.config.lower():
                config = load_yaml_config(
                    upload_files.raw_file_object(file_info.path).os_path
                )
                config['llm_engine_config'] = {
                    'api_key': input_data.api_token.get_secret_value(),
                    'model_name': input_data.model,
                }

                return config
    raise FileNotFoundError(
        f'Configuration file {input_data.config} not found in the upload.'
    )


@activity.defn(name=f'{ACTION_NAME}.get_text_from_pdf')
def get_text_from_pdf(
    input_data: ActionFileHandlerInput,
) -> tuple[str | None, str | None]:
    from nomad.actions.manager import get_upload_files

    from nomad_llm_extraction.pipeline.input_sources.paper import PDFParser
    from nomad_llm_extraction.utils.utils import extract_doi_from_pdf

    logger = get_logger(__name__).bind(
        workflow=activity.info().workflow_type, activity=activity.info().activity_type
    )
    upload_files = get_upload_files(
        input_data.upload_id,
        input_data.user_id,
    )
    if upload_files is None:
        error_msg = f'Upload files not found or can not be accessed for upload ID: {input_data.upload_id}'
        logger.error(error_msg)
        return None, None
    pdf_path = upload_files.raw_file_object(input_data.name).os_path
    parser = PDFParser()
    text = parser.parse_pdf(pdf_path)
    doi = extract_doi_from_pdf(pdf_path)
    if not text:
        logger.warning(f'Failed to extract text from PDF: {input_data.name}')
    return text, doi


@activity.defn(name=f'{ACTION_NAME}.dump_extractions')
async def dump_extractions(input_data: ActionFileHandlerInput):
    from nomad.actions.manager import get_upload_files

    upload_files = get_upload_files(
        input_data.upload_id,
        input_data.user_id,
    )
    temp_dir = 'temp_results'
    fname = f'{temp_dir}/{input_data.name}'
    save_paths = []
    extractions = input_data.data or []
    for index, extracted_instance in enumerate(extractions):
        if not upload_files.raw_path_exists(temp_dir):
            upload_files.raw_create_directory(temp_dir)
        with upload_files.raw_file(
            file_path=fname + f'_{index}.archive.json', mode='w', encoding='utf-8'
        ) as f:
            json.dump(extracted_instance, f, indent=4)
            save_paths.append(fname + f'_{index}.archive.json')
    return save_paths


def get_upload(upload_id: str, user_id: str):
    from nomad.actions.manager import get_upload_files
    from nomad.processing.data import Upload

    upload_files = get_upload_files(
        upload_id,
        user_id,
    )
    if upload_files is None:
        return (
            None,
            None,
            f'Upload files not found or can not be accessed for upload ID: {upload_id}',
        )
    for i in range(MAX_ATTEMPT_NUM):
        upload = Upload.get(upload_id)

        if not upload.process_running:
            break
        else:
            # reload if upload is busy
            time.sleep(0.5)
    else:
        return (
            None,
            None,
            f'Upload with ID {upload_id} is busy or could not be accessed.',
        )
    return upload, upload_files, None


@activity.defn(name=f'{ACTION_NAME}.process_new_files')
async def process_new_files(data: ProcessNewFilesInput) -> dict:
    """Process newly created entries in the upload, then return their references."""
    from nomad.utils import generate_entry_id

    logger = get_logger(__name__).bind(
        workflow=activity.info().workflow_type, activity=activity.info().activity_type
    )
    upload, upload_files, error = get_upload(data.upload_id, data.user_id)
    if error:
        logger.error(error)
        return {'refs': [], 'success': False, 'errors': [error]}

    file_operations = []
    proc_file_paths = []
    for path in data.results['paths']:
        file_operations.append(
            dict(
                op='ADD',
                path=upload_files.raw_file_object(path).os_path,
                target_dir='results',
                temporary=False,
            )
        )
        proc_file_paths.append([path, path.replace('temp_results/', 'results/')])

    handle = upload.process_upload(
        file_operations=file_operations,
        path_filter='results',
        only_updated_files=True,
    )

    await handle.result()  # type: ignore

    result_entry_refs = []
    cleaned_paths = []
    for temp_file_path, final_file_path in proc_file_paths:
        if upload_files.raw_path_exists(
            final_file_path
        ) and upload_files.raw_path_is_file(final_file_path):
            result_entry_refs.append(
                f'../uploads/{upload.upload_id}/archive/{generate_entry_id(str(upload.upload_id), final_file_path)}#/data'
            )
            if upload_files.raw_path_exists(temp_file_path):
                upload_files.delete_rawfiles(temp_file_path)
                cleaned_paths.append(temp_file_path)
    if (
        upload_files.raw_path_exists('temp_results')
        and cleaned_paths == data.results['paths']
    ):
        upload_files.delete_rawfiles('temp_results')
    return {'refs': result_entry_refs, 'success': True, 'errors': []}


@activity.defn(name=f'{ACTION_NAME}.save_extraction_output')
async def save_extraction_output(input_data: ActionFileHandlerInput) -> dict:
    """
    Save the extraction output to a JSON file in the upload.
    """

    logger = get_logger(__name__).bind(
        workflow=activity.info().workflow_type, activity=activity.info().activity_type
    )
    upload, upload_files, error = get_upload(input_data.upload_id, input_data.user_id)
    if error:
        logger.error(error)
        return {'success': False, 'errors': [error]}
    file_name = input_data.name or f'extraction_output_{int(time.time())}.json'
    output_path = f'temp/{file_name}'
    if not upload_files.raw_path_exists('temp'):
        upload_files.raw_create_directory('temp')

    with upload_files.raw_file(file_path=output_path, mode='w', encoding='utf-8') as f:
        json.dump(input_data.data[0], f, indent=4)
    file_operations = [
        dict(
            op='ADD',
            path=upload_files.raw_file_object(output_path).os_path,
            target_dir='',
            temporary=False,
        )
    ]
    handle = upload.process_upload(
        file_operations=file_operations,
        path_filter='',
        only_updated_files=True,
    )
    await handle.result()  # type: ignore
    if upload_files.raw_path_exists(file_name):
        upload_files.delete_rawfiles('temp')
        return {'success': True, 'errors': []}
    logger.error(f'Failed to save extraction output to {file_name}')
    return {
        'success': False,
        'errors': [f'Failed to save extraction output to {file_name}'],
    }


@activity.defn(name=f'{ACTION_NAME}.remove_source_pdfs')
def remove_source_pdfs(input_data: CleanupInput) -> None:
    """
    Remove source PDF files from the upload after extraction.
    """
    from nomad.actions.manager import get_upload_files

    logger = get_logger(__name__).bind(
        workflow=activity.info().workflow_type, activity=activity.info().activity_type
    )
    upload_files = get_upload_files(
        input_data.upload_id,
        input_data.user_id,
    )
    if upload_files is None:
        logger.error(
            f'Upload files not found or can not be accessed for upload ID: {input_data.upload_id}'
        )
        return

    for pdf in input_data.pdfs:
        upload_files.delete_rawfiles(
            path=pdf
        )  # Delete the PDF after extraction for copyright reasons
