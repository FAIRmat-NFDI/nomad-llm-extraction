import json
import time

from temporalio import activity

from nomad_llm_extraction.actions.llm_extractor.models import (
    ActionFileHandlerInput,
    CleanupInput,
    ExtractWorkflowInput,
    ProcessNewFilesInput,
)

MAX_ATTEMPT_NUM = 100  # attempts to reprocess upload with new entries


@activity.defn
def get_list_of_pdfs(input_data: ActionFileHandlerInput) -> dict:
    """
    Find all PDF files in the upload if authorized user has access to the upload.
    """
    from nomad.actions.manager import get_upload_files

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

    return {
        'pdfs': pdfs,
    }


@activity.defn
def get_config(input_data: ExtractWorkflowInput):
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


@activity.defn
def get_text_from_pdf(
    input_data: ActionFileHandlerInput,
) -> tuple[str | None, str | None]:
    from nomad.actions.manager import get_upload_files

    from nomad_llm_extraction.pipeline.input_sources.paper import PDFParser
    from nomad_llm_extraction.utils.utils import extract_doi_from_pdf

    upload_files = get_upload_files(
        input_data.upload_id,
        input_data.user_id,
    )
    if upload_files is None:
        error_msg = f'Upload files not found or can not be accessed for upload ID: {input_data.upload_id}'
        activity.logger.error(error_msg)
        return None, None
    pdf_path = upload_files.raw_file_object(input_data.name).os_path
    parser = PDFParser()
    text = parser.parse_pdf(pdf_path)
    doi = extract_doi_from_pdf(pdf_path)
    return text, doi


@activity.defn
def dump_extractions(input_data: ActionFileHandlerInput):
    from nomad.actions.manager import get_upload_files

    upload_files = get_upload_files(
        input_data.upload_id,
        input_data.user_id,
    )
    fname = f'results/{input_data.name}_extracted-' + '{{index}}.archive.json'
    save_paths = []
    extractions = input_data.data or []
    for index, extracted_instance in enumerate(extractions):
        if not upload_files.raw_path_exists('results'):
            upload_files.raw_create_directory('results')
        with upload_files.raw_file(
            file_path=fname.format(index=index), mode='w', encoding='utf-8'
        ) as f:
            json.dump(extracted_instance, f, indent=4)
            save_paths.append(fname.format(index=index))

    return save_paths


@activity.defn
async def process_new_files(data: ProcessNewFilesInput) -> dict:
    """Process newly created entries in the upload, then return their references."""
    from nomad.actions.manager import get_upload_files
    from nomad.app.v1.routers.uploads import get_upload_with_read_access
    from nomad.datamodel import User
    from nomad.utils import generate_entry_id

    upload_files = get_upload_files(
        data.upload_id,
        data.user_id,
    )
    if upload_files is None:
        error_msg = f'Upload files not found or can not be accessed for upload ID: {data.upload_id}'
        activity.logger.error(error_msg)
        return {'refs': [], 'success': False, 'errors': [error_msg]}

    file_operations = []

    for path in data.results['paths']:
        file_operations.append(
            dict(
                op='ADD',
                path=upload_files.raw_file_object(path).os_path,
                target_dir='results',
                temporary=False,
            )
        )

    # Wait until the upload is not busy
    for i in range(MAX_ATTEMPT_NUM):
        upload = get_upload_with_read_access(
            data.upload_id,
            User(user_id=data.user_id),
            include_others=True,
        )

        if not upload.process_running:
            break
        else:
            # reload if upload is busy
            time.sleep(0.5)
            activity.logger.warning('Upload is currently being processed. Waiting...')
    else:
        error_msg = (
            f'Upload {data.upload_id} is busy for too long. Cannot process new files.'
        )
        activity.logger.error(error_msg)
        return {'refs': [], 'success': False, 'errors': [error_msg]}

    handle = upload.process_upload(
        file_operations=file_operations,
        path_filter='results',
        only_updated_files=True,
    )

    await handle.result()  # type: ignore

    result_entry_refs = []

    for path in data.results['paths']:
        if upload_files.raw_path_exists(path) and upload_files.raw_path_is_file(path):
            result_entry_refs.append(
                f'../uploads/{upload.upload_id}/archive/{generate_entry_id(str(upload.upload_id), path)}#/data'
            )

    return {'refs': result_entry_refs, 'success': True, 'errors': []}


@activity.defn
def remove_source_pdfs(input_data: CleanupInput) -> None:
    """
    Remove source PDF files from the upload after extraction.
    """
    from nomad.actions.manager import get_upload_files

    upload_files = get_upload_files(
        input_data.upload_id,
        input_data.user_id,
    )
    if upload_files is None:
        activity.logger.error(
            f'Upload files not found or can not be accessed for upload ID: {input_data.upload_id}'
        )
        return

    for pdf in input_data.pdfs:
        upload_files.delete_rawfiles(
            path=pdf
        )  # Delete the PDF after extraction for copyright reasons
