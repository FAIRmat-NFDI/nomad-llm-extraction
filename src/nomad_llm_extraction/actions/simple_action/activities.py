import os
import importlib
import PyPDF2
import instructor
import google.generativeai as genai
from temporalio import activity
from nomad.config import config
from pathlib import Path
import json

def _load_class_from_path(class_path: str):
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

@activity.defn
async def read_pdf_activity(upload_id: str) -> str:
    activity.logger.info(f"Automatically searching upload {upload_id} for a PDF file...")

    prefix = upload_id[:2] 
    staging_dir = config.fs.staging 
    upload_folder_path = None

    current_file_path = Path(__file__).resolve()
    for parent_dir in current_file_path.parents:
        potential_staging = parent_dir / staging_dir / prefix / upload_id
        if potential_staging.exists():
            upload_folder_path = str(potential_staging)
            break
            
    if not upload_folder_path:
        upload_folder_path = f"/Users/uday/projects/nomad-distro-dev/{staging_dir}/{prefix}/{upload_id}"

    if not os.path.exists(upload_folder_path):
        raise FileNotFoundError(f"Could not find the hashed upload folder: {upload_folder_path}")

    # searching for pdfs
    pdf_path = None
    for root, dirs, files in os.walk(upload_folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                break
        if pdf_path:
            break
            
    if not pdf_path:
        raise FileNotFoundError(f"Could not find any .pdf files inside {upload_folder_path}!")

    activity.logger.info(f"Success! Automatically found PDF at: {pdf_path}")
    
    # reading the pdf
    extracted_text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = min(3, len(reader.pages))
            for i in range(num_pages):
                page_text = reader.pages[i].extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF at {pdf_path}: {str(e)}")
        
    return extracted_text

@activity.defn
async def extract_simple_data_activity(text: str, config_dict: dict) -> str:
    ExtractionSchema = _load_class_from_path(config_dict['extraction_schema_path'])
    
    plugin_config = config.get_plugin_entry_point('nomad_llm_extraction.actions.simple_action:simple_action_entry_point')
    api_key = os.environ.get("GEMINI_API_KEY") or (plugin_config.gemini_api_key if plugin_config else None)
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable or plugin option is not set.")
    genai.configure(api_key=api_key)
    
    client = instructor.from_gemini(
        client=genai.GenerativeModel('gemini-2.5-flash'),
        mode=instructor.Mode.GEMINI_JSON
    )
    
    prompt = config_dict['extraction_prompt_template'].format(text=text)
    
    extracted_data = client.chat.completions.create(
        response_model=ExtractionSchema, 
        messages=[
            {"role": "user", "content": config_dict['system_prompt'] + "\n\n" + prompt}
        ]
    )
    
    return extracted_data.model_dump_json()

@activity.defn
async def save_extracted_data_activity(upload_id: str, extracted_json: str) -> str:
    activity.logger.info("Saving extracted data to NOMAD archive format...")
    
    prefix = upload_id[:2] 
    staging_dir = config.fs.staging
    current_file_path = Path(__file__).resolve()
    upload_folder_path = None
    
    for parent_dir in current_file_path.parents:
        potential_staging = parent_dir / staging_dir / prefix / upload_id
        if potential_staging.exists():
            upload_folder_path = str(potential_staging)
            break
            
    if not upload_folder_path:
        upload_folder_path = f"/Users/uday/projects/nomad-distro-dev/{staging_dir}/{prefix}/{upload_id}"
        
    raw_folder = os.path.join(upload_folder_path, 'raw')
    
    data = json.loads(extracted_json)
    
    archive_data = {
        "data": {
            # This links it to your schema_package.py!
            "m_def": "nomad_llm_extraction.schema_packages.schema_package.BatteryExtraction",
            **data
        }
    }
    
    file_name = "llm_results.archive.json"
    file_path = os.path.join(raw_folder, file_name)
    
    with open(file_path, 'w') as f:
        json.dump(archive_data, f, indent=4)
        
    activity.logger.info(f"Saved archive file to {file_path}")
    return file_path