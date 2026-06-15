import json

from nomad_llm_extraction.pipeline.extract import extract as extract_workflow
from nomad_llm_extraction.pipeline.extract import upload_extraction_to_nomad
from nomad_llm_extraction.utils.utils import load_yaml_config


class CLI:
    def __init__(self):
        pass
        # self.nomad = upload_extraction_to_nomad

    def extract(self, config_path: str, nomad: bool = False):
        config = load_yaml_config(config_path)
        result = extract_workflow(config)
        if result.extracted_data:
            with open(config.get('output_path', 'extraction_result.json'), 'w') as f:
                json.dump(result.extracted_data, f, indent=2)
            if nomad:
                nomad_upload_config = config.get('nomad_upload_config', {})
                upload_extraction_to_nomad(result.extracted_data, nomad_upload_config)


def main_cli():
    import fire

    fire.Fire(CLI)


if __name__ == '__main__':
    main_cli()
