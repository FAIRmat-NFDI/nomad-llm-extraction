import json
from typing import Any

from nomad_llm_extraction.pipeline.extract import (
    ConfigError,
    build_effective_config,
    upload_extraction_to_nomad,
    validate_workflow_config,
    write_yaml_config,
)
from nomad_llm_extraction.pipeline.extract import extract as extract_workflow


class CLI:
    def extract(
        self,
        config_path: str,
        override_file: str | None = None,
        pdf_path: str | None = None,
        text: str | None = None,
        model_name: str | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
        m_def: str | None = None,
        output_path: str | None = None,
        llm_extraction_method: str | None = None,
        write_config: str | None = None,
        nomad: bool = False,
        set_values: str | list[str] | None = None,
        **kwargs: Any,
    ):
        direct_set_values = kwargs.pop('set', None)
        if direct_set_values is not None:
            if set_values is not None:
                raise ConfigError('Use either --set or --set_values, not both.')
            set_values = direct_set_values
        if kwargs:
            unknown = ', '.join(sorted(kwargs))
            raise ConfigError(f'Unknown extract option(s): {unknown}')
        config = build_effective_config(
            config_path,
            override_file=override_file,
            pdf_path=pdf_path,
            text=text,
            model_name=model_name,
            api_url=api_url,
            api_key=api_key,
            m_def=m_def,
            output_path=output_path,
            llm_extraction_method=llm_extraction_method,
            set_values=set_values,
        )
        workflow_input = validate_workflow_config(config)
        if write_config is not None:
            write_yaml_config(config, write_config)
        result = extract_workflow(workflow_input)
        if result.extracted_data is not None:
            with open(config.get('output_path', 'extraction_result.json'), 'w') as f:
                json.dump(result.extracted_data, f, indent=2)
            if nomad:
                nomad_upload_config = config.get('nomad_upload_config', {})
                upload_extraction_to_nomad(result.extracted_data, nomad_upload_config)
        return result


def collect_repeatable_set_flags(arguments: list[str]) -> list[str]:
    remaining_arguments = []
    settings = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == '--':
            remaining_arguments.extend(arguments[index:])
            break
        if argument == '--set':
            if index + 1 == len(arguments):
                raise ConfigError('--set requires a path=value argument.')
            settings.append(arguments[index + 1])
            index += 2
            continue
        if argument.startswith('--set='):
            settings.append(argument.removeprefix('--set='))
        else:
            remaining_arguments.append(argument)
        index += 1
    if settings:
        try:
            separator_index = remaining_arguments.index('--')
        except ValueError:
            separator_index = len(remaining_arguments)
        remaining_arguments[separator_index:separator_index] = [
            '--set_values',
            repr(settings),
        ]
    return remaining_arguments


def main_cli():
    import sys

    import fire

    fire.Fire(CLI, command=collect_repeatable_set_flags(sys.argv[1:]))


if __name__ == '__main__':
    main_cli()
