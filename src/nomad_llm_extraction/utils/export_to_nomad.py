import io
import json
import os
from typing import Any

import requests
from loguru import logger

NOMAD_URL = os.getenv('NOMAD_URL', 'https://nomad-lab.eu/prod/v1/')
NOMAD_USERNAME = os.getenv('NOMAD_USERNAME', 'your_nomad_username')
NOMAD_PASSWORD = os.getenv('NOMAD_PASSWORD', 'your_nomad_password')


def get_authentication_token(
    nomad_url: str = NOMAD_URL,
    username: str = NOMAD_USERNAME,
    password: str = NOMAD_PASSWORD,
) -> str | None:
    """Get the token for accessing your NOMAD unpublished uploads remotely"""
    body = {'username': username, 'password': password, 'grant_type': 'password'}
    try:
        response = requests.post(nomad_url + 'auth/token', data=body, timeout=10)
        token = response.json().get('access_token')
        if token:
            return token

        logger.error('response is missing token: ')
        logger.error(response.json())
        return None
    except Exception:
        logger.error('something went wrong trying to get authentication token')
        return None


def push_to_nomad(
    entry_name_format: str,
    response: list[Any],
    token: str,
    upload_id: str | None = None,
):
    """Push extraction to Nomad"""
    if len(response) == 0:
        logger.warning('No extracted data instances, skipping upload to NOMAD')
        return
    for index, instance in enumerate(response):
        transformed_json = json.dumps(instance, indent=4)
        file = io.StringIO(transformed_json)
        file_name = entry_name_format.format(index=index) + '.archive.json'
        if upload_id is None:
            res = requests.post(
                f'{NOMAD_URL}uploads/',
                headers={
                    'Authorization': f'Bearer {token}',
                    'Accept': 'application/json',
                },
                params={'wait_for_processing': 'true'},
                files={'file': (file_name, file)},
                timeout=30,
            )
        else:
            res = requests.put(
                f'{NOMAD_URL}uploads/{upload_id}/raw/',
                headers={
                    'Authorization': f'Bearer {token}',
                    'Accept': 'application/json',
                },
                params={'wait_for_processing': 'true'},
                files={'file': (file_name, file)},
                timeout=30,
            )
        upload_id = res.json().get('upload_id')
        if upload_id:
            logger.info(
                f'Entry:{entry_name_format.format(index=index)} Upload Id:{upload_id} Status Code:{res.status_code}'
            )
        else:
            logger.error(
                f'Response is missing upload_id for Entry:{entry_name_format.format(index=index)}'
            )
            logger.error(f'Response:{res.json()}')
            raise Exception('Upload failed, missing upload_id in response')


def process_to_nomad(m_def, data, doi, model_name, multi_instance_field=None):
    """Wraps the transformed data in the specific NOMAD schema envelope."""
    doi_url = 'https://www.doi.org/' + doi
    # Run the transformation    output_entries = []

    # The input is usually a dict with "cells": [...]
    output_entries = []
    if multi_instance_field and multi_instance_field in data:
        for instance in data[multi_instance_field]:
            entry = {
                'data': {
                    'm_def': m_def,
                    'DOI_number': doi_url,
                    'extraction_metadata': {
                        'model': model_name,
                        'model_version': model_name,
                    },
                    **instance,  # Unpack the processed instance data here
                }
            }
            output_entries.append(entry)

    return output_entries


def upload_extraction_to_nomad(
    m_def: str,
    data: dict[str, Any],
    doi: str,
    model_name: str,
    multi_instance_field: str | None = None,
    entry_name_format: str | None = None,
    upload_id: str | None = None,
):
    token = get_authentication_token()
    if not token:
        logger.error('Failed to get authentication token, cannot upload to NOMAD')
        return

    entries = process_to_nomad(m_def, data, doi, model_name, multi_instance_field)

    entry_name_format = (
        f'{doi.replace("/", "_")}_{model_name}_' + '{{index}}'
        if entry_name_format is None
        else entry_name_format
    )
    push_to_nomad(entry_name_format, entries, token, upload_id=upload_id)
