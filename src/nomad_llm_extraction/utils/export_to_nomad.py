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
