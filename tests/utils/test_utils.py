from dataclasses import dataclass, field
from pathlib import Path

import pytest

from nomad_llm_extraction.utils import utils


def test_convert_to_nomad_unit_and_validate_with_schema():
    converted = utils.convert_to_nomad_unit(100, 'centimeter', 'meter')

    assert converted == {'value': 1.0, 'unit': 'meter'}
    assert utils.validate_with_schema({'value': 1}, {'type': 'object'}) == (True, None)
    valid, message = utils.validate_with_schema('wrong', {'type': 'object'})
    assert valid is False
    assert "'wrong' is not of type 'object'" in message


@pytest.mark.parametrize(
    ('pdf2doi_result', 'expected'),
    [
        (
            {'identifier_type': 'DOI', 'identifier': '10.1234/example'},
            '10.1234/example',
        ),
        ([{'identifier_type': 'DOI', 'identifier': '10.1234/list'}], '10.1234/list'),
        ({'identifier_type': 'ISBN', 'identifier': '978-0'}, 'NOT_FOUND'),
        (None, 'NOT_FOUND'),
    ],
)
def test_extract_doi_from_pdf_handles_supported_results(
    monkeypatch, pdf2doi_result, expected
):
    monkeypatch.setattr(utils.pdf2doi, 'pdf2doi', lambda path: pdf2doi_result)

    assert utils.extract_doi_from_pdf('paper.pdf') == expected


def test_extract_doi_from_pdf_returns_not_found_after_library_error(monkeypatch):
    def raise_error(path):
        raise RuntimeError('corrupt PDF')

    monkeypatch.setattr(utils.pdf2doi, 'pdf2doi', raise_error)

    assert utils.extract_doi_from_pdf('paper.pdf') == 'NOT_FOUND'


def test_get_hash_load_yaml_and_safe_json_helpers(tmp_path):
    content = tmp_path / 'content.txt'
    content.write_text('hello')
    config = tmp_path / 'config.yaml'
    config.write_text('enabled: true\nitems:\n  - one\n')

    @dataclass
    class Config:
        included: str
        excluded: str = field(metadata={'serialize': False})

    assert utils.get_hash(content) == '5d41402abc4b2a76b9719d911017c592'
    assert utils.load_yaml_config(str(config)) == {'enabled': True, 'items': ['one']}
    assert utils.safe_asdict(Config('yes', 'no')) == {'included': 'yes'}
    assert utils.get_safe_ctx({'config': Config('yes', 'no')}) == {
        'config': {'included': 'yes', 'excluded': 'no'}
    }
    assert utils.safe_json_default(None) is None
    assert '<unserializable: PosixPath >' in utils.safe_json_default(Path('file.txt'))


def test_verify_activity_signature_accepts_and_rejects_expected_parameters():
    def expected(value: 'int', enabled: 'bool'):
        return value if enabled else 0

    def missing(value: 'int'):
        return value

    def wrong_type(value: 'str'):
        return value

    assert (
        utils.verify_activity_signature(expected, {'value': int, 'enabled': bool})
        is True
    )
    with pytest.raises(TypeError, match="missing.*'enabled'"):
        utils.verify_activity_signature(missing, {'value': int, 'enabled': bool})
    with pytest.raises(TypeError, match='must be annotated as int, got str'):
        utils.verify_activity_signature(wrong_type, {'value': int})


def test_get_temporal_activities_reuses_previously_decorated_function():
    from temporalio import activity

    def plain(value):
        return value

    decorated = activity.defn(name='existing')(lambda value: value)

    activities = utils.get_temporal_activities(
        [('plain', plain), ('existing', decorated)]
    )

    assert activities[0][0] == 'plain'
    assert activities[1] == ('existing', decorated)

