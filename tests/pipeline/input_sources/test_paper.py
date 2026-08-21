from pathlib import Path

import pytest

from nomad_llm_extraction.pipeline.input_sources import paper


def test_pymupdf_parse_concatenates_text_from_each_page(monkeypatch):
    class Page:
        def __init__(self, text):
            self.text = text

        def get_text(self):
            return self.text

    monkeypatch.setattr(
        paper.pymupdf,
        'open',
        lambda path: [Page('first page'), Page('second page')],
    )

    assert paper.pymupdf_parse('paper.pdf') == 'first page\n\nsecond page\n\n'


def test_pdf_parser_rejects_unknown_parse_method():
    with pytest.raises(ValueError, match='Unsupported parse method: unknown'):
        paper.PDFParser(parse_method='unknown', use_cache=False)


def test_pdf_parser_returns_none_for_missing_file(tmp_path, monkeypatch):
    calls = []

    def parse(path):
        calls.append(path)

    monkeypatch.setitem(paper.parsers, 'fake', parse)
    parser = paper.PDFParser(parse_method='fake', use_cache=False)

    assert parser.parse_pdf(tmp_path / 'missing.pdf') is None
    assert calls == []


def test_pdf_parser_checks_missing_file_before_hashing_with_cache(
    tmp_path, monkeypatch
):
    class Cache:
        def __init__(self, cache_dir):
            self.cache_dir = cache_dir

        def get(self, key):
            return None

    monkeypatch.setattr(paper, 'Cache', Cache)
    monkeypatch.setattr(
        paper,
        'get_hash',
        lambda path: pytest.fail(f'get_hash called for missing file: {path}'),
    )
    parser = paper.PDFParser(parse_method='pymupdf', cache_dir=tmp_path)

    assert parser.parse_pdf(tmp_path / 'missing.pdf') is None


@pytest.mark.parametrize('parser_result', ['', '   ', None])
def test_pdf_parser_returns_none_when_parser_extracts_no_text(
    tmp_path, monkeypatch, parser_result
):
    source = tmp_path / 'paper.pdf'
    source.write_text('not a pdf')
    monkeypatch.setitem(paper.parsers, 'fake', lambda path: parser_result)
    parser = paper.PDFParser(parse_method='fake', use_cache=False)

    assert parser.parse_pdf(source) is None


def test_pdf_parser_returns_none_when_parser_raises(tmp_path, monkeypatch):
    source = tmp_path / 'paper.pdf'
    source.write_text('not a pdf')

    def fail(path):
        raise RuntimeError('unreadable')

    monkeypatch.setitem(paper.parsers, 'fake', fail)
    parser = paper.PDFParser(parse_method='fake', use_cache=False)

    assert parser.parse_pdf(source) is None


def test_pdf_parser_caches_successful_parse_in_supplied_cache_dir(
    tmp_path, monkeypatch
):
    source = tmp_path / 'paper.pdf'
    source.write_text('not a pdf')
    parse_calls = []
    monkeypatch.setitem(
        paper.parsers,
        'fake',
        lambda path: parse_calls.append(path) or 'extracted text',
    )

    class Cache:
        instances = []

        def __init__(self, cache_dir):
            self.cache_dir = cache_dir
            self.values = {}
            self.instances.append(self)

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

    monkeypatch.setattr(paper, 'Cache', Cache)
    monkeypatch.setattr(paper, 'get_hash', lambda path: 'content-hash')
    parser = paper.PDFParser(
        parse_method='fake', cache_dir=tmp_path / 'cache', use_cache=True
    )

    assert parser.parse_pdf(source) == 'extracted text'
    assert parser.parse_pdf(source) == 'extracted text'
    assert parse_calls == [source]
    assert Cache.instances[-1].cache_dir == tmp_path / 'cache'
    assert Cache.instances[-1].values == {'content-hash': 'extracted text'}


def test_pdf_parser_uses_cache_hit_without_reading_source(tmp_path, monkeypatch):
    source = tmp_path / 'paper.pdf'
    source.write_text('not a pdf')

    class Cache:
        def __init__(self, cache_dir):
            self.cache_dir = cache_dir

        def get(self, key):
            assert key == 'content-hash'
            return 'cached text'

    monkeypatch.setattr(paper, 'Cache', Cache)
    monkeypatch.setattr(paper, 'get_hash', lambda path: 'content-hash')
    monkeypatch.setitem(
        paper.parsers,
        'fake',
        lambda path: pytest.fail('parser should not be called for a cache hit'),
    )
    parser = paper.PDFParser(
        parse_method='fake', cache_dir=tmp_path / 'cache', use_cache=True
    )

    assert parser.parse_pdf(Path(source)) == 'cached text'
