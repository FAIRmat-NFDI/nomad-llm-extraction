import pdf2doi


def extract_doi_from_pdf(filepath) -> str:
    doi = 'NOT_FOUND'
    try:
        pdf2doi_results = pdf2doi.pdf2doi(filepath)
        if pdf2doi_results is None:
            return doi
        pdf2doi_results = (
            pdf2doi_results[0] if isinstance(pdf2doi_results, list) else pdf2doi_results
        )
        if pdf2doi_results.get('identifier_type') == 'DOI':
            doi = pdf2doi_results.get('identifier', doi)
    except Exception as e:
        print(f'Could not extract DOI from {filepath}: {e}')
    return doi
