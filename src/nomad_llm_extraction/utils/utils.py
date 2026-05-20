from nomad.units import ureg as nomad_ureg


def convert_to_nomad_unit(value, from_unit, to_unit):
    quantity = nomad_ureg.Quantity(value, from_unit)
    converted_quantity = quantity.to(to_unit)
    return {'value': converted_quantity.magnitude, 'unit': to_unit}


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
