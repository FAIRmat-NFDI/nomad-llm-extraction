import copy
import math
import re

from nomad_llm_extraction.pipeline.models import StageContext, StageResult


def remove_pce_check(data: dict) -> dict:
    new_data = {'cells': []}
    for i, cell in enumerate(data['cells'] or []):
        # PCE metrics filter
        if (
            ((cell.get('pce') or {'value': 28}).get('value') or 28) < 27.5
            and ((cell.get('voc') or {'value': 1}).get('value') or 1)
            < 1.56  # Voltage > 1.56 are tandems. Voltage<4 cuz above 4 are modules
        ):
            if (
                (cell.get('pce', {'value': 0}) or {'value': 0}).get('value', 0) == 0
                or (cell.get('jsc', {'value': 0}) or {'value': 0}).get('value', 0) == 0
                or (cell.get('voc', {'value': 0}) or {'value': 0}).get('value', 0) == 0
                or (cell.get('ff', {'value': 0}) or {'value': 0}).get('value', 0) == 0
            ):
                new_data['cells'].append(data['cells'][i])
                continue
            if math.isclose(
                ((cell.get('pce') or {'value': 99}).get('value') or 99),
                (
                    ((cell.get('jsc') or {'value': 0}).get('value') or 0)
                    * ((cell.get('voc') or {'value': 0}).get('value') or 0)
                    * ((cell.get('ff') or {'value': 0}).get('value') or 0)
                )
                / 100,
                abs_tol=0.2,
            ):
                new_data['cells'].append(data['cells'][i])
                continue
    return new_data


results = []
total_found = total_hallucinated = total_missing = total_expected = 0
from decimal import ROUND_HALF_UP, Decimal


def normalize_float(val, max_decimals=6):
    return float(
        Decimal(str(val)).quantize(
            Decimal(f'1.{"0" * max_decimals}'), rounding=ROUND_HALF_UP
        )
    )


def number_matches_text(val, text):
    val = normalize_float(val)

    canonical = format(val, 'f').rstrip('0').rstrip('.')

    # 1. Strict decimal equivalence (1.13 == 1.130)
    if re.search(rf'\b{re.escape(canonical)}0*\b', text):
        return f'strict decimal match: {canonical}'

    # 2. Exact normalized match
    if re.search(rf'\b{re.escape(str(val))}\b', text):
        return f'exact match: {val}'

    # 3. Fixed 2-decimal formatting
    fmt2 = f'{val:.2f}'
    if re.search(rf'\b{re.escape(fmt2)}\b', text):
        return f'2-decimal match: {fmt2}'

    # 4. Strict ÷100 shift (77.6 -> 0.776)
    if val >= 1:
        shifted = val / 100
        if 0 < shifted < 1:
            shifted_str = format(shifted, 'f').rstrip('0').rstrip('.')
            if '.' in shifted_str and len(shifted_str.split('.')[-1]) <= 3:
                if re.search(rf'\b{re.escape(shifted_str)}0*\b', text):
                    return f'strict /100 match: {val}->{shifted_str}'

    # 5. Integer fallback (17.0 → 17 or 17%)
    if math.isclose(val, round(val)):
        int_val = str(int(round(val)))
        if re.search(rf'\b{int_val}\b', text):
            return f'integer match: {int_val}'
        if re.search(rf'\b{int_val}\s*%', text):
            return f'integer percent match: {int_val}%'

    # 6. Scale ×1000 fallback (0.548 → 548)
    if abs(val) < 1:
        scaled = val * 1000
        if math.isclose(scaled, round(scaled)):
            scaled_int = str(int(round(scaled)))
            if re.search(rf'\b{scaled_int}\b', text):
                return f'scaled x1000 match: {val}->{scaled_int}'

    return None


def check_values_in_text(pdf_text, values):
    found = {}

    for key, val in values.items():
        if val is None:
            continue

        match_info = number_matches_text(val, pdf_text)

        if match_info:
            found[key] = {'status': 'found', 'value': match_info}
        else:
            found[key] = {'status': 'hallucination', 'value': val}

    return found


def extract_numeric_values(obj, prefix=''):
    values = {}

    if isinstance(obj, dict):
        # Case 1: dict with a numeric "value"
        if isinstance(obj.get('value'), (float, int)):
            values[prefix.rstrip('.')] = obj['value']

        # Case 2: keep walking
        for key, val in obj.items():
            new_prefix = f'{prefix}{key}.'
            values.update(extract_numeric_values(val, new_prefix))

    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            values.update(extract_numeric_values(item, f'{prefix}{idx}.'))

    return values


def remove_hallucinated_big_four_area(data, pdf_text):
    cells = copy.deepcopy(data['cells'])
    for i, cell in enumerate(cells):
        values = {
            key: data.get('value')
            for key, data in cell.items()
            if isinstance(data, dict) and isinstance(data.get('value'), (float, int))
        }

        if not values:
            continue

        found_values = check_values_in_text(pdf_text, values)

        for key, info in found_values.items():
            if info['status'] == 'hallucination':
                del cells[i][key]
    return {'cells': cells}


def filter_unwanted(data: dict, pdf_text) -> dict:
    p_data = remove_pce_check(data)
    return remove_hallucinated_big_four_area(p_data, pdf_text)
