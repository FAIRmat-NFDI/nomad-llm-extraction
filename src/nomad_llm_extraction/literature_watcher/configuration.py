import re
from pathlib import Path

from pint import UnitRegistry
from platformdirs import user_data_dir

# Initialize the UnitRegistry
ureg = UnitRegistry()

ureg.define('sun = 1 kW/m^2')

pint = {
    'default_units_by_type': {
        ureg.percent.dimensionality: (ureg.percent, '%'),  # Efficiency, humidity, etc.
        (ureg.ampere / (ureg.centimeter**2)).dimensionality: (
            ureg.milliampere / (ureg.centimeter**2),
            'mA cm^-2',
        ),  # Current density
        ureg.volt.dimensionality: (ureg.volt, 'V'),  # Voltage
        ureg.nanometer.dimensionality: (ureg.nanometer, 'nm'),  # Thickness,
        (ureg.meter**2).dimensionality: (ureg.centimeter**2, 'cm^2'),
        ureg.day.dimensionality: (
            ureg.second,
            's',
        ),  # Time (converted to hours for finer granularity)
        ureg.celsius.dimensionality: (
            ureg.celsius,
            '°C',
        ),  # Temperature converted from Celsius
        (1 * ureg.mg / ureg.mL).dimensionality: (ureg.mg / ureg.mL, 'mg/mL'),
        (ureg.mW / ureg.cm**2).dimensionality: ((ureg.mW / ureg.cm**2), 'mW cm^-2'),
        (ureg.mW / ureg.cm**2).dimensionality: ((ureg.mW / ureg.cm**2), 'mW cm^-2'),
        (ureg.mol / ureg.L).dimensionality: ((ureg.mol / ureg.L), 'mol/L'),
        (ureg.eV).dimensionality: ((ureg.eV), 'eV'),
        (ureg.meter**3).dimensionality: (ureg.milliliter, 'mL'),
    }
}

default_units = {
    'thickness': 'nm',
    'light_intensity': 'mW cm^-2',
    'duration': 's',
    'temperature': '°C',
    'time': 'h',
    'PCE_after_1000_hours': '%',
    'humidity': '%',
    'PCE_at_the_start_of_the_experiment': '%',
    'PCE_at_the_end_of_description': '%',
    'PCE_T80': '%',
    'bandgap': 'eV',
    'concentration': 'mol/L',
    'volume': 'mL',
}


papersbot_runs_path = Path(user_data_dir('papersbot_run', 'perla-extractor') + '/runs')


retry_dates = [30, 90, 180, 360]  # days

RELAXED_REGEX = re.compile(
    r"""
    # CONDITION 1: Must contain a perovskite-related term
    (?=.*\b(?:
        perovskite[s]?|
        PSC[s]?|
        halide|
        CsPb|
        MAPb|
        FAPb|
        CH3NH3|
        formamidinium
    )\b)

    # CONDITION 2: Must ALSO contain a solar-cell-related term
    (?=.*(?:
        # --- List of all word-based terms from all 3 regexes ---
        \b(?:
            # Solar/PV terms
            solar[\s-]?cell[s]?|
            photovoltaic[s]?|
            PV|
            solar|

            # Performance Metrics
            efficiency|
            PCE|
            power[\s-]conversion[\s-]efficiency|
            energy[\s-]conversion|
            conversion|
            performance|
            V(?:OC|oc)|
            open[\s-]circuit[\s-]voltage|
            J(?:SC|sc)|
            short[\s-]circuit[\s-]current|
            fill[\s-]factor|
            FF|
            quantum[\s-]efficiency|
            EQE|
            IPCE|
            hysteresis|

            # Device/Material terms
            device[s]?|
            cell[s]?|
            hole[\s-]transport|
            electron[\s-]transport|
            HTL|
            ETL|
            absorber|
            active[\s-]layer|
            photoactive|

            # Electrical terms
            conductivity|
            carrier|
            charge|
            recombination|
            extraction|
            transport|
            energy|
            power|
            voltage|
            current|

            # Stability/Environmental terms
            stability|
            lifetime|
            degradation|
            aging|
            operational|
            moisture|
            thermal|
            illumination|
            light|
            AM1\.5|
            sun|

            # Processing terms
            fabrication|
            processing|
            preparation|
            synthesis|

            # Application terms
            commercialization|
            applications?
        )\b|

        # --- List of all numeric/unit-based terms from all 3 regexes ---
        [\d]+(?:\.\d*)?[\s]*%|                    # Percentage (e.g., 25.5%)
        [\d]+(?:\.\d*)?[\s]*mA[\/\s]*cm[2²]?|    # Current density (e.g., 22 mA/cm2)
        [\d]+(?:\.\d*)?[\s]*V|                    # Voltage (e.g., 1.1 V)
        (?:[\d]+(?:\.\d*)?[\s]*)?mW(?:[\/\s]*cm)?| # Power density (e.g., 100 mW/cm or mW)
        (?:[\d]+(?:\.\d*)?[\s]*)?W[\/\s]*g        # Power-to-weight (e.g., 10 W/g)
    ))
    
    # If both conditions are met, match the entire string
    .*?
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# This is the regular expression that selects the papers of interest
STRICT_REGEX = re.compile(
    r"""
  (
    # Perovskite cell variations
    \b(perovskite(?:[\s-](?:solar|photovoltaic|PV))?(?:\s*cell[s]?|\s*device[s]?)|PSC[s]?)\b
    # Single junction specific terms
    |single[\s-]?(?:junction|layer|absorber|heterojunction|stack)
    # Architecture variations for single junction
    |(?:planar|mesoscopic|inverted|flexible|rigid|printable)[\s-]?perovskite
    # Exclude explicit mentions of tandem/multi-junction
    (?<!tandem[\s-])(?<!multi[\s-])(?<!double[\s-])(?<!triple[\s-])
  )
  .*?
  (
    # Performance metrics
    \b(?:efficiency|PCE|power[\s-]conversion[\s-]efficiency
    |V(?:OC|oc)|open[\s-]circuit[\s-]voltage
    |J(?:SC|sc)|short[\s-]circuit[\s-]current(?:[\s-]density)?
    |fill[\s-]factor|FF
    |stability|lifetime|degradation|performance
    |I[- ]?V(?:[\s-]curve)?|J[- ]?V(?:[\s-]curve)?|current[\s-](?:voltage|density)
    |hysteresis|quantum[\s-]efficiency|(?:internal|external)[\s-]quantum[\s-]efficiency|(?:IPCE|EQE)
    |[\d]+(?:\.\d+)?%|[\d]+(?:\.\d+)?[\s]?mA\/cm2|[\d]+(?:\.\d+)?[\s]?V)\b
  )
""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


def is_doi_good_to_go(doi, pdf_text, metadata=None) -> bool:
    def remove_conjunctions(text):
        # Convert to lowercase
        text = text.lower()

        # Replace HTML entity &amp; with &
        text = text.replace('&amp;', '&')

        # Remove 'and' as a word and '&' symbols with optional spaces around them
        # This also handles cases like "Tom&Jerry" or "Tom & Jerry"
        text = re.sub(r'\b(and)\b', '', text)
        text = re.sub(r'\s*&\s*', ' ', text)

        # Clean up any extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def journal_filter(journal, publisher):
        words_not_allowed = [
            'reviews',
            'theory',
            'computation',
            'catalysis',
            'review',
            'ceramic',
            'toxicology',
            'bio',
        ]
        if any(word in journal.lower() for word in words_not_allowed):
            return False

        allowed_journals = [
            journal_entry['Source title']
            for journal_entry in read_csv_to_dict(
                files('perla_extract').joinpath('allowed_journals.csv')
            )
        ]
        if journal not in allowed_journals:
            return remove_conjunctions(journal) in [
                remove_conjunctions(j) for j in allowed_journals
            ]

        return True

    def extract_words(text):
        # 1. Remove HTML tags (e.g., <a href="...">)
        text = re.sub(r'<[^>]+>', ' ', text)

        # 2. Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # 3. Extract words (letters, numbers, underscores)
        words = re.findall(r'\b\w+\b', text)

        return words

    def word_count(text):
        return len(extract_words(text))

    def non_solar_filter(text):
        non_solar_keywords = {
            'LED': [
                r'\bLED\b',
                r'light\s*-?\s*emitting\s+diode',
                r'electroluminescen\w*',
            ],
            'Battery': [r'\bbattery\b', r'energy storage', r'rechargeable'],
            'Photodetector': [r'photodetector', r'X\s*-?\s*ray detector'],
            'Catalyst': [
                r'catalys\w*',
                r'photocatalys\w*',
                r'water splitting',
                r'hydrogen evolution',
            ],
            'Other': [
                r'sens\w*',
                r'sensor',
                r'transistor',
                r'laser',
                r'memory',
                r'thermoelectric',
                r'capacitor',
            ],
        }

        solar_cell_keywords = [
            r'solar cell',
            r'photovoltaic',
            r'\bPV\b',
            r'\bPSC\b',
            r'\bPCE\b',
        ]

        # Compile patterns
        exclude_patterns = [
            re.compile(p, re.IGNORECASE)
            for sublist in non_solar_keywords.values()
            for p in sublist
        ]

        include_patterns = [re.compile(p, re.IGNORECASE) for p in solar_cell_keywords]

        # Count matches
        non_solar_count = sum(len(p.findall(text)) for p in exclude_patterns)

        solar_count = sum(len(p.findall(text)) for p in include_patterns)

        # Decision rule:
        # 1. Must mention solar at least once
        # 2. Solar mentions must be >= non-solar mentions
        return solar_count > 0 and solar_count >= non_solar_count

    def theory_filter(text):
        theory_keywords = [
            r'\bDFT\b',
            r'\bSCAPS\b',
            r'\bSCAPS-1D\b',
            r'density functional',
            r'first.?principles',
            r'ab.?initio',
            r'molecular dynamics',
            r'\bMD\b simulation',
            r'VASP',
            r'Gaussian',
            r'Quantum ESPRESSO',
            r'CASTEP',
            r'SIESTA',
            r'computational study',
            r'theoretical study',
            r'theoretical investigation',
            r'numerical simulation',
            r'numerical investigation',
            r'device simulation',
            r'theoretical analysis',
            r'computational analysis',
            r'theoretical modell?ing',
            r'computational modell?ing',
            r'simulated',
            r'simulation of',
            r'wxAMPS',
            r'AMPS-1D',
            r'PC1D',
            r'AFORS-HET',
            r'theoretical optimization',
            r'computational optimization',
            r'numerical modeling',
            r'simulated performance',
            r'theoretical efficiency',
            r'predicted efficiency',
            r'simulation',
            r'\bMD\b.*simulation',
            r'\bMD\b.*simulation',
            # --- Machine Learning / AI ---
            r'machine learning',
            r'\bML\b',
            r'deep learning',
            r'\bDL\b',
            r'artificial intelligence',
            r'\bAI\b',
            r'neural network',
            r'neural networks',
            r'\bNN\b',
            r'\bANN\b',
            r'convolutional neural network',
            r'\bCNN\b',
            r'recurrent neural network',
            r'\bRNN\b',
            r'graph neural network',
            r'\bGNN\b',
            r'support vector machine',
            r'\bSVM\b',
            r'random forest',
            r'decision tree',
            r'k[- ]?nearest neighbors?',
            r'\bKNN\b',
            r'Gaussian process',
            r'\bGP\b',
            r'data[- ]?driven',
            r'surrogate model',
            r'meta[- ]?model',
            r'predictive model',
            r'statistical learning',
            r'learning[- ]?based',
            r'model training',
            r'model prediction',
            r'feature engineering',
            r'dimensionality reduction',
        ]
        pattern = re.compile('|'.join(theory_keywords), re.IGNORECASE)
        match = pattern.search(text)
        return match is None

    def review_article_filter(text):
        review_patterns = [
            r'^Review\b',
            r'^Perspective\b',
            r'^Overview\b',
            r'^Outlook\b',
            r'^Minireview\b',
            r'^Critical [Rr]eview\b',
            r': [Aa] [Rr]eview\b',
            r': [Aa] [Pp]erspective\b',
            r'\b[Rr]eview of\b',
            r'\b[Rr]eview on\b',
            r'^Progress in\b',
            r'^Recent [Aa]dvances\b',
            r'^Advances in\b',
            r'^State of the art\b',
            r'^Current status\b',
            # Explicit review types
            r'\b(review|minireview|perspective|overview)\b',
            r'\bcritical review\b',
            r'\bstate[- ]of[- ]the[- ]art\b',
            r'\bis (discussed|reviewed|summarized|outlined)\b',
            r'\bare reviewed\b',
            r'\bwe review\b',
            r'\bthis (review|work) reviews\b',
        ]
        pattern = re.compile('|'.join(review_patterns), re.IGNORECASE)
        is_review_article = pattern.search(text)
        return not is_review_article

    metadata = metadata or get_doi_summary(doi)['consolidated']
    title = metadata.get('title', '')
    abstract = metadata.get('abstract', '')
    journal = metadata.get('journal', '')
    publisher = metadata.get('publisher', '')

    if word_count(abstract) < 100 and pdf_text:
        text_to_filter = pdf_text[: int(len(pdf_text) * 0.05)]
    else:
        text_to_filter = title + ' ' + abstract

    if not journal:
        return (
            theory_filter(text_to_filter)
            and non_solar_filter(text_to_filter)
            and review_article_filter(text_to_filter)
        )

    return (
        journal_filter(journal, publisher)
        and theory_filter(text_to_filter)
        and non_solar_filter(text_to_filter)
        and review_article_filter(text_to_filter)
    )
