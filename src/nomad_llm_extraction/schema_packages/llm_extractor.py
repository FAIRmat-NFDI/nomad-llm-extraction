from nomad.datamodel.data import EntryData, Schema, UseCaseElnCategory
from nomad.datamodel.metainfo.annotations import SectionProperties
from nomad.datamodel.metainfo.eln import ELNAnnotation
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection

m_package = SchemaPackage()


class LLMExtractionInput(Schema):
    m_def = Section(
        label='LLM Extraction Input',
        categories=[UseCaseElnCategory],
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=[
                    'model',
                    'api_base_url',
                    'extraction_m_def',
                    'text',
                    'delete_source_pdfs',
                    'extract_multiple_instances',
                ]
            )
        ),
    )
    model = Quantity(
        type=str,
        description='LLM model to use for extraction.',
        default='Claude Sonnet 4.6',
    )
    api_base_url = Quantity(
        type=str,
        description='Custom API URL for the LLM endpoint, if applicable.',
    )
    extraction_m_def = Quantity(
        type=str,
        description='Nomad Section m_def to be used for extraction.',
    )
    text = Quantity(
        type=str,
        description='Text to run the extraction on. If not provided, the action will extract text from all PDFs in the project.',
    )
    delete_source_pdfs = Quantity(
        type=bool,
        default=True,
        description='Whether to delete the source PDF files after processing.',
    )
    extract_multiple_instances = Quantity(
        type=bool,
        default=True,
        description='Whether to extract multiple instances of the schema from the text.',
    )


class LLMExtractionOutput(Schema):
    m_def = Section(
        label='LLM Extraction Output',
        categories=[UseCaseElnCategory],
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=['action_id', 'extracted_data', 'input_data']
            )
        ),
    )
    extracted_data = Quantity(
        type=EntryData,
        shape=['*'],
        description='List of extracted data from the paper',
    )
    action_id = Quantity(
        type=str,
        description='ID of the LLM Extraction action.',
    )
    input_data = SubSection(
        section_def=LLMExtractionInput,
        description='Input data used for the LLM Extraction action.',
        a_eln=ELNAnnotation(label='Action Input Data'),
    )


m_package.__init_metainfo__()
