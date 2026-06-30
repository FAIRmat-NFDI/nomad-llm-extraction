from nomad.config.models.plugins import SchemaPackageEntryPoint
from pydantic import Field


class LlmExtractorPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_llm_extraction.schema_packages.llm_extractor import m_package

        return m_package


llm_extractor = LlmExtractorPackageEntryPoint(
    name='LlmExtractor',
    description='Schema package defined for the LLM extractor.',
)
