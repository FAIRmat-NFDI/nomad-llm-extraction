from nomad.config.models.plugins import SchemaPackageEntryPoint

class BatteryExtractionSchemaPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_llm_extraction.schema_packages.schema_package import m_package

        return m_package


schema_package_entry_point = BatteryExtractionSchemaPackageEntryPoint(
    name='BatteryExtractionSchemaPackage',
    description='Schema package for storing extracted battery data.',
)
