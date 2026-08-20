from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad.config import config
from nomad.datamodel.data import Schema
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.metainfo import Quantity, SchemaPackage

configuration = config.get_plugin_entry_point(
    'nomad_llm_extraction.schema_packages:schema_package_entry_point'
)

m_package = SchemaPackage()

class BatteryExtraction(Schema):
    """NOMAD schema for storing extracted battery data."""
    
    active_material = Quantity(
        type=str, 
        description="The main active material of the electrode.",
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity)
    )
    
    battery_types = Quantity(
        type=str, 
        shape=['*'], 
        description="The types of batteries tested.",
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity)
    )
    
    binder_percentage = Quantity(
        type=float, 
        description="The weight percentage of the binder.",
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity)
    )
    
    voltage_window_v = Quantity(
        type=str, 
        description="The voltage range used for testing.",
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity)
    )
    
    characterization_techniques = Quantity(
        type=str, 
        shape=['*'], 
        description="The analytical techniques used.",
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity)
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
        logger.info('BatteryExtraction normalized successfully.')

m_package.__init_metainfo__()