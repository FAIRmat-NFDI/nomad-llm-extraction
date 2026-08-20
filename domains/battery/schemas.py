from pydantic import BaseModel, Field


class BatteryData(BaseModel):
    """Schema to extract basic battery experimental data."""

    active_material: str = Field(
        ..., description="Main active material of the electrode (e.g., 'SnO2')."
    )
    battery_types: list[str] = Field(
        ...,
        description="Types of batteries tested (e.g., 'lithium-ion', 'sodium-ion').",
    )
    binder_percentage: float = Field(
        ..., description='Weight percentage of the binder.'
    )
    voltage_window_v: str = Field(
        ..., description="Voltage range used for testing (e.g., '0.01 to 3.0 V')."
    )
    characterization_techniques: list[str] = Field(
        ..., description="Analytical techniques used (e.g., 'XAS', 'XRD')."
    )


class ExperimentIdentifiers(BaseModel):
    """Used in first pass to find all cell/experiment names."""

    identifiers: list[str] = Field(
        default_factory=lambda: ['default'],
        description='Unique cell or experiment identifiers found in the text',
    )
