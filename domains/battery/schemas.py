from pydantic import BaseModel, Field
from typing import List

class BatteryData(BaseModel):
    """Schema to extract basic battery experimental data."""
    active_material: str = Field(..., description="Main active material of the electrode (e.g., 'SnO2').")
    battery_types: List[str] = Field(..., description="Types of batteries tested (e.g., 'lithium-ion', 'sodium-ion').")
    binder_percentage: float = Field(..., description="Weight percentage of the binder.")
    voltage_window_v: str = Field(..., description="Voltage range used for testing (e.g., '0.01 to 3.0 V').")
    characterization_techniques: List[str] = Field(..., description="Analytical techniques used (e.g., 'XAS', 'XRD').")