from nomad_llm_extraction.actions.simple_action.models import SimpleDomainConfig

BATTERY_CONFIG = SimpleDomainConfig(
    domain_name="battery",
    system_prompt="You are an expert materials scientist. Extract the requested experimental parameters.",
    extraction_prompt_template="Extract the experimental data from the following text:\n\n{text}",
    extraction_schema_path="nomad_llm_extraction.domains.battery.schemas.BatteryData" 
)