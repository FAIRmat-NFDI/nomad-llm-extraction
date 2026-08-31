

def trim_ids_refs(schema: dict[str, any]) -> dict[str, any]:
    """trim $id and $ref fields from the schema recursively"""

    def trim(id_str: str) -> str:
        return id_str.rsplit('/schemas/', maxsplit=1)[-1].split('@', maxsplit=1)[0]

    if '$id' in schema:
        schema['$id'] = trim(schema['$id'])
    if '$ref' in schema:
        schema['$ref'] = trim(schema['$ref'])
    for key, value in schema.items():
        if isinstance(value, dict):
            schema[key] = trim_ids_refs(value)
        elif isinstance(value, list):
            schema[key] = [
                trim_ids_refs(item) if isinstance(item, dict) else item
                for item in value
            ]
    return schema


def edit_unit_value_in_schema(schema: dict[str, any]) -> dict[str, any]:
    """Edit the unit value in the schema recursively"""
    if 'unit' in schema and isinstance(schema['unit'], dict):
        schema['unit'].pop('enum', [])
        schema['unit']['description'] = 'Unit of the quantity.'
    for key, value in schema.items():
        if isinstance(value, dict):
            schema[key] = edit_unit_value_in_schema(value)
        elif isinstance(value, list):
            schema[key] = [
                edit_unit_value_in_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
    return schema


