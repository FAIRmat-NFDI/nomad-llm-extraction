from nomad.units import ureg as nomad_ureg


def convert_to_nomad_unit(value, from_unit, to_unit):
    quantity = nomad_ureg.Quantity(value, from_unit)
    converted_quantity = quantity.to(to_unit)
    return {'value': converted_quantity.magnitude, 'unit': to_unit}
