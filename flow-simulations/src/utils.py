def validate_inputs(radius, length, viscosity, pressure_difference):
    if radius <= 0:
        raise ValueError("Radius must be greater than zero.")
    if length <= 0:
        raise ValueError("Length must be greater than zero.")
    if viscosity <= 0:
        raise ValueError("Viscosity must be greater than zero.")
    if pressure_difference < 0:
        raise ValueError("Pressure difference must be non-negative.")

def calculate_flow_rate(radius, length, viscosity, pressure_difference):
    validate_inputs(radius, length, viscosity, pressure_difference)
    flow_rate = (3.14159 * radius**4 * pressure_difference) / (8 * viscosity * length)
    return flow_rate

def calculate_pressure_drop(flow_rate, radius, length, viscosity):
    if flow_rate <= 0:
        raise ValueError("Flow rate must be greater than zero.")
    pressure_drop = (8 * viscosity * length * flow_rate) / (3.14159 * radius**4)
    return pressure_drop