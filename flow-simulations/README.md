# Poiseuille Flow Simulation

This project simulates Poiseuille flow, which describes the motion of a viscous fluid in a cylindrical pipe. The simulation calculates flow rates, pressure drops, and other relevant fluid dynamics parameters based on the principles of fluid mechanics.

## Project Structure

```
poiseuille-flow-simulation
├── src
│   ├── poiseuille.py      # Main implementation for simulating Poiseuille flow
│   └── utils.py           # Utility functions for calculations and data handling
├── requirements.txt        # Python dependencies required for the project
└── README.md               # Documentation for the project
```

## Installation

To set up the project, ensure you have Python installed on your machine. Then, install the required dependencies by running:

```
pip install -r requirements.txt
```

## Usage

1. Import the necessary functions from the `poiseuille` module in your Python script.
2. Use the provided functions to calculate flow rates and pressure drops based on your input parameters.

### Example

```python
from src.poiseuille import calculate_flow_rate

# Example parameters
radius = 0.01  # radius in meters
length = 1.0   # length in meters
viscosity = 0.001  # viscosity in Pa.s
pressure_drop = 1000  # pressure drop in Pascals

flow_rate = calculate_flow_rate(radius, length, viscosity, pressure_drop)
print(f"Flow Rate: {flow_rate} m^3/s")
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.