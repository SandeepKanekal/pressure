# Flow Simulations (Pressure and Fluid Flow)

Collection of self-contained Python scripts used to generate figures and animations for the accompanying LaTeX document “Pressure and Fluid Flow”. The simulations cover canonical problems (1D Burgers, forced turbulence, and 2D incompressible Navier–Stokes) and simple analysis tools (structure functions).

Outputs are saved to the local images/ and videos/ folders for easy inclusion in LaTeX.

## Structure

```
flow-simulations/
├── burgers1d.py              # 1D Burgers (finite difference, adaptive dt); saves videos/burgers1d.mp4
├── burgers1d_spectral.py     # 1D Burgers (Fourier–Galerkin, pseudospectral; two ν values); saves an MP4 in 
├── stable_turbulence.py      # Forced 1D Burgers to maintain steady turbulence; spectra and dissipation; 
├── structure_function.py     # Compute S1, S2, S3 structure functions from a 1D field; prints scaling exponents
├── incompressible_2D_NS.py   # 2D incompressible NS in vorticity form (spectral); KE/enstrophy over time; optional animation
├── turbulence_with_linearfriction_2D.py # 2D incompressible NS with linear friction and forcing; steady-state spectra
├── requirements.txt          # Minimal dependencies: numpy, matplotlib
├── images/                   # Optional still figures
└── videos/                   # Animation outputs (*.mp4)
```

## Requirements

- Python 3
- Dependencies listed in requirements.txt

Install dependencies:

```
pip install -r requirements.txt
```

## Quick Start

From the repository root:

```
python3 flow-simulations/burgers1d.py
python3 flow-simulations/burgers1d_spectral.py
python3 flow-simulations/stable_turbulence.py
python3 flow-simulations/structure_function.py
python3 flow-simulations/incompressible_2D_NS.py
```

Notes:
- Animations are written to videos/ as MP4 files. Ensure the videos/ directory exists (it is included here).
- `incompressible_2D_NS.py` runs a time loop and plots kinetic energy and enstrophy; animation code is present but commented—uncomment to export an MP4 to videos/.
- Tweak resolution, viscosity, and runtime via the constants near the top of each script: `N`, `nu`, `tmax`, `CFL`, etc.

## What Each Script Does

- burgers1d.py: Evolves 1D viscous Burgers with periodic BCs using centered differences + explicit diffusion; shows u(x,t), E(t), and ε(t); saves videos/burgers1d.mp4.
- burgers1d_spectral.py: Fourier–Galerkin method with dealiased nonlinear term. Compares two viscosities (ν=0.02, 0.01) and tracks energy/dissipation; saves an MP4 to videos/.
- stable_turbulence.py: Adds large-scale stochastic forcing to sustain a statistically steady state; shows velocity, energy, dissipation, spectra, and running-average dissipation; saves an MP4 to videos/.
- incompressible_2D_NS.py: Spectral vorticity solver on a periodic square; computes kinetic energy and enstrophy in time; optional animation to videos/ (commented section).
- turbulence_with_linearfriction_2D.py: 2D incompressible NS with linear friction and large-scale forcing; runs to steady state and plots time-averaged energy spectra.
- structure_function.py: After evolving a 1D field, computes structure functions S_p(r)=⟨|u(x+r)−u(x)|^p⟩ for p=1,2,3, prints scaling exponents, and shows log–log plots.

## Tips

- For reproducibility or different regimes, change `N`, `nu`, and `tmax` to explore resolution/viscosity effects.
- For LaTeX integration, point to files in videos/ (animations) or save figures into images/ with `plt.savefig()`.