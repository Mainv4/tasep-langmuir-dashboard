# TASEP-Langmuir Polymer Dashboard

Interactive dashboard for exploring simulation data from active polymer chains with TASEP-Langmuir kinetics.

## Overview

This dashboard visualizes correlations between local and global observables in polymer simulations where active sites migrate along the chain following TASEP dynamics coupled with Langmuir kinetics (attachment/detachment).

## Observables

### System Parameters
- **Segment position**: Position along the chain (30 segments of 10 beads)
- **Ω (Langmuir rate)**: Attachment/detachment rate

### Global (chain-level)
- **Rg_chain**: Radius of gyration of the full chain
- **D_chain**: Center-of-mass diffusion coefficient
- **ρ_act_chain**: Mean TASEP occupation density

### Local (segment-level)
- **ρ_act**: Local TASEP occupation density
- **D_local**: Local segment diffusion
- **Rg_local**: Local radius of gyration
- **ρ_spatial**: Spatial bead density (neighbors within 3σ)
- **MSD(t)**: Mean square displacement at various timelags

### Fluctuations
- **σ_ρ_act**: Measured density fluctuations
- **σ_ρ_theory**: Theoretical TASEP-Langmuir fluctuations

## Parameters

- Chain length: 300 beads
- α_in ∈ {0.1, 0.2}: TASEP entry rate
- α_out ∈ {0.1, 0.2}: TASEP exit rate
- Ω ∈ {0.001, 0.01, 0.1, 1.0, 10.0}: Langmuir rate
- Pe = 10: Péclet number (activity strength)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

Data compiled from LAMMPS molecular dynamics simulations coupled with Gillespie algorithm for TASEP dynamics.
