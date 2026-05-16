# Ecological Modeling Course Repository

This repository contains the Python code used for the final ecological modeling project in the course **Computational Marine Ecological Modelling**.

The project implements a vertically resolved **SNPZDO model**:  
**Silicate – Nutrients – Phytoplankton – Zooplankton – Detritus – Oxygen**.

The model extends a baseline NPZDO water-column model by adding:

- dissolved silicate,
- diatom nitrogen biomass,
- living diatom silica,
- biogenic silica detritus,
- an internal silicon quota formulation,
- quota-dependent diatom growth,
- and optional quota-dependent grazing protection.

The model can also be run in an NPZDO-like configuration by setting the initial silicate concentration to zero. In that case, diatom growth is suppressed and the system behaves like a non-silicate model.

---

## Main scripts

### `ModelCode.py`

`ModelCode.py` is the main single-run version of the model.

It contains:

- the full SNPZDO model implementation,
- the state variables,
- the process equations,
- the parameter setup,
- initial-condition construction,
- numerical integration,
- diagnostics,
- and saving/loading of cached model runs.

This script is useful for running one representative simulation and checking the seasonal behaviour of the model in detail.

---

### `ModelSweep.py`

`ModelSweep.py` runs the model repeatedly over different initial nutrient and silicate conditions.

It is used for nutrient--silicate experiments where the model is evaluated across many combinations of initial total nitrogen and initial total silicate.

The sweep script uses the same model structure as `ModelCode.py`, but repeats the simulation for many cases and saves each completed run.

---

### `PlotDepthProfiles.py`

`PlotDepthProfiles.py` is used to plot selected depth profiles from saved model runs.

It can access cached outputs created by either `ModelCode.py` or `ModelSweep.py`.

---

## Cached model runs

Both `ModelCode.py` and `ModelSweep.py` create hashed model runs.

Each run is saved using a parameter-based hash and a case-specific identifier. This means that completed simulations can be accessed again without rerunning the full model.

The caching system allows:

- single runs to be reused by sweep scripts,
- sweep outputs to be opened again for plotting,
- diagnostics to be regenerated from saved results,
- and repeated model experiments to avoid unnecessary recomputation.

In practice, `ModelCode.py` and `ModelSweep.py` can access each other's saved outputs as long as the parameter hash and initial-condition case match. If a run already exists, it can be loaded from cache; if not, rerunning the script will create the missing cached run.

---

## Repository structure

```text
Ecological-Modeling-Course/
├── ModelCode.py
├── ModelSweep.py
├── PlotDepthProfiles.py
├── README.md
├── exercises/
├── project/
└── results_files/
