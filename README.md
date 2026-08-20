# SNPZDO Marine Ecological Model

> **Project type:** DTU University Course Project  
> **Course:** Computational Marine Ecological Modelling  
> **Field:** Marine ecology · biogeochemistry · numerical modelling  
> **Language:** Python

## Overview

This project develops a vertically resolved **SNPZDO marine ecosystem model**:

**Silicate – Nutrients – Phytoplankton – Zooplankton – Detritus – Oxygen**

The model extends a traditional NPZDO water-column ecosystem model by explicitly introducing **silicate and diatom dynamics**.

The aim is to investigate how nutrient and silicate availability influence phytoplankton community structure, biological production and oxygen dynamics in the water column.

## Scientific Motivation

Diatoms are major contributors to marine primary production and differ from many other phytoplankton groups because they require **silicate** to construct their frustules.

Adding silicate therefore allows the model to represent competition between phytoplankton groups and investigate how changes in nutrient supply affect:

- diatom abundance
- primary production
- detritus formation
- remineralisation
- oxygen consumption
- seasonal ecosystem dynamics

## Model Extension

The baseline NPZDO model was extended with:

- dissolved silicate
- diatom nitrogen biomass
- living diatom silica
- biogenic silica detritus
- an internal silicon quota
- quota-dependent diatom growth
- optional quota-dependent grazing protection

The model can also approximate the original NPZDO system by setting initial silicate concentrations to zero, thereby suppressing diatom growth.

## Main Components

### `ModelCode.py`

Primary single-simulation model.

Contains:

- state variables
- model equations
- parameters
- initial conditions
- numerical integration
- diagnostics
- model-output caching

This script is useful for analysing the seasonal development of an individual simulation.

### `ModelSweep.py`

Runs the model across combinations of initial nutrient and silicate concentrations.

This allows systematic exploration of how ecosystem behaviour changes across different nutrient environments.

### `PlotDepthProfiles.py`

Loads saved model simulations and generates selected vertical profiles for analysing water-column structure.

## Repository Structure

```text
Ecological-Modeling-Course/
├── ModelCode.py
├── ModelSweep.py
├── PlotDepthProfiles.py
├── README.md
├── exercises/
├── project/
└── results_files/
```

## Model Experiments

The repository supports experiments including:

- changing initial nitrogen availability
- changing initial silicate availability
- comparing NPZDO and SNPZDO configurations
- evaluating diatom growth
- analysing nutrient limitation
- examining vertical oxygen dynamics
- comparing ecosystem responses across parameter combinations

## Cached Simulation System

Model runs are saved using parameter-based hashes and case identifiers.

This allows previously completed simulations to be reused without repeating computationally expensive integrations.

Cached simulations can be accessed by both the single-run and parameter-sweep workflows.

## Typical Workflow

1. Define model parameters and initial conditions.
2. Run a representative simulation using `ModelCode.py`.
3. Inspect seasonal ecosystem behaviour.
4. Run nutrient–silicate experiments using `ModelSweep.py`.
5. Reuse cached simulations for additional diagnostics.
6. Plot selected vertical profiles using `PlotDepthProfiles.py`.
7. Compare ecosystem responses across nutrient regimes.

## What This Project Demonstrates

The project provides experience in:

- marine ecological modelling
- NPZ / NPZDO ecosystem models
- nutrient–phytoplankton interactions
- diatom and silicate dynamics
- oxygen modelling
- vertical water-column models
- numerical integration
- parameter sweeps
- scientific Python
- model diagnostics and visualisation

## Status

Completed DTU university course project.

The repository contains research-oriented course code and model experiments rather than a production software package.
