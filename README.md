# Ecological Modeling Course Repository

This repository contains step-by-step programming exercises developed during a university course in **computational aquatic ecology**. The codebase is focused on learning how to build and evaluate ecological models of a water column, progressing from basic numerical methods to integrated ecosystem simulations.

## Course purpose

The overall goal of the course is to train students to:

- formulate new ecological models and sub-models,
- implement numerical solutions to process-based equations,
- build and analyze **NPZD** models (Nutrients–Phytoplankton–Zooplankton–Detritus),
- understand advection/diffusion in partial differential equations,
- parameterize and interpret 1D water-column ecosystem models,
- evaluate strengths/limitations of NPZD approaches (e.g., spring/autumn bloom dynamics),
- gain practical familiarity with professional tools such as **MIKE EcoLab**.

## What kind of code is written here?

The repository consists of educational Python scripts, each script corresponding to one exercise in the course progression. The code is intended for learning and experimentation rather than production software.

Typical code in this repo includes:

- numerical modeling workflows for ecological and hydrographic processes,
- implementations of finite-difference style updates for time/space dynamics,
- parameter setup and unit handling for ecological state variables,
- simulation-oriented scripts that help build toward a full 1D NPZD water-column model.

## Repository structure

```text
Ecological-Modeling-Course/
├── README.md
├── modelcode.py
└── exercises/
    ├── 1/
    │   ├── Exercise 1.py
    │   └── Exercise 1.pdf
    ├── 2/
    │   ├── Exercise 2.py
    │   └── Exercise 2.pdf
    ├── 3/
    │   ├── Exercise 3.py
    │   └── Exercise 3.pdf
    ├── 4/
    │   ├── Exercise 4.py
    │   └── Exercise 4.pdf
    ├── 5/
    │   ├── Exercise 5.py
    │   └── Exercise 5.pdf
    ├── 6/
    │   ├── Exercise 6.py
    │   └── Exercise 6.pdf
    ├── 7/
    │   ├── plotting_7.py
    │   └── Exercise 7.pdf
    ├── 8/
    │   ├── plotting_8.py
    │   └── Exercise 8.pdf
    └── 9/
        ├── plotting_9.py
        └── Exercise 9.pdf
```

### Folder and file roles

- `exercises/<n>/Exercise <n>.pdf`  
  Assignment description, theory, and expected outcomes for each exercise.

- `exercises/<n>/Exercise <n>.py`  
  Python implementation corresponding to that exercise.

- `modelcode.py`  
  Main ecological model code developed for the course.

## Step-by-step development path in this course

The exercises are organized in increasing complexity. A typical learning trajectory is:

1. **Foundations of numerical ecological modeling**  
   Set up basic variables, equations, and simple numerical updates.

2. **Advection and diffusion in 1D**  
   Implement finite-difference approximations for transport and mixing processes.

3. **Core ecosystem process representation**  
   Add biological process terms relevant to NPZD dynamics.

4. **Water-column NPZD integration**  
   Combine physical and biological components into a vertically resolved model.

5. **Seasonal simulation and interpretation**  
   Use forcing (e.g., wind/temperature seasonality), run scenarios, and interpret bloom and production patterns.

In the broader course context, this hands-on coding phase is complemented by work with **MIKE EcoLab** and a final project where students either extend these models or apply professional tools to a practical ecological problem.

## How to use this repository

1. Start with `exercises/1/Exercise 1.pdf` and read the task description.
2. Open and run `exercises/1/Exercise 1.py`.
3. Continue sequentially through the exercises.
4. Compare your implementation choices and outputs with the conceptual expectations in each PDF.
5. Use later exercises as templates for your own extended project model.

## Important files for the reports

### Report 1

The most important file for **Report 1** is the **model code in the main directory**, as this contains the core ecological model implementation used in the report.

### Report 2

The most important files for **Report 2** are the **plotting and analysis files in `exercises/7` to `exercises/9`**, since these folders contain the scripts used for visualization, scenario comparison, and interpretation of results.

## Notes

- Scripts are arranged for course progression and may depend on concepts introduced in earlier exercises.
- File names include spaces to match assignment naming where relevant.
- This repository is best read as a learning portfolio that documents model development over the semester.
