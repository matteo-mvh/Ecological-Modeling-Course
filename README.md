# Ecological Modeling Course Repository

This repository contains code developed during a university course on computational aquatic ecosystem modeling. The project focuses on building and analyzing a 1D water-column ecosystem model, progressing from basic numerical exercises to a complete NPZD-style implementation.

The repository serves both as:

a record of the step-by-step development of the model, and

the final model implementation used for the course project.

## Repository structure

```text
Ecological-Modeling-Course/
├── README.md
├── ModelCode/
│   └── (final ecosystem model implementation)
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
    └── 5/
        ├── Exercise 5.py
        └── Exercise 5.pdf
```

## Main model (ModelCode)

The ModelCode directory contains the final ecosystem model developed for the course project.
The model represents a vertically resolved 1D water column and includes biological and physical processes relevant for marine ecosystem dynamics.

The implementation builds on the classical NPZD framework, representing:

- nutrients
- phytoplankton
- zooplankton
- detritus

and their interactions through growth, grazing, mortality, remineralization, and vertical transport.

The model is used to simulate seasonal ecosystem dynamics and explore how physical forcing and biological parameters influence bloom development and nutrient cycling.

Further details about the model structure and results are described in the final project report included in the repository.

## Exercises (exercises/)

The exercises folder documents the development process throughout the course.
Each folder contains:

- the exercise description (Exercise n.pdf)
- the corresponding Python implementation (Exercise n.py)

The exercises gradually introduce the components required for the final model, including:

- numerical time stepping
- vertical transport processes (advection and diffusion)
- biological source and sink terms
- integration of physical and biological processes in a water-column model

These scripts should be viewed primarily as learning steps and intermediate implementations, not as production-ready code.

## How to navigate the repository

If you want to understand how the model was developed:

1. Start with the exercises (exercises/1 → exercises/5).
2. Follow how physical and biological components are introduced step by step.
3. Look at ModelCode/ to see the final integrated model.

## Notes

- The code was written for educational purposes during the course.
- Scripts prioritize clarity and experimentation over software engineering structure.
- The repository therefore reflects the model development workflow rather than a polished software package.
