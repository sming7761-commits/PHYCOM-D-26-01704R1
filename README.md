# PhyGuard

Code and reproducibility artifacts for the accepted paper:

**PhyGuard: Physics-Guided Machine Learning for Selective Diagnosis of Multivariate OFDM Link Anomalies**

Accepted in *Physical Communication*.

## Overview

PhyGuard is a physics-guided machine-learning framework for selective diagnosis
of multivariate OFDM link anomalies. This repository contains the cleaned core
implementation, formal validation scripts, experiment configurations, compact
result tables, and author-generated figures associated with the accepted study.

## Repository structure

- `src/core/` — core PhyGuard implementation and evaluation utilities.
- `src/validation/` — formal independent-link validation, robustness, sensitivity,
  ablation, complexity, and temporal-baseline experiments.
- `configs/` — frozen experiment and protocol configurations.
- `contracts/` — KPI/interface contracts used by the evaluation pipeline.
- `manifests/` — final experiment/reproducibility manifests.
- `reference/source_baseline/` — compact source-baseline data and summary results.
- `results/` — compact final result tables and summaries.
- `figures/` — author-generated manuscript figures.
- `environment/` — recorded environment specification for temporal baselines.

## Data

Large third-party raw measurement files are not redistributed in this repository.
The validation scripts and configuration files document the corresponding public
data workflow used in the study.

## Reproducibility scope

This public release intentionally excludes internal revision logs, failed/debug
runs, duplicated checkpoints, journal-submission archives, large third-party raw
binary files, and regenerated model binaries.

## Citation

Please cite the PhyGuard paper in *Physical Communication*. The final DOI and
bibliographic metadata can be added here once they are available.

## License

No open-source license is granted by default in this repository. Please contact
the authors before reusing code beyond what is permitted by applicable law.
