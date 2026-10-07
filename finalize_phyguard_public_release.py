# -*- coding: utf-8 -*-
from pathlib import Path

ROOT = Path(r"C:\Users\symin\Desktop\PhyGuard_GitHub_Release")

# Files that are useful internally but should not be part of the public repository.
REMOVE_NAMES = {
    "BUILD_REPORT.txt",
    "release_tree.txt",
    "manuscript_results_draft.txt",
    "reviewer_response_draft.txt",
    "REVIEWER_RESPONSE_TEMPORAL_BASELINES.md",
    "SCIENTIFIC_DIAGNOSIS.md",
    "old_vs_corrected_metrics.csv",
    "excluded_debug_artifacts.json",
    "progress.json",
}

removed = []
for p in ROOT.rglob("*"):
    if p.is_file() and p.name in REMOVE_NAMES:
        p.unlink()
        removed.append(str(p.relative_to(ROOT)))

# Rewrite README to remove TODO-style internal notes while keeping claims conservative.
readme = """# PhyGuard

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
"""
(ROOT / "README.md").write_text(readme, encoding="utf-8")

print("Public-release cleanup complete.")
print(f"Removed {len(removed)} internal-only files:")
for x in removed:
    print(" -", x)
print("\nREADME.md refreshed.")
print("Repository root:", ROOT)
