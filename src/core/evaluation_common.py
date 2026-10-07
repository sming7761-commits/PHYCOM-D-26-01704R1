"""Shared evaluation utilities for the PhyGuard reconstruction.

This module implements transparent reconstruction choices. It is not claimed to
recover the unavailable original implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

PHYS = np.array(["interference", "blockage", "mobility", "adaptation_mismatch"], dtype=object)
CONTROLS = np.array(["normal", "nonphysical_goodput"], dtype=object)
ABSTAIN = "ABSTAIN"


def selective_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    physical = np.isin(y, PHYS)
    control = ~physical
    emitted = pred != ABSTAIN

    selected_accuracy = (
        float(np.mean(pred[physical & emitted] == y[physical & emitted]))
        if np.any(physical & emitted)
        else 0.0
    )
    coverage = float(np.mean(emitted[physical])) if np.any(physical) else 0.0
    false_specific_rate = float(np.mean(emitted[control])) if np.any(control) else 0.0
    macro_f1 = float(
        f1_score(y[physical], pred[physical], labels=list(PHYS), average="macro", zero_division=0)
    )
    balanced_accuracy = float(
        recall_score(y[physical], pred[physical], labels=list(PHYS), average="macro", zero_division=0)
    )
    return {
        "selected_accuracy": selected_accuracy,
        "coverage": coverage,
        "false_specific_rate": false_specific_rate,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
    }


def operating_point_rows(
    y_val: np.ndarray,
    labels: np.ndarray,
    gate_confidence: np.ndarray,
    mechanism_confidence: np.ndarray,
    gate_grid: np.ndarray | None = None,
    mechanism_grid: np.ndarray | None = None,
    fsr_max: float = 0.05,
    selected_accuracy_min: float = 0.90,
) -> tuple[dict, list[dict]]:
    """Choose two thresholds under the manuscript-stated validation constraints."""
    gate_grid = np.round(np.linspace(0.05, 0.95, 19), 2) if gate_grid is None else gate_grid
    mechanism_grid = np.round(np.linspace(0.05, 0.95, 19), 2) if mechanism_grid is None else mechanism_grid
    rows: list[dict] = []
    for tau_g in gate_grid:
        for tau_c in mechanism_grid:
            pred = np.where(
                (gate_confidence >= tau_g) & (mechanism_confidence >= tau_c),
                labels,
                ABSTAIN,
            )
            met = selective_metrics(y_val, pred)
            feasible = (
                met["false_specific_rate"] <= fsr_max
                and met["selected_accuracy"] >= selected_accuracy_min
            )
            penalty = (
                max(0.0, met["false_specific_rate"] - fsr_max) / max(fsr_max, 1e-12)
                + max(0.0, selected_accuracy_min - met["selected_accuracy"])
                / max(selected_accuracy_min, 1e-12)
            )
            rows.append(
                {
                    **met,
                    "tau_g": float(tau_g),
                    "tau_c": float(tau_c),
                    "feasible": bool(feasible),
                    "penalty": float(penalty),
                }
            )
    feasible_rows = [row for row in rows if row["feasible"]]
    if feasible_rows:
        best = max(
            feasible_rows,
            key=lambda row: (
                row["coverage"],
                row["selected_accuracy"],
                -row["false_specific_rate"],
                -(row["tau_g"] + row["tau_c"]),
            ),
        )
    else:
        best = min(
            rows,
            key=lambda row: (
                row["penalty"],
                -row["coverage"],
                -row["selected_accuracy"],
                row["false_specific_rate"],
                row["tau_g"] + row["tau_c"],
            ),
        )
    return best, rows


def single_threshold_rows(
    y_val: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    thresholds: np.ndarray,
    fsr_max: float = 0.05,
    selected_accuracy_min: float = 0.90,
) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    for tau in thresholds:
        pred = np.where(confidence >= tau, labels, ABSTAIN)
        met = selective_metrics(y_val, pred)
        feasible = (
            met["false_specific_rate"] <= fsr_max
            and met["selected_accuracy"] >= selected_accuracy_min
        )
        penalty = (
            max(0.0, met["false_specific_rate"] - fsr_max) / max(fsr_max, 1e-12)
            + max(0.0, selected_accuracy_min - met["selected_accuracy"])
            / max(selected_accuracy_min, 1e-12)
        )
        rows.append({**met, "tau": float(tau), "feasible": bool(feasible), "penalty": float(penalty)})
    feasible_rows = [row for row in rows if row["feasible"]]
    if feasible_rows:
        best = max(
            feasible_rows,
            key=lambda row: (
                row["coverage"],
                row["selected_accuracy"],
                -row["false_specific_rate"],
                -row["tau"],
            ),
        )
    else:
        best = min(
            rows,
            key=lambda row: (
                row["penalty"],
                -row["coverage"],
                -row["selected_accuracy"],
                row["false_specific_rate"],
                row["tau"],
            ),
        )
    return best, rows


def stratification_keys(metadata: pd.DataFrame, include_regime: bool = True) -> np.ndarray:
    columns = ["label", "severity"]
    if include_regime:
        columns.insert(0, "regime")
    return metadata[columns].astype(str).agg("|".join, axis=1).to_numpy()


def confidence_from_class_probabilities(probabilities: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a generic six-class classifier to selective physical predictions.

    Control predictions are assigned zero physical confidence. A physical class
    is emitted only if it is the highest-probability class and its probability
    clears the validation-selected threshold.
    """
    argmax = probabilities.argmax(axis=1)
    top_label = classes[argmax].astype(object)
    top_prob = probabilities[np.arange(len(probabilities)), argmax]
    is_physical = np.isin(top_label, PHYS)
    physical_label = np.where(is_physical, top_label, PHYS[0]).astype(object)
    physical_confidence = np.where(is_physical, top_prob, 0.0)
    return physical_label, physical_confidence


def summary_from_rows(df: pd.DataFrame, group_column: str = "method") -> pd.DataFrame:
    metrics = ["selected_accuracy", "coverage", "false_specific_rate", "macro_f1", "balanced_accuracy"]
    rows = []
    for group, part in df.groupby(group_column, sort=False):
        row = {group_column: group, "n_runs": int(len(part))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)
