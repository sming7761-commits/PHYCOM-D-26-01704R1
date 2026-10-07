"""Derive severity, confusion, risk-coverage and paired statistics from frozen predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

from evaluation_common import ABSTAIN, PHYS, selective_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.data_dir / "metadata.csv").set_index("global_index", drop=False)
    predictions = pd.read_csv(args.evaluation_dir / "predictions" / "all_test_predictions.csv")
    phy = predictions[predictions["method"] == "PhyGuard"].copy()
    phy = phy.join(metadata[["severity", "regime", "label"]], on="index", rsuffix="_meta")

    severity_rows = []
    for (repeat, severity), part in phy.groupby(["repeat", "severity"]):
        mask = part["truth"].isin(PHYS)
        met = selective_metrics(part.loc[mask, "truth"].to_numpy(), part.loc[mask, "prediction"].to_numpy())
        severity_rows.append({"repeat": int(repeat), "severity": severity, **met, "n": int(mask.sum())})
    severity_df = pd.DataFrame(severity_rows)
    severity_df.to_csv(args.out_dir / "severity_per_repeat.csv", index=False)
    severity_summary = severity_df.groupby("severity", sort=False).agg(
        coverage_mean=("coverage", "mean"), coverage_std=("coverage", lambda x: x.std(ddof=0)),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", lambda x: x.std(ddof=0)),
        selected_accuracy_mean=("selected_accuracy", "mean"), selected_accuracy_std=("selected_accuracy", lambda x: x.std(ddof=0)),
    ).reset_index()
    severity_summary.to_csv(args.out_dir / "severity_summary.csv", index=False)

    emitted = phy[phy["truth"].isin(PHYS) & (phy["prediction"] != ABSTAIN)]
    confusion = pd.crosstab(emitted["truth"], emitted["prediction"], normalize="index").reindex(index=PHYS, columns=PHYS, fill_value=0.0)
    confusion.to_csv(args.out_dir / "emitted_confusion_matrix.csv")
    counts = pd.crosstab(emitted["truth"], emitted["prediction"]).reindex(index=PHYS, columns=PHYS, fill_value=0)
    counts.to_csv(args.out_dir / "emitted_confusion_counts.csv")

    coverages = np.linspace(0.05, 1.0, 20)
    risk_rows = []
    for repeat, part in phy[phy["truth"].isin(PHYS)].groupby("repeat"):
        part = part.sort_values("confidence", ascending=False).reset_index(drop=True)
        n = len(part)
        for retained in coverages:
            k = max(1, int(np.ceil(retained * n)))
            top = part.iloc[:k]
            correct = top["prediction"].to_numpy(dtype=object) == top["truth"].to_numpy(dtype=object)
            risk_rows.append({"repeat": int(repeat), "retained_coverage": float(k / n), "risk": float(1.0 - correct.mean()), "n_retained": int(k)})
    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(args.out_dir / "risk_coverage_per_repeat.csv", index=False)
    risk_df.groupby("retained_coverage").agg(risk_mean=("risk", "mean"), risk_std=("risk", lambda x: x.std(ddof=0))).reset_index().to_csv(args.out_dir / "risk_coverage_summary.csv", index=False)

    metric_df = pd.read_csv(args.evaluation_dir / "per_repeat_all_methods.csv")
    pivot = metric_df.pivot(index="repeat", columns="method", values="macro_f1")
    pair_rows = []
    for comparator in ["Gradient boosting", "Random forest", "Two-stage without physics"]:
        diff = (pivot["PhyGuard"] - pivot[comparator]).dropna().to_numpy()
        mean = float(diff.mean())
        if len(diff) >= 2:
            sem = stats.sem(diff)
            ci = stats.t.interval(0.95, len(diff)-1, loc=mean, scale=sem)
            t_stat, p_value = stats.ttest_rel(pivot.loc[pivot.index, "PhyGuard"], pivot.loc[pivot.index, comparator])
        else:
            ci = (np.nan, np.nan); t_stat = np.nan; p_value = np.nan
        pair_rows.append({
            "comparator": comparator,
            "n_repeats": int(len(diff)),
            "mean_difference_pp": mean * 100.0,
            "ci95_low_pp": float(ci[0] * 100.0),
            "ci95_high_pp": float(ci[1] * 100.0),
            "paired_t_stat": float(t_stat),
            "paired_t_p": float(p_value),
        })
    pd.DataFrame(pair_rows).to_csv(args.out_dir / "paired_macro_f1_statistics.csv", index=False)

    audit = {
        "n_phyguard_test_predictions": int(len(phy)),
        "n_emitted_physical_predictions": int(len(emitted)),
        "severity_levels": severity_summary["severity"].tolist(),
        "risk_points_per_repeat": int(len(coverages)),
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(severity_summary.to_string(index=False))
    print(confusion.to_string())


if __name__ == "__main__":
    main()
