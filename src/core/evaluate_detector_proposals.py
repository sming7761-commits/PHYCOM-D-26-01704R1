"""Evaluate PhyGuard end-to-end with detector-proposed intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from evaluation_common import ABSTAIN, operating_point_rows, selective_metrics
from evaluate_suite import fit_two_stage, split_indices, two_stage_probabilities

DETECTORS = ["robust_energy", "pca_reconstruction", "isolation_forest"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--oracle-feature-dir", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260705)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "models").mkdir(exist_ok=True)
    (args.out_dir / "predictions").mkdir(exist_ok=True)

    metadata = pd.read_csv(args.data_dir / "metadata.csv")
    oracle = np.load(args.oracle_feature_dir / "features.npz")
    y = metadata["label"].to_numpy(dtype=object)
    metric_rows, prediction_rows, threshold_rows = [], [], []

    for detector in DETECTORS:
        proposal_features = np.load(args.proposal_dir / detector / "features.npz")
        proposals = pd.read_csv(args.proposal_dir / detector / "proposals.csv")
        physical_proposals = proposals[proposals["label"].isin(["interference", "blockage", "mobility", "adaptation_mismatch"])]
        mean_iou = float(physical_proposals["iou"].mean())
        event_recall = float((physical_proposals["iou"] >= 0.3).mean())

        for repeat in range(args.repeats):
            train, val, test = split_indices(metadata, args.seed + repeat)
            gate, resolver = fit_two_stage(
                oracle["raw"][train], oracle["physical"][train], oracle["combined"][train], y[train], args.seed + repeat, True
            )
            gv, cv, lv = two_stage_probabilities(
                gate, resolver,
                proposal_features["raw"][val], proposal_features["physical"][val], proposal_features["combined"][val], True
            )
            best, rows = operating_point_rows(y[val], lv, gv, cv)
            threshold_rows.extend({"detector": detector, "repeat": repeat, **row} for row in rows)
            gt, ct, lt = two_stage_probabilities(
                gate, resolver,
                proposal_features["raw"][test], proposal_features["physical"][test], proposal_features["combined"][test], True
            )
            pred = np.where((gt >= best["tau_g"]) & (ct >= best["tau_c"]), lt, ABSTAIN)
            met = selective_metrics(y[test], pred)
            metric_rows.append({
                "detector": detector,
                "repeat": repeat,
                "mean_iou": mean_iou,
                "event_recall_iou_ge_0_3": event_recall,
                **met,
                "validation_feasible": best["feasible"],
                "tau_g": best["tau_g"],
                "tau_c": best["tau_c"],
            })
            proposal_by_idx = proposals.set_index("global_index")
            prediction_rows.extend(
                {
                    "detector": detector,
                    "repeat": repeat,
                    "index": int(i),
                    "truth": str(y[i]),
                    "prediction": str(p),
                    "gate_confidence": float(g),
                    "mechanism_confidence": float(c),
                    "confidence": float(g*c),
                    "iou": float(proposal_by_idx.loc[int(i), "iou"]),
                    "severity": str(metadata.iloc[int(i)].severity),
                }
                for i, p, g, c in zip(test, pred, gt, ct)
            )
            joblib.dump({"gate": gate, "resolver": resolver}, args.out_dir / "models" / f"{detector}_repeat_{repeat}.joblib")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.out_dir / "per_repeat_detector_metrics.csv", index=False)
    summary = metrics.groupby("detector", sort=False).agg(
        mean_iou=("mean_iou", "mean"),
        event_recall_iou_ge_0_3=("event_recall_iou_ge_0_3", "mean"),
        selected_accuracy_mean=("selected_accuracy", "mean"), selected_accuracy_std=("selected_accuracy", lambda x: x.std(ddof=0)),
        coverage_mean=("coverage", "mean"), coverage_std=("coverage", lambda x: x.std(ddof=0)),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", lambda x: x.std(ddof=0)),
        false_specific_rate_mean=("false_specific_rate", "mean"), false_specific_rate_std=("false_specific_rate", lambda x: x.std(ddof=0)),
    ).reset_index()
    summary.to_csv(args.out_dir / "detector_summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.out_dir / "predictions" / "all_detector_predictions.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(args.out_dir / "threshold_search.csv", index=False)
    (args.out_dir / "audit.json").write_text(json.dumps({"detectors": DETECTORS, "repeats": args.repeats}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
