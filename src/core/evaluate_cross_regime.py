"""Leave-one-regime-out reconstruction for PhyGuard and selected baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluation_common import (
    ABSTAIN,
    PHYS,
    confidence_from_class_probabilities,
    operating_point_rows,
    selective_metrics,
    single_threshold_rows,
    stratification_keys,
    summary_from_rows,
)
from evaluate_suite import generic_model, fit_two_stage, two_stage_probabilities

METHODS = ["Random forest", "Gradient boosting", "Two-stage without physics", "PhyGuard"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260705)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "models").mkdir(exist_ok=True)
    (args.out_dir / "predictions").mkdir(exist_ok=True)

    metadata = pd.read_csv(args.data_dir / "metadata.csv")
    features = np.load(args.feature_dir / "features.npz")
    raw, physical, combined = features["raw"], features["physical"], features["combined"]
    y = metadata["label"].to_numpy(dtype=object)

    metric_rows, prediction_rows, threshold_rows = [], [], []
    regimes = list(metadata["regime"].drop_duplicates())
    for fold, held_out in enumerate(regimes):
        test = np.flatnonzero(metadata["regime"].to_numpy() == held_out)
        source = np.flatnonzero(metadata["regime"].to_numpy() != held_out)
        source_meta = metadata.iloc[source]
        keys = stratification_keys(source_meta, include_regime=True)
        train_local, val_local = train_test_split(
            np.arange(len(source)), test_size=0.25, random_state=args.seed + fold, stratify=keys
        )
        train, val = source[train_local], source[val_local]
        np.savez_compressed(args.out_dir / f"split_{held_out}.npz", train=train, val=val, test=test)

        for method in ["Random forest", "Gradient boosting"]:
            model = generic_model(method, args.seed + fold)
            model.fit(raw[train], y[train])
            pv = model.predict_proba(raw[val])
            pt = model.predict_proba(raw[test])
            lv, cv = confidence_from_class_probabilities(pv, model.classes_)
            lt, ct = confidence_from_class_probabilities(pt, model.classes_)
            best, rows = single_threshold_rows(y[val], lv, cv, np.round(np.linspace(0.05, 0.95, 19), 2))
            threshold_rows.extend({"held_out_regime": held_out, "method": method, **row} for row in rows)
            pred = np.where(ct >= best["tau"], lt, ABSTAIN)
            metric_rows.append({"held_out_regime": held_out, "method": method, **selective_metrics(y[test], pred), "validation_feasible": best["feasible"], "tau": best["tau"]})
            prediction_rows.extend(
                {"held_out_regime": held_out, "method": method, "index": int(i), "truth": str(y[i]), "prediction": str(p), "confidence": float(c)}
                for i, p, c in zip(test, pred, ct)
            )
            joblib.dump(model, args.out_dir / "models" / f"{held_out}_{method.lower().replace(' ', '_')}.joblib")

        for method, use_physics in [("Two-stage without physics", False), ("PhyGuard", True)]:
            gate, resolver = fit_two_stage(raw[train], physical[train], combined[train], y[train], args.seed + fold, use_physics)
            gv, cv, lv = two_stage_probabilities(gate, resolver, raw[val], physical[val], combined[val], use_physics)
            best, rows = operating_point_rows(y[val], lv, gv, cv)
            threshold_rows.extend({"held_out_regime": held_out, "method": method, **row} for row in rows)
            gt, ct, lt = two_stage_probabilities(gate, resolver, raw[test], physical[test], combined[test], use_physics)
            pred = np.where((gt >= best["tau_g"]) & (ct >= best["tau_c"]), lt, ABSTAIN)
            metric_rows.append({"held_out_regime": held_out, "method": method, **selective_metrics(y[test], pred), "validation_feasible": best["feasible"], "tau_g": best["tau_g"], "tau_c": best["tau_c"]})
            prediction_rows.extend(
                {"held_out_regime": held_out, "method": method, "index": int(i), "truth": str(y[i]), "prediction": str(p), "gate_confidence": float(g), "mechanism_confidence": float(c), "confidence": float(g*c)}
                for i, p, g, c in zip(test, pred, gt, ct)
            )
            joblib.dump({"gate": gate, "resolver": resolver}, args.out_dir / "models" / f"{held_out}_{method.lower().replace(' ', '_')}.joblib")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.out_dir / "per_regime_metrics.csv", index=False)
    summary_from_rows(metrics_df, "method").to_csv(args.out_dir / "cross_regime_summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.out_dir / "predictions" / "all_cross_regime_predictions.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(args.out_dir / "threshold_search.csv", index=False)
    (args.out_dir / "audit.json").write_text(json.dumps({"held_out_regimes": regimes, "methods": METHODS}, indent=2), encoding="utf-8")
    print(summary_from_rows(metrics_df, "method").to_string(index=False))


if __name__ == "__main__":
    main()
