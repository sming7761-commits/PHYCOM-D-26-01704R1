"""Reconstruct the complete repeated main comparison suite for PhyGuard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

METHOD_ORDER = [
    "Single-KPI rules",
    "Logistic regression",
    "Random forest",
    "Gradient boosting",
    "Two-stage without physics",
    "PhyGuard without abstention",
    "PhyGuard",
]


def fit_two_stage(raw_train, physical_train, combined_train, y_train, seed, use_physics: bool):
    gate_x = physical_train if use_physics else raw_train
    resolver_x = combined_train if use_physics else raw_train
    gate = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=seed),
    )
    gate.fit(gate_x, np.isin(y_train, PHYS).astype(int))
    physical_mask = np.isin(y_train, PHYS)
    resolver = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    resolver.fit(resolver_x[physical_mask], y_train[physical_mask])
    return gate, resolver


def two_stage_probabilities(gate, resolver, raw, physical, combined, use_physics: bool):
    gate_x = physical if use_physics else raw
    resolver_x = combined if use_physics else raw
    gate_probability = gate.predict_proba(gate_x)[:, 1]
    mechanism_probability = resolver.predict_proba(resolver_x)
    mechanism_confidence = mechanism_probability.max(axis=1)
    mechanism_label = resolver.classes_[mechanism_probability.argmax(axis=1)]
    return gate_probability, mechanism_confidence, mechanism_label


def generic_model(name: str, seed: int):
    if name == "Logistic regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            ),
        )
    if name == "Random forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    if name == "Gradient boosting":
        return HistGradientBoostingClassifier(max_iter=220, random_state=seed)
    raise ValueError(name)


def split_indices(metadata: pd.DataFrame, seed: int):
    idx = np.arange(len(metadata))
    keys = stratification_keys(metadata, include_regime=True)
    trainval, test = train_test_split(idx, test_size=0.20, random_state=seed, stratify=keys)
    train, val = train_test_split(
        trainval,
        test_size=0.25,
        random_state=seed + 100,
        stratify=keys[trainval],
    )
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260705)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "splits").mkdir(exist_ok=True)
    (args.out_dir / "models").mkdir(exist_ok=True)
    (args.out_dir / "predictions").mkdir(exist_ok=True)

    metadata = pd.read_csv(args.data_dir / "metadata.csv")
    feature_data = np.load(args.feature_dir / "features.npz")
    raw = feature_data["raw"]
    physical = feature_data["physical"]
    combined = feature_data["combined"]
    y = metadata["label"].to_numpy(dtype=object)

    metric_rows = []
    threshold_rows = []
    all_predictions = []

    for repeat in range(args.repeats):
        run_seed = args.seed + repeat
        train, val, test = split_indices(metadata, run_seed)
        np.savez_compressed(args.out_dir / "splits" / f"repeat_{repeat}.npz", train=train, val=val, test=test)

        # Single-KPI rules: the largest physical score is proposed. The margin
        # over the non-physical score is a transparent evidence requirement.
        val_scores = physical[val, 12:16]
        test_scores = physical[test, 12:16]
        val_labels = PHYS[val_scores.argmax(axis=1)]
        test_labels = PHYS[test_scores.argmax(axis=1)]
        val_conf = val_scores.max(axis=1) - physical[val, 16]
        test_conf = test_scores.max(axis=1) - physical[test, 16]
        thresholds = np.unique(np.quantile(val_conf, np.linspace(0.0, 1.0, 41)))
        best, rows = single_threshold_rows(y[val], val_labels, val_conf, thresholds)
        for row in rows:
            threshold_rows.append({"repeat": repeat, "method": "Single-KPI rules", **row})
        pred = np.where(test_conf >= best["tau"], test_labels, ABSTAIN)
        met = selective_metrics(y[test], pred)
        metric_rows.append({"repeat": repeat, "method": "Single-KPI rules", **met, "validation_feasible": best["feasible"], "tau": best["tau"]})
        all_predictions.extend(
            {"repeat": repeat, "method": "Single-KPI rules", "index": int(i), "truth": str(y[i]), "prediction": str(p), "confidence": float(c)}
            for i, p, c in zip(test, pred, test_conf)
        )

        # Generic raw-statistics classifiers.
        for method in ["Logistic regression", "Random forest", "Gradient boosting"]:
            model = generic_model(method, run_seed)
            model.fit(raw[train], y[train])
            val_prob = model.predict_proba(raw[val])
            test_prob = model.predict_proba(raw[test])
            val_label, val_confidence = confidence_from_class_probabilities(val_prob, model.classes_)
            test_label, test_confidence = confidence_from_class_probabilities(test_prob, model.classes_)
            best, rows = single_threshold_rows(
                y[val], val_label, val_confidence, np.round(np.linspace(0.05, 0.95, 19), 2)
            )
            for row in rows:
                threshold_rows.append({"repeat": repeat, "method": method, **row})
            pred = np.where(test_confidence >= best["tau"], test_label, ABSTAIN)
            met = selective_metrics(y[test], pred)
            metric_rows.append({"repeat": repeat, "method": method, **met, "validation_feasible": best["feasible"], "tau": best["tau"]})
            joblib.dump(model, args.out_dir / "models" / f"repeat_{repeat}_{method.lower().replace(' ', '_')}.joblib")
            all_predictions.extend(
                {"repeat": repeat, "method": method, "index": int(i), "truth": str(y[i]), "prediction": str(p), "confidence": float(c)}
                for i, p, c in zip(test, pred, test_confidence)
            )

        # Two-stage without physics.
        gate_raw, resolver_raw = fit_two_stage(raw[train], physical[train], combined[train], y[train], run_seed, use_physics=False)
        gv, cv, lv = two_stage_probabilities(gate_raw, resolver_raw, raw[val], physical[val], combined[val], use_physics=False)
        best, rows = operating_point_rows(y[val], lv, gv, cv)
        for row in rows:
            threshold_rows.append({"repeat": repeat, "method": "Two-stage without physics", **row})
        gt, ct, lt = two_stage_probabilities(gate_raw, resolver_raw, raw[test], physical[test], combined[test], use_physics=False)
        pred = np.where((gt >= best["tau_g"]) & (ct >= best["tau_c"]), lt, ABSTAIN)
        met = selective_metrics(y[test], pred)
        metric_rows.append({"repeat": repeat, "method": "Two-stage without physics", **met, "validation_feasible": best["feasible"], "tau_g": best["tau_g"], "tau_c": best["tau_c"]})
        joblib.dump({"gate": gate_raw, "resolver": resolver_raw}, args.out_dir / "models" / f"repeat_{repeat}_two_stage_without_physics.joblib")
        all_predictions.extend(
            {"repeat": repeat, "method": "Two-stage without physics", "index": int(i), "truth": str(y[i]), "prediction": str(p), "gate_confidence": float(g), "mechanism_confidence": float(c), "confidence": float(g*c)}
            for i, p, g, c in zip(test, pred, gt, ct)
        )

        # Complete PhyGuard and forced-output variant use the same resolver.
        gate, resolver = fit_two_stage(raw[train], physical[train], combined[train], y[train], run_seed, use_physics=True)
        gv, cv, lv = two_stage_probabilities(gate, resolver, raw[val], physical[val], combined[val], use_physics=True)
        best, rows = operating_point_rows(y[val], lv, gv, cv)
        for row in rows:
            threshold_rows.append({"repeat": repeat, "method": "PhyGuard", **row})
        gt, ct, lt = two_stage_probabilities(gate, resolver, raw[test], physical[test], combined[test], use_physics=True)
        pred = np.where((gt >= best["tau_g"]) & (ct >= best["tau_c"]), lt, ABSTAIN)
        met = selective_metrics(y[test], pred)
        metric_rows.append({"repeat": repeat, "method": "PhyGuard", **met, "validation_feasible": best["feasible"], "tau_g": best["tau_g"], "tau_c": best["tau_c"]})
        all_predictions.extend(
            {"repeat": repeat, "method": "PhyGuard", "index": int(i), "truth": str(y[i]), "prediction": str(p), "gate_confidence": float(g), "mechanism_confidence": float(c), "confidence": float(g*c)}
            for i, p, g, c in zip(test, pred, gt, ct)
        )

        forced_pred = lt.astype(object)
        forced_met = selective_metrics(y[test], forced_pred)
        metric_rows.append({"repeat": repeat, "method": "PhyGuard without abstention", **forced_met, "validation_feasible": True})
        all_predictions.extend(
            {"repeat": repeat, "method": "PhyGuard without abstention", "index": int(i), "truth": str(y[i]), "prediction": str(p), "mechanism_confidence": float(c), "confidence": float(c)}
            for i, p, c in zip(test, forced_pred, ct)
        )
        joblib.dump({"gate": gate, "resolver": resolver}, args.out_dir / "models" / f"repeat_{repeat}_phyguard.joblib")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df["method"] = pd.Categorical(metrics_df["method"], categories=METHOD_ORDER, ordered=True)
    metrics_df = metrics_df.sort_values(["method", "repeat"])
    metrics_df.to_csv(args.out_dir / "per_repeat_all_methods.csv", index=False)
    summary = summary_from_rows(metrics_df, "method")
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    summary = summary.sort_values("method")
    summary.to_csv(args.out_dir / "main_summary.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(args.out_dir / "threshold_search_all_methods.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(args.out_dir / "predictions" / "all_test_predictions.csv", index=False)

    audit = {
        "n_sequences": int(len(y)),
        "repeats": int(args.repeats),
        "methods": METHOD_ORDER,
        "all_phyguard_validation_points_feasible": bool(
            metrics_df.loc[metrics_df["method"] == "PhyGuard", "validation_feasible"].all()
        ),
        "reconstruction_notice": "Generic baseline rejection and single-KPI evidence thresholds are transparent reconstruction choices because the submission did not preserve exact code.",
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
