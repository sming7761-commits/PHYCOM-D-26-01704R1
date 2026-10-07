#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/phyguard_revision")

R0 = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
)

CORE_METRICS_PATH = (
    R0
    / "results"
    / "full"
    / "evaluation"
    / "per_repeat_metrics.csv"
)

SUITE_METRICS_PATH = (
    R0
    / "results"
    / "full"
    / "evaluation_suite"
    / "per_repeat_all_methods.csv"
)

SUITE_PREDICTIONS_PATH = (
    R0
    / "results"
    / "full"
    / "evaluation_suite"
    / "predictions"
    / "all_test_predictions.csv"
)

EVALUATION_COMMON_PATH = (
    R0
    / "scripts"
    / "evaluation_common.py"
)

REPAIR_ROOT = (
    ROOT
    / "artifacts"
    / "temporal_baseline_metric_repair_audit_v1"
)

REPAIR_SUMMARY_PATH = REPAIR_ROOT / "summary.json"
BASELINE_REPEAT_PATH = REPAIR_ROOT / "corrected_per_repeat_metrics.csv"
BASELINE_AGGREGATE_PATH = REPAIR_ROOT / "corrected_aggregate_metrics.csv"
BASELINE_PREDICTIONS_PATH = REPAIR_ROOT / "corrected_predictions.csv"

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "phyguard_temporal_comparison_audit_v1"
)

PHYGUARD_REPEAT_PATH = OUTPUT_ROOT / "phyguard_per_repeat_metrics.csv"
PHYGUARD_AGGREGATE_PATH = OUTPUT_ROOT / "phyguard_aggregate_metrics.csv"
PAIRWISE_PATH = OUTPUT_ROOT / "pairwise_comparison.csv"
METHOD_MATCH_PATH = OUTPUT_ROOT / "phyguard_method_identification.csv"
ALIGNMENT_PATH = OUTPUT_ROOT / "prediction_alignment.json"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
NOTES_PATH = OUTPUT_ROOT / "MANUSCRIPT_INTERPRETATION.md"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "phyguard_temporal_comparison_audit_v1.json"
)

PHYSICAL_LABELS = [
    "interference",
    "blockage",
    "mobility",
    "adaptation_mismatch",
]

CONTROL_LABELS = [
    "normal",
    "nonphysical_goodput",
]

CORE_METRIC_COLUMNS = [
    "selected_accuracy",
    "coverage",
    "false_specific_rate",
    "macro_f1",
    "balanced_accuracy",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))

    spec = importlib.util.spec_from_file_location(name, path)

    require(
        spec is not None and spec.loader is not None,
        f"Could not load module: {path}",
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def additional_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    abstain_token: str,
) -> dict[str, float]:
    physical_mask = np.isin(truth, PHYSICAL_LABELS)
    control_mask = np.isin(truth, CONTROL_LABELS)

    return {
        "physical_selection_rate": float(
            np.mean(prediction[physical_mask] != abstain_token)
        ),
        "physical_exact_match": float(
            np.mean(prediction[physical_mask] == truth[physical_mask])
        ),
        "control_abstention": float(
            np.mean(prediction[control_mask] == abstain_token)
        ),
    }


def aggregate_repeat_frame(
    frame: pd.DataFrame,
    name_column: str,
) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in frame.columns
        if column not in {name_column, "repeat"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]

    rows = []

    for name, group in frame.groupby(name_column, sort=False):
        row = {name_column: name}

        for column in numeric_columns:
            values = group[column].dropna()
            row[f"{column}_mean"] = (
                float(values.mean()) if len(values) else math.nan
            )
            row[f"{column}_std"] = (
                float(values.std(ddof=0)) if len(values) else math.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


for path in [
    CORE_METRICS_PATH,
    SUITE_METRICS_PATH,
    SUITE_PREDICTIONS_PATH,
    EVALUATION_COMMON_PATH,
    REPAIR_SUMMARY_PATH,
    BASELINE_REPEAT_PATH,
    BASELINE_AGGREGATE_PATH,
    BASELINE_PREDICTIONS_PATH,
]:
    require(path.exists(), f"Required input is missing: {path}")

for path in [OUTPUT_ROOT, MANIFEST_PATH]:
    require(not path.exists(), f"Audit output already exists: {path}")


repair_summary = json.loads(
    REPAIR_SUMMARY_PATH.read_text(encoding="utf-8")
)

require(
    repair_summary.get("status") == "PASS",
    "Stage 64C2 repair audit is not PASS.",
)

abstain_token = str(
    repair_summary["detected_abstain_token"]
)

core_metrics = pd.read_csv(CORE_METRICS_PATH)
suite_metrics = pd.read_csv(SUITE_METRICS_PATH)
suite_predictions = pd.read_csv(SUITE_PREDICTIONS_PATH)
baseline_repeat = pd.read_csv(BASELINE_REPEAT_PATH)
baseline_aggregate = pd.read_csv(BASELINE_AGGREGATE_PATH)
baseline_predictions = pd.read_csv(BASELINE_PREDICTIONS_PATH)

for column in ["repeat", *CORE_METRIC_COLUMNS]:
    require(
        column in core_metrics.columns,
        f"Core metrics lack column: {column}",
    )

for column in ["repeat", "method", *CORE_METRIC_COLUMNS]:
    require(
        column in suite_metrics.columns,
        f"Suite metrics lack column: {column}",
    )

for column in [
    "repeat",
    "method",
    "index",
    "truth",
    "prediction",
]:
    require(
        column in suite_predictions.columns,
        f"Suite predictions lack column: {column}",
    )

for column in [
    "model_id",
    "repeat",
    "index",
    "truth",
    "corrected_prediction",
]:
    require(
        column in baseline_predictions.columns,
        f"Corrected baseline predictions lack column: {column}",
    )


match_rows = []

core_sorted = core_metrics.sort_values("repeat").reset_index(drop=True)

for method, group in suite_metrics.groupby("method", sort=True):
    group_sorted = group.sort_values("repeat").reset_index(drop=True)

    if len(group_sorted) != len(core_sorted):
        match_rows.append(
            {
                "method": method,
                "repeat_count": len(group_sorted),
                "maximum_absolute_metric_difference": math.inf,
                "exact_core_match": False,
                "name_contains_phyguard": (
                    "phyguard" in str(method).lower()
                ),
            }
        )
        continue

    merged = core_sorted[
        ["repeat", *CORE_METRIC_COLUMNS]
    ].merge(
        group_sorted[
            ["repeat", *CORE_METRIC_COLUMNS]
        ],
        on="repeat",
        how="inner",
        suffixes=("_core", "_suite"),
        validate="one_to_one",
    )

    differences = []

    for metric in CORE_METRIC_COLUMNS:
        differences.extend(
            np.abs(
                merged[f"{metric}_core"].to_numpy(dtype=float)
                - merged[f"{metric}_suite"].to_numpy(dtype=float)
            ).tolist()
        )

    maximum_difference = float(np.max(differences))

    match_rows.append(
        {
            "method": method,
            "repeat_count": len(group_sorted),
            "maximum_absolute_metric_difference": maximum_difference,
            "exact_core_match": maximum_difference <= 1e-10,
            "name_contains_phyguard": (
                "phyguard" in str(method).lower()
            ),
        }
    )

match_frame = pd.DataFrame(match_rows)

exact_matches = match_frame[
    match_frame["exact_core_match"]
].copy()

require(
    len(exact_matches) >= 1,
    (
        "No evaluation-suite method exactly matches the core "
        "PhyGuard per-repeat metrics."
    ),
)

if len(exact_matches) == 1:
    phyguard_method = str(exact_matches.iloc[0]["method"])
else:
    named_matches = exact_matches[
        exact_matches["name_contains_phyguard"]
    ]

    require(
        len(named_matches) == 1,
        (
            "Core metric match is ambiguous and cannot be "
            f"resolved by method name: {exact_matches.to_dict('records')}"
        ),
    )

    phyguard_method = str(named_matches.iloc[0]["method"])


evaluation_common = load_module(
    "phyguard_evaluation_common_stage64c3",
    EVALUATION_COMMON_PATH,
)

require(
    hasattr(evaluation_common, "selective_metrics"),
    "evaluation_common lacks selective_metrics.",
)

phyguard_predictions = suite_predictions[
    suite_predictions["method"].astype(str) == phyguard_method
].copy()

require(
    len(phyguard_predictions) == 5 * 216,
    (
        "Unexpected PhyGuard prediction count: "
        f"{len(phyguard_predictions)}"
    ),
)

require(
    not phyguard_predictions.duplicated(
        subset=["repeat", "index"]
    ).any(),
    "PhyGuard predictions contain duplicate repeat/index rows.",
)

phyguard_repeat_rows = []

for repeat, group in phyguard_predictions.groupby("repeat", sort=True):
    truth = group["truth"].astype(str).to_numpy(dtype=object)
    prediction = (
        group["prediction"]
        .astype(str)
        .to_numpy(dtype=object)
    )

    metrics = evaluation_common.selective_metrics(
        truth,
        prediction,
    )

    metrics.update(
        additional_metrics(
            truth,
            prediction,
            abstain_token,
        )
    )

    phyguard_repeat_rows.append(
        {
            "method": phyguard_method,
            "repeat": int(repeat),
            **{
                key: float(value)
                for key, value in metrics.items()
            },
        }
    )

phyguard_repeat_frame = pd.DataFrame(phyguard_repeat_rows)

recomputed_core = core_metrics[
    ["repeat", *CORE_METRIC_COLUMNS]
].merge(
    phyguard_repeat_frame[
        ["repeat", *CORE_METRIC_COLUMNS]
    ],
    on="repeat",
    how="inner",
    suffixes=("_core", "_recomputed"),
    validate="one_to_one",
)

recomputed_differences = []

for metric in CORE_METRIC_COLUMNS:
    recomputed_differences.extend(
        np.abs(
            recomputed_core[f"{metric}_core"].to_numpy(dtype=float)
            - recomputed_core[f"{metric}_recomputed"].to_numpy(dtype=float)
        ).tolist()
    )

maximum_recomputed_difference = float(
    np.max(recomputed_differences)
)

require(
    maximum_recomputed_difference <= 1e-10,
    (
        "Recomputed PhyGuard metrics do not reproduce the "
        f"core table: max difference={maximum_recomputed_difference}"
    ),
)

phyguard_aggregate_frame = aggregate_repeat_frame(
    phyguard_repeat_frame,
    "method",
)

phyguard_row = phyguard_aggregate_frame.iloc[0]

baseline_keys = baseline_predictions[
    ["model_id", "repeat", "index", "truth"]
].copy()

alignment_records = []

for model_id, group in baseline_keys.groupby("model_id", sort=True):
    joined = group.merge(
        phyguard_predictions[
            ["repeat", "index", "truth"]
        ],
        on=["repeat", "index"],
        how="outer",
        suffixes=("_baseline", "_phyguard"),
        indicator=True,
        validate="one_to_one",
    )

    truth_match = bool(
        np.all(
            joined["truth_baseline"].astype(str)
            == joined["truth_phyguard"].astype(str)
        )
    )

    alignment_records.append(
        {
            "model_id": model_id,
            "joined_rows": int(len(joined)),
            "all_rows_both": bool(
                (joined["_merge"] == "both").all()
            ),
            "truth_match": truth_match,
        }
    )

require(
    all(
        record["joined_rows"] == 1080
        and record["all_rows_both"]
        and record["truth_match"]
        for record in alignment_records
    ),
    f"Prediction alignment failed: {alignment_records}",
)

comparison_rows = []

for _, baseline in baseline_aggregate.iterrows():
    model_id = str(baseline["model_id"])

    comparison = {
        "model_id": model_id,
        "phyguard_method": phyguard_method,
    }

    for metric in [
        "selected_accuracy",
        "coverage",
        "false_specific_rate",
        "macro_f1",
        "balanced_accuracy",
        "physical_selection_rate",
        "physical_exact_match",
        "control_abstention",
    ]:
        baseline_value = float(
            baseline[f"{metric}_mean"]
        )

        phyguard_value = float(
            phyguard_row[f"{metric}_mean"]
        )

        comparison[f"baseline_{metric}"] = baseline_value
        comparison[f"phyguard_{metric}"] = phyguard_value
        comparison[f"baseline_minus_phyguard_{metric}"] = (
            baseline_value - phyguard_value
        )

    core_three_dominates = (
        comparison["baseline_selected_accuracy"]
        >= comparison["phyguard_selected_accuracy"]
        and comparison["baseline_coverage"]
        >= comparison["phyguard_coverage"]
        and comparison["baseline_false_specific_rate"]
        <= comparison["phyguard_false_specific_rate"]
        and (
            comparison["baseline_selected_accuracy"]
            > comparison["phyguard_selected_accuracy"]
            or comparison["baseline_coverage"]
            > comparison["phyguard_coverage"]
            or comparison["baseline_false_specific_rate"]
            < comparison["phyguard_false_specific_rate"]
        )
    )

    full_five_dominates = (
        core_three_dominates
        and comparison["baseline_physical_exact_match"]
        >= comparison["phyguard_physical_exact_match"]
        and comparison["baseline_control_abstention"]
        >= comparison["phyguard_control_abstention"]
        and (
            comparison["baseline_physical_exact_match"]
            > comparison["phyguard_physical_exact_match"]
            or comparison["baseline_control_abstention"]
            > comparison["phyguard_control_abstention"]
        )
    )

    comparison["baseline_pareto_dominates_core_three"] = bool(
        core_three_dominates
    )
    comparison["baseline_pareto_dominates_full_five"] = bool(
        full_five_dominates
    )

    comparison_rows.append(comparison)

pairwise_frame = pd.DataFrame(comparison_rows)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

match_frame.to_csv(METHOD_MATCH_PATH, index=False)
phyguard_repeat_frame.to_csv(PHYGUARD_REPEAT_PATH, index=False)
phyguard_aggregate_frame.to_csv(PHYGUARD_AGGREGATE_PATH, index=False)
pairwise_frame.to_csv(PAIRWISE_PATH, index=False)

alignment_payload = {
    "status": "PASS",
    "phyguard_method": phyguard_method,
    "expected_rows_per_model": 1080,
    "records": alignment_records,
}

ALIGNMENT_PATH.write_text(
    json.dumps(alignment_payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

core_dominators = pairwise_frame[
    pairwise_frame["baseline_pareto_dominates_core_three"]
]["model_id"].astype(str).tolist()

full_dominators = pairwise_frame[
    pairwise_frame["baseline_pareto_dominates_full_five"]
]["model_id"].astype(str).tolist()

higher_selected_accuracy = pairwise_frame[
    pairwise_frame[
        "baseline_minus_phyguard_selected_accuracy"
    ] > 0
]["model_id"].astype(str).tolist()

lower_false_specific_rate = pairwise_frame[
    pairwise_frame[
        "baseline_minus_phyguard_false_specific_rate"
    ] < 0
]["model_id"].astype(str).tolist()

higher_control_abstention = pairwise_frame[
    pairwise_frame[
        "baseline_minus_phyguard_control_abstention"
    ] > 0
]["model_id"].astype(str).tolist()

summary = {
    "schema": "phyguard.temporal_comparison_audit.v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS",
    "phyguard_method_identified": phyguard_method,
    "method_match_maximum_difference": float(
        exact_matches[
            exact_matches["method"].astype(str) == phyguard_method
        ]["maximum_absolute_metric_difference"].iloc[0]
    ),
    "recomputed_core_maximum_difference": (
        maximum_recomputed_difference
    ),
    "prediction_alignment": "PASS",
    "phyguard_aggregate": {
        metric: float(phyguard_row[f"{metric}_mean"])
        for metric in [
            "selected_accuracy",
            "coverage",
            "false_specific_rate",
            "macro_f1",
            "balanced_accuracy",
            "physical_selection_rate",
            "physical_exact_match",
            "control_abstention",
        ]
    },
    "baseline_models_with_higher_selected_accuracy": (
        higher_selected_accuracy
    ),
    "baseline_models_with_lower_false_specific_rate": (
        lower_false_specific_rate
    ),
    "baseline_models_with_higher_control_abstention": (
        higher_control_abstention
    ),
    "baseline_pareto_dominators_core_three": core_dominators,
    "baseline_pareto_dominators_full_five": full_dominators,
    "formal_hardware": "NVIDIA GeForce RTX 3080 10 GB",
    "interpretation_boundary": (
        "This is an exact same-split descriptive comparison. "
        "No statistical superiority claim is made at this stage."
    ),
}

SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ),
    encoding="utf-8",
)

notes = [
    "# PhyGuard versus modern temporal baselines",
    "",
    "This audit identifies the exact PhyGuard row by matching the",
    "evaluation-suite metrics to the frozen core per-repeat table.",
    "",
    "All comparisons use the same five repeats, the same 216 test",
    "samples per repeat, and exact repeat/index/truth alignment.",
    "",
    "The table is descriptive. A later paired uncertainty analysis",
    "is required before any superiority or non-inferiority claim.",
    "",
    "A baseline with higher selected accuracy alone does not establish",
    "overall dominance. Coverage, false-specific rate, physical exact",
    "match, and control abstention must be considered jointly.",
    "",
]

NOTES_PATH.write_text(
    "\n".join(notes),
    encoding="utf-8",
)

files = []

for path in sorted(OUTPUT_ROOT.rglob("*")):
    if path.is_file():
        files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

manifest = {
    "schema": "phyguard.temporal_comparison_audit_manifest.v1",
    "status": "PASS",
    "core_metrics_sha256": sha256_file(CORE_METRICS_PATH),
    "suite_metrics_sha256": sha256_file(SUITE_METRICS_PATH),
    "suite_predictions_sha256": sha256_file(
        SUITE_PREDICTIONS_PATH
    ),
    "corrected_baseline_predictions_sha256": sha256_file(
        BASELINE_PREDICTIONS_PATH
    ),
    "files": files,
    "formal_hardware": "NVIDIA GeForce RTX 3080 10 GB",
}

MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

display_columns = [
    "model_id",
    "baseline_selected_accuracy",
    "phyguard_selected_accuracy",
    "baseline_minus_phyguard_selected_accuracy",
    "baseline_coverage",
    "phyguard_coverage",
    "baseline_false_specific_rate",
    "phyguard_false_specific_rate",
    "baseline_physical_exact_match",
    "phyguard_physical_exact_match",
    "baseline_control_abstention",
    "phyguard_control_abstention",
    "baseline_pareto_dominates_core_three",
    "baseline_pareto_dominates_full_five",
]

print("status: PASS")
print("phyguard_method_identified:", phyguard_method)
print(
    "method_match_maximum_difference:",
    summary["method_match_maximum_difference"],
)
print(
    "recomputed_core_maximum_difference:",
    maximum_recomputed_difference,
)
print("prediction_alignment: PASS")
print()
print("PHYGUARD AGGREGATE")
print(
    phyguard_aggregate_frame.to_string(index=False)
)
print()
print("PAIRWISE DESCRIPTIVE COMPARISON")
print(
    pairwise_frame[display_columns]
    .sort_values(
        "baseline_minus_phyguard_selected_accuracy",
        ascending=False,
    )
    .to_string(index=False)
)
print()
print(
    "higher_selected_accuracy:",
    higher_selected_accuracy,
)
print(
    "lower_false_specific_rate:",
    lower_false_specific_rate,
)
print(
    "higher_control_abstention:",
    higher_control_abstention,
)
print(
    "pareto_dominators_core_three:",
    core_dominators,
)
print(
    "pareto_dominators_full_five:",
    full_dominators,
)
print()
print("summary:", SUMMARY_PATH)
print("pairwise:", PAIRWISE_PATH)
print("alignment:", ALIGNMENT_PATH)
print("manifest:", MANIFEST_PATH)
print()
print("PHYGUARD_TEMPORAL_COMPARISON_AUDIT_V1_PASS")
