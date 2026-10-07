#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT = Path("/root/phyguard_revision")
RESULT_ROOT = ROOT / "artifacts" / "temporal_baseline_results_v1"
TASK_ROOT = RESULT_ROOT / "tasks"
PREDICTION_ROOT = RESULT_ROOT / "predictions"

R0 = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
)

EVALUATION_COMMON_PATH = R0 / "scripts" / "evaluation_common.py"

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "temporal_baseline_metric_repair_audit_v1"
)

SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
REPEAT_PATH = OUTPUT_ROOT / "corrected_per_repeat_metrics.csv"
AGGREGATE_PATH = OUTPUT_ROOT / "corrected_aggregate_metrics.csv"
COMPARISON_PATH = OUTPUT_ROOT / "old_vs_corrected_metrics.csv"
PREDICTIONS_PATH = OUTPUT_ROOT / "corrected_predictions.csv"
HELPER_SOURCE_PATH = OUTPUT_ROOT / "selective_metrics_source.txt"
NOTES_PATH = OUTPUT_ROOT / "SCIENTIFIC_DIAGNOSIS.md"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "temporal_baseline_metric_repair_audit_v1.json"
)

MODEL_ORDER = [
    "ROCKET",
    "MINIROCKET",
    "MULTIROCKET",
    "HYDRA",
    "MULTIROCKET_HYDRA",
    "ARSENAL",
    "FCN1D",
    "RESNET1D",
    "INCEPTIONTIME_LITE",
    "LITE1D",
    "TCN",
    "BIGRU",
    "BILSTM",
    "TINY_TRANSFORMER",
]

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


def detect_abstain_token(module) -> tuple[str, list[dict]]:
    candidates: list[str] = []

    for name, value in vars(module).items():
        if "ABSTAIN" in str(name).upper() and isinstance(value, str):
            candidates.append(value)

    source = EVALUATION_COMMON_PATH.read_text(
        encoding="utf-8",
        errors="replace",
    )

    for match in re.finditer(
        r"""["']([^"']*abstain[^"']*)["']""",
        source,
        flags=re.IGNORECASE,
    ):
        candidates.append(match.group(1))

    candidates.extend(["ABSTAIN", "abstain", "Abstain", "__ABSTAIN__"])

    ordered: list[str] = []
    seen = set()

    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)

    y_true = np.asarray(
        [
            "interference",
            "blockage",
            "mobility",
            "adaptation_mismatch",
            "normal",
            "nonphysical_goodput",
        ],
        dtype=object,
    )

    records = []

    for candidate in ordered:
        try:
            metrics = module.selective_metrics(
                y_true,
                np.full(len(y_true), candidate, dtype=object),
            )
            coverage = float(metrics["coverage"])
            records.append(
                {
                    "candidate": candidate,
                    "coverage": coverage,
                    "metrics": metrics,
                }
            )
        except Exception as error:
            records.append(
                {
                    "candidate": candidate,
                    "coverage": None,
                    "error": repr(error),
                }
            )

    zero_candidates = [
        record["candidate"]
        for record in records
        if record.get("coverage") is not None
        and abs(float(record["coverage"])) <= 1e-12
    ]

    require(
        zero_candidates,
        f"No candidate abstention token produced zero coverage: {records}",
    )

    token = "ABSTAIN" if "ABSTAIN" in zero_candidates else zero_candidates[0]
    return token, records


def recompute_predictions(
    frame: pd.DataFrame,
    abstain_token: str,
) -> np.ndarray:
    required = {
        "truth",
        "candidate_prediction",
        "confidence",
        "tau",
    }

    require(
        required.issubset(frame.columns),
        f"Prediction file lacks columns: {required - set(frame.columns)}",
    )

    tau_values = frame["tau"].dropna().unique()
    require(
        len(tau_values) == 1,
        f"Expected one threshold; found {tau_values}.",
    )

    tau = float(tau_values[0])
    candidate = (
        frame["candidate_prediction"]
        .astype(str)
        .to_numpy(dtype=object)
    )
    confidence = frame["confidence"].to_numpy(dtype=float)

    emit = np.isin(candidate, PHYSICAL_LABELS) & (confidence >= tau)

    return np.where(
        emit,
        candidate,
        abstain_token,
    ).astype(object)


def additional_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    abstain_token: str,
) -> dict:
    physical_mask = np.isin(truth, PHYSICAL_LABELS)
    control_mask = np.isin(truth, CONTROL_LABELS)
    selected_mask = prediction != abstain_token

    return {
        "selected_count": int(np.sum(selected_mask)),
        "sample_count": int(len(truth)),
        "physical_selection_rate": float(
            np.mean(prediction[physical_mask] != abstain_token)
        ),
        "physical_exact_match": float(
            np.mean(prediction[physical_mask] == truth[physical_mask])
        ),
        "control_abstention": float(
            np.mean(prediction[control_mask] == abstain_token)
        ),
        "all_sample_accuracy_with_abstain_as_error": float(
            accuracy_score(truth, prediction)
        ),
        "all_sample_macro_f1_with_abstain_class": float(
            f1_score(
                truth,
                prediction,
                labels=PHYSICAL_LABELS + CONTROL_LABELS + [abstain_token],
                average="macro",
                zero_division=0,
            )
        ),
        "all_sample_balanced_accuracy_with_abstain_as_error": float(
            balanced_accuracy_score(truth, prediction)
        ),
    }


for path in [
    RESULT_ROOT,
    TASK_ROOT,
    PREDICTION_ROOT,
    EVALUATION_COMMON_PATH,
]:
    require(path.exists(), f"Required input is missing: {path}")

for path in [OUTPUT_ROOT, MANIFEST_PATH]:
    require(not path.exists(), f"Audit output already exists: {path}")

evaluation_common = load_module(
    "phyguard_evaluation_common_stage64c2",
    EVALUATION_COMMON_PATH,
)

require(
    hasattr(evaluation_common, "selective_metrics"),
    "evaluation_common lacks selective_metrics.",
)

abstain_token, token_records = detect_abstain_token(evaluation_common)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

HELPER_SOURCE_PATH.write_text(
    inspect.getsource(evaluation_common.selective_metrics),
    encoding="utf-8",
)

repeat_rows = []
comparison_rows = []
corrected_prediction_frames = []
old_coverages = []
old_false_specific_rates = []

for model_id in MODEL_ORDER:
    for repeat in range(5):
        task_name = f"{model_id}__repeat_{repeat}"
        task_path = TASK_ROOT / f"{task_name}.json"
        prediction_path = PREDICTION_ROOT / f"{task_name}.csv"

        require(task_path.exists(), f"Missing task result: {task_path}")
        require(
            prediction_path.exists(),
            f"Missing prediction file: {prediction_path}",
        )

        task = json.loads(task_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(prediction_path)

        truth = frame["truth"].astype(str).to_numpy(dtype=object)
        repaired_prediction = recompute_predictions(
            frame,
            abstain_token,
        )

        repaired_metrics = evaluation_common.selective_metrics(
            truth,
            repaired_prediction,
        )
        repaired_metrics.update(
            additional_metrics(
                truth,
                repaired_prediction,
                abstain_token,
            )
        )

        old_metrics = task["metrics"]
        old_coverages.append(float(old_metrics["coverage"]))
        old_false_specific_rates.append(
            float(old_metrics["false_specific_rate"])
        )

        repeat_row = {
            "model_id": model_id,
            "repeat": repeat,
            "tau": float(task["tau"]),
            **{
                key: float(value)
                for key, value in repaired_metrics.items()
                if isinstance(
                    value,
                    (int, float, np.integer, np.floating),
                )
            },
            "training_seconds": float(task["training_seconds"]),
            "task_seconds": float(task["task_seconds"]),
            "inference_median_ms_per_interval": float(
                task["inference_median_ms_per_interval"]
            ),
            "parameter_count": task["parameter_count"],
            "serialized_model_bytes": int(
                task["serialized_model_bytes"]
            ),
        }
        repeat_rows.append(repeat_row)

        metric_names = sorted(set(old_metrics) | set(repaired_metrics))

        for metric_name in metric_names:
            old_value = old_metrics.get(metric_name)
            repaired_value = repaired_metrics.get(metric_name)

            if isinstance(
                old_value,
                (int, float, np.integer, np.floating),
            ):
                old_value = float(old_value)

            if isinstance(
                repaired_value,
                (int, float, np.integer, np.floating),
            ):
                repaired_value = float(repaired_value)

            delta = None
            if (
                isinstance(old_value, float)
                and isinstance(repaired_value, float)
                and math.isfinite(old_value)
                and math.isfinite(repaired_value)
            ):
                delta = repaired_value - old_value

            comparison_rows.append(
                {
                    "model_id": model_id,
                    "repeat": repeat,
                    "metric": metric_name,
                    "old_value": old_value,
                    "corrected_value": repaired_value,
                    "corrected_minus_old": delta,
                }
            )

        repaired_frame = frame.copy()
        repaired_frame["old_prediction"] = repaired_frame["prediction"]
        repaired_frame["corrected_prediction"] = repaired_prediction
        repaired_frame["abstain_token"] = abstain_token
        repaired_frame["mapping_changed"] = (
            repaired_frame["old_prediction"].astype(str)
            != repaired_frame["corrected_prediction"].astype(str)
        )
        corrected_prediction_frames.append(repaired_frame)


repeat_frame = pd.DataFrame(repeat_rows)
comparison_frame = pd.DataFrame(comparison_rows)
predictions_frame = pd.concat(
    corrected_prediction_frames,
    ignore_index=True,
)

aggregate_rows = []

numeric_columns = [
    column
    for column in repeat_frame.columns
    if column not in {"model_id", "repeat"}
    and pd.api.types.is_numeric_dtype(repeat_frame[column])
]

for model_id, group in repeat_frame.groupby("model_id", sort=False):
    row = {"model_id": model_id}

    for column in numeric_columns:
        values = group[column].dropna()
        row[f"{column}_mean"] = (
            float(values.mean()) if len(values) else math.nan
        )
        row[f"{column}_std"] = (
            float(values.std(ddof=0)) if len(values) else math.nan
        )

    aggregate_rows.append(row)

aggregate_frame = pd.DataFrame(aggregate_rows)

repeat_frame.to_csv(REPEAT_PATH, index=False)
aggregate_frame.to_csv(AGGREGATE_PATH, index=False)
comparison_frame.to_csv(COMPARISON_PATH, index=False)
predictions_frame.to_csv(PREDICTIONS_PATH, index=False)

old_all_coverage_one = bool(
    np.allclose(old_coverages, 1.0, rtol=0.0, atol=0.0)
)
old_all_fsr_zero = bool(
    np.allclose(old_false_specific_rates, 0.0, rtol=0.0, atol=0.0)
)

changed_count = int(predictions_frame["mapping_changed"].sum())
prediction_count = int(len(predictions_frame))

summary = {
    "schema": "phyguard.temporal_baseline_metric_repair_audit.v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS",
    "scientific_status": "V1_METRICS_REQUIRE_REPAIR_BEFORE_FREEZE",
    "detected_abstain_token": abstain_token,
    "abstain_token_detection_records": token_records,
    "task_count": int(len(repeat_frame)),
    "prediction_count": prediction_count,
    "changed_prediction_count": changed_count,
    "changed_prediction_fraction": changed_count / prediction_count,
    "old_metric_red_flags": {
        "all_70_coverages_equal_one": old_all_coverage_one,
        "all_70_false_specific_rates_equal_zero": old_all_fsr_zero,
    },
    "corrected_coverage_range": [
        float(repeat_frame["coverage"].min()),
        float(repeat_frame["coverage"].max()),
    ],
    "retraining_required": False,
    "threshold_reselection_required": False,
    "threshold_reason": (
        "Validation thresholds were selected by the existing "
        "single_threshold_rows helper. The defect is confined to "
        "the runner's test-time mapping, which wrote 'normal' instead "
        "of the helper-compatible abstention token."
    ),
    "formal_hardware": "NVIDIA GeForce RTX 3080 10 GB",
    "outputs": {
        "corrected_per_repeat_metrics": str(REPEAT_PATH.relative_to(ROOT)),
        "corrected_aggregate_metrics": str(
            AGGREGATE_PATH.relative_to(ROOT)
        ),
        "old_vs_corrected_metrics": str(COMPARISON_PATH.relative_to(ROOT)),
        "corrected_predictions": str(PREDICTIONS_PATH.relative_to(ROOT)),
    },
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

NOTES_PATH.write_text(
    "\n".join(
        [
            "# Stage 64C2 scientific diagnosis",
            "",
            "The 70 model-training tasks completed successfully.",
            "Retraining is not required.",
            "",
            "The v1 test mapper wrote `normal` for rejected outputs,",
            "while the locked metric helper uses a dedicated abstention",
            f"token: `{abstain_token}`.",
            "",
            "Therefore the v1 aggregate table must not be frozen or used",
            "in the manuscript. This audit recomputes test predictions",
            "and metrics with the original validation-selected threshold.",
            "",
        ]
    ),
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
    "schema": (
        "phyguard.temporal_baseline_metric_repair_audit_manifest.v1"
    ),
    "status": "PASS",
    "evaluation_common_sha256": sha256_file(EVALUATION_COMMON_PATH),
    "result_summary_sha256": sha256_file(RESULT_ROOT / "summary.json"),
    "files": files,
    "retraining_required": False,
    "threshold_reselection_required": False,
    "formal_hardware": "NVIDIA GeForce RTX 3080 10 GB",
}

MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

display_columns = [
    "model_id",
    "selected_accuracy_mean",
    "selected_accuracy_std",
    "coverage_mean",
    "coverage_std",
    "false_specific_rate_mean",
    "physical_selection_rate_mean",
    "physical_exact_match_mean",
    "control_abstention_mean",
]

display_columns = [
    column
    for column in display_columns
    if column in aggregate_frame.columns
]

print("status: PASS")
print("scientific_status: V1_METRICS_REQUIRE_REPAIR_BEFORE_FREEZE")
print("detected_abstain_token:", abstain_token)
print("old_all_70_coverages_equal_one:", old_all_coverage_one)
print("old_all_70_false_specific_rates_equal_zero:", old_all_fsr_zero)
print("changed_predictions:", f"{changed_count}/{prediction_count}")
print("retraining_required: False")
print("threshold_reselection_required: False")
print()
print("CORRECTED AGGREGATE METRICS")
print(
    aggregate_frame[display_columns]
    .sort_values("selected_accuracy_mean", ascending=False)
    .to_string(index=False)
)
print()
print("summary:", SUMMARY_PATH)
print("corrected_aggregate:", AGGREGATE_PATH)
print("old_vs_corrected:", COMPARISON_PATH)
print("corrected_predictions:", PREDICTIONS_PATH)
print("manifest:", MANIFEST_PATH)
print()
print("TEMPORAL_BASELINE_METRIC_REPAIR_AUDIT_V1_PASS")
