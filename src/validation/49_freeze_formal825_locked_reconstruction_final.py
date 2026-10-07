import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


ROOT = Path("/root/phyguard_revision")

RESULT_ROOT = (
    ROOT
    / "results"
    / "sionna_formal_test840_locked_phyguard"
)

EXTERNAL_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna24_external_validation_final.zip"
)

FORMAL_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal_test840_causal_final.zip"
)

PLAN_PATH = (
    ROOT
    / "configs"
    / "sionna_formal_test840_plan_v1.json"
)

POLICY_PATH = (
    ROOT
    / "configs"
    / "sionna_causal_validation_policy_v2.json"
)

INFERENCE_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_locked_reconstruction_inference.json"
)

STATISTICS_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal825_locked_reconstruction_statistics.json"
)

CAUSAL_FREEZE_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_causal_final_freeze_manifest.json"
)

PREFLIGHT_MANIFEST = (
    ROOT
    / "manifests"
    / "formal_test840_locked_reconstruction_preflight.json"
)

CONTRACT_AUDIT_MANIFEST = (
    ROOT
    / "manifests"
    / "locked_reconstruction_contract_audit.json"
)

FINAL_FREEZE_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal825_locked_reconstruction_final_freeze_manifest.json"
)

ARCHIVE_PATH = (
    ROOT
    / "results"
    / "phyguard_sionna_formal825_locked_reconstruction_final.zip"
)

ARCHIVE_SHA_PATH = Path(
    str(ARCHIVE_PATH) + ".sha256"
)

R0_ROOT = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
)

FEATURE_PATH = (
    R0_ROOT
    / "scripts"
    / "features.py"
)

EVALUATION_COMMON_PATH = (
    R0_ROOT
    / "scripts"
    / "evaluation_common.py"
)

MODEL_DIR = (
    R0_ROOT
    / "results"
    / "full"
    / "evaluation_suite"
    / "models"
)

THRESHOLD_CONTRACT_PATH = (
    RESULT_ROOT
    / "contracts"
    / "source_fixed_thresholds_locked.json"
)

PROVENANCE_PATH = (
    RESULT_ROOT
    / "formal825_provenance.json"
)

ENVIRONMENT_PATH = (
    RESULT_ROOT
    / "freeze_environment.json"
)

EXPECTED_EXTERNAL_ARCHIVE_SHA256 = (
    "238980731f09b758ef834e4961ba1427"
    "665a352aac279f164f02efba5a033aad"
)

EXPECTED_FORMAL_ARCHIVE_SHA256 = (
    "b6c5f400b7d55949fb1e1f39c0c092d"
    "a8f780eed100866f30176e26e462ff2f5"
)

EXPECTED_PLAN_SHA256 = (
    "b767fe07cd17eef0c42ec58cfe801732"
    "d38f7124ed5859915195e21223083437"
)

EXPECTED_POLICY_SHA256 = (
    "510b0ba97393a63a199d65414c2b44660"
    "e7140c1696582202b1603c7197f879d"
)

EXPECTED_SOURCE_EVIDENCE_SHA256 = (
    "2a003bc8ecce4243309b297be36740669"
    "c04a8e7dfa27384a5f2de33ec3cc297"
)

EXPECTED_FEATURE_SHA256 = (
    "b665765bbdb317f0568d77a5004e4107"
    "5e4a2e037fc5f6caa8102f6c932fc4f7"
)

EXPECTED_EVALUATION_COMMON_SHA256 = (
    "afa1f3387c272dee075f72f481e01257"
    "2813892976c54dc860901114f8080cdd"
)

EXPECTED_THRESHOLD_SHA256 = (
    "cfdf67e558196294557e33b9f736c606"
    "c53aeea2ea0b532b0f08222806423107"
)

EXPECTED_MODELS = {
    "repeat_0_phyguard.joblib":
        "6877df3e027ec8e7b1964e835c706c96"
        "f54a4a9bdbfcc5b230914773cc7332ff",

    "repeat_1_phyguard.joblib":
        "2a009ff1f590500d930e3eec0b6209697"
        "8ba73d1b1eaf2b5fc3c320c5a70f627",

    "repeat_2_phyguard.joblib":
        "c875168c7d92449cc08335920723df562"
        "4bd066365c39eb36fab184111055e75",

    "repeat_3_phyguard.joblib":
        "ee74feb3e43e34f9149eaeee9797b949"
        "43275344c84b919a973f30fadc17ea99",

    "repeat_4_phyguard.joblib":
        "eb1319f18489287ae2aef3313824613e"
        "30108a95bd6e70d096e3c0d45bd84f2d",
}

EXPECTED_METRICS = {
    "selected_accuracy": 0.60414713,
    "coverage": 0.66345609,
    "false_specific_rate": 0.09915966,
    "macro_f1": 0.40410566,
    "balanced_accuracy": 0.39396409,
    "physical_exact_diagnosis_rate": 0.39830028,
    "control_abstention_rate": 0.90084034,
}

THRESHOLD_MEMBER = (
    "manifests/"
    "source_fixed_thresholds_locked.json"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


required_paths = [
    RESULT_ROOT,
    EXTERNAL_ARCHIVE,
    FORMAL_ARCHIVE,
    PLAN_PATH,
    POLICY_PATH,
    INFERENCE_MANIFEST,
    STATISTICS_MANIFEST,
    CAUSAL_FREEZE_MANIFEST,
    PREFLIGHT_MANIFEST,
    CONTRACT_AUDIT_MANIFEST,
    FEATURE_PATH,
    EVALUATION_COMMON_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)


errors = []

external_archive_hash = sha256_file(
    EXTERNAL_ARCHIVE
)

formal_archive_hash = sha256_file(
    FORMAL_ARCHIVE
)

if (
    external_archive_hash
    != EXPECTED_EXTERNAL_ARCHIVE_SHA256
):
    errors.append(
        "External-validation archive SHA256 mismatch."
    )

if (
    formal_archive_hash
    != EXPECTED_FORMAL_ARCHIVE_SHA256
):
    errors.append(
        "Formal causal archive SHA256 mismatch."
    )


plan = load_json(PLAN_PATH)
policy = load_json(POLICY_PATH)
inference = load_json(INFERENCE_MANIFEST)
statistics = load_json(STATISTICS_MANIFEST)
preflight = load_json(PREFLIGHT_MANIFEST)


if plan.get("plan_sha256") != EXPECTED_PLAN_SHA256:
    errors.append("Formal-plan SHA256 mismatch.")

if (
    policy.get("policy_sha256")
    != EXPECTED_POLICY_SHA256
):
    errors.append("Causal-policy SHA256 mismatch.")

if policy.get("status") != "LOCKED":
    errors.append("Causal policy is not LOCKED.")

if inference.get("status") != "PASS":
    errors.append("Inference manifest is not PASS.")

if inference.get("eligible_sample_count") != 825:
    errors.append("Inference eligible count is not 825.")

if inference.get("model_count") != 5:
    errors.append("Inference model count is not five.")

if inference.get("prediction_count") != 4125:
    errors.append("Prediction count is not 4125.")

if (
    inference
    .get("determinism", {})
    .get("feature_replay")
    != "825/825"
):
    errors.append("Feature replay is not 825/825.")

if (
    inference
    .get("determinism", {})
    .get("prediction_replay")
    != "4125/4125"
):
    errors.append("Prediction replay is not 4125/4125.")

if (
    inference
    .get("original_evidence", {})
    .get("aggregate_sha256")
    != EXPECTED_SOURCE_EVIDENCE_SHA256
):
    errors.append("Original-evidence SHA256 mismatch.")

if (
    inference
    .get("original_evidence", {})
    .get("unchanged")
    is not True
):
    errors.append("Original evidence is not marked unchanged.")

if statistics.get("status") != "PASS":
    errors.append("Statistics manifest is not PASS.")

if (
    statistics
    .get("integrity", {})
    .get("sample_count")
    != 825
):
    errors.append("Statistics sample count is not 825.")

if (
    statistics
    .get("integrity", {})
    .get("repeat_count")
    != 5
):
    errors.append("Statistics repeat count is not five.")

if (
    statistics
    .get("integrity", {})
    .get("prediction_count")
    != 4125
):
    errors.append("Statistics prediction count is not 4125.")

if (
    statistics
    .get("integrity", {})
    .get("stored_repeat_metrics_recomputed_exactly")
    is not True
):
    errors.append(
        "Stored repeat metrics were not recomputed exactly."
    )

if (
    statistics
    .get("integrity", {})
    .get("sample_alignment_across_repeats")
    is not True
):
    errors.append(
        "Sample alignment across repeats is not PASS."
    )

if preflight.get("status") != "PASS":
    errors.append("Locked reconstruction preflight is not PASS.")


if sha256_file(FEATURE_PATH) != EXPECTED_FEATURE_SHA256:
    errors.append("Feature-builder SHA256 mismatch.")

if (
    sha256_file(EVALUATION_COMMON_PATH)
    != EXPECTED_EVALUATION_COMMON_SHA256
):
    errors.append("evaluation_common.py SHA256 mismatch.")


THRESHOLD_CONTRACT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with zipfile.ZipFile(
    EXTERNAL_ARCHIVE,
    mode="r",
) as archive:
    bad_file = archive.testzip()

    if bad_file is not None:
        errors.append(
            f"External archive ZIP failure: {bad_file}"
        )

    if THRESHOLD_MEMBER not in archive.namelist():
        errors.append(
            "Frozen threshold contract is missing."
        )
        threshold_bytes = b""
    else:
        threshold_bytes = archive.read(
            THRESHOLD_MEMBER
        )


if threshold_bytes:
    if (
        sha256_bytes(threshold_bytes)
        != EXPECTED_THRESHOLD_SHA256
    ):
        errors.append(
            "Frozen threshold-contract SHA256 mismatch."
        )

    THRESHOLD_CONTRACT_PATH.write_bytes(
        threshold_bytes
    )


model_records = []
model_hashes_before = {}

for filename, expected_hash in (
    EXPECTED_MODELS.items()
):
    path = MODEL_DIR / filename

    if not path.exists():
        errors.append(
            f"Missing locked model: {filename}"
        )
        continue

    actual_hash = sha256_file(path)

    model_hashes_before[
        filename
    ] = actual_hash

    if actual_hash != expected_hash:
        errors.append(
            f"Locked model SHA256 mismatch: {filename}"
        )

    model_records.append(
        {
            "filename": filename,
            "path": str(path),
            "sha256": actual_hash,
            "size_bytes": path.stat().st_size,
        }
    )


repeat_metrics_path = (
    RESULT_ROOT
    / "per_repeat_metrics.csv"
)

bootstrap_ci_path = (
    RESULT_ROOT
    / "sample_cluster_bootstrap_confidence_intervals.csv"
)

confusion_path = (
    RESULT_ROOT
    / "physical_outcome_confusion_counts.csv"
)

predictions_path = (
    RESULT_ROOT
    / "predictions.csv"
)

for path in (
    repeat_metrics_path,
    bootstrap_ci_path,
    confusion_path,
    predictions_path,
):
    if not path.exists():
        errors.append(
            f"Missing final result file: {path}"
        )


if not errors:
    repeat_metrics = pd.read_csv(
        repeat_metrics_path
    )

    bootstrap_ci = pd.read_csv(
        bootstrap_ci_path
    )

    confusion = pd.read_csv(
        confusion_path,
        index_col=0,
    )

    predictions = pd.read_csv(
        predictions_path
    )

    if len(repeat_metrics) != 5:
        errors.append(
            "Per-repeat metric row count is not five."
        )

    if len(predictions) != 4125:
        errors.append(
            "Prediction CSV row count is not 4125."
        )

    if int(confusion.to_numpy().sum()) != 3530:
        errors.append(
            "Physical confusion total is not 3530."
        )

    bootstrap_by_metric = {
        row["metric"]: row
        for row in bootstrap_ci.to_dict(
            orient="records"
        )
    }

    for metric, expected_value in (
        EXPECTED_METRICS.items()
    ):
        row = bootstrap_by_metric.get(metric)

        if row is None:
            errors.append(
                f"Missing bootstrap metric: {metric}"
            )
            continue

        actual_value = float(
            row["point_estimate"]
        )

        if not np.isclose(
            actual_value,
            expected_value,
            rtol=0.0,
            atol=5e-9,
        ):
            errors.append(
                f"Metric mismatch for {metric}: "
                f"{actual_value}"
            )


environment = {
    "python_version": sys.version,
    "platform": platform.platform(),
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "scikit_learn_version": sklearn.__version__,
    "joblib_version": joblib.__version__,
}

ENVIRONMENT_PATH.write_text(
    json.dumps(
        environment,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


provenance = {
    "schema":
        "phyguard.sionna.formal825."
        "locked_reconstruction.provenance.v1",

    "status": "PASS",

    "provenance_class":
        "revision-time locked reconstruction",

    "parent_artifacts": {
        "external_smoke24_archive_sha256":
            EXPECTED_EXTERNAL_ARCHIVE_SHA256,

        "formal_test840_causal_archive_sha256":
            EXPECTED_FORMAL_ARCHIVE_SHA256,

        "formal_plan_sha256":
            EXPECTED_PLAN_SHA256,

        "causal_validation_policy_sha256":
            EXPECTED_POLICY_SHA256,

        "original_evidence_aggregate_sha256":
            EXPECTED_SOURCE_EVIDENCE_SHA256,
    },

    "formal_scope": {
        "preregistered_samples": 840,
        "artifact_qc_eligible": 839,
        "causal_valid_samples": 825,
        "locked_models": 5,
        "predictions": 4125,
    },

    "metric_semantics": {
        "selected_accuracy":
            "Exact mechanism accuracy among selected "
            "physical-anomaly cases.",

        "coverage":
            "Selection rate among physical-anomaly cases; "
            "not overall output rate across all samples.",

        "false_specific_rate":
            "Fraction of control cases receiving a specific "
            "physical-mechanism output.",

        "physical_exact_diagnosis_rate":
            "Exact mechanism correctness over all eligible "
            "physical-anomaly cases, including abstentions.",

        "control_abstention_rate":
            "Fraction of normal and nonphysical controls "
            "receiving abstention.",
    },

    "headline_results": {
        "selected_accuracy_mean":
            EXPECTED_METRICS[
                "selected_accuracy"
            ],

        "physical_selection_rate_mean":
            EXPECTED_METRICS[
                "coverage"
            ],

        "false_specific_rate_mean":
            EXPECTED_METRICS[
                "false_specific_rate"
            ],

        "physical_exact_diagnosis_rate_mean":
            EXPECTED_METRICS[
                "physical_exact_diagnosis_rate"
            ],

        "control_abstention_rate_mean":
            EXPECTED_METRICS[
                "control_abstention_rate"
            ],
    },

    "claim_boundary": {
        "diagnosis_conditioned_on_locked_candidate_interval":
            True,

        "end_to_end_anomaly_localization_claim":
            False,

        "original_submission_model_replay_claim":
            False,

        "revision_time_locked_reconstruction_claim":
            True,

        "training_on_formal_test":
            False,

        "threshold_tuning_on_formal_test":
            False,

        "calibration_on_formal_test":
            False,
    },
}

PROVENANCE_PATH.write_text(
    json.dumps(
        provenance,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


if errors:
    print("status: FAIL")

    for error in errors:
        print("-", error)

    raise RuntimeError(
        "Formal825 final freeze validation failed."
    )


entries = {}


def add_entry(path, archive_name=None):
    if not path.exists():
        raise FileNotFoundError(path)

    name = (
        archive_name
        if archive_name is not None
        else str(path.relative_to(ROOT))
    )

    if name in entries:
        raise RuntimeError(
            f"Duplicate archive entry: {name}"
        )

    entries[name] = path


for path in RESULT_ROOT.rglob("*"):
    if path.is_file():
        add_entry(path)


project_files = [
    PLAN_PATH,
    POLICY_PATH,
    INFERENCE_MANIFEST,
    STATISTICS_MANIFEST,
    CAUSAL_FREEZE_MANIFEST,
    PREFLIGHT_MANIFEST,
    CONTRACT_AUDIT_MANIFEST,
]

for path in project_files:
    add_entry(path)


for script_number in range(45, 50):
    path = (
        ROOT
        / "scripts"
        / f"{script_number}_"
    )

    matches = list(
        path.parent.glob(
            f"{script_number}_*.py"
        )
    )

    for match in matches:
        add_entry(match)


for log_number in range(45, 49):
    matches = list(
        (ROOT / "logs").glob(
            f"{log_number}_*.log"
        )
    )

    for match in matches:
        add_entry(match)


add_entry(
    FEATURE_PATH,
    "locked_assets/features.py",
)

add_entry(
    EVALUATION_COMMON_PATH,
    "locked_assets/evaluation_common.py",
)

for filename in EXPECTED_MODELS:
    add_entry(
        MODEL_DIR / filename,
        f"locked_assets/models/{filename}",
    )


file_records = []

for archive_name, path in sorted(
    entries.items()
):
    file_records.append(
        {
            "archive_name":
                archive_name,

            "source_path":
                str(path),

            "sha256":
                sha256_file(path),

            "size_bytes":
                path.stat().st_size,
        }
    )


freeze_manifest = {
    "schema":
        "phyguard.sionna.formal825."
        "locked_reconstruction.final_freeze.v1",

    "status": "PASS",

    "purpose":
        "Freeze the 825-sample causal-valid formal "
        "evaluation, five locked reconstruction bundles, "
        "fixed operating points, deterministic predictions, "
        "and statistical analyses.",

    "provenance_class":
        "revision-time locked reconstruction",

    "formal_scope": {
        "preregistered_sample_count": 840,
        "artifact_qc_eligible_count": 839,
        "causal_valid_sample_count": 825,
        "model_count": 5,
        "prediction_count": 4125,
    },

    "determinism": {
        "feature_replay": "825/825",
        "prediction_replay": "4125/4125",
    },

    "headline_metrics": {
        "selected_accuracy_mean":
            0.60414713,

        "physical_selection_rate_mean":
            0.66345609,

        "false_specific_rate_mean":
            0.09915966,

        "physical_exact_diagnosis_rate_mean":
            0.39830028,

        "control_abstention_rate_mean":
            0.90084034,
    },

    "parent_hashes": {
        "external_smoke24_archive":
            EXPECTED_EXTERNAL_ARCHIVE_SHA256,

        "formal_test840_causal_archive":
            EXPECTED_FORMAL_ARCHIVE_SHA256,

        "formal_plan":
            EXPECTED_PLAN_SHA256,

        "causal_policy":
            EXPECTED_POLICY_SHA256,

        "original_evidence":
            EXPECTED_SOURCE_EVIDENCE_SHA256,
    },

    "model_records":
        model_records,

    "methodological_boundary": {
        "original_submission_model_replay":
            False,

        "revision_time_locked_reconstruction":
            True,

        "diagnosis_conditioned_on_locked_candidate_interval":
            True,

        "training_performed":
            False,

        "threshold_tuning_performed":
            False,

        "threshold_calibration_performed":
            False,

        "model_selection_performed":
            False,

        "formal_predictions_modified":
            False,

        "failed_samples_replaced":
            False,
    },

    "archive_entry_count":
        len(file_records) + 1,

    "files":
        file_records,
}


FINAL_FREEZE_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL_FREEZE_MANIFEST.write_text(
    json.dumps(
        freeze_manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

entries[
    str(
        FINAL_FREEZE_MANIFEST.relative_to(
            ROOT
        )
    )
] = FINAL_FREEZE_MANIFEST


ARCHIVE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if ARCHIVE_PATH.exists():
    ARCHIVE_PATH.unlink()


fixed_time = (
    1980,
    1,
    1,
    0,
    0,
    0,
)

with zipfile.ZipFile(
    ARCHIVE_PATH,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=9,
) as archive:
    for archive_name, path in sorted(
        entries.items()
    ):
        info = zipfile.ZipInfo(
            archive_name,
            date_time=fixed_time,
        )

        info.compress_type = (
            zipfile.ZIP_DEFLATED
        )

        info.external_attr = (
            0o644 << 16
        )

        archive.writestr(
            info,
            path.read_bytes(),
        )


archive_sha256 = sha256_file(
    ARCHIVE_PATH
)

ARCHIVE_SHA_PATH.write_text(
    f"{archive_sha256}  "
    f"{ARCHIVE_PATH.name}\n",
    encoding="utf-8",
)


with zipfile.ZipFile(
    ARCHIVE_PATH,
    mode="r",
) as archive:
    bad_file = archive.testzip()
    archive_names = archive.namelist()


if bad_file is not None:
    raise RuntimeError(
        f"ZIP integrity failure: {bad_file}"
    )

if len(archive_names) != len(entries):
    raise RuntimeError(
        f"Archive entry count="
        f"{len(archive_names)}, "
        f"expected {len(entries)}."
    )


for filename, before_hash in (
    model_hashes_before.items()
):
    path = MODEL_DIR / filename

    if sha256_file(path) != before_hash:
        raise RuntimeError(
            f"Model changed during freeze: {filename}"
        )


print("status: PASS")
print("provenance_class: revision-time locked reconstruction")
print("formal_preregistered_count: 840")
print("artifact_qc_eligible_count: 839")
print("causal_valid_sample_count: 825")
print("model_count: 5")
print("prediction_count: 4125")
print("deterministic_feature_replay: 825/825")
print("deterministic_prediction_replay: 4125/4125")
print("bootstrap_repetitions: 10000")
print("selected_accuracy_mean: 0.60414713")
print("physical_selection_rate_mean: 0.66345609")
print("false_specific_rate_mean: 0.09915966")
print("physical_exact_diagnosis_rate_mean: 0.39830028")
print("control_abstention_rate_mean: 0.90084034")
print("models_unchanged: True")
print("zip_test: PASS")
print("archive:", ARCHIVE_PATH)
print("archive_sha256:", archive_sha256)
print("archive_entry_count:", len(archive_names))
print("freeze_manifest:", FINAL_FREEZE_MANIFEST)
print("sha256_file:", ARCHIVE_SHA_PATH)

print(
    "\nSIONNA_FORMAL825_LOCKED_"
    "RECONSTRUCTION_FINAL_FREEZE_PASS"
)
