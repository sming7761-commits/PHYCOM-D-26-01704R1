import hashlib
import importlib.util
import json
import platform
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest


ROOT = Path("/root/phyguard_revision")

R0_ROOT = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
)

DETECTOR_SCRIPT = (
    R0_ROOT
    / "scripts"
    / "detector_proposals.py"
)

FEATURE_SCRIPT = (
    R0_ROOT
    / "scripts"
    / "features.py"
)

SOURCE_TELEMETRY = (
    R0_ROOT
    / "data"
    / "full"
    / "telemetry.npz"
)

SOURCE_METADATA = (
    R0_ROOT
    / "data"
    / "full"
    / "metadata.csv"
)

ASSET_AUDIT = (
    ROOT
    / "manifests"
    / "formal825_upstream_detector_asset_audit.json"
)

FORMAL_CAUSAL_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_causal_validation_v2.json"
)

FORMAL825_FINAL_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal825_locked_reconstruction_final.zip"
)

MODEL_DIR = (
    ROOT
    / "artifacts"
    / "frozen_upstream_detectors"
)

IFOREST_MODEL = (
    MODEL_DIR
    / "formal825_source_fitted_isolation_forest_v1.joblib"
)

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "sionna_formal825_upstream_detector_protocol_v1.json"
)

LOCK_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal825_upstream_detector_protocol_lock_v1.json"
)

ARCHIVE_PATH = (
    ROOT
    / "results"
    / "phyguard_sionna_formal825_upstream_detector_protocol_locked.zip"
)

ARCHIVE_SHA_PATH = Path(
    str(ARCHIVE_PATH) + ".sha256"
)

EXPECTED_DETECTOR_SCRIPT_SHA256 = (
    "f38672d30a769aebeb0b0e156f187dce"
    "b9daa7317fe97fc32faaec0186f54311"
)

EXPECTED_FEATURE_SCRIPT_SHA256 = (
    "b665765bbdb317f0568d77a5004e4107"
    "5e4a2e037fc5f6caa8102f6c932fc4f7"
)

EXPECTED_FORMAL825_ARCHIVE_SHA256 = (
    "5d08fc3b8265789819b0f49c1f93a9c7"
    "9c8fe44ce7ec4aec8f1fa603a08c241b"
)

EXPECTED_DETECTORS = [
    "robust_energy",
    "pca_reconstruction",
    "isolation_forest",
]

REFERENCE_PREFIX = 24
CANDIDATE_WINDOW = 24
EARLIEST_START = 24
IFOREST_SEED = 20260705
IFOREST_ESTIMATORS = 200
IFOREST_MAX_SAMPLES_CAP = 4096


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


def import_module(path, module_name):
    script_directory = str(path.parent)

    if script_directory not in sys.path:
        sys.path.insert(
            0,
            script_directory,
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import module: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


def contains_isolation_forest(value, visited=None):
    if visited is None:
        visited = set()

    object_id = id(value)

    if object_id in visited:
        return False

    visited.add(object_id)

    if isinstance(value, IsolationForest):
        return True

    if isinstance(value, dict):
        return any(
            contains_isolation_forest(
                child,
                visited,
            )
            for child in value.values()
        )

    if isinstance(value, (list, tuple, set)):
        return any(
            contains_isolation_forest(
                child,
                visited,
            )
            for child in value
        )

    return False


def array_contract_sha256(array):
    array = np.ascontiguousarray(array)

    digest = hashlib.sha256()

    digest.update(
        str(array.dtype).encode("ascii")
    )
    digest.update(b"\0")

    digest.update(
        json.dumps(
            list(array.shape),
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")

    digest.update(
        array.tobytes(order="C")
    )

    return digest.hexdigest()


required = [
    DETECTOR_SCRIPT,
    FEATURE_SCRIPT,
    SOURCE_TELEMETRY,
    SOURCE_METADATA,
    ASSET_AUDIT,
    FORMAL_CAUSAL_MANIFEST,
    FORMAL825_FINAL_ARCHIVE,
]

for path in required:
    if not path.exists():
        raise FileNotFoundError(path)


errors = []

if sklearn.__version__ != "1.8.0":
    errors.append(
        f"scikit-learn={sklearn.__version__}, expected 1.8.0."
    )


detector_script_hash = sha256_file(
    DETECTOR_SCRIPT
)

feature_script_hash = sha256_file(
    FEATURE_SCRIPT
)

formal825_archive_hash = sha256_file(
    FORMAL825_FINAL_ARCHIVE
)


if (
    detector_script_hash
    != EXPECTED_DETECTOR_SCRIPT_SHA256
):
    errors.append(
        "detector_proposals.py SHA256 mismatch."
    )

if (
    feature_script_hash
    != EXPECTED_FEATURE_SCRIPT_SHA256
):
    errors.append(
        "features.py SHA256 mismatch."
    )

if (
    formal825_archive_hash
    != EXPECTED_FORMAL825_ARCHIVE_SHA256
):
    errors.append(
        "Formal825 final archive SHA256 mismatch."
    )


audit = load_json(
    ASSET_AUDIT
)

causal = load_json(
    FORMAL_CAUSAL_MANIFEST
)


if audit.get("status") != "PASS":
    errors.append(
        "Upstream detector asset audit is not PASS."
    )


formal_eligible_records = [
    record
    for record in causal.get(
        "records",
        [],
    )
    if record.get(
        "causal_v2_status"
    ) == "PASS"
]

if len(formal_eligible_records) != 825:
    errors.append(
        "Formal causal-valid count is not 825."
    )


detector_module = import_module(
    DETECTOR_SCRIPT,
    "phyguard_detector_protocol_lock",
)

detectors = list(
    detector_module.DETECTORS
)

if detectors != EXPECTED_DETECTORS:
    errors.append(
        f"Detector list mismatch: {detectors}"
    )


candidate_records = audit.get(
    "frozen_iforest_candidates",
    [],
)

candidate_reclassification = []

valid_existing_iforest = []


for record in candidate_records:
    path = Path(
        record["path"]
    )

    if not path.exists():
        errors.append(
            f"Candidate file missing: {path}"
        )
        continue

    actual_hash = sha256_file(path)

    if actual_hash != record.get("sha256"):
        errors.append(
            f"Candidate hash changed: {path}"
        )
        continue

    value = joblib.load(path)

    contains_iforest = (
        contains_isolation_forest(value)
    )

    candidate_reclassification.append(
        {
            "path": str(path),
            "sha256": actual_hash,
            "python_type":
                (
                    type(value).__module__
                    + "."
                    + type(value).__name__
                ),
            "dict_keys":
                (
                    sorted(
                        str(key)
                        for key in value.keys()
                    )
                    if isinstance(value, dict)
                    else None
                ),
            "contains_isolation_forest":
                contains_iforest,
            "classification":
                (
                    "VALID_FROZEN_ISOLATION_FOREST"
                    if contains_iforest
                    else
                    "FALSE_POSITIVE_DOWNSTREAM_MODEL"
                ),
        }
    )

    if contains_iforest:
        valid_existing_iforest.append(
            path
        )


if valid_existing_iforest:
    errors.append(
        "Unexpected valid frozen IsolationForest found; "
        "the source-fitting branch must not proceed silently."
    )


with np.load(
    SOURCE_TELEMETRY,
    allow_pickle=False,
) as archive:
    if "X" not in archive.files:
        raise RuntimeError(
            "Source telemetry does not contain X."
        )

    source_x = np.asarray(
        archive["X"],
    )


source_metadata = pd.read_csv(
    SOURCE_METADATA
)


if source_x.shape != (1080, 80, 10):
    errors.append(
        f"Source telemetry shape={source_x.shape}, "
        "expected (1080, 80, 10)."
    )

if len(source_metadata) != 1080:
    errors.append(
        "Source metadata row count is not 1080."
    )

if not np.isfinite(source_x).all():
    errors.append(
        "Source telemetry contains NaN or Inf."
    )

if "event_start" not in source_metadata.columns:
    errors.append(
        "Source metadata has no event_start column."
    )
else:
    minimum_event_start = int(
        source_metadata[
            "event_start"
        ].min()
    )

    if minimum_event_start < REFERENCE_PREFIX:
        errors.append(
            "At least one source event starts inside "
            "the fixed 24-frame reference prefix."
        )


if errors:
    print("status: FAIL")

    for error in errors:
        print("-", error)

    raise RuntimeError(
        "Upstream detector protocol preconditions failed."
    )


robust_standardize = (
    detector_module.robust_standardize
)


prefix_blocks = []

for sequence in source_x:
    standardized = robust_standardize(
        sequence,
        prefix=REFERENCE_PREFIX,
    )

    prefix_blocks.append(
        standardized[
            :REFERENCE_PREFIX
        ]
    )


prefix_frames = np.concatenate(
    prefix_blocks,
    axis=0,
)


if prefix_frames.shape != (
    1080 * REFERENCE_PREFIX,
    10,
):
    raise RuntimeError(
        f"Source-prefix matrix shape="
        f"{prefix_frames.shape}."
    )

if not np.isfinite(
    prefix_frames
).all():
    raise RuntimeError(
        "Source-prefix matrix contains NaN or Inf."
    )


prefix_contract_hash = (
    array_contract_sha256(
        prefix_frames
    )
)

max_samples = min(
    IFOREST_MAX_SAMPLES_CAP,
    len(prefix_frames),
)


source_iforest = IsolationForest(
    n_estimators=
        IFOREST_ESTIMATORS,

    max_samples=
        max_samples,

    contamination=
        "auto",

    random_state=
        IFOREST_SEED,

    n_jobs=
        -1,
)


source_iforest.fit(
    prefix_frames
)


if int(source_iforest.n_features_in_) != 10:
    raise RuntimeError(
        "Fitted IsolationForest feature dimension is not 10."
    )

if len(source_iforest.estimators_) != IFOREST_ESTIMATORS:
    raise RuntimeError(
        "Fitted IsolationForest estimator count is not 200."
    )


probe = prefix_frames[
    :256
]

probe_score_1 = source_iforest.score_samples(
    probe
)

probe_score_2 = source_iforest.score_samples(
    probe
)

if not np.array_equal(
    probe_score_1,
    probe_score_2,
):
    raise RuntimeError(
        "IsolationForest score replay is not deterministic."
    )


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if IFOREST_MODEL.exists():
    IFOREST_MODEL.unlink()


joblib.dump(
    source_iforest,
    IFOREST_MODEL,
    compress=3,
)


iforest_model_hash = sha256_file(
    IFOREST_MODEL
)


reloaded_iforest = joblib.load(
    IFOREST_MODEL
)

if not isinstance(
    reloaded_iforest,
    IsolationForest,
):
    raise RuntimeError(
        "Reloaded object is not IsolationForest."
    )


reload_score = (
    reloaded_iforest.score_samples(
        probe
    )
)

if not np.array_equal(
    probe_score_1,
    reload_score,
):
    raise RuntimeError(
        "Saved/reloaded IsolationForest "
        "does not reproduce exact scores."
    )


if (
    sha256_file(IFOREST_MODEL)
    != iforest_model_hash
):
    raise RuntimeError(
        "IsolationForest model file changed "
        "during read-only reload."
    )


protocol = {
    "schema":
        "phyguard.sionna.formal825."
        "upstream_detector_protocol.v1",

    "status":
        "LOCKED",

    "purpose":
        "Evaluate sensitivity of the locked PhyGuard "
        "diagnostic pipeline to non-oracle upstream "
        "candidate intervals on the 825 causal-valid "
        "Formal Test samples.",

    "lock_timing":
        "Protocol and source-fitted IsolationForest "
        "were frozen before any Formal825 detector "
        "proposal or detector-conditioned diagnostic "
        "result was generated.",

    "formal_scope": {
        "preregistered_sample_count":
            840,

        "artifact_qc_eligible_count":
            839,

        "causal_valid_sample_count":
            825,

        "formal_sample_selection_rule":
            "causal_v2_status == PASS",
    },

    "detectors": {
        "robust_energy": {
            "fitting":
                "None",

            "reference_prefix_frames":
                REFERENCE_PREFIX,

            "standardization":
                "Per-sequence median and MAD over "
                "the first 24 frames; scale=1.4826*MAD+1e-6; "
                "z clipped to [-10,10].",

            "frame_score":
                "sqrt(mean(z^2 across 10 KPIs))",
        },

        "pca_reconstruction": {
            "fitting":
                "Per-sequence unsupervised PCA fitted "
                "only on the first 24 reference frames.",

            "n_components":
                5,

            "svd_solver":
                "full",

            "frame_score":
                "Mean squared reconstruction error "
                "across 10 KPIs.",

            "formal_label_access":
                False,

            "formal_event_interval_access":
                False,
        },

        "isolation_forest": {
            "fitting":
                "Exactly once on R0 source-domain "
                "standardized reference-prefix frames.",

            "formal_test_fitting":
                False,

            "source_sequence_count":
                1080,

            "reference_frames_per_sequence":
                REFERENCE_PREFIX,

            "source_prefix_frame_count":
                int(
                    len(prefix_frames)
                ),

            "source_prefix_contract_sha256":
                prefix_contract_hash,

            "n_estimators":
                IFOREST_ESTIMATORS,

            "max_samples":
                max_samples,

            "contamination":
                "auto",

            "random_state":
                IFOREST_SEED,

            "n_jobs":
                -1,

            "model_path":
                str(IFOREST_MODEL),

            "model_sha256":
                iforest_model_hash,

            "frame_score":
                "-score_samples(z)",
        },
    },

    "candidate_interval_rule": {
        "sequence_length":
            80,

        "reference_prefix_frames":
            REFERENCE_PREFIX,

        "candidate_window_frames":
            CANDIDATE_WINDOW,

        "earliest_candidate_start":
            EARLIEST_START,

        "latest_candidate_start":
            56,

        "selection":
            "Select the first 24-frame window attaining "
            "the maximum mean detector score.",

        "tie_breaking":
            "Earliest start through numpy.argmax.",
    },

    "locked_downstream_pipeline": {
        "feature_builder":
            str(FEATURE_SCRIPT),

        "feature_builder_sha256":
            feature_script_hash,

        "feature_dimensions": {
            "raw": 60,
            "physical": 19,
            "combined": 79,
        },

        "model_repeat_count":
            5,

        "operating_points_modified":
            False,

        "models_modified":
            False,
    },

    "evaluation_plan": {
        "proposal_metrics": [
            "interval IoU",
            "mean IoU for physical anomalies",
            "event recall at IoU >= 0.3",
            "event recall at IoU >= 0.5",
            "absolute start error",
            "absolute end error",
        ],

        "stratification": [
            "detector",
            "label",
            "severity",
            "channel_model",
            "detector x severity",
            "detector x label",
        ],

        "diagnostic_comparison": [
            "locked-event-interval diagnosis",
            "detector-proposed-interval diagnosis",
            "difference in physical selection rate",
            "difference in selected accuracy",
            "difference in exact diagnosis rate",
            "difference in control abstention rate",
        ],

        "association_analysis": [
            "IoU versus exact diagnosis",
            "IoU versus selection",
            "IoU versus abstention",
            "mild versus moderate versus severe",
        ],

        "confidence_intervals":
            "Sample-cluster bootstrap with all five "
            "locked model repeats kept together.",

        "bootstrap_repetitions":
            10000,

        "bootstrap_seed":
            20260731,
    },

    "decision_rules": {
        "primary_detectors":
            EXPECTED_DETECTORS,

        "detector_selection_after_results":
            False,

        "detector_parameter_tuning":
            False,

        "candidate_window_tuning":
            False,

        "post_result_threshold_tuning":
            False,

        "failed_sample_replacement":
            False,

        "all_detector_results_reported":
            True,
    },

    "claim_boundary": {
        "upstream_detector_sensitivity_analysis":
            True,

        "new_anomaly_detector_contribution":
            False,

        "end_to_end_detector_training_claim":
            False,

        "formal_labels_used_for_proposal_generation":
            False,

        "formal_event_intervals_used_for_proposal_generation":
            False,

        "formal_event_intervals_used_for_evaluation_only":
            True,
    },
}


PROTOCOL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

PROTOCOL_PATH.write_text(
    json.dumps(
        protocol,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


protocol_hash = sha256_file(
    PROTOCOL_PATH
)


source_records = {
    "detector_script": {
        "path":
            str(DETECTOR_SCRIPT),

        "sha256":
            detector_script_hash,
    },

    "feature_script": {
        "path":
            str(FEATURE_SCRIPT),

        "sha256":
            feature_script_hash,
    },

    "source_telemetry": {
        "path":
            str(SOURCE_TELEMETRY),

        "sha256":
            sha256_file(
                SOURCE_TELEMETRY
            ),

        "shape":
            list(source_x.shape),
    },

    "source_metadata": {
        "path":
            str(SOURCE_METADATA),

        "sha256":
            sha256_file(
                SOURCE_METADATA
            ),

        "rows":
            len(source_metadata),

        "minimum_event_start":
            int(
                source_metadata[
                    "event_start"
                ].min()
            ),
    },

    "formal825_final_archive": {
        "path":
            str(
                FORMAL825_FINAL_ARCHIVE
            ),

        "sha256":
            formal825_archive_hash,
    },
}


lock_manifest = {
    "schema":
        "phyguard.sionna.formal825."
        "upstream_detector_protocol_lock.v1",

    "status":
        "PASS",

    "protocol_path":
        str(PROTOCOL_PATH),

    "protocol_sha256":
        protocol_hash,

    "valid_existing_frozen_isolation_forest_count":
        0,

    "misclassified_candidate_count":
        len(
            candidate_reclassification
        ),

    "candidate_reclassification":
        candidate_reclassification,

    "source_fitted_isolation_forest": {
        "path":
            str(IFOREST_MODEL),

        "sha256":
            iforest_model_hash,

        "size_bytes":
            IFOREST_MODEL.stat().st_size,

        "python_type":
            (
                type(
                    reloaded_iforest
                ).__module__
                + "."
                + type(
                    reloaded_iforest
                ).__name__
            ),

        "n_features_in":
            int(
                reloaded_iforest.n_features_in_
            ),

        "n_estimators":
            len(
                reloaded_iforest.estimators_
            ),

        "max_samples":
            int(
                reloaded_iforest.max_samples_
            ),

        "deterministic_score_replay":
            "256/256 exact",
    },

    "source_prefix_matrix": {
        "shape":
            list(
                prefix_frames.shape
            ),

        "dtype":
            str(
                prefix_frames.dtype
            ),

        "aggregate_sha256":
            prefix_contract_hash,

        "all_finite":
            bool(
                np.isfinite(
                    prefix_frames
                ).all()
            ),
    },

    "source_records":
        source_records,

    "environment": {
        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "numpy_version":
            np.__version__,

        "pandas_version":
            pd.__version__,

        "scikit_learn_version":
            sklearn.__version__,

        "joblib_version":
            joblib.__version__,
    },

    "methodological_boundary": {
        "formal_sequences_opened":
            False,

        "formal_detector_predictions_generated":
            False,

        "formal_labels_used":
            False,

        "formal_event_intervals_used":
            False,

        "formal_test_model_fitting":
            False,

        "source_domain_prefix_fitting_only":
            True,

        "detector_parameter_tuning":
            False,

        "candidate_window_tuning":
            False,

        "post_result_detector_selection":
            False,
    },
}


LOCK_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True,
)

LOCK_MANIFEST.write_text(
    json.dumps(
        lock_manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


archive_entries = {
    "configs/"
    "sionna_formal825_upstream_detector_protocol_v1.json":
        PROTOCOL_PATH,

    "manifests/"
    "sionna_formal825_upstream_detector_protocol_lock_v1.json":
        LOCK_MANIFEST,

    "manifests/"
    "formal825_upstream_detector_asset_audit.json":
        ASSET_AUDIT,

    "locked_assets/"
    "formal825_source_fitted_isolation_forest_v1.joblib":
        IFOREST_MODEL,

    "locked_assets/"
    "detector_proposals.py":
        DETECTOR_SCRIPT,

    "locked_assets/"
    "features.py":
        FEATURE_SCRIPT,

    "scripts/"
    "51_lock_formal825_upstream_detector_protocol.py":
        ROOT
        / "scripts"
        / "51_lock_formal825_upstream_detector_protocol.py",
}


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
        archive_entries.items()
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


archive_hash = sha256_file(
    ARCHIVE_PATH
)


ARCHIVE_SHA_PATH.write_text(
    f"{archive_hash}  "
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

if len(archive_names) != len(
    archive_entries
):
    raise RuntimeError(
        "Protocol archive entry count mismatch."
    )


print("status: PASS")
print(
    "valid_existing_frozen_isolation_forest_count:",
    0,
)
print(
    "misclassified_candidate_count:",
    len(
        candidate_reclassification
    ),
)
print(
    "source_prefix_matrix_shape:",
    list(
        prefix_frames.shape
    ),
)
print(
    "source_prefix_contract_sha256:",
    prefix_contract_hash,
)
print(
    "isolation_forest_model:",
    IFOREST_MODEL,
)
print(
    "isolation_forest_model_sha256:",
    iforest_model_hash,
)
print(
    "isolation_forest_n_estimators:",
    len(
        reloaded_iforest.estimators_
    ),
)
print(
    "isolation_forest_max_samples:",
    int(
        reloaded_iforest.max_samples_
    ),
)
print(
    "isolation_forest_score_replay:",
    "256/256 exact",
)
print(
    "formal_sequences_opened:",
    False,
)
print(
    "formal_detector_predictions_generated:",
    False,
)
print(
    "protocol:",
    PROTOCOL_PATH,
)
print(
    "protocol_sha256:",
    protocol_hash,
)
print(
    "lock_manifest:",
    LOCK_MANIFEST,
)
print(
    "archive:",
    ARCHIVE_PATH,
)
print(
    "archive_sha256:",
    archive_hash,
)
print(
    "archive_entry_count:",
    len(
        archive_names
    ),
)
print("zip_test: PASS")

print(
    "\nFORMAL825_UPSTREAM_"
    "DETECTOR_PROTOCOL_LOCK_PASS"
)
