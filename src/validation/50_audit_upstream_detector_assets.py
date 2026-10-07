import ast
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


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

FORMAL_CAUSAL_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_causal_validation_v2.json"
)

FORMAL_FINAL_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal825_locked_reconstruction_final.zip"
)

OUTPUT_MANIFEST = (
    ROOT
    / "manifests"
    / "formal825_upstream_detector_asset_audit.json"
)

EXPECTED_DETECTOR_SCRIPT_SHA256 = (
    "f38672d30a769aebeb0b0e156f187dce"
    "b9daa7317fe97fc32faaec0186f54311"
)

EXPECTED_FEATURE_SCRIPT_SHA256 = (
    "b665765bbdb317f0568d77a5004e4107"
    "5e4a2e037fc5f6caa8102f6c932fc4f7"
)

EXPECTED_FORMAL_ARCHIVE_SHA256 = (
    "5d08fc3b8265789819b0f49c1f93a9c7"
    "9c8fe44ce7ec4aec8f1fa603a08c241b"
)

EXPECTED_DETECTORS = [
    "robust_energy",
    "pca_reconstruction",
    "isolation_forest",
]


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


required = [
    DETECTOR_SCRIPT,
    FEATURE_SCRIPT,
    SOURCE_TELEMETRY,
    SOURCE_METADATA,
    FORMAL_CAUSAL_MANIFEST,
    FORMAL_FINAL_ARCHIVE,
]

for path in required:
    if not path.exists():
        raise FileNotFoundError(path)


errors = []

detector_script_hash = sha256_file(
    DETECTOR_SCRIPT
)

feature_script_hash = sha256_file(
    FEATURE_SCRIPT
)

formal_archive_hash = sha256_file(
    FORMAL_FINAL_ARCHIVE
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
    formal_archive_hash
    != EXPECTED_FORMAL_ARCHIVE_SHA256
):
    errors.append(
        "Formal825 final archive SHA256 mismatch."
    )


source_code = DETECTOR_SCRIPT.read_text(
    encoding="utf-8"
)

tree = ast.parse(source_code)


function_records = {}

for node in tree.body:
    if isinstance(
        node,
        ast.FunctionDef,
    ):
        function_records[node.name] = {
            "line_start": node.lineno,
            "line_end":
                getattr(
                    node,
                    "end_lineno",
                    node.lineno,
                ),
            "arguments": [
                argument.arg
                for argument
                in node.args.args
            ],
            "defaults": [
                ast.unparse(value)
                for value
                in node.args.defaults
            ],
        }


required_functions = [
    "robust_standardize",
    "sliding_window_start",
    "proposal_for_sequence",
]

for name in required_functions:
    if name not in function_records:
        errors.append(
            f"Required detector function missing: {name}"
        )


detector_script_directory = str(
    DETECTOR_SCRIPT.parent
)

if detector_script_directory not in sys.path:
    sys.path.insert(
        0,
        detector_script_directory,
    )

namespace = {
    "__name__":
        "phyguard_detector_proposals_asset_audit",
    "__file__":
        str(DETECTOR_SCRIPT),
}

exec(
    compile(
        source_code,
        str(DETECTOR_SCRIPT),
        "exec",
    ),
    namespace,
)


detectors = list(
    namespace.get(
        "DETECTORS",
        [],
    )
)

if detectors != EXPECTED_DETECTORS:
    errors.append(
        f"Detector list mismatch: {detectors}"
    )


robust_standardize = namespace[
    "robust_standardize"
]

sliding_window_start = namespace[
    "sliding_window_start"
]

proposal_for_sequence = namespace[
    "proposal_for_sequence"
]


probe = np.zeros(
    (80, 10),
    dtype=np.float32,
)

probe[40:64, 2] = -3.0

standardized = robust_standardize(
    probe,
    prefix=24,
)

if standardized.shape != (80, 10):
    errors.append(
        "robust_standardize output shape mismatch."
    )

score = np.zeros(
    80,
    dtype=float,
)

score[40:64] = 1.0

probe_start = sliding_window_start(
    score,
    window=24,
    earliest=24,
)

if probe_start != 40:
    errors.append(
        f"Sliding-window probe start={probe_start}, "
        "expected 40."
    )


with np.load(
    SOURCE_TELEMETRY,
    allow_pickle=False,
) as archive:
    if "X" not in archive.files:
        errors.append(
            "Source telemetry does not contain X."
        )
        source_x = None
    else:
        source_x = np.asarray(
            archive["X"],
            dtype=np.float32,
        )


source_metadata = pd.read_csv(
    SOURCE_METADATA
)


if source_x is not None:
    if source_x.ndim != 3:
        errors.append(
            f"Source telemetry ndim={source_x.ndim}, expected 3."
        )

    if source_x.shape[1:] != (80, 10):
        errors.append(
            "Source telemetry sample shape "
            f"is {source_x.shape[1:]}, expected (80, 10)."
        )

    if len(source_x) != len(source_metadata):
        errors.append(
            "Source telemetry/metadata count mismatch."
        )

    if not np.isfinite(source_x).all():
        errors.append(
            "Source telemetry contains NaN or Inf."
        )


required_metadata_columns = {
    "label",
    "seed",
    "event_start",
    "event_end",
}

missing_columns = (
    required_metadata_columns
    - set(source_metadata.columns)
)

if missing_columns:
    errors.append(
        "Source metadata missing columns: "
        + str(sorted(missing_columns))
    )


source_label_counts = (
    source_metadata[
        "label"
    ]
    .value_counts()
    .to_dict()
)


causal = load_json(
    FORMAL_CAUSAL_MANIFEST
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
        "Formal causal-valid sample count "
        f"is {len(formal_eligible_records)}, expected 825."
    )


candidate_artifacts = []

search_roots = [
    R0_ROOT / "results",
    R0_ROOT / "artifacts",
    R0_ROOT / "models",
]

keywords = (
    "isolation",
    "iforest",
    "detector",
    "proposal",
    "pca",
    "robust",
)


for search_root in search_roots:
    if not search_root.exists():
        continue

    for path in search_root.rglob("*"):
        if not path.is_file():
            continue

        lower_name = path.name.lower()

        if not any(
            keyword in lower_name
            for keyword in keywords
        ):
            continue

        record = {
            "path": str(path),
            "relative_path":
                str(
                    path.relative_to(
                        R0_ROOT
                    )
                ),
            "suffix":
                path.suffix.lower(),
            "size_bytes":
                path.stat().st_size,
            "sha256":
                sha256_file(path),
            "load_status":
                "NOT_ATTEMPTED",
            "python_type":
                None,
        }

        if path.suffix.lower() in {
            ".joblib",
            ".pkl",
            ".pickle",
        }:
            try:
                value = joblib.load(
                    path
                )

                record[
                    "load_status"
                ] = "PASS"

                record[
                    "python_type"
                ] = (
                    type(value).__module__
                    + "."
                    + type(value).__name__
                )

                if isinstance(value, dict):
                    record[
                        "dict_keys"
                    ] = sorted(
                        str(key)
                        for key
                        in value.keys()
                    )

            except Exception as exc:
                record[
                    "load_status"
                ] = "FAIL"

                record[
                    "load_error"
                ] = repr(exc)

        candidate_artifacts.append(
            record
        )


frozen_iforest_candidates = [
    record
    for record in candidate_artifacts
    if (
        record["suffix"]
        in {
            ".joblib",
            ".pkl",
            ".pickle",
        }
        and (
            "isolation"
            in record[
                "relative_path"
            ].lower()
            or "iforest"
            in record[
                "relative_path"
            ].lower()
        )
        and record[
            "load_status"
        ] == "PASS"
    )
]


source_prefix_training_available = bool(
    source_x is not None
    and source_x.shape[1:] == (80, 10)
    and len(source_x) > 0
)


protocol_decision = {
    "robust_energy": {
        "eligible_for_primary_analysis":
            True,

        "fitting":
            "None",

        "reference_frames":
            24,

        "candidate_window":
            24,

        "earliest_candidate_start":
            24,
    },

    "pca_reconstruction": {
        "eligible_for_primary_analysis":
            True,

        "fitting":
            (
                "Per-sequence PCA fitted only on "
                "the first 24 reference frames"
            ),

        "label_access":
            False,

        "event_interval_access":
            False,

        "candidate_window":
            24,
    },

    "isolation_forest": {
        "existing_frozen_candidate_count":
            len(
                frozen_iforest_candidates
            ),

        "source_prefix_training_available":
            source_prefix_training_available,

        "formal_test_fitting_allowed":
            False,

        "eligible_protocol":
            (
                "Use an existing hash-verified frozen "
                "IsolationForest when available; otherwise "
                "fit exactly once on R0 source-domain "
                "reference prefixes, freeze it before "
                "Formal825 inference, and never fit it "
                "on Formal825."
            ),

        "candidate_window":
            24,
    },
}


output = {
    "schema":
        "phyguard.sionna.formal825."
        "upstream_detector_asset_audit.v1",

    "status":
        "PASS"
        if not errors
        else "FAIL",

    "sklearn_version":
        sklearn.__version__,

    "formal_scope": {
        "causal_valid_sample_count":
            len(
                formal_eligible_records
            ),

        "formal_archive_sha256":
            formal_archive_hash,
    },

    "locked_source_code": {
        "detector_script": {
            "path":
                str(DETECTOR_SCRIPT),

            "sha256":
                detector_script_hash,

            "functions":
                function_records,
        },

        "feature_script": {
            "path":
                str(FEATURE_SCRIPT),

            "sha256":
                feature_script_hash,
        },
    },

    "detectors":
        detectors,

    "fixed_operating_contract": {
        "reference_prefix_frames":
            24,

        "candidate_window_frames":
            24,

        "earliest_candidate_start":
            24,

        "sequence_length":
            80,

        "kpi_count":
            10,

        "selection_rule":
            (
                "Choose the 24-frame window "
                "with maximum mean detector score."
            ),
    },

    "source_domain_assets": {
        "telemetry_path":
            str(SOURCE_TELEMETRY),

        "telemetry_sha256":
            sha256_file(
                SOURCE_TELEMETRY
            ),

        "telemetry_shape":
            (
                list(source_x.shape)
                if source_x is not None
                else None
            ),

        "metadata_path":
            str(SOURCE_METADATA),

        "metadata_sha256":
            sha256_file(
                SOURCE_METADATA
            ),

        "metadata_rows":
            len(
                source_metadata
            ),

        "label_counts":
            source_label_counts,
    },

    "candidate_artifacts":
        candidate_artifacts,

    "frozen_iforest_candidates":
        frozen_iforest_candidates,

    "protocol_decision":
        protocol_decision,

    "methodological_boundary": {
        "formal_labels_used":
            False,

        "formal_event_intervals_used":
            False,

        "formal_predictions_generated":
            False,

        "detector_parameter_tuning":
            False,

        "candidate_window_tuning":
            False,

        "formal_test_model_fitting":
            False,

        "post_result_detector_selection":
            False,
    },

    "errors":
        errors,
}


OUTPUT_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_MANIFEST.write_text(
    json.dumps(
        output,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "status:",
    output["status"],
)

print(
    "formal_causal_valid_count:",
    len(
        formal_eligible_records
    ),
)

print(
    "detector_script_sha256:",
    detector_script_hash,
)

print(
    "feature_script_sha256:",
    feature_script_hash,
)

print(
    "detectors:",
    detectors,
)

print(
    "source_telemetry_shape:",
    (
        list(source_x.shape)
        if source_x is not None
        else None
    ),
)

print(
    "source_metadata_rows:",
    len(
        source_metadata
    ),
)

print(
    "source_label_counts:",
    source_label_counts,
)

print(
    "candidate_artifact_count:",
    len(
        candidate_artifacts
    ),
)

print(
    "existing_frozen_iforest_candidate_count:",
    len(
        frozen_iforest_candidates
    ),
)

print(
    "source_prefix_training_available:",
    source_prefix_training_available,
)


print("\nFROZEN IFOREST CANDIDATES")

if frozen_iforest_candidates:
    for record in frozen_iforest_candidates:
        print(record)
else:
    print("<none>")


print("\nPROTOCOL DECISION")

for detector, value in (
    protocol_decision.items()
):
    print(
        detector,
        json.dumps(
            value,
            ensure_ascii=False,
        ),
    )


print("\nmanifest:", OUTPUT_MANIFEST)


if errors:
    print("\nERRORS")

    for error in errors:
        print("-", error)

    raise RuntimeError(
        "Upstream detector asset audit failed."
    )


print(
    "\nFORMAL825_UPSTREAM_"
    "DETECTOR_ASSET_AUDIT_PASS"
)
