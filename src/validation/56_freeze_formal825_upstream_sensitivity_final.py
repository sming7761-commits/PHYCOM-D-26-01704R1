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

FORMAL825_PARENT_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal825_locked_reconstruction_final.zip"
)

FORMAL825_PARENT_SHA_PATH = Path(
    str(FORMAL825_PARENT_ARCHIVE) + ".sha256"
)

PROTOCOL_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal825_upstream_detector_protocol_locked.zip"
)

PROTOCOL_ARCHIVE_SHA_PATH = Path(
    str(PROTOCOL_ARCHIVE) + ".sha256"
)

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "sionna_formal825_upstream_detector_protocol_v1.json"
)

PLAN_PATH = (
    ROOT
    / "configs"
    / "sionna_formal_test840_plan_v1.json"
)

CAUSAL_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_causal_validation_v2.json"
)

ASSET_AUDIT_PATH = (
    ROOT
    / "manifests"
    / "formal825_upstream_detector_asset_audit.json"
)

PROTOCOL_LOCK_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal825_upstream_detector_protocol_lock_v1.json"
)

PROPOSAL_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal825_upstream_detector_proposals_v1.json"
)

PROPOSAL_EVALUATION_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal825_upstream_detector_evaluation_v1.json"
)

DIAGNOSIS_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal825_detector_conditioned_diagnosis_v1.json"
)

STATISTICS_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal825_detector_conditioned_statistics_v1.json"
)

PROPOSAL_ROOT = (
    ROOT
    / "results"
    / "sionna_formal825_upstream_detector_proposals_v1"
)

PROPOSAL_EVALUATION_ROOT = (
    ROOT
    / "results"
    / "sionna_formal825_upstream_detector_evaluation_v1"
)

DIAGNOSIS_ROOT = (
    ROOT
    / "results"
    / "sionna_formal825_detector_conditioned_diagnosis_v1"
)

STATISTICS_ROOT = (
    ROOT
    / "results"
    / "sionna_formal825_detector_conditioned_statistics_v1"
)

IFOREST_MODEL_PATH = (
    ROOT
    / "artifacts"
    / "frozen_upstream_detectors"
    / "formal825_source_fitted_isolation_forest_v1.joblib"
)

R0_ROOT = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
)

DETECTOR_SCRIPT_PATH = (
    R0_ROOT
    / "scripts"
    / "detector_proposals.py"
)

FEATURE_SCRIPT_PATH = (
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
    ROOT
    / "results"
    / "sionna_formal_test840_locked_phyguard"
    / "contracts"
    / "source_fixed_thresholds_locked.json"
)

CLAIMS_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal825_upstream_sensitivity_claims_v1.json"
)

ENVIRONMENT_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal825_upstream_sensitivity_environment_v1.json"
)

FREEZE_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal825_upstream_sensitivity_final_freeze_manifest.json"
)

ARCHIVE_PATH = (
    ROOT
    / "results"
    / "phyguard_sionna_formal825_upstream_sensitivity_final.zip"
)

ARCHIVE_SHA_PATH = Path(
    str(ARCHIVE_PATH) + ".sha256"
)

EXPECTED_FORMAL825_PARENT_SHA256 = (
    "5d08fc3b8265789819b0f49c1f93a9c7"
    "9c8fe44ce7ec4aec8f1fa603a08c241b"
)

EXPECTED_PROTOCOL_ARCHIVE_SHA256 = (
    "a0aecfcb798a4c6113992abb39686c4d"
    "9140e03bc6a5c3ee9e4d6946c62d9053"
)

EXPECTED_PROTOCOL_SHA256 = (
    "4a9b09b82e82efa4940602a73a25cdd"
    "aa05361af4c5edee2ae73f46860996aa2"
)

EXPECTED_IFOREST_SHA256 = (
    "9ad52cf66ffff12f423959db37ff9ed7"
    "c5d7292d620023c531217c42c5040dd1"
)

EXPECTED_DETECTOR_SCRIPT_SHA256 = (
    "f38672d30a769aebeb0b0e156f187dce"
    "b9daa7317fe97fc32faaec0186f54311"
)

EXPECTED_FEATURE_SCRIPT_SHA256 = (
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

DETECTORS = [
    "robust_energy",
    "pca_reconstruction",
    "isolation_forest",
]

EXPECTED_RESULTS = {
    "robust_energy": {
        "mean_iou": 0.723518,
        "selected_accuracy": 0.477969,
        "delta_selected_accuracy": -0.126179,
        "physical_exact": 0.314164,
        "delta_physical_exact": -0.084136,
        "control_abstention": 0.606723,
        "delta_control_abstention": -0.294118,
        "false_specific": 0.393277,
        "delta_false_specific": 0.294118,
        "iou_zero_delta_task": -0.421053,
        "iou_high_delta_task": 0.001815,
        "rho_task_success": 0.476284,
        "rho_prediction_change": -0.703504,
        "net_task_change": -0.114424,
    },

    "pca_reconstruction": {
        "mean_iou": 0.744961,
        "selected_accuracy": 0.501019,
        "delta_selected_accuracy": -0.103128,
        "physical_exact": 0.317847,
        "delta_physical_exact": -0.080453,
        "control_abstention": 0.734454,
        "delta_control_abstention": -0.166387,
        "false_specific": 0.265546,
        "delta_false_specific": 0.166387,
        "iou_zero_delta_task": -0.241237,
        "iou_high_delta_task": -0.046127,
        "rho_task_success": 0.429206,
        "rho_prediction_change": -0.620393,
        "net_task_change": -0.092848,
    },

    "isolation_forest": {
        "mean_iou": 0.667693,
        "selected_accuracy": 0.429093,
        "delta_selected_accuracy": -0.175054,
        "physical_exact": 0.264873,
        "delta_physical_exact": -0.133428,
        "control_abstention": 0.647059,
        "delta_control_abstention": -0.253782,
        "false_specific": 0.352941,
        "delta_false_specific": 0.253782,
        "iou_zero_delta_task": -0.476571,
        "iou_high_delta_task": -0.005138,
        "rho_task_success": 0.548397,
        "rho_prediction_change": -0.748486,
        "net_task_change": -0.150788,
    },
}

TOLERANCE = 5e-6


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


def assert_close(name, actual, expected):
    if not np.isclose(
        float(actual),
        float(expected),
        rtol=0.0,
        atol=TOLERANCE,
    ):
        raise RuntimeError(
            f"{name}: actual={actual}, expected={expected}"
        )


def one_row(frame, **conditions):
    selected = frame

    for column, value in conditions.items():
        selected = selected[
            selected[column] == value
        ]

    if len(selected) != 1:
        raise RuntimeError(
            "Expected exactly one row for "
            f"{conditions}, found {len(selected)}."
        )

    return selected.iloc[0]


required_paths = [
    FORMAL825_PARENT_ARCHIVE,
    PROTOCOL_ARCHIVE,
    PROTOCOL_PATH,
    PLAN_PATH,
    CAUSAL_MANIFEST_PATH,
    ASSET_AUDIT_PATH,
    PROTOCOL_LOCK_PATH,
    PROPOSAL_MANIFEST_PATH,
    PROPOSAL_EVALUATION_MANIFEST_PATH,
    DIAGNOSIS_MANIFEST_PATH,
    STATISTICS_MANIFEST_PATH,
    PROPOSAL_ROOT,
    PROPOSAL_EVALUATION_ROOT,
    DIAGNOSIS_ROOT,
    STATISTICS_ROOT,
    IFOREST_MODEL_PATH,
    DETECTOR_SCRIPT_PATH,
    FEATURE_SCRIPT_PATH,
    EVALUATION_COMMON_PATH,
    THRESHOLD_CONTRACT_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)


if ARCHIVE_PATH.exists():
    raise RuntimeError(
        "Final upstream-sensitivity archive already exists; "
        "refusing to overwrite it."
    )


if sklearn.__version__ != "1.8.0":
    raise RuntimeError(
        f"scikit-learn={sklearn.__version__}, "
        "expected 1.8.0."
    )


hash_checks = {
    "formal825_parent_archive":
        (
            FORMAL825_PARENT_ARCHIVE,
            EXPECTED_FORMAL825_PARENT_SHA256,
        ),

    "protocol_archive":
        (
            PROTOCOL_ARCHIVE,
            EXPECTED_PROTOCOL_ARCHIVE_SHA256,
        ),

    "protocol":
        (
            PROTOCOL_PATH,
            EXPECTED_PROTOCOL_SHA256,
        ),

    "source_fitted_isolation_forest":
        (
            IFOREST_MODEL_PATH,
            EXPECTED_IFOREST_SHA256,
        ),

    "detector_script":
        (
            DETECTOR_SCRIPT_PATH,
            EXPECTED_DETECTOR_SCRIPT_SHA256,
        ),

    "feature_script":
        (
            FEATURE_SCRIPT_PATH,
            EXPECTED_FEATURE_SCRIPT_SHA256,
        ),

    "evaluation_common":
        (
            EVALUATION_COMMON_PATH,
            EXPECTED_EVALUATION_COMMON_SHA256,
        ),

    "threshold_contract":
        (
            THRESHOLD_CONTRACT_PATH,
            EXPECTED_THRESHOLD_SHA256,
        ),
}


for name, (
    path,
    expected_hash,
) in hash_checks.items():
    actual_hash = sha256_file(path)

    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{name} SHA256 mismatch: {actual_hash}"
        )


model_records = []

for filename, expected_hash in EXPECTED_MODELS.items():
    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    actual_hash = sha256_file(path)

    if actual_hash != expected_hash:
        raise RuntimeError(
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


asset_audit = load_json(
    ASSET_AUDIT_PATH
)

protocol_lock = load_json(
    PROTOCOL_LOCK_PATH
)

proposal_manifest = load_json(
    PROPOSAL_MANIFEST_PATH
)

proposal_evaluation_manifest = load_json(
    PROPOSAL_EVALUATION_MANIFEST_PATH
)

diagnosis_manifest = load_json(
    DIAGNOSIS_MANIFEST_PATH
)

statistics_manifest = load_json(
    STATISTICS_MANIFEST_PATH
)


for name, value in {
    "asset audit": asset_audit,
    "protocol lock": protocol_lock,
    "proposal generation": proposal_manifest,
    "proposal evaluation": proposal_evaluation_manifest,
    "detector-conditioned diagnosis": diagnosis_manifest,
    "detector-conditioned statistics": statistics_manifest,
}.items():
    if value.get("status") != "PASS":
        raise RuntimeError(
            f"{name} manifest is not PASS."
        )


if (
    protocol_lock.get(
        "valid_existing_frozen_isolation_forest_count"
    )
    != 0
):
    raise RuntimeError(
        "Valid existing frozen IsolationForest count "
        "is not zero."
    )

if (
    protocol_lock.get(
        "misclassified_candidate_count"
    )
    != 3
):
    raise RuntimeError(
        "Misclassified detector asset count is not three."
    )

if proposal_manifest.get("total_proposal_count") != 2475:
    raise RuntimeError(
        "Proposal count is not 2475."
    )

if (
    proposal_evaluation_manifest.get(
        "localization_evaluation_count"
    )
    != 2118
):
    raise RuntimeError(
        "Localization evaluation count is not 2118."
    )

if diagnosis_manifest.get("prediction_count") != 12375:
    raise RuntimeError(
        "Detector-conditioned prediction count "
        "is not 12375."
    )

if (
    diagnosis_manifest.get(
        "deterministic_prediction_replay"
    )
    != "12375/12375"
):
    raise RuntimeError(
        "Detector-conditioned prediction replay mismatch."
    )

if (
    statistics_manifest
    .get("bootstrap", {})
    .get("repetitions")
    != 10000
):
    raise RuntimeError(
        "Paired-bootstrap repetition count "
        "is not 10000."
    )


proposal_overall = pd.read_csv(
    PROPOSAL_EVALUATION_ROOT
    / "proposal_metrics_overall.csv"
)

proposal_stratified = pd.read_csv(
    PROPOSAL_EVALUATION_ROOT
    / "proposal_metrics_stratified.csv"
)

diagnosis_summary = pd.read_csv(
    DIAGNOSIS_ROOT
    / "detector_summary.csv"
)

paired_bootstrap = pd.read_csv(
    STATISTICS_ROOT
    / "paired_sample_cluster_bootstrap.csv"
)

diagnostic_stratified = pd.read_csv(
    STATISTICS_ROOT
    / "stratified_diagnostic_metrics.csv"
)

associations = pd.read_csv(
    STATISTICS_ROOT
    / "iou_diagnostic_associations.csv"
)

transitions = pd.read_csv(
    STATISTICS_ROOT
    / "paired_outcome_transitions.csv"
)


validated_results = {}


for detector in DETECTORS:
    expected = EXPECTED_RESULTS[
        detector
    ]

    proposal_row = one_row(
        proposal_overall,
        detector=detector,
    )

    summary_row = one_row(
        diagnosis_summary,
        detector=detector,
    )

    selected_row = one_row(
        paired_bootstrap,
        detector=detector,
        metric="selected_accuracy",
    )

    physical_exact_row = one_row(
        paired_bootstrap,
        detector=detector,
        metric="physical_exact_diagnosis_rate",
    )

    control_abstention_row = one_row(
        paired_bootstrap,
        detector=detector,
        metric="control_abstention_rate",
    )

    false_specific_row = one_row(
        paired_bootstrap,
        detector=detector,
        metric="false_specific_rate",
    )

    zero_iou_row = one_row(
        diagnostic_stratified,
        dimension="iou_bin",
        detector=detector,
        iou_bin="IoU = 0",
    )

    high_iou_row = one_row(
        diagnostic_stratified,
        dimension="iou_bin",
        detector=detector,
        iou_bin="IoU >= 0.5",
    )

    rho_task_row = one_row(
        associations,
        detector=detector,
        predictor="interval_iou",
        outcome="detector_task_success_rate",
    )

    rho_change_row = one_row(
        associations,
        detector=detector,
        predictor="interval_iou",
        outcome="prediction_change_rate",
    )

    transition_row = one_row(
        transitions,
        dimension="overall",
        detector=detector,
    )


    assert_close(
        f"{detector}.mean_iou",
        proposal_row["mean_iou"],
        expected["mean_iou"],
    )

    assert_close(
        f"{detector}.selected_accuracy",
        summary_row[
            "selected_accuracy_mean"
        ],
        expected["selected_accuracy"],
    )

    assert_close(
        f"{detector}.delta_selected_accuracy",
        selected_row["delta_estimate"],
        expected["delta_selected_accuracy"],
    )

    assert_close(
        f"{detector}.physical_exact",
        summary_row[
            "physical_exact_diagnosis_rate_mean"
        ],
        expected["physical_exact"],
    )

    assert_close(
        f"{detector}.delta_physical_exact",
        physical_exact_row["delta_estimate"],
        expected["delta_physical_exact"],
    )

    assert_close(
        f"{detector}.control_abstention",
        summary_row[
            "control_abstention_rate_mean"
        ],
        expected["control_abstention"],
    )

    assert_close(
        f"{detector}.delta_control_abstention",
        control_abstention_row["delta_estimate"],
        expected["delta_control_abstention"],
    )

    assert_close(
        f"{detector}.false_specific",
        summary_row[
            "false_specific_rate_mean"
        ],
        expected["false_specific"],
    )

    assert_close(
        f"{detector}.delta_false_specific",
        false_specific_row["delta_estimate"],
        expected["delta_false_specific"],
    )

    assert_close(
        f"{detector}.iou_zero_delta_task",
        zero_iou_row[
            "delta_task_success_rate"
        ],
        expected["iou_zero_delta_task"],
    )

    assert_close(
        f"{detector}.iou_high_delta_task",
        high_iou_row[
            "delta_task_success_rate"
        ],
        expected["iou_high_delta_task"],
    )

    assert_close(
        f"{detector}.rho_task_success",
        rho_task_row["spearman_rho"],
        expected["rho_task_success"],
    )

    assert_close(
        f"{detector}.rho_prediction_change",
        rho_change_row["spearman_rho"],
        expected["rho_prediction_change"],
    )

    assert_close(
        f"{detector}.net_task_change",
        transition_row[
            "net_task_success_change_rate"
        ],
        expected["net_task_change"],
    )


    if float(
        selected_row[
            "delta_upper_95"
        ]
    ) >= 0.0:
        raise RuntimeError(
            f"{detector}: selected-accuracy "
            "delta CI does not remain below zero."
        )

    if float(
        physical_exact_row[
            "delta_upper_95"
        ]
    ) >= 0.0:
        raise RuntimeError(
            f"{detector}: physical-exact "
            "delta CI does not remain below zero."
        )

    if float(
        control_abstention_row[
            "delta_upper_95"
        ]
    ) >= 0.0:
        raise RuntimeError(
            f"{detector}: control-abstention "
            "delta CI does not remain below zero."
        )

    if float(
        false_specific_row[
            "delta_lower_95"
        ]
    ) <= 0.0:
        raise RuntimeError(
            f"{detector}: false-specific "
            "delta CI does not remain above zero."
        )


    severity_rows = proposal_stratified[
        (
            proposal_stratified[
                "dimension"
            ] == "severity"
        )
        & (
            proposal_stratified[
                "detector"
            ] == detector
        )
    ]

    severity_iou = {
        str(row["severity"]):
            float(row["mean_iou"])
        for _, row in severity_rows.iterrows()
    }

    if not (
        severity_iou["mild"]
        < severity_iou["moderate"]
        < severity_iou["severe"]
    ):
        raise RuntimeError(
            f"{detector}: proposal IoU does not "
            "increase from mild to severe."
        )


    diagnosis_severity_rows = diagnostic_stratified[
        (
            diagnostic_stratified[
                "dimension"
            ] == "severity"
        )
        & (
            diagnostic_stratified[
                "detector"
            ] == detector
        )
    ]

    severity_delta = {
        str(row["severity"]):
            float(
                row[
                    "delta_task_success_rate"
                ]
            )
        for _, row
        in diagnosis_severity_rows.iterrows()
    }

    if not (
        severity_delta["severe"]
        < severity_delta["mild"]
    ):
        raise RuntimeError(
            f"{detector}: severe paired diagnostic "
            "loss is not larger than mild loss."
        )


    validated_results[
        detector
    ] = {
        "localization": {
            "mean_iou":
                float(
                    proposal_row[
                        "mean_iou"
                    ]
                ),

            "event_recall_iou_ge_0_3":
                float(
                    proposal_row[
                        "event_recall_iou_ge_0_3"
                    ]
                ),

            "event_recall_iou_ge_0_5":
                float(
                    proposal_row[
                        "event_recall_iou_ge_0_5"
                    ]
                ),
        },

        "diagnosis": {
            "selected_accuracy":
                float(
                    summary_row[
                        "selected_accuracy_mean"
                    ]
                ),

            "delta_selected_accuracy":
                float(
                    selected_row[
                        "delta_estimate"
                    ]
                ),

            "delta_selected_accuracy_ci95":
                [
                    float(
                        selected_row[
                            "delta_lower_95"
                        ]
                    ),
                    float(
                        selected_row[
                            "delta_upper_95"
                        ]
                    ),
                ],

            "physical_exact_diagnosis_rate":
                float(
                    summary_row[
                        "physical_exact_diagnosis_rate_mean"
                    ]
                ),

            "delta_physical_exact_diagnosis_rate":
                float(
                    physical_exact_row[
                        "delta_estimate"
                    ]
                ),

            "control_abstention_rate":
                float(
                    summary_row[
                        "control_abstention_rate_mean"
                    ]
                ),

            "delta_control_abstention_rate":
                float(
                    control_abstention_row[
                        "delta_estimate"
                    ]
                ),

            "false_specific_rate":
                float(
                    summary_row[
                        "false_specific_rate_mean"
                    ]
                ),

            "delta_false_specific_rate":
                float(
                    false_specific_row[
                        "delta_estimate"
                    ]
                ),
        },

        "interval_sensitivity": {
            "iou_zero_delta_task_success":
                float(
                    zero_iou_row[
                        "delta_task_success_rate"
                    ]
                ),

            "iou_ge_0_5_delta_task_success":
                float(
                    high_iou_row[
                        "delta_task_success_rate"
                    ]
                ),

            "iou_vs_task_success_spearman":
                float(
                    rho_task_row[
                        "spearman_rho"
                    ]
                ),

            "iou_vs_prediction_change_spearman":
                float(
                    rho_change_row[
                        "spearman_rho"
                    ]
                ),
        },

        "severity": {
            "proposal_mean_iou":
                severity_iou,

            "paired_task_success_delta":
                severity_delta,
        },
    }


claims = {
    "schema":
        "phyguard.sionna.formal825."
        "upstream_sensitivity.claims.v1",

    "status":
        "PASS",

    "reviewer_comment_scope":
        "Sensitivity of the downstream selective "
        "diagnostic pipeline to realistic upstream "
        "detector-proposed candidate intervals.",

    "formal_scope": {
        "causal_valid_samples":
            825,

        "physical_localization_samples":
            706,

        "control_samples":
            119,

        "upstream_detectors":
            3,

        "locked_downstream_models":
            5,

        "detector_conditioned_predictions":
            12375,

        "paired_locked_interval_predictions":
            4125,

        "paired_bootstrap_repetitions":
            10000,
    },

    "validated_results":
        validated_results,

    "supported_claims": [
        (
            "Candidate-interval quality materially affects "
            "mechanism discrimination and selective-risk "
            "control under the frozen reconstruction."
        ),

        (
            "All three detector-proposed conditions reduced "
            "selected accuracy, physical exact diagnosis, "
            "and control abstention relative to the locked "
            "event-interval reference; paired 95% intervals "
            "excluded zero."
        ),

        (
            "PCA reconstruction achieved the highest overall "
            "localization quality and the smallest downstream "
            "diagnostic degradation among the three detectors "
            "that were frozen before formal evaluation."
        ),

        (
            "Non-overlapping proposals produced large paired "
            "diagnostic losses, whereas IoU >= 0.5 largely "
            "preserved the locked-interval result for Robust "
            "Energy and Isolation Forest."
        ),

        (
            "Mild events had the lowest proposal overlap for "
            "all three detectors."
        ),

        (
            "Downstream paired diagnostic loss was not confined "
            "to mild events; severe strata also showed substantial "
            "absolute degradation relative to their locked-interval "
            "reference."
        ),
    ],

    "excluded_claims": [
        "The upstream detectors constitute a new detector contribution.",
        "PCA was selected after viewing Formal825 results.",
        "IoU was proven to causally determine diagnostic correctness.",
        "The analysis constitutes SDR or over-the-air validation.",
        "The analysis proves end-to-end field deployment readiness.",
        "Diagnostic degradation occurs only in mild events.",
        "The results generalize beyond the evaluated 1x2 SIMO CDL-A/B/C scope.",
    ],

    "recommended_manuscript_wording": {
        "main_result":
            (
                "Across three upstream detectors frozen before "
                "formal evaluation, detector-proposed intervals "
                "reduced selected mechanism accuracy and exact "
                "physical diagnosis relative to the locked-event-"
                "interval reference. The degradation was accompanied "
                "by a marked increase in false specific diagnoses "
                "on control sequences."
            ),

        "interval_quality":
            (
                "Diagnostic stability was strongly associated with "
                "candidate-interval overlap. Non-overlapping proposals "
                "caused substantial paired losses, whereas proposals "
                "with IoU at least 0.5 largely preserved the locked-"
                "interval result for two of the three detectors."
            ),

        "severity_boundary":
            (
                "Mild events were the most difficult to localize, "
                "but diagnostic degradation was not restricted to "
                "mild events; severe strata also exhibited substantial "
                "absolute loss relative to the stronger locked-interval "
                "reference."
            ),

        "method_boundary":
            (
                "This analysis evaluates diagnosis conditioned on "
                "frozen detector-proposed intervals. It is a sensitivity "
                "analysis rather than a new anomaly-detector contribution."
            ),
    },
}


CLAIMS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

CLAIMS_PATH.write_text(
    json.dumps(
        claims,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


environment = {
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
}

ENVIRONMENT_PATH.write_text(
    json.dumps(
        environment,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


entries = {}


def add_entry(path, archive_name=None):
    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_file():
        raise RuntimeError(
            f"Archive source is not a file: {path}"
        )

    name = (
        archive_name
        if archive_name is not None
        else str(
            path.relative_to(ROOT)
        )
    )

    if name in entries:
        raise RuntimeError(
            f"Duplicate archive entry: {name}"
        )

    entries[name] = path


def add_tree(directory):
    for path in sorted(
        directory.rglob("*")
    ):
        if path.is_file():
            add_entry(path)


for directory in [
    PROPOSAL_ROOT,
    PROPOSAL_EVALUATION_ROOT,
    DIAGNOSIS_ROOT,
    STATISTICS_ROOT,
]:
    add_tree(directory)


direct_project_files = [
    FORMAL825_PARENT_ARCHIVE,
    PROTOCOL_ARCHIVE,
    PROTOCOL_PATH,
    PLAN_PATH,
    CAUSAL_MANIFEST_PATH,
    ASSET_AUDIT_PATH,
    PROTOCOL_LOCK_PATH,
    PROPOSAL_MANIFEST_PATH,
    PROPOSAL_EVALUATION_MANIFEST_PATH,
    DIAGNOSIS_MANIFEST_PATH,
    STATISTICS_MANIFEST_PATH,
    CLAIMS_PATH,
    ENVIRONMENT_PATH,
]

for path in direct_project_files:
    add_entry(path)


for optional_path in [
    FORMAL825_PARENT_SHA_PATH,
    PROTOCOL_ARCHIVE_SHA_PATH,
]:
    if optional_path.exists():
        add_entry(optional_path)


add_entry(
    IFOREST_MODEL_PATH,
    "locked_assets/upstream/"
    "formal825_source_fitted_isolation_forest_v1.joblib",
)

add_entry(
    DETECTOR_SCRIPT_PATH,
    "locked_assets/source_code/"
    "detector_proposals.py",
)

add_entry(
    FEATURE_SCRIPT_PATH,
    "locked_assets/source_code/"
    "features.py",
)

add_entry(
    EVALUATION_COMMON_PATH,
    "locked_assets/source_code/"
    "evaluation_common.py",
)

add_entry(
    THRESHOLD_CONTRACT_PATH,
    "locked_assets/contracts/"
    "source_fixed_thresholds_locked.json",
)


for filename in EXPECTED_MODELS:
    add_entry(
        MODEL_DIR / filename,
        "locked_assets/downstream_models/"
        + filename,
    )


for script_number in range(50, 57):
    matches = sorted(
        (ROOT / "scripts").glob(
            f"{script_number}_*.py"
        )
    )

    if not matches:
        raise RuntimeError(
            f"No script found for stage {script_number}."
        )

    for path in matches:
        add_entry(path)


for log_number in range(50, 56):
    matches = sorted(
        (ROOT / "logs").glob(
            f"{log_number}_*.log"
        )
    )

    if not matches:
        raise RuntimeError(
            f"No log found for stage {log_number}."
        )

    for path in matches:
        add_entry(path)


source_hashes_before = {
    archive_name:
        sha256_file(path)
    for archive_name, path
    in entries.items()
}


file_records = [
    {
        "archive_name":
            archive_name,

        "source_path":
            str(path),

        "sha256":
            source_hashes_before[
                archive_name
            ],

        "size_bytes":
            path.stat().st_size,
    }
    for archive_name, path
    in sorted(
        entries.items()
    )
]


freeze_manifest = {
    "schema":
        "phyguard.sionna.formal825."
        "upstream_sensitivity.final_freeze.v1",

    "status":
        "PASS",

    "purpose":
        (
            "Freeze the preregistered three-detector "
            "candidate-interval sensitivity analysis, "
            "paired detector-conditioned diagnosis, "
            "and uncertainty analyses used to address "
            "the reviewer concern about realistic upstream "
            "candidate intervals."
        ),

    "formal_scope": {
        "causal_valid_sample_count":
            825,

        "physical_localization_sample_count":
            706,

        "control_sample_count":
            119,

        "detector_count":
            3,

        "model_repeat_count":
            5,

        "proposal_count":
            2475,

        "localization_evaluation_count":
            2118,

        "detector_conditioned_prediction_count":
            12375,

        "paired_locked_interval_prediction_count":
            4125,

        "paired_bootstrap_repetitions":
            10000,
    },

    "lineage": {
        "formal825_parent_archive_sha256":
            EXPECTED_FORMAL825_PARENT_SHA256,

        "upstream_protocol_archive_sha256":
            EXPECTED_PROTOCOL_ARCHIVE_SHA256,

        "upstream_protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,

        "source_fitted_isolation_forest_sha256":
            EXPECTED_IFOREST_SHA256,

        "detector_script_sha256":
            EXPECTED_DETECTOR_SCRIPT_SHA256,

        "feature_script_sha256":
            EXPECTED_FEATURE_SCRIPT_SHA256,

        "evaluation_common_sha256":
            EXPECTED_EVALUATION_COMMON_SHA256,

        "threshold_contract_sha256":
            EXPECTED_THRESHOLD_SHA256,
    },

    "validated_results":
        validated_results,

    "model_records":
        model_records,

    "integrity": {
        "proposal_replay":
            "2475/2475 exact",

        "proposal_feature_replay":
            "2475/2475 exact",

        "detector_conditioned_prediction_replay":
            "12375/12375 exact",

        "paired_cluster_unit":
            "sample_id",

        "five_model_repeats_kept_together":
            True,

        "protocol_locked_before_formal_proposals":
            True,

        "all_three_detector_results_preserved":
            True,
    },

    "methodological_boundary": {
        "formal_test_model_fitting":
            False,

        "downstream_model_training":
            False,

        "threshold_tuning":
            False,

        "threshold_calibration":
            False,

        "detector_parameter_tuning":
            False,

        "candidate_window_tuning":
            False,

        "post_result_detector_selection":
            False,

        "failed_sample_replacement":
            False,

        "new_detector_contribution_claim":
            False,

        "causal_iou_claim":
            False,

        "sdr_or_ota_claim":
            False,
    },

    "archive_entry_count":
        len(file_records) + 1,

    "files":
        file_records,
}


FREEZE_MANIFEST_PATH.write_text(
    json.dumps(
        freeze_manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


add_entry(
    FREEZE_MANIFEST_PATH
)


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
        f"Archive entry count={len(archive_names)}, "
        f"expected {len(entries)}."
    )


for archive_name, path in entries.items():
    if archive_name == str(
        FREEZE_MANIFEST_PATH.relative_to(ROOT)
    ):
        continue

    current_hash = sha256_file(path)

    if (
        current_hash
        != source_hashes_before[
            archive_name
        ]
    ):
        raise RuntimeError(
            "Source changed during freeze: "
            f"{archive_name}"
        )


for filename, expected_hash in EXPECTED_MODELS.items():
    if (
        sha256_file(
            MODEL_DIR / filename
        )
        != expected_hash
    ):
        raise RuntimeError(
            f"Model changed during freeze: {filename}"
        )


print("status: PASS")
print(
    "reviewer_comment_scope:",
    "upstream candidate-interval sensitivity",
)
print("causal_valid_sample_count: 825")
print("physical_localization_sample_count: 706")
print("control_sample_count: 119")
print("detector_count: 3")
print("model_repeat_count: 5")
print("proposal_count: 2475")
print("localization_evaluation_count: 2118")
print("detector_conditioned_prediction_count: 12375")
print("paired_locked_interval_prediction_count: 4125")
print("paired_bootstrap_repetitions: 10000")
print("proposal_replay: 2475/2475 exact")
print("proposal_feature_replay: 2475/2475 exact")
print(
    "detector_conditioned_prediction_replay:",
    "12375/12375 exact",
)
print("post_result_detector_selection: False")
print("models_unchanged: True")
print("zip_test: PASS")
print("archive:", ARCHIVE_PATH)
print("archive_sha256:", archive_sha256)
print("archive_entry_count:", len(archive_names))
print("claims:", CLAIMS_PATH)
print("freeze_manifest:", FREEZE_MANIFEST_PATH)
print("sha256_file:", ARCHIVE_SHA_PATH)

print(
    "\nSIONNA_FORMAL825_UPSTREAM_"
    "SENSITIVITY_FINAL_FREEZE_PASS"
)
