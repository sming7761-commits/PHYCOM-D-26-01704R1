import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


ROOT = Path("/root/phyguard_revision")

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "sionna_formal825_upstream_detector_protocol_v1.json"
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

LOCKED_INFERENCE_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_locked_reconstruction_inference.json"
)

FORMAL825_FINAL_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal825_locked_reconstruction_final.zip"
)

PROPOSAL_ROOT = (
    ROOT
    / "results"
    / "sionna_formal825_upstream_detector_proposals_v1"
)

PROPOSAL_EVALUATION_PATH = (
    ROOT
    / "results"
    / "sionna_formal825_upstream_detector_evaluation_v1"
    / "proposal_evaluations.csv"
)

ORACLE_ROOT = (
    ROOT
    / "results"
    / "sionna_formal_test840_locked_phyguard"
)

ORACLE_METADATA_PATH = (
    ORACLE_ROOT
    / "metadata.csv"
)

ORACLE_PREDICTIONS_PATH = (
    ORACLE_ROOT
    / "predictions.csv"
)

R0_ROOT = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
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

IFOREST_MODEL_PATH = (
    ROOT
    / "artifacts"
    / "frozen_upstream_detectors"
    / "formal825_source_fitted_isolation_forest_v1.joblib"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "sionna_formal825_detector_conditioned_diagnosis_v1"
)

TEMP_ROOT = Path(
    str(OUTPUT_ROOT) + "_tmp"
)

OUTPUT_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal825_detector_conditioned_diagnosis_v1.json"
)

EXPECTED_PROTOCOL_SHA256 = (
    "4a9b09b82e82efa4940602a73a25cdd"
    "aa05361af4c5edee2ae73f46860996aa2"
)

EXPECTED_FORMAL825_ARCHIVE_SHA256 = (
    "5d08fc3b8265789819b0f49c1f93a9c7"
    "9c8fe44ce7ec4aec8f1fa603a08c241b"
)

EXPECTED_IFOREST_SHA256 = (
    "9ad52cf66ffff12f423959db37ff9ed7"
    "c5d7292d620023c531217c42c5040dd1"
)

EXPECTED_EVALUATION_COMMON_SHA256 = (
    "afa1f3387c272dee075f72f481e01257"
    "2813892976c54dc860901114f8080cdd"
)

DETECTORS = [
    "robust_energy",
    "pca_reconstruction",
    "isolation_forest",
]

PHYSICAL_CLASSES = [
    "adaptation_mismatch",
    "blockage",
    "interference",
    "mobility",
]

CONTROL_CLASSES = [
    "normal",
    "nonphysical_goodput",
]

EXPECTED_MODELS = [
    {
        "repeat": 0,
        "filename": "repeat_0_phyguard.joblib",
        "sha256":
            "6877df3e027ec8e7b1964e835c706c96"
            "f54a4a9bdbfcc5b230914773cc7332ff",
        "tau_g": 0.70,
        "tau_c": 0.35,
    },
    {
        "repeat": 1,
        "filename": "repeat_1_phyguard.joblib",
        "sha256":
            "2a009ff1f590500d930e3eec0b6209697"
            "8ba73d1b1eaf2b5fc3c320c5a70f627",
        "tau_g": 0.35,
        "tau_c": 0.45,
    },
    {
        "repeat": 2,
        "filename": "repeat_2_phyguard.joblib",
        "sha256":
            "c875168c7d92449cc08335920723df562"
            "4bd066365c39eb36fab184111055e75",
        "tau_g": 0.40,
        "tau_c": 0.45,
    },
    {
        "repeat": 3,
        "filename": "repeat_3_phyguard.joblib",
        "sha256":
            "ee74feb3e43e34f9149eaeee9797b949"
            "43275344c84b919a973f30fadc17ea99",
        "tau_g": 0.50,
        "tau_c": 0.40,
    },
    {
        "repeat": 4,
        "filename": "repeat_4_phyguard.joblib",
        "sha256":
            "eb1319f18489287ae2aef3313824613e"
            "30108a95bd6e70d096e3c0d45bd84f2d",
        "tau_g": 0.50,
        "tau_c": 0.45,
    },
]

CORE_METRICS = [
    "selected_accuracy",
    "coverage",
    "false_specific_rate",
    "macro_f1",
    "balanced_accuracy",
    "physical_exact_diagnosis_rate",
    "physical_selection_rate",
    "control_abstention_rate",
    "normal_abstention_rate",
    "nonphysical_abstention_rate",
    "overall_output_rate",
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


def import_module(path, name):
    directory = str(path.parent)

    if directory not in sys.path:
        sys.path.insert(0, directory)

    spec = importlib.util.spec_from_file_location(
        name,
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


def to_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .astype(bool)
    )


def clean_value(value):
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def compute_metrics(
    truth,
    prediction,
    abstain,
    selective_metrics,
):
    truth = np.asarray(
        truth,
        dtype=object,
    )

    prediction = np.asarray(
        prediction,
        dtype=object,
    )

    selected = prediction != abstain

    physical_mask = np.isin(
        truth,
        PHYSICAL_CLASSES,
    )

    control_mask = np.isin(
        truth,
        CONTROL_CLASSES,
    )

    normal_mask = truth == "normal"

    nonphysical_mask = (
        truth == "nonphysical_goodput"
    )

    metrics = {
        key: clean_value(value)
        for key, value in selective_metrics(
            truth,
            prediction,
        ).items()
    }

    metrics.update(
        {
            "physical_exact_diagnosis_rate":
                float(
                    np.mean(
                        prediction[physical_mask]
                        == truth[physical_mask]
                    )
                ),

            "physical_selection_rate":
                float(
                    np.mean(
                        selected[physical_mask]
                    )
                ),

            "control_abstention_rate":
                float(
                    np.mean(
                        prediction[control_mask]
                        == abstain
                    )
                ),

            "normal_abstention_rate":
                float(
                    np.mean(
                        prediction[normal_mask]
                        == abstain
                    )
                ),

            "nonphysical_abstention_rate":
                float(
                    np.mean(
                        prediction[nonphysical_mask]
                        == abstain
                    )
                ),

            "overall_output_rate":
                float(
                    np.mean(selected)
                ),
        }
    )

    return metrics


required_paths = [
    PROTOCOL_PATH,
    PROPOSAL_MANIFEST_PATH,
    PROPOSAL_EVALUATION_MANIFEST_PATH,
    LOCKED_INFERENCE_MANIFEST_PATH,
    FORMAL825_FINAL_ARCHIVE,
    PROPOSAL_ROOT,
    PROPOSAL_EVALUATION_PATH,
    ORACLE_METADATA_PATH,
    ORACLE_PREDICTIONS_PATH,
    EVALUATION_COMMON_PATH,
    IFOREST_MODEL_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)


if sklearn.__version__ != "1.8.0":
    raise RuntimeError(
        f"scikit-learn={sklearn.__version__}, "
        "expected 1.8.0."
    )

if (
    sha256_file(PROTOCOL_PATH)
    != EXPECTED_PROTOCOL_SHA256
):
    raise RuntimeError(
        "Upstream protocol SHA256 mismatch."
    )

if (
    sha256_file(FORMAL825_FINAL_ARCHIVE)
    != EXPECTED_FORMAL825_ARCHIVE_SHA256
):
    raise RuntimeError(
        "Formal825 parent archive SHA256 mismatch."
    )

if (
    sha256_file(IFOREST_MODEL_PATH)
    != EXPECTED_IFOREST_SHA256
):
    raise RuntimeError(
        "Frozen IsolationForest SHA256 mismatch."
    )

if (
    sha256_file(EVALUATION_COMMON_PATH)
    != EXPECTED_EVALUATION_COMMON_SHA256
):
    raise RuntimeError(
        "evaluation_common.py SHA256 mismatch."
    )


protocol = load_json(
    PROTOCOL_PATH
)

proposal_manifest = load_json(
    PROPOSAL_MANIFEST_PATH
)

proposal_evaluation_manifest = load_json(
    PROPOSAL_EVALUATION_MANIFEST_PATH
)

locked_inference_manifest = load_json(
    LOCKED_INFERENCE_MANIFEST_PATH
)


if protocol.get("status") != "LOCKED":
    raise RuntimeError(
        "Upstream protocol is not LOCKED."
    )

if proposal_manifest.get("status") != "PASS":
    raise RuntimeError(
        "Proposal manifest is not PASS."
    )

if (
    proposal_evaluation_manifest.get("status")
    != "PASS"
):
    raise RuntimeError(
        "Proposal evaluation manifest is not PASS."
    )

if (
    locked_inference_manifest.get("status")
    != "PASS"
):
    raise RuntimeError(
        "Locked-interval inference manifest is not PASS."
    )

if (
    proposal_manifest.get("total_proposal_count")
    != 2475
):
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

if (
    locked_inference_manifest.get(
        "prediction_count"
    )
    != 4125
):
    raise RuntimeError(
        "Locked-interval prediction count is not 4125."
    )


for record in proposal_manifest.get(
    "files",
    [],
):
    path = ROOT / record["relative_path"]

    if not path.exists():
        raise FileNotFoundError(path)

    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(
            "Proposal asset changed: "
            f"{record['relative_path']}"
        )


if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Final detector-conditioned diagnosis output "
        "already exists; refusing to overwrite it."
    )

if TEMP_ROOT.exists():
    shutil.rmtree(TEMP_ROOT)

TEMP_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


evaluation_module = import_module(
    EVALUATION_COMMON_PATH,
    "phyguard_detector_conditioned_evaluation_common",
)

selective_metrics = (
    evaluation_module.selective_metrics
)

ABSTAIN = str(
    evaluation_module.ABSTAIN
)


metadata = pd.read_csv(
    ORACLE_METADATA_PATH
).sort_values(
    "sample_id"
).reset_index(drop=True)

if len(metadata) != 825:
    raise RuntimeError(
        "Locked metadata count is not 825."
    )

if metadata["sample_id"].nunique() != 825:
    raise RuntimeError(
        "Locked metadata sample IDs are not unique."
    )


sample_ids = metadata[
    "sample_id"
].astype(str).tolist()

truth = metadata[
    "label"
].astype(str).to_numpy(dtype=object)


oracle_predictions = pd.read_csv(
    ORACLE_PREDICTIONS_PATH
)

oracle_predictions["selected"] = to_bool(
    oracle_predictions["selected"]
)

if len(oracle_predictions) != 4125:
    raise RuntimeError(
        "Oracle prediction count is not 4125."
    )


proposal_evaluations = pd.read_csv(
    PROPOSAL_EVALUATION_PATH
)

if len(proposal_evaluations) != 2475:
    raise RuntimeError(
        "Proposal-evaluation row count is not 2475."
    )


prediction_rows = []
metric_rows = []
delta_rows = []
model_records = []

model_hashes_before = {}


for expected in EXPECTED_MODELS:
    repeat = expected["repeat"]

    model_path = (
        MODEL_DIR
        / expected["filename"]
    )

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model_hash = sha256_file(
        model_path
    )

    model_hashes_before[
        repeat
    ] = model_hash

    if model_hash != expected["sha256"]:
        raise RuntimeError(
            f"Repeat {repeat}: model SHA256 mismatch."
        )

    bundle = joblib.load(model_path)

    if set(bundle) != {
        "gate",
        "resolver",
    }:
        raise RuntimeError(
            f"Repeat {repeat}: model keys mismatch."
        )

    gate = bundle["gate"]
    resolver = bundle["resolver"]

    if int(gate.n_features_in_) != 19:
        raise RuntimeError(
            f"Repeat {repeat}: Gate dimension mismatch."
        )

    if int(resolver.n_features_in_) != 79:
        raise RuntimeError(
            f"Repeat {repeat}: Resolver dimension mismatch."
        )

    resolver_classes = [
        str(value)
        for value in resolver.classes_
    ]

    if resolver_classes != PHYSICAL_CLASSES:
        raise RuntimeError(
            f"Repeat {repeat}: Resolver class order mismatch."
        )


    oracle_repeat = (
        oracle_predictions[
            oracle_predictions[
                "repeat"
            ] == repeat
        ]
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    if (
        oracle_repeat[
            "sample_id"
        ].astype(str).tolist()
        != sample_ids
    ):
        raise RuntimeError(
            f"Repeat {repeat}: oracle sample alignment mismatch."
        )

    oracle_prediction = (
        oracle_repeat[
            "prediction"
        ]
        .astype(str)
        .to_numpy(dtype=object)
    )

    oracle_selected = (
        oracle_repeat[
            "selected"
        ].to_numpy(dtype=bool)
    )

    oracle_metrics = compute_metrics(
        truth,
        oracle_prediction,
        ABSTAIN,
        selective_metrics,
    )


    for detector in DETECTORS:
        feature_path = (
            PROPOSAL_ROOT
            / detector
            / "features.npz"
        )

        proposal_path = (
            PROPOSAL_ROOT
            / detector
            / "proposals.csv"
        )

        if not feature_path.exists():
            raise FileNotFoundError(feature_path)

        if not proposal_path.exists():
            raise FileNotFoundError(proposal_path)


        with np.load(
            feature_path,
            allow_pickle=False,
        ) as archive:
            raw_all = np.asarray(
                archive["raw"],
                dtype=np.float32,
            )

            physical_all = np.asarray(
                archive["physical"],
                dtype=np.float32,
            )

            combined_all = np.asarray(
                archive["combined"],
                dtype=np.float32,
            )

            feature_ids = [
                str(value)
                for value in archive[
                    "sample_ids"
                ].tolist()
            ]


        if raw_all.shape != (825, 60):
            raise RuntimeError(
                f"{detector}: raw shape mismatch."
            )

        if physical_all.shape != (825, 19):
            raise RuntimeError(
                f"{detector}: physical shape mismatch."
            )

        if combined_all.shape != (825, 79):
            raise RuntimeError(
                f"{detector}: combined shape mismatch."
            )

        if len(set(feature_ids)) != 825:
            raise RuntimeError(
                f"{detector}: feature IDs are not unique."
            )


        feature_index = {
            sample_id: index
            for index, sample_id
            in enumerate(feature_ids)
        }

        try:
            order = np.asarray(
                [
                    feature_index[
                        sample_id
                    ]
                    for sample_id
                    in sample_ids
                ],
                dtype=int,
            )
        except KeyError as exc:
            raise RuntimeError(
                f"{detector}: missing feature sample ID "
                f"{exc}"
            )


        physical = physical_all[
            order
        ]

        combined = combined_all[
            order
        ]


        proposals = (
            pd.read_csv(
                proposal_path
            )
            .set_index("sample_id")
            .loc[sample_ids]
            .reset_index()
        )

        if len(proposals) != 825:
            raise RuntimeError(
                f"{detector}: proposal alignment failed."
            )


        evaluation = (
            proposal_evaluations[
                proposal_evaluations[
                    "detector"
                ] == detector
            ]
            .set_index("sample_id")
            .loc[sample_ids]
            .reset_index()
        )

        if len(evaluation) != 825:
            raise RuntimeError(
                f"{detector}: evaluation alignment failed."
            )


        gate_probability_1 = (
            gate.predict_proba(
                physical
            )[:, 1]
        )

        resolver_probability_1 = (
            resolver.predict_proba(
                combined
            )
        )

        gate_probability_2 = (
            gate.predict_proba(
                physical
            )[:, 1]
        )

        resolver_probability_2 = (
            resolver.predict_proba(
                combined
            )
        )

        if not np.array_equal(
            gate_probability_1,
            gate_probability_2,
        ):
            raise RuntimeError(
                f"{detector}/repeat {repeat}: "
                "Gate replay mismatch."
            )

        if not np.array_equal(
            resolver_probability_1,
            resolver_probability_2,
        ):
            raise RuntimeError(
                f"{detector}/repeat {repeat}: "
                "Resolver replay mismatch."
            )


        mechanism_index = (
            resolver_probability_1.argmax(
                axis=1
            )
        )

        mechanism_label = np.asarray(
            resolver.classes_,
            dtype=object,
        )[mechanism_index]

        mechanism_confidence = (
            resolver_probability_1.max(
                axis=1
            )
        )

        selected = (
            (
                gate_probability_1
                >= expected["tau_g"]
            )
            & (
                mechanism_confidence
                >= expected["tau_c"]
            )
        )

        prediction = np.where(
            selected,
            mechanism_label,
            ABSTAIN,
        ).astype(object)


        detector_metrics = compute_metrics(
            truth,
            prediction,
            ABSTAIN,
            selective_metrics,
        )

        metric_rows.append(
            {
                "detector":
                    detector,

                "repeat":
                    repeat,

                "sample_count":
                    825,

                "selected_count":
                    int(
                        selected.sum()
                    ),

                "abstained_count":
                    int(
                        825
                        - selected.sum()
                    ),

                "tau_g":
                    expected["tau_g"],

                "tau_c":
                    expected["tau_c"],

                **detector_metrics,
            }
        )


        delta_row = {
            "detector":
                detector,

            "repeat":
                repeat,

            "comparison":
                "detector_proposed_interval_minus_locked_event_interval",
        }

        for metric in CORE_METRICS:
            delta_row[
                f"detector_{metric}"
            ] = detector_metrics[
                metric
            ]

            delta_row[
                f"locked_interval_{metric}"
            ] = oracle_metrics[
                metric
            ]

            delta_row[
                f"delta_{metric}"
            ] = (
                detector_metrics[
                    metric
                ]
                - oracle_metrics[
                    metric
                ]
            )

        delta_rows.append(
            delta_row
        )


        for index, sample_id in enumerate(
            sample_ids
        ):
            label = str(
                truth[index]
            )

            is_physical = (
                label
                in PHYSICAL_CLASSES
            )

            detector_task_success = bool(
                (
                    prediction[index]
                    == label
                )
                if is_physical
                else (
                    prediction[index]
                    == ABSTAIN
                )
            )

            oracle_task_success = bool(
                (
                    oracle_prediction[index]
                    == label
                )
                if is_physical
                else (
                    oracle_prediction[index]
                    == ABSTAIN
                )
            )

            probability_map = {
                resolver_classes[
                    class_index
                ]:
                    float(
                        resolver_probability_1[
                            index,
                            class_index,
                        ]
                    )
                for class_index in range(
                    len(resolver_classes)
                )
            }

            iou_value = (
                None
                if pd.isna(
                    evaluation.iloc[
                        index
                    ]["iou"]
                )
                else float(
                    evaluation.iloc[
                        index
                    ]["iou"]
                )
            )

            prediction_rows.append(
                {
                    "detector":
                        detector,

                    "repeat":
                        repeat,

                    "sample_id":
                        sample_id,

                    "truth":
                        label,

                    "severity":
                        str(
                            metadata.iloc[
                                index
                            ]["severity"]
                        ),

                    "channel_model":
                        str(
                            metadata.iloc[
                                index
                            ]["channel_model"]
                        ),

                    "proposal_start":
                        int(
                            proposals.iloc[
                                index
                            ][
                                "proposal_start"
                            ]
                        ),

                    "proposal_end":
                        int(
                            proposals.iloc[
                                index
                            ][
                                "proposal_end"
                            ]
                        ),

                    "locked_event_start":
                        int(
                            metadata.iloc[
                                index
                            ][
                                "event_start"
                            ]
                        ),

                    "locked_event_end":
                        int(
                            metadata.iloc[
                                index
                            ][
                                "event_end"
                            ]
                        ),

                    "localization_eligible":
                        bool(
                            is_physical
                        ),

                    "interval_iou":
                        iou_value,

                    "detector_prediction":
                        str(
                            prediction[index]
                        ),

                    "detector_selected":
                        bool(
                            selected[index]
                        ),

                    "detector_exact_correct":
                        bool(
                            prediction[index]
                            == label
                        ),

                    "detector_task_success":
                        detector_task_success,

                    "locked_interval_prediction":
                        str(
                            oracle_prediction[
                                index
                            ]
                        ),

                    "locked_interval_selected":
                        bool(
                            oracle_selected[
                                index
                            ]
                        ),

                    "locked_interval_exact_correct":
                        bool(
                            oracle_prediction[
                                index
                            ]
                            == label
                        ),

                    "locked_interval_task_success":
                        oracle_task_success,

                    "selection_changed":
                        bool(
                            selected[index]
                            != oracle_selected[
                                index
                            ]
                        ),

                    "prediction_changed":
                        bool(
                            prediction[index]
                            != oracle_prediction[
                                index
                            ]
                        ),

                    "task_success_changed":
                        bool(
                            detector_task_success
                            != oracle_task_success
                        ),

                    "gate_probability":
                        float(
                            gate_probability_1[
                                index
                            ]
                        ),

                    "mechanism_confidence":
                        float(
                            mechanism_confidence[
                                index
                            ]
                        ),

                    "joint_confidence":
                        float(
                            gate_probability_1[
                                index
                            ]
                            * mechanism_confidence[
                                index
                            ]
                        ),

                    "tau_g":
                        expected["tau_g"],

                    "tau_c":
                        expected["tau_c"],

                    "resolver_adaptation_mismatch":
                        probability_map[
                            "adaptation_mismatch"
                        ],

                    "resolver_blockage":
                        probability_map[
                            "blockage"
                        ],

                    "resolver_interference":
                        probability_map[
                            "interference"
                        ],

                    "resolver_mobility":
                        probability_map[
                            "mobility"
                        ],
                }
            )


    model_records.append(
        {
            "repeat":
                repeat,

            "model_path":
                str(model_path),

            "model_sha256":
                model_hash,

            "tau_g":
                expected["tau_g"],

            "tau_c":
                expected["tau_c"],

            "model_file_unchanged":
                sha256_file(
                    model_path
                )
                == model_hash,
        }
    )


predictions = pd.DataFrame(
    prediction_rows
)

per_repeat_metrics = pd.DataFrame(
    metric_rows
)

per_repeat_deltas = pd.DataFrame(
    delta_rows
)


if len(predictions) != 12375:
    raise RuntimeError(
        f"Prediction count={len(predictions)}, "
        "expected 12375."
    )

if len(per_repeat_metrics) != 15:
    raise RuntimeError(
        "Per-repeat metric row count is not 15."
    )

if len(per_repeat_deltas) != 15:
    raise RuntimeError(
        "Per-repeat delta row count is not 15."
    )


summary_rows = []

for detector in DETECTORS:
    frame = per_repeat_metrics[
        per_repeat_metrics[
            "detector"
        ] == detector
    ]

    delta_frame = per_repeat_deltas[
        per_repeat_deltas[
            "detector"
        ] == detector
    ]

    row = {
        "detector":
            detector,

        "repeat_count":
            5,
    }

    for metric in CORE_METRICS:
        values = frame[
            metric
        ].to_numpy(dtype=float)

        deltas = delta_frame[
            f"delta_{metric}"
        ].to_numpy(dtype=float)

        row[
            f"{metric}_mean"
        ] = float(
            values.mean()
        )

        row[
            f"{metric}_std"
        ] = float(
            values.std(ddof=1)
        )

        row[
            f"delta_{metric}_mean"
        ] = float(
            deltas.mean()
        )

        row[
            f"delta_{metric}_std"
        ] = float(
            deltas.std(ddof=1)
        )

    detector_predictions = predictions[
        predictions[
            "detector"
        ] == detector
    ]

    row[
        "prediction_change_rate"
    ] = float(
        detector_predictions[
            "prediction_changed"
        ].mean()
    )

    row[
        "selection_change_rate"
    ] = float(
        detector_predictions[
            "selection_changed"
        ].mean()
    )

    row[
        "task_success_change_rate"
    ] = float(
        detector_predictions[
            "task_success_changed"
        ].mean()
    )

    summary_rows.append(row)


detector_summary = pd.DataFrame(
    summary_rows
)


predictions_path = (
    TEMP_ROOT
    / "predictions.csv"
)

per_repeat_metrics_path = (
    TEMP_ROOT
    / "per_repeat_metrics.csv"
)

per_repeat_deltas_path = (
    TEMP_ROOT
    / "per_repeat_deltas_vs_locked_interval.csv"
)

detector_summary_path = (
    TEMP_ROOT
    / "detector_summary.csv"
)

audit_path = (
    TEMP_ROOT
    / "audit.json"
)


predictions.to_csv(
    predictions_path,
    index=False,
)

per_repeat_metrics.to_csv(
    per_repeat_metrics_path,
    index=False,
)

per_repeat_deltas.to_csv(
    per_repeat_deltas_path,
    index=False,
)

detector_summary.to_csv(
    detector_summary_path,
    index=False,
)


for expected in EXPECTED_MODELS:
    repeat = expected["repeat"]

    model_path = (
        MODEL_DIR
        / expected["filename"]
    )

    if (
        sha256_file(model_path)
        != model_hashes_before[
            repeat
        ]
    ):
        raise RuntimeError(
            f"Repeat {repeat}: model changed "
            "during inference."
        )


audit = {
    "schema":
        "phyguard.sionna.formal825."
        "detector_conditioned_diagnosis.audit.v1",

    "status":
        "PASS",

    "sklearn_version":
        sklearn.__version__,

    "sample_count":
        825,

    "detector_count":
        3,

    "model_repeat_count":
        5,

    "prediction_count":
        12375,

    "locked_interval_reference_prediction_count":
        4125,

    "deterministic_prediction_replay":
        "12375/12375",

    "model_records":
        model_records,

    "methodological_boundary": {
        "formal_sequences_opened":
            False,

        "frozen_proposal_features_only":
            True,

        "model_training":
            False,

        "model_weight_update":
            False,

        "threshold_tuning":
            False,

        "threshold_calibration":
            False,

        "detector_selection":
            False,

        "all_three_detectors_reported":
            True,

        "ground_truth_used_for_evaluation_only":
            True,
    },
}

audit_path.write_text(
    json.dumps(
        audit,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


TEMP_ROOT.rename(
    OUTPUT_ROOT
)


output_files = sorted(
    path
    for path in OUTPUT_ROOT.rglob("*")
    if path.is_file()
)


manifest = {
    "schema":
        "phyguard.sionna.formal825."
        "detector_conditioned_diagnosis_manifest.v1",

    "status":
        "PASS",

    "protocol_sha256":
        EXPECTED_PROTOCOL_SHA256,

    "formal825_parent_archive_sha256":
        EXPECTED_FORMAL825_ARCHIVE_SHA256,

    "proposal_manifest_sha256":
        sha256_file(
            PROPOSAL_MANIFEST_PATH
        ),

    "proposal_evaluation_manifest_sha256":
        sha256_file(
            PROPOSAL_EVALUATION_MANIFEST_PATH
        ),

    "locked_inference_manifest_sha256":
        sha256_file(
            LOCKED_INFERENCE_MANIFEST_PATH
        ),

    "sample_count":
        825,

    "detectors":
        DETECTORS,

    "model_repeat_count":
        5,

    "prediction_count":
        12375,

    "paired_locked_interval_prediction_count":
        4125,

    "deterministic_prediction_replay":
        "12375/12375",

    "methodological_boundary": {
        "formal_test_model_fitting":
            False,

        "threshold_tuning":
            False,

        "threshold_calibration":
            False,

        "detector_parameter_tuning":
            False,

        "candidate_window_tuning":
            False,

        "detector_selection":
            False,

        "all_detector_results_preserved":
            True,

        "raw_formal_sequences_opened":
            False,
    },

    "files": [
        {
            "relative_path":
                str(
                    path.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                sha256_file(path),

            "size_bytes":
                path.stat().st_size,
        }
        for path in output_files
    ],
}


OUTPUT_MANIFEST_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_MANIFEST_PATH.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print("status: PASS")
print("sample_count: 825")
print("detector_count: 3")
print("model_repeat_count: 5")
print("prediction_count: 12375")
print(
    "paired_locked_interval_prediction_count: 4125"
)
print(
    "deterministic_prediction_replay: "
    "12375/12375"
)
print("model_training: False")
print("threshold_tuning: False")
print("detector_selection: False")
print("raw_formal_sequences_opened: False")


print("\nDETECTOR SUMMARY")

summary_columns = [
    "detector",
    "selected_accuracy_mean",
    "delta_selected_accuracy_mean",
    "coverage_mean",
    "delta_coverage_mean",
    "physical_exact_diagnosis_rate_mean",
    "delta_physical_exact_diagnosis_rate_mean",
    "control_abstention_rate_mean",
    "delta_control_abstention_rate_mean",
    "false_specific_rate_mean",
    "delta_false_specific_rate_mean",
    "prediction_change_rate",
    "selection_change_rate",
    "task_success_change_rate",
]

print(
    detector_summary[
        summary_columns
    ].to_string(
        index=False
    )
)


print("\nPER-REPEAT DELTAS")

delta_columns = [
    "detector",
    "repeat",
    "delta_selected_accuracy",
    "delta_coverage",
    "delta_physical_exact_diagnosis_rate",
    "delta_control_abstention_rate",
    "delta_false_specific_rate",
    "delta_macro_f1",
    "delta_balanced_accuracy",
]

print(
    per_repeat_deltas[
        delta_columns
    ].to_string(
        index=False
    )
)


print("\noutput_root:", OUTPUT_ROOT)
print("predictions:", OUTPUT_ROOT / "predictions.csv")
print("summary:", OUTPUT_ROOT / "detector_summary.csv")
print("manifest:", OUTPUT_MANIFEST_PATH)

print(
    "\nFORMAL825_DETECTOR_CONDITIONED_"
    "DIAGNOSIS_PASS"
)
