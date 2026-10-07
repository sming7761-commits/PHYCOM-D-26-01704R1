#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
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

DATA_ROOT = R0 / "data" / "full"

TELEMETRY_PATH = (
    DATA_ROOT
    / "telemetry.npz"
)

EVALUATE_SUITE_PATH = (
    R0
    / "scripts"
    / "evaluate_suite.py"
)

THRESHOLD_SOURCE_PATH = (
    R0
    / "results"
    / "full"
    / "evaluation_suite"
    / "threshold_search_all_methods.csv"
)

INTERFACE_AUDIT_MANIFEST = (
    ROOT
    / "manifests"
    / "temporal_baseline_interface_audit_v1.json"
)

LOCKED_RECONSTRUCTION_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal825_locked_reconstruction_final.zip"
)

PER_ABLATION_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_per_indicator_ablation_v1_final.zip"
)

COMPLEXITY_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_complexity_scaling_v1_final.zip"
)

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "temporal_baseline_protocol_v1.json"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "temporal_baseline_protocol_v1"
)

SPLIT_PATH = (
    OUTPUT_ROOT
    / "split_memberships.csv"
)

DATASET_PATH = (
    OUTPUT_ROOT
    / "dataset_contract.json"
)

MODEL_PATH = (
    OUTPUT_ROOT
    / "model_registry.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)

READY_PATH = (
    OUTPUT_ROOT
    / "PROTOCOL_READY.md"
)

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "temporal_baseline_protocol_v1.json"
)

EXPECTED_SHA256 = {
    str(LOCKED_RECONSTRUCTION_ARCHIVE):
        "5d08fc3b8265789819b0f49c1f93a9c79c8fe44ce7ec4aec8f1fa603a08c241b",

    str(PER_ABLATION_ARCHIVE):
        "c49c4465dead8422348d03d80b76420ed2396cca9663d9bbfa2301e2e8550555",

    str(COMPLEXITY_ARCHIVE):
        "cdf7b1a8f3415cde2c7c5f8c0b26c9421e4fc707f80de14d5c227e00afc61e9c",
}

BASE_SEED = 20260705
REPEATS = 5

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
        for block in iter(
            lambda: stream.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_module(
    module_name: str,
    path: Path,
):
    if str(path.parent) not in sys.path:
        sys.path.insert(
            0,
            str(path.parent),
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    require(
        spec is not None
        and spec.loader is not None,
        f"Could not load module: {path}",
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def locate_metadata() -> Path:
    candidates = []

    for path in sorted(
        DATA_ROOT.rglob("*.csv")
    ):
        try:
            frame = pd.read_csv(
                path,
                nrows=5,
            )
        except Exception:
            continue

        columns = {
            str(column).lower()
            for column in frame.columns
        }

        if (
            "label" in columns
            or "truth" in columns
            or "dataset_label" in columns
        ):
            candidates.append(
                path
            )

    require(
        len(candidates) > 0,
        "No metadata CSV with a label column was found.",
    )

    exact = []

    for path in candidates:
        frame = pd.read_csv(
            path
        )

        if len(frame) == 1080:
            exact.append(
                path
            )

    require(
        len(exact) == 1,
        (
            "Expected exactly one 1080-row metadata CSV; "
            f"found {[str(path) for path in exact]}."
        ),
    )

    return exact[0]


def unpack_split_result(result):
    if isinstance(result, dict):
        aliases = {
            "train": [
                "train",
                "train_idx",
                "train_indices",
            ],

            "validation": [
                "val",
                "valid",
                "validation",
                "val_idx",
                "validation_idx",
            ],

            "test": [
                "test",
                "test_idx",
                "test_indices",
            ],
        }

        output = {}

        for role, names in aliases.items():
            key = next(
                (
                    name
                    for name in names
                    if name in result
                ),
                None,
            )

            require(
                key is not None,
                f"Split dictionary lacks {role}.",
            )

            output[role] = np.asarray(
                result[key],
                dtype=int,
            ).reshape(-1)

        return output

    require(
        isinstance(result, tuple)
        and len(result) == 3,
        (
            "split_indices must return a 3-tuple "
            "or a train/validation/test dictionary."
        ),
    )

    return {
        "train":
            np.asarray(
                result[0],
                dtype=int,
            ).reshape(-1),

        "validation":
            np.asarray(
                result[1],
                dtype=int,
            ).reshape(-1),

        "test":
            np.asarray(
                result[2],
                dtype=int,
            ).reshape(-1),
    }


required_paths = [
    TELEMETRY_PATH,
    EVALUATE_SUITE_PATH,
    THRESHOLD_SOURCE_PATH,
    INTERFACE_AUDIT_MANIFEST,
    LOCKED_RECONSTRUCTION_ARCHIVE,
    PER_ABLATION_ARCHIVE,
    COMPLEXITY_ARCHIVE,
]

for path in required_paths:
    require(
        path.exists(),
        f"Required path is missing: {path}",
    )

for path in [
    LOCKED_RECONSTRUCTION_ARCHIVE,
    PER_ABLATION_ARCHIVE,
    COMPLEXITY_ARCHIVE,
]:
    require(
        sha256_file(path)
        == EXPECTED_SHA256[str(path)],
        f"Frozen asset checksum mismatch: {path}",
    )

for path in [
    PROTOCOL_PATH,
    OUTPUT_ROOT,
    MANIFEST_PATH,
]:
    require(
        not path.exists(),
        f"Protocol output already exists: {path}",
    )


metadata_path = locate_metadata()

metadata = pd.read_csv(
    metadata_path
)

with np.load(
    TELEMETRY_PATH,
    allow_pickle=False,
) as archive:
    X = np.asarray(
        archive[
            "X"
        ]
    )

    kpi_names = [
        str(value)
        for value in archive[
            "kpi_names"
        ].tolist()
    ]

require(
    X.shape == (1080, 80, 10),
    f"Unexpected telemetry shape: {X.shape}",
)

label_column = next(
    (
        column
        for column in [
            "label",
            "truth",
            "dataset_label",
        ]
        if column in metadata.columns
    ),
    None,
)

require(
    label_column is not None,
    "Metadata label column is missing.",
)

labels = (
    metadata[
        label_column
    ]
    .astype(str)
    .to_numpy(
        dtype=object
    )
)

observed_labels = sorted(
    set(
        labels.tolist()
    )
)

expected_labels = sorted(
    PHYSICAL_LABELS
    + CONTROL_LABELS
)

require(
    observed_labels == expected_labels,
    (
        "Observed labels do not match the six-class contract: "
        f"{observed_labels}"
    ),
)


evaluate_module = load_module(
    "phyguard_evaluate_suite_stage64b",
    EVALUATE_SUITE_PATH,
)

require(
    hasattr(
        evaluate_module,
        "split_indices",
    ),
    "evaluate_suite.py lacks split_indices.",
)


split_rows = []
split_hashes = {}

for repeat in range(
    REPEATS
):
    seed = (
        BASE_SEED
        + repeat
    )

    split = unpack_split_result(
        evaluate_module.split_indices(
            metadata,
            seed,
        )
    )

    concatenated = np.concatenate(
        [
            split["train"],
            split["validation"],
            split["test"],
        ]
    )

    require(
        len(concatenated) == 1080,
        (
            f"Repeat {repeat} does not assign all samples: "
            f"{len(concatenated)}."
        ),
    )

    require(
        len(
            np.unique(
                concatenated
            )
        )
        == 1080,
        f"Repeat {repeat} has duplicate split memberships.",
    )

    require(
        int(
            concatenated.min()
        )
        == 0
        and int(
            concatenated.max()
        )
        == 1079,
        f"Repeat {repeat} split indices are out of range.",
    )

    for role in [
        "train",
        "validation",
        "test",
    ]:
        indices = split[
            role
        ]

        label_counts = {
            label:
                int(
                    np.sum(
                        labels[
                            indices
                        ]
                        == label
                    )
                )
            for label in expected_labels
        }

        payload = (
            ",".join(
                str(int(index))
                for index in indices
            )
        ).encode(
            "utf-8"
        )

        digest = hashlib.sha256(
            payload
        ).hexdigest()

        split_hashes[
            f"repeat_{repeat}_{role}"
        ] = digest

        for index in indices:
            split_rows.append(
                {
                    "repeat":
                        repeat,

                    "seed":
                        seed,

                    "role":
                        role,

                    "index":
                        int(
                            index
                        ),

                    "label":
                        str(
                            labels[
                                index
                            ]
                        ),

                    "role_membership_sha256":
                        digest,

                    "role_size":
                        int(
                            len(
                                indices
                            )
                        ),

                    "role_label_counts":
                        json.dumps(
                            label_counts,
                            sort_keys=True,
                        ),
                }
            )


threshold_grid_frame = pd.read_csv(
    THRESHOLD_SOURCE_PATH,
    usecols=[
        "tau",
    ],
)

threshold_grid = sorted(
    {
        float(value)
        for value in threshold_grid_frame[
            "tau"
        ].dropna()
    }
)

require(
    len(threshold_grid) > 0,
    "Existing single-threshold grid is empty.",
)


models = [
    {
        "model_id":
            "ROCKET",

        "family":
            "random_convolution_kernel",

        "implementation":
            "aeon.RocketClassifier",

        "mandatory":
            True,

        "main_table":
            False,

        "hyperparameters":
            {
                "n_kernels":
                    5000,

                "random_state":
                    "repeat_seed",
            },
    },
    {
        "model_id":
            "MINIROCKET",

        "family":
            "random_convolution_kernel",

        "implementation":
            "aeon.MiniRocketClassifier",

        "mandatory":
            True,

        "main_table":
            True,

        "hyperparameters":
            {
                "n_kernels":
                    9996,

                "random_state":
                    "repeat_seed",
            },
    },
    {
        "model_id":
            "MULTIROCKET",

        "family":
            "random_convolution_kernel",

        "implementation":
            "aeon.MultiRocketClassifier",

        "mandatory":
            True,

        "main_table":
            False,

        "hyperparameters":
            {
                "n_kernels":
                    6250,

                "random_state":
                    "repeat_seed",
            },
    },
    {
        "model_id":
            "HYDRA",

        "family":
            "dictionary_convolution_kernel",

        "implementation":
            "aeon.HydraClassifier",

        "mandatory":
            True,

        "main_table":
            False,

        "hyperparameters":
            {
                "n_kernels":
                    8,

                "n_groups":
                    64,

                "random_state":
                    "repeat_seed",
            },
    },
    {
        "model_id":
            "MULTIROCKET_HYDRA",

        "family":
            "hybrid_convolution_kernel",

        "implementation":
            "aeon.MultiRocketHydraClassifier",

        "mandatory":
            True,

        "main_table":
            True,

        "hyperparameters":
            {
                "n_kernels":
                    5000,

                "random_state":
                    "repeat_seed",
            },
    },
    {
        "model_id":
            "ARSENAL",

        "family":
            "ensemble_convolution_kernel",

        "implementation":
            "aeon.Arsenal",

        "mandatory":
            True,

        "main_table":
            False,

        "hyperparameters":
            {
                "n_estimators":
                    10,

                "n_kernels":
                    2000,

                "random_state":
                    "repeat_seed",
            },
    },
    {
        "model_id":
            "FCN1D",

        "family":
            "fully_convolutional_network",

        "implementation":
            "custom_pytorch",

        "mandatory":
            True,

        "main_table":
            True,

        "hyperparameters":
            {
                "channels":
                    [
                        64,
                        128,
                        64,
                    ],

                "kernels":
                    [
                        8,
                        5,
                        3,
                    ],

                "dropout":
                    0.1,
            },
    },
    {
        "model_id":
            "RESNET1D",

        "family":
            "residual_convolutional_network",

        "implementation":
            "custom_pytorch",

        "mandatory":
            True,

        "main_table":
            False,

        "hyperparameters":
            {
                "base_channels":
                    64,

                "residual_blocks":
                    3,

                "dropout":
                    0.1,
            },
    },
    {
        "model_id":
            "INCEPTIONTIME_LITE",

        "family":
            "multi_scale_convolutional_network",

        "implementation":
            "custom_pytorch",

        "mandatory":
            True,

        "main_table":
            True,

        "hyperparameters":
            {
                "inception_blocks":
                    3,

                "filters":
                    32,

                "bottleneck":
                    32,

                "dropout":
                    0.1,
            },
    },
    {
        "model_id":
            "LITE1D",

        "family":
            "lightweight_convolutional_network",

        "implementation":
            "custom_pytorch",

        "mandatory":
            True,

        "main_table":
            False,

        "hyperparameters":
            {
                "filters":
                    32,

                "depthwise":
                    True,

                "dropout":
                    0.1,
            },
    },
    {
        "model_id":
            "TCN",

        "family":
            "dilated_temporal_convolution",

        "implementation":
            "custom_pytorch",

        "mandatory":
            True,

        "main_table":
            True,

        "hyperparameters":
            {
                "channels":
                    [
                        64,
                        64,
                        64,
                    ],

                "kernel_size":
                    3,

                "dilations":
                    [
                        1,
                        2,
                        4,
                    ],

                "dropout":
                    0.1,
            },
    },
    {
        "model_id":
            "BIGRU",

        "family":
            "recurrent_network",

        "implementation":
            "custom_pytorch",

        "mandatory":
            True,

        "main_table":
            False,

        "hyperparameters":
            {
                "hidden_size":
                    64,

                "layers":
                    1,

                "bidirectional":
                    True,

                "dropout":
                    0.0,
            },
    },
    {
        "model_id":
            "BILSTM",

        "family":
            "recurrent_network",

        "implementation":
            "custom_pytorch",

        "mandatory":
            True,

        "main_table":
            True,

        "hyperparameters":
            {
                "hidden_size":
                    64,

                "layers":
                    1,

                "bidirectional":
                    True,

                "dropout":
                    0.0,
            },
    },
    {
        "model_id":
            "TINY_TRANSFORMER",

        "family":
            "attention_network",

        "implementation":
            "custom_pytorch",

        "mandatory":
            True,

        "main_table":
            True,

        "hyperparameters":
            {
                "d_model":
                    64,

                "heads":
                    4,

                "layers":
                    2,

                "feedforward":
                    128,

                "dropout":
                    0.1,
            },
    },
]


status = (
    "LOCKED_BEFORE_TEMPORAL_BASELINE_EXECUTION"
)

protocol = {
    "schema":
        "phyguard.temporal_baseline_protocol.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        status,

    "purpose":
        (
            "Compare PhyGuard with a broad, preregistered set "
            "of modern multivariate time-series classifiers "
            "under identical five-repeat memberships and a "
            "validation-only selective-output protocol."
        ),

    "dataset_contract": {
        "telemetry_path":
            str(
                TELEMETRY_PATH.relative_to(
                    ROOT
                )
            ),

        "telemetry_sha256":
            sha256_file(
                TELEMETRY_PATH
            ),

        "metadata_path":
            str(
                metadata_path.relative_to(
                    ROOT
                )
            ),

        "metadata_sha256":
            sha256_file(
                metadata_path
            ),

        "shape":
            [
                1080,
                80,
                10,
            ],

        "dtype":
            str(
                X.dtype
            ),

        "kpi_names":
            kpi_names,

        "label_column":
            label_column,

        "labels":
            expected_labels,

        "label_counts":
            {
                label:
                    int(
                        np.sum(
                            labels == label
                        )
                    )
                for label in expected_labels
            },

        "physical_labels":
            PHYSICAL_LABELS,

        "control_labels":
            CONTROL_LABELS,
    },

    "split_contract": {
        "implementation":
            (
                "Existing evaluate_suite.py::split_indices"
            ),

        "base_seed":
            BASE_SEED,

        "repeats":
            REPEATS,

        "repeat_seeds":
            [
                BASE_SEED
                + repeat
                for repeat in range(
                    REPEATS
                )
            ],

        "split_membership_path":
            str(
                SPLIT_PATH.relative_to(
                    ROOT
                )
            ),

        "split_hashes":
            split_hashes,

        "no_test_membership_changes":
            True,
    },

    "task_contract": {
        "training_task":
            "six-class multivariate time-series classification",

        "input_orientation_for_aeon":
            "(n_cases, n_channels, n_timepoints)",

        "input_orientation_for_pytorch":
            "(batch, n_channels, n_timepoints)",

        "output_mapping":
            (
                "Predicted normal or nonphysical_goodput maps "
                "to ABSTAIN. A physical prediction is emitted "
                "only when the maximum physical-class "
                "probability is at least tau."
            ),

        "confidence":
            "maximum physical-class probability",

        "threshold_grid":
            threshold_grid,

        "threshold_selection":
            (
                "Validation only. Choose a feasible threshold "
                "with false-specific rate <= 0.10 and selected "
                "accuracy >= 0.80; among feasible thresholds "
                "maximize coverage, then selected accuracy, "
                "then use the larger threshold. If none is "
                "feasible, minimize the same locked penalty "
                "used by the existing evaluation contract."
            ),

        "test_threshold_selection":
            False,
    },

    "model_registry":
        models,

    "training_contract": {
        "normalization":
            (
                "Per-KPI mean and standard deviation fitted "
                "on the training split only and applied to "
                "validation and test."
            ),

        "deep_optimizer":
            "AdamW",

        "deep_learning_rate":
            0.001,

        "deep_weight_decay":
            0.0001,

        "deep_batch_size":
            64,

        "deep_max_epochs":
            100,

        "deep_early_stopping_patience":
            15,

        "deep_scheduler":
            "ReduceLROnPlateau on validation cross-entropy",

        "deep_gradient_clip_norm":
            1.0,

        "deep_class_weights":
            "inverse-frequency from the training split only",

        "deep_seed":
            "repeat_seed",

        "checkpoint_selection":
            (
                "Lowest validation cross-entropy. Test metrics "
                "must not influence checkpoint selection."
            ),

        "rocket_family_threads":
            1,

        "gpu_for_pytorch":
            True,

        "mixed_precision":
            True,

        "failed_run_policy":
            (
                "One deterministic retry is permitted only for "
                "documented numerical or memory failure using "
                "the preregistered reduced batch size 32. A "
                "model may not be removed because its result "
                "is unfavorable."
            ),
    },

    "metrics": {
        "primary":
            "selected_accuracy",

        "secondary": [
            "physical_selection_rate",
            "coverage",
            "false_specific_rate",
            "macro_f1",
            "balanced_accuracy",
            "physical_exact_match",
            "control_abstention",
            "risk_coverage_auc",
            "parameter_count",
            "training_seconds",
            "inference_median_ms_per_interval",
            "peak_gpu_memory_bytes",
        ],

        "aggregation":
            "mean and standard deviation across five repeats",

        "paired_comparison":
            (
                "Paired repeat-wise and sample-cluster "
                "bootstrap differences versus PhyGuard."
            ),

        "bootstrap_replicates":
            10000,

        "bootstrap_seed":
            64021,
    },

    "reporting_contract": {
        "all_models_mandatory":
            True,

        "full_table_location":
            "main manuscript appendix or supplementary table",

        "main_table_models": [
            model[
                "model_id"
            ]
            for model in models
            if model[
                "main_table"
            ]
        ],

        "main_table_selection_basis":
            (
                "Fixed a priori to cover random-kernel, hybrid, "
                "fully convolutional, multi-scale convolution, "
                "dilated convolution, recurrent, and attention "
                "families. It is not selected from test results."
            ),

        "unfavorable_results_must_be_reported":
            True,

        "failed_models_must_be_reported":
            True,

        "no_posthoc_model_omission":
            True,
    },

    "interpretation_boundaries": [
        (
            "PhyGuard need not outperform every temporal model "
            "on every metric; its claim is selective, "
            "physics-guided diagnosis with abstention."
        ),
        (
            "A temporal baseline with higher forced "
            "classification accuracy does not invalidate "
            "selective-risk or control-abstention advantages."
        ),
        (
            "Main-text compression may use the fixed "
            "representative subset, but the complete "
            "preregistered table must remain available."
        ),
    ],

    "prohibited_posthoc_actions": [
        "dropping a model because it outperforms PhyGuard",
        "dropping a model because it underperforms",
        "changing test memberships",
        "selecting thresholds on test data",
        "selecting deep checkpoints on test data",
        "changing the primary metric after viewing results",
        "reporting only favorable repeats",
        "changing architecture size after viewing test results",
        "using different control-label mappings for different models",
        "claiming statistical superiority when the interval includes zero",
    ],

    "methodological_boundary": {
        "models_trained":
            False,

        "predictions_generated":
            False,

        "performance_metrics_computed":
            False,

        "test_results_viewed":
            False,

        "splits_and_model_registry_locked":
            True,
    },
}


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

PROTOCOL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with SPLIT_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    fieldnames = [
        "repeat",
        "seed",
        "role",
        "index",
        "label",
        "role_membership_sha256",
        "role_size",
        "role_label_counts",
    ]

    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        split_rows
    )


with MODEL_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    fieldnames = [
        "model_id",
        "family",
        "implementation",
        "mandatory",
        "main_table",
        "hyperparameters",
    ]

    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for model in models:
        writer.writerow(
            {
                **{
                    key:
                        model[key]
                    for key in [
                        "model_id",
                        "family",
                        "implementation",
                        "mandatory",
                        "main_table",
                    ]
                },

                "hyperparameters":
                    json.dumps(
                        model[
                            "hyperparameters"
                        ],
                        sort_keys=True,
                    ),
            }
        )


DATASET_PATH.write_text(
    json.dumps(
        protocol[
            "dataset_contract"
        ],
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

PROTOCOL_PATH.write_text(
    json.dumps(
        protocol,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

summary = {
    "schema":
        "phyguard.temporal_baseline_protocol_summary.v1",

    "status":
        status,

    "sample_count":
        1080,

    "sequence_shape":
        [
            80,
            10,
        ],

    "class_count":
        6,

    "repeat_count":
        REPEATS,

    "model_count":
        len(
            models
        ),

    "model_ids":
        [
            model[
                "model_id"
            ]
            for model in models
        ],

    "main_table_models":
        protocol[
            "reporting_contract"
        ][
            "main_table_models"
        ],

    "all_models_mandatory":
        True,

    "unfavorable_results_must_be_reported":
        True,

    "protocol_sha256":
        sha256_file(
            PROTOCOL_PATH
        ),

    "split_membership_sha256":
        sha256_file(
            SPLIT_PATH
        ),

    "model_registry_sha256":
        sha256_file(
            MODEL_PATH
        ),

    "methodological_boundary":
        protocol[
            "methodological_boundary"
        ],
}

SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

ready_text = f"""# Stage 64B temporal-baseline protocol

Status: **LOCKED BEFORE EXECUTION**

- Dataset: 1080 multivariate sequences, each 80×10.
- Repeats: 5, using the existing `split_indices` implementation.
- Models: {len(models)}.
- Primary endpoint: selected accuracy.
- Thresholds: validation only.
- All preregistered models must be reported.
- Main table uses a fixed a-priori representative subset.
- The complete result table must be included in an appendix or supplement.
- No model may be hidden because it wins or loses.
"""

READY_PATH.write_text(
    ready_text,
    encoding="utf-8",
)


manifest_files = []

for path in sorted(
    OUTPUT_ROOT.rglob("*")
):
    if path.is_file():
        manifest_files.append(
            {
                "path":
                    str(
                        path.relative_to(
                            ROOT
                        )
                    ),

                "size_bytes":
                    path.stat().st_size,

                "sha256":
                    sha256_file(
                        path
                    ),
            }
        )

manifest = {
    "schema":
        "phyguard.temporal_baseline_protocol_manifest.v1",

    "status":
        status,

    "protocol_sha256":
        sha256_file(
            PROTOCOL_PATH
        ),

    "interface_audit_manifest_sha256":
        sha256_file(
            INTERFACE_AUDIT_MANIFEST
        ),

    "locked_assets":
        {
            str(
                path.relative_to(
                    ROOT
                )
            ):
                EXPECTED_SHA256[
                    str(path)
                ]
            for path in [
                LOCKED_RECONSTRUCTION_ARCHIVE,
                PER_ABLATION_ARCHIVE,
                COMPLEXITY_ARCHIVE,
            ]
        },

    "files":
        manifest_files,

    "methodological_boundary":
        protocol[
            "methodological_boundary"
        ],
}

MANIFEST_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

MANIFEST_PATH.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print("status:", status)
print("metadata_path:", metadata_path)
print("telemetry_shape:", list(X.shape))
print("kpi_names:", kpi_names)
print("label_counts:", protocol["dataset_contract"]["label_counts"])
print("repeat_seeds:", protocol["split_contract"]["repeat_seeds"])
print("threshold_grid_count:", len(threshold_grid))
print("model_count:", len(models))
print(
    "model_ids:",
    [
        model["model_id"]
        for model in models
    ],
)
print(
    "main_table_models:",
    protocol[
        "reporting_contract"
    ][
        "main_table_models"
    ],
)
print("all_models_mandatory: True")
print("unfavorable_results_must_be_reported: True")
print("models_trained: False")
print("test_results_viewed: False")
print("protocol:", PROTOCOL_PATH)
print("split_memberships:", SPLIT_PATH)
print("model_registry:", MODEL_PATH)
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)
print()
print(
    "TEMPORAL_BASELINE_PROTOCOL_V1_PASS"
)
