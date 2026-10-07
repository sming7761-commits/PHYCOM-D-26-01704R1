#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from aeon.classification.convolution_based import (
    Arsenal,
    HydraClassifier,
    MiniRocketClassifier,
    MultiRocketClassifier,
    MultiRocketHydraClassifier,
    RocketClassifier,
)


ROOT = Path("/root/phyguard_revision")

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "temporal_baseline_protocol_v1.json"
)

SPLIT_PATH = (
    ROOT
    / "artifacts"
    / "temporal_baseline_protocol_v1"
    / "split_memberships.csv"
)

ENVIRONMENT_SUMMARY = (
    ROOT
    / "artifacts"
    / "temporal_baseline_environment_v1"
    / "summary.json"
)

FORMAL_HARDWARE = (
    ROOT
    / "configs"
    / "formal_hardware_configuration_v1.json"
)

R0 = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
)

EVALUATION_COMMON_PATH = (
    R0
    / "scripts"
    / "evaluation_common.py"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "temporal_baseline_api_metric_audit_v1"
)

SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
AEON_PATH = OUTPUT_ROOT / "aeon_constructor_audit.csv"
SPLIT_SUMMARY_PATH = OUTPUT_ROOT / "split_summary.csv"
METRIC_AUDIT_PATH = OUTPUT_ROOT / "metric_helper_audit.json"
CONTEXT_PATH = OUTPUT_ROOT / "API_AUDIT_NOTES.md"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "temporal_baseline_api_metric_audit_v1.json"
)

AEON_CLASSES = {
    "ROCKET":
        RocketClassifier,

    "MINIROCKET":
        MiniRocketClassifier,

    "MULTIROCKET":
        MultiRocketClassifier,

    "HYDRA":
        HydraClassifier,

    "MULTIROCKET_HYDRA":
        MultiRocketHydraClassifier,

    "ARSENAL":
        Arsenal,
}

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


def normalize_signature_default(value):
    if value is inspect._empty:
        return "__REQUIRED__"

    try:
        json.dumps(
            value
        )
        return value
    except TypeError:
        return repr(
            value
        )


def constructor_parameters(model_class):
    signature = inspect.signature(
        model_class
    )

    parameters = {}

    accepts_var_kwargs = False

    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var_kwargs = True

        parameters[name] = {
            "kind":
                str(
                    parameter.kind
                ),

            "default":
                normalize_signature_default(
                    parameter.default
                ),
        }

    return (
        signature,
        parameters,
        accepts_var_kwargs,
    )


def map_protocol_kwargs(
    model_id: str,
    protocol_hyperparameters: dict,
    signature_parameters: dict,
    accepts_var_kwargs: bool,
):
    aliases = {
        "n_kernels":
            [
                "n_kernels",
                "num_kernels",
            ],

        "num_kernels":
            [
                "num_kernels",
                "n_kernels",
            ],

        "n_estimators":
            [
                "n_estimators",
            ],

        "n_groups":
            [
                "n_groups",
            ],

        "random_state":
            [
                "random_state",
            ],

        "n_jobs":
            [
                "n_jobs",
            ],
    }

    desired = dict(
        protocol_hyperparameters
    )

    desired[
        "random_state"
    ] = 20260705

    desired[
        "n_jobs"
    ] = 1

    accepted = {}
    rejected = {}
    mapping = {}

    for source_key, value in desired.items():
        candidate_names = aliases.get(
            source_key,
            [
                source_key
            ],
        )

        target_key = next(
            (
                candidate
                for candidate in candidate_names
                if candidate
                in signature_parameters
            ),
            None,
        )

        if target_key is None and accepts_var_kwargs:
            target_key = source_key

        if target_key is None:
            rejected[
                source_key
            ] = value
            continue

        accepted[
            target_key
        ] = value

        mapping[
            source_key
        ] = target_key

    semantic_flags = []

    if (
        model_id
        == "MULTIROCKET_HYDRA"
        and "n_kernels"
        in accepted
        and float(
            accepted[
                "n_kernels"
            ]
        )
        > 256
    ):
        semantic_flags.append(
            (
                "The protocol value n_kernels is very large "
                "for the constructor parameter with the same "
                "name. Confirm whether this parameter controls "
                "Hydra kernels rather than MultiRocket features "
                "before training."
            )
        )

    return (
        accepted,
        rejected,
        mapping,
        semantic_flags,
    )


for path in [
    PROTOCOL_PATH,
    SPLIT_PATH,
    ENVIRONMENT_SUMMARY,
    FORMAL_HARDWARE,
    EVALUATION_COMMON_PATH,
]:
    require(
        path.exists(),
        f"Required input is missing: {path}",
    )

for path in [
    OUTPUT_ROOT,
    MANIFEST_PATH,
]:
    require(
        not path.exists(),
        f"Audit output already exists: {path}",
    )


protocol = json.loads(
    PROTOCOL_PATH.read_text(
        encoding="utf-8"
    )
)

environment = json.loads(
    ENVIRONMENT_SUMMARY.read_text(
        encoding="utf-8"
    )
)

hardware = json.loads(
    FORMAL_HARDWARE.read_text(
        encoding="utf-8"
    )
)

require(
    protocol.get("status")
    == "LOCKED_BEFORE_TEMPORAL_BASELINE_EXECUTION",
    "Protocol status mismatch.",
)

require(
    len(
        protocol[
            "model_registry"
        ]
    )
    == 14,
    "The locked model count is not 14.",
)

require(
    environment.get("status")
    == "PASS",
    "Temporal environment is not PASS.",
)

require(
    environment.get("formal_hardware")
    == "NVIDIA GeForce RTX 3080 10 GB",
    "Formal environment hardware mismatch.",
)

require(
    hardware.get(
        "user_confirmed_configuration"
    )
    == "NVIDIA GeForce RTX 3080 10 GB",
    "Formal hardware contract mismatch.",
)


registry = {
    model[
        "model_id"
    ]:
        model
    for model in protocol[
        "model_registry"
    ]
}

constructor_rows = []
semantic_flags_all = []

for model_id, model_class in AEON_CLASSES.items():
    require(
        model_id in registry,
        f"Protocol registry lacks {model_id}.",
    )

    signature, parameters, accepts_var_kwargs = constructor_parameters(
        model_class
    )

    protocol_hyperparameters = registry[
        model_id
    ][
        "hyperparameters"
    ]

    (
        accepted,
        rejected,
        mapping,
        semantic_flags,
    ) = map_protocol_kwargs(
        model_id,
        protocol_hyperparameters,
        parameters,
        accepts_var_kwargs,
    )

    instantiated = False
    instantiation_error = None
    resolved_parameters = None

    try:
        model = model_class(
            **accepted
        )

        instantiated = True

        if hasattr(
            model,
            "get_params",
        ):
            resolved_parameters = model.get_params(
                deep=False
            )
    except Exception as error:
        instantiation_error = repr(
            error
        )

    require(
        instantiated,
        (
            f"Constructor instantiation failed for "
            f"{model_id}: {instantiation_error}"
        ),
    )

    semantic_flags_all.extend(
        [
            {
                "model_id":
                    model_id,

                "flag":
                    flag,
            }
            for flag in semantic_flags
        ]
    )

    constructor_rows.append(
        {
            "model_id":
                model_id,

            "resolved_class":
                (
                    f"{model_class.__module__}."
                    f"{model_class.__name__}"
                ),

            "signature":
                str(
                    signature
                ),

            "signature_parameters":
                json.dumps(
                    parameters,
                    sort_keys=True,
                    default=str,
                ),

            "protocol_hyperparameters":
                json.dumps(
                    protocol_hyperparameters,
                    sort_keys=True,
                ),

            "accepted_kwargs":
                json.dumps(
                    accepted,
                    sort_keys=True,
                ),

            "rejected_kwargs":
                json.dumps(
                    rejected,
                    sort_keys=True,
                ),

            "source_to_constructor_mapping":
                json.dumps(
                    mapping,
                    sort_keys=True,
                ),

            "resolved_get_params":
                json.dumps(
                    resolved_parameters,
                    sort_keys=True,
                    default=str,
                ),

            "semantic_flags":
                json.dumps(
                    semantic_flags,
                    ensure_ascii=False,
                ),

            "instantiation_pass":
                instantiated,

            "instantiation_error":
                instantiation_error,
        }
    )


split_frame = pd.read_csv(
    SPLIT_PATH
)

required_split_columns = {
    "repeat",
    "seed",
    "role",
    "index",
    "label",
}

require(
    required_split_columns.issubset(
        set(
            split_frame.columns
        )
    ),
    "Split-membership columns are incomplete.",
)

split_rows = []

for (
    repeat,
    seed,
    role,
), group in split_frame.groupby(
    [
        "repeat",
        "seed",
        "role",
    ],
    sort=True,
):
    counts = (
        group[
            "label"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    split_rows.append(
        {
            "repeat":
                int(
                    repeat
                ),

            "seed":
                int(
                    seed
                ),

            "role":
                str(
                    role
                ),

            "size":
                int(
                    len(
                        group
                    )
                ),

            "unique_indices":
                int(
                    group[
                        "index"
                    ].nunique()
                ),

            "label_counts":
                json.dumps(
                    {
                        str(
                            key
                        ):
                            int(
                                value
                            )
                        for key, value in counts.items()
                    },
                    sort_keys=True,
                ),
        }
    )

for repeat, group in split_frame.groupby(
    "repeat"
):
    require(
        len(
            group
        )
        == 1080,
        f"Repeat {repeat} does not contain 1080 assignments.",
    )

    require(
        group[
            "index"
        ].nunique()
        == 1080,
        f"Repeat {repeat} contains duplicate assignments.",
    )


evaluation_common = load_module(
    "phyguard_evaluation_common_stage64c0",
    EVALUATION_COMMON_PATH,
)

require(
    hasattr(
        evaluation_common,
        "selective_metrics",
    ),
    "evaluation_common lacks selective_metrics.",
)

require(
    hasattr(
        evaluation_common,
        "single_threshold_rows",
    ),
    "evaluation_common lacks single_threshold_rows.",
)

selective_signature = inspect.signature(
    evaluation_common.selective_metrics
)

threshold_signature = inspect.signature(
    evaluation_common.single_threshold_rows
)

synthetic_truth = np.asarray(
    [
        "interference",
        "blockage",
        "mobility",
        "adaptation_mismatch",
        "normal",
        "nonphysical_goodput",
        "interference",
        "blockage",
        "mobility",
        "adaptation_mismatch",
        "normal",
        "nonphysical_goodput",
    ],
    dtype=object,
)

synthetic_labels = np.asarray(
    [
        "interference",
        "blockage",
        "normal",
        "adaptation_mismatch",
        "interference",
        "normal",
        "blockage",
        "blockage",
        "mobility",
        "nonphysical_goodput",
        "normal",
        "mobility",
    ],
    dtype=object,
)

synthetic_confidence = np.asarray(
    [
        0.95,
        0.88,
        0.40,
        0.91,
        0.72,
        0.31,
        0.63,
        0.84,
        0.79,
        0.36,
        0.28,
        0.68,
    ],
    dtype=float,
)

synthetic_thresholds = np.asarray(
    [
        0.0,
        0.5,
        0.8,
    ],
    dtype=float,
)

metric_result = evaluation_common.selective_metrics(
    synthetic_truth,
    synthetic_labels,
)

threshold_result = evaluation_common.single_threshold_rows(
    synthetic_truth,
    synthetic_labels,
    synthetic_confidence,
    synthetic_thresholds,
    0.10,
    0.80,
)

if isinstance(
    threshold_result,
    pd.DataFrame,
):
    threshold_records = threshold_result.to_dict(
        orient="records"
    )

    threshold_columns = list(
        threshold_result.columns
    )
elif isinstance(
    threshold_result,
    list,
):
    threshold_records = threshold_result

    threshold_columns = sorted(
        {
            key
            for row in threshold_result
            if isinstance(
                row,
                dict,
            )
            for key in row
        }
    )
else:
    threshold_records = [
        {
            "repr":
                repr(
                    threshold_result
                )
        }
    ]

    threshold_columns = []


metric_audit = {
    "selective_metrics_signature":
        str(
            selective_signature
        ),

    "single_threshold_rows_signature":
        str(
            threshold_signature
        ),

    "synthetic_selective_metrics_type":
        type(
            metric_result
        ).__name__,

    "synthetic_selective_metrics":
        metric_result,

    "synthetic_threshold_result_type":
        type(
            threshold_result
        ).__name__,

    "synthetic_threshold_columns":
        threshold_columns,

    "synthetic_threshold_records":
        threshold_records,
}


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

with AEON_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            constructor_rows[
                0
            ].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        constructor_rows
    )


with SPLIT_SUMMARY_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            split_rows[
                0
            ].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        split_rows
    )


METRIC_AUDIT_PATH.write_text(
    json.dumps(
        metric_audit,
        ensure_ascii=False,
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)


notes = [
    "# Stage 64C0 API and metric audit",
    "",
    "No model was fitted and no prediction was generated.",
    "",
    "The audit instantiated the six Aeon constructors, resolved protocol",
    "hyperparameters against the installed Aeon 1.5.0 signatures, verified",
    "all five split-membership tables, and executed the existing metric",
    "helpers only on a synthetic twelve-row example.",
    "",
    "Any semantic flag must be resolved before full training.",
    "",
]

CONTEXT_PATH.write_text(
    "\n".join(
        notes
    ),
    encoding="utf-8",
)


summary = {
    "schema":
        "phyguard.temporal_baseline_api_metric_audit.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "PASS",

    "model_count_locked":
        14,

    "aeon_constructor_count":
        len(
            constructor_rows
        ),

    "aeon_instantiation_pass_count":
        int(
            sum(
                bool(
                    row[
                        "instantiation_pass"
                    ]
                )
                for row in constructor_rows
            )
        ),

    "semantic_flags":
        semantic_flags_all,

    "split_repeat_count":
        int(
            split_frame[
                "repeat"
            ].nunique()
        ),

    "split_rows":
        int(
            len(
                split_frame
            )
        ),

    "metric_helper_audit":
        metric_audit,

    "formal_hardware":
        "NVIDIA GeForce RTX 3080 10 GB",

    "methodological_boundary": {
        "models_fitted":
            False,

        "predictions_generated":
            False,

        "performance_metrics_computed_on_real_data":
            False,

        "test_results_viewed":
            False,

        "synthetic_metric_smoke_only":
            True,
    },
}

SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)


files = []

for path in sorted(
    OUTPUT_ROOT.rglob("*")
):
    if path.is_file():
        files.append(
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
        "phyguard.temporal_baseline_api_metric_audit_manifest.v1",

    "status":
        "PASS",

    "protocol_sha256":
        sha256_file(
            PROTOCOL_PATH
        ),

    "environment_summary_sha256":
        sha256_file(
            ENVIRONMENT_SUMMARY
        ),

    "formal_hardware_sha256":
        sha256_file(
            FORMAL_HARDWARE
        ),

    "files":
        files,

    "methodological_boundary":
        summary[
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


print("status: PASS")
print("formal_hardware: NVIDIA GeForce RTX 3080 10 GB")
print("model_count_locked: 14")
print("models_fitted: False")
print("test_results_viewed: False")
print()
print("AEON CONSTRUCTORS")
for row in constructor_rows:
    print(
        row[
            "model_id"
        ],
        "| signature=",
        row[
            "signature"
        ],
    )
    print(
        " accepted_kwargs=",
        row[
            "accepted_kwargs"
        ],
    )
    print(
        " rejected_kwargs=",
        row[
            "rejected_kwargs"
        ],
    )
    print(
        " semantic_flags=",
        row[
            "semantic_flags"
        ],
    )

print()
print("SPLIT SUMMARY")
for row in split_rows:
    print(
        "repeat=",
        row[
            "repeat"
        ],
        "role=",
        row[
            "role"
        ],
        "size=",
        row[
            "size"
        ],
        "label_counts=",
        row[
            "label_counts"
        ],
    )

print()
print("METRIC HELPERS")
print(
    "selective_metrics_signature:",
    metric_audit[
        "selective_metrics_signature"
    ],
)
print(
    "single_threshold_rows_signature:",
    metric_audit[
        "single_threshold_rows_signature"
    ],
)
print(
    "synthetic_selective_metrics:",
    metric_audit[
        "synthetic_selective_metrics"
    ],
)
print(
    "synthetic_threshold_columns:",
    metric_audit[
        "synthetic_threshold_columns"
    ],
)
print(
    "synthetic_threshold_records:",
    metric_audit[
        "synthetic_threshold_records"
    ],
)

print()
print("summary:", SUMMARY_PATH)
print("constructor_audit:", AEON_PATH)
print("split_summary:", SPLIT_SUMMARY_PATH)
print("metric_audit:", METRIC_AUDIT_PATH)
print("manifest:", MANIFEST_PATH)
print()
print(
    "TEMPORAL_BASELINE_API_METRIC_AUDIT_V1_PASS"
)
