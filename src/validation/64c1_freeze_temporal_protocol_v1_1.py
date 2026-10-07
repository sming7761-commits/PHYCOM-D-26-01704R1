#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

PROTOCOL_V1 = (
    ROOT
    / "configs"
    / "temporal_baseline_protocol_v1.json"
)

API_AUDIT = (
    ROOT
    / "artifacts"
    / "temporal_baseline_api_metric_audit_v1"
    / "summary.json"
)

PROTOCOL_V11 = (
    ROOT
    / "configs"
    / "temporal_baseline_protocol_v1_1.json"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "temporal_baseline_protocol_v1_1"
)

SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
AMENDMENT_PATH = OUTPUT_ROOT / "AMENDMENT.md"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "temporal_baseline_protocol_v1_1.json"
)


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


for path in [
    PROTOCOL_V1,
    API_AUDIT,
]:
    require(
        path.exists(),
        f"Required input is missing: {path}",
    )

for path in [
    PROTOCOL_V11,
    OUTPUT_ROOT,
    MANIFEST_PATH,
]:
    require(
        not path.exists(),
        f"Protocol v1.1 output already exists: {path}",
    )


protocol = json.loads(
    PROTOCOL_V1.read_text(
        encoding="utf-8"
    )
)

audit = json.loads(
    API_AUDIT.read_text(
        encoding="utf-8"
    )
)

require(
    protocol.get("status")
    == "LOCKED_BEFORE_TEMPORAL_BASELINE_EXECUTION",
    "Protocol v1 status mismatch.",
)

require(
    audit.get("status") == "PASS",
    "API audit status is not PASS.",
)

require(
    audit.get(
        "methodological_boundary",
        {}
    ).get(
        "models_fitted"
    )
    is False,
    "API audit does not confirm zero fitted models.",
)

require(
    audit.get(
        "methodological_boundary",
        {}
    ).get(
        "test_results_viewed"
    )
    is False,
    "API audit does not confirm zero viewed test results.",
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

require(
    len(
        registry
    )
    == 14,
    "The 14-model registry is incomplete.",
)

require(
    "MULTIROCKET_HYDRA"
    in registry,
    "MULTIROCKET_HYDRA is missing.",
)

old_hyperparameters = dict(
    registry[
        "MULTIROCKET_HYDRA"
    ][
        "hyperparameters"
    ]
)

require(
    int(
        old_hyperparameters[
            "n_kernels"
        ]
    )
    == 5000,
    "Unexpected v1 MultiRocket-Hydra n_kernels.",
)

registry[
    "MULTIROCKET_HYDRA"
][
    "hyperparameters"
] = {
    "n_kernels":
        8,

    "n_groups":
        64,

    "random_state":
        "repeat_seed",
}

protocol[
    "schema"
] = "phyguard.temporal_baseline_protocol.v1.1"

protocol[
    "created_at_utc"
] = datetime.now(
    timezone.utc
).isoformat()

protocol[
    "status"
] = "LOCKED_BEFORE_TEMPORAL_BASELINE_EXECUTION"

protocol[
    "supersedes"
] = {
    "protocol":
        str(
            PROTOCOL_V1.relative_to(
                ROOT
            )
        ),

    "protocol_sha256":
        sha256_file(
            PROTOCOL_V1
        ),

    "reason":
        (
            "The installed Aeon 1.5.0 constructor audit showed "
            "that MultiRocketHydraClassifier.n_kernels is the "
            "Hydra kernel count and defaults to 8. The v1 value "
            "5000 was a parameter-semantics error. It is corrected "
            "to n_kernels=8 and n_groups=64 before any model fitting "
            "or test-result inspection."
        ),
}

protocol[
    "api_binding"
] = {
    "aeon_version":
        "1.5.0",

    "multirocket_hydra_signature":
        (
            "(n_kernels: int = 8, n_groups: int = 64, "
            "class_weight=None, n_jobs: int = 1, "
            "random_state=None)"
        ),

    "api_audit_path":
        str(
            API_AUDIT.relative_to(
                ROOT
            )
        ),

    "api_audit_sha256":
        sha256_file(
            API_AUDIT
        ),
}

protocol[
    "progress_contract"
] = {
    "total_model_repeat_tasks":
        70,

    "progress_file":
        (
            "artifacts/temporal_baseline_results_v1/"
            "progress.json"
        ),

    "progress_bar":
        True,

    "dynamic_eta":
        True,

    "update_frequency":
        (
            "At every completed model-repeat task and every "
            "deep-learning epoch."
        ),

    "resume":
        True,
}

protocol[
    "initial_runtime_estimate"
] = {
    "lower_hours":
        6,

    "upper_hours":
        12,

    "basis":
        (
            "Thirty Aeon CPU fits and forty GPU deep-model fits "
            "on 1080 sequences of shape 80x10. The live ETA "
            "supersedes this range after completed tasks."
        ),
}

protocol[
    "formal_hardware_contract"
] = {
    "manuscript_hardware":
        "NVIDIA GeForce RTX 3080 10 GB",

    "runtime_autodetection_used_for_manuscript":
        False,
}

protocol[
    "methodological_boundary"
] = {
    "models_trained":
        False,

    "predictions_generated":
        False,

    "performance_metrics_computed":
        False,

    "test_results_viewed":
        False,

    "api_parameter_error_corrected_before_execution":
        True,

    "model_count_unchanged":
        14,
}


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

PROTOCOL_V11.parent.mkdir(
    parents=True,
    exist_ok=True,
)

PROTOCOL_V11.write_text(
    json.dumps(
        protocol,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

summary = {
    "schema":
        "phyguard.temporal_baseline_protocol_summary.v1.1",

    "status":
        "LOCKED_BEFORE_TEMPORAL_BASELINE_EXECUTION",

    "model_count":
        len(
            registry
        ),

    "model_ids":
        list(
            registry.keys()
        ),

    "changed_model":
        "MULTIROCKET_HYDRA",

    "old_hyperparameters":
        old_hyperparameters,

    "new_hyperparameters":
        registry[
            "MULTIROCKET_HYDRA"
        ][
            "hyperparameters"
        ],

    "models_trained":
        False,

    "test_results_viewed":
        False,

    "protocol_v1_sha256":
        sha256_file(
            PROTOCOL_V1
        ),

    "protocol_v1_1_sha256":
        sha256_file(
            PROTOCOL_V11
        ),

    "api_audit_sha256":
        sha256_file(
            API_AUDIT
        ),

    "initial_runtime_estimate_hours":
        [
            6,
            12,
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

AMENDMENT_PATH.write_text(
    "\n".join(
        [
            "# Temporal baseline protocol v1.1 amendment",
            "",
            "The model set remains unchanged at fourteen.",
            "",
            "The only model-level correction is:",
            "",
            "- MultiRocket-Hydra: `n_kernels=5000` -> `n_kernels=8`",
            "- MultiRocket-Hydra: add `n_groups=64`",
            "",
            "The change follows the installed Aeon 1.5.0 constructor",
            "signature and occurred before any model fitting, prediction",
            "generation, metric computation, or test-result inspection.",
            "",
        ]
    ),
    encoding="utf-8",
)

manifest = {
    "schema":
        "phyguard.temporal_baseline_protocol_manifest.v1.1",

    "status":
        "LOCKED_BEFORE_TEMPORAL_BASELINE_EXECUTION",

    "protocol_v1_sha256":
        sha256_file(
            PROTOCOL_V1
        ),

    "protocol_v1_1_sha256":
        sha256_file(
            PROTOCOL_V11
        ),

    "summary_sha256":
        sha256_file(
            SUMMARY_PATH
        ),

    "amendment_sha256":
        sha256_file(
            AMENDMENT_PATH
        ),

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

print(
    "status: LOCKED_BEFORE_TEMPORAL_BASELINE_EXECUTION"
)
print(
    "model_count: 14"
)
print(
    "changed_model: MULTIROCKET_HYDRA"
)
print(
    "old_hyperparameters:",
    old_hyperparameters,
)
print(
    "new_hyperparameters:",
    registry[
        "MULTIROCKET_HYDRA"
    ][
        "hyperparameters"
    ],
)
print(
    "models_trained: False"
)
print(
    "test_results_viewed: False"
)
print(
    "initial_runtime_estimate_hours: [6, 12]"
)
print(
    "protocol_v1_1:",
    PROTOCOL_V11,
)
print(
    "manifest:",
    MANIFEST_PATH,
)
print()
print(
    "TEMPORAL_BASELINE_PROTOCOL_V1_1_PASS"
)
