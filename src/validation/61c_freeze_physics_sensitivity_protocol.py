#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/root/phyguard_revision")

R0 = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
)

FEATURES_PATH = R0 / "scripts" / "features.py"
EVALUATION_COMMON_PATH = R0 / "scripts" / "evaluation_common.py"

STAGE61B_SUMMARY = (
    ROOT
    / "artifacts"
    / "physics_score_weight_penalty_exact_interface_v1"
    / "summary.json"
)

TEMPORAL_FREEZE = (
    ROOT
    / "results"
    / "phyguard_temporal_baselines_final.zip"
)

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "physics_sensitivity_protocol_v1.json"
)

ARTIFACT_ROOT = (
    ROOT
    / "artifacts"
    / "physics_sensitivity_protocol_v1"
)

SUMMARY_PATH = ARTIFACT_ROOT / "summary.json"
EQUIVALENCE_PATH = ARTIFACT_ROOT / "baseline_equivalence.json"
SOURCE_ASSERTIONS_PATH = ARTIFACT_ROOT / "source_assertions.json"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "physics_sensitivity_protocol_v1.json"
)


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
        f"Could not import module: {path}",
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameterized_build_one(
    module,
    x: np.ndarray,
    start: int,
    end: int,
    *,
    eps: float = 1e-6,
    phi_scale: float = 1.5,
    adapt_gap_weight: float = 1.0,
    mismatch_mcs_drop_penalty: float = 0.35,
):
    ref = x[:24]
    cand = x[start:end]

    med = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - med), axis=0)
    scale = 1.4826 * mad + eps

    z = np.clip((x - med) / scale, -10, 10)
    zc = z[start:end]

    raw = []

    for j in range(x.shape[1]):
        v = zc[:, j]

        raw.extend(
            [
                v.mean(),
                v.std(),
                np.median(v),
                np.quantile(v, 0.1),
                np.quantile(v, 0.9),
                module.slope(v),
            ]
        )

    raw = np.asarray(raw, float)

    dz = zc.mean(axis=0)

    rsrp_drop = -dz[0]
    rssi_rise = dz[1]
    sinr_drop = -dz[2]
    evm_rise = dz[3]
    ber_rise = dz[4]
    per_rise = dz[5]
    cqi_drop = -dz[6]
    mcs_drop = -dz[7]
    goodput_drop = -dz[8]
    rho_drop = -dz[9]

    gap_rise = dz[1] - dz[0]

    used_mcs = float(np.mean(cand[:, 7]))

    supported = float(
        np.mean(
            module.supported_mcs_from_sinr(
                cand[:, 2]
            )
        )
    )

    adapt_gap = max(
        0.0,
        used_mcs - supported,
    )

    # Baseline equivalence at adapt_gap_weight=1.0:
    # original adapt_gap_z = adapt_gap / 0.75
    adapt_gap_z = (
        adapt_gap_weight
        * adapt_gap
        / 0.75
    )

    pos = lambda q: max(float(q), 0.0)

    phi = lambda q: float(
        np.exp(
            -abs(float(q))
            / phi_scale
        )
    )

    p_int = (
        pos(gap_rise)
        + pos(sinr_drop)
        + phi(rsrp_drop)
        + pos(evm_rise)
        + pos(ber_rise)
    ) / 5.0

    p_blk = (
        pos(rsrp_drop)
        + pos(sinr_drop)
        + pos(evm_rise)
        + pos(cqi_drop)
        + phi(gap_rise)
    ) / 5.0

    p_mob = (
        pos(rho_drop)
        + pos(evm_rise)
        + pos(ber_rise)
        + phi(rsrp_drop)
        + phi(sinr_drop)
    ) / 5.0

    p_mis = (
        pos(adapt_gap_z)
        + pos(sinr_drop)
        + pos(evm_rise)
        + pos(ber_rise)
        + pos(cqi_drop)
        - mismatch_mcs_drop_penalty
        * pos(mcs_drop)
    ) / 5.0

    p_non = (
        pos(goodput_drop)
        + phi(rsrp_drop)
        + phi(sinr_drop)
        + phi(evm_rise)
        + phi(ber_rise)
    ) / 5.0

    scores = np.array(
        [
            p_int,
            p_blk,
            p_mob,
            p_mis,
            p_non,
        ],
        float,
    )

    sorted_scores = np.sort(
        scores[:4]
    )

    underlying = np.array(
        [
            rsrp_drop,
            rssi_rise,
            sinr_drop,
            evm_rise,
            ber_rise,
            per_rise,
            cqi_drop,
            mcs_drop,
            goodput_drop,
            rho_drop,
            gap_rise,
            adapt_gap_z,
        ],
        float,
    )

    phys = np.concatenate(
        [
            underlying,
            scores,
            [
                scores[:4].max(),
                sorted_scores[-1]
                - sorted_scores[-2],
            ],
        ]
    )

    require(
        raw.shape == (60,),
        f"Unexpected raw shape: {raw.shape}",
    )

    require(
        phys.shape == (19,),
        f"Unexpected physical shape: {phys.shape}",
    )

    return raw, phys


for path in [
    FEATURES_PATH,
    EVALUATION_COMMON_PATH,
    STAGE61B_SUMMARY,
    TEMPORAL_FREEZE,
]:
    require(
        path.exists(),
        f"Required input is missing: {path}",
    )

for path in [
    PROTOCOL_PATH,
    ARTIFACT_ROOT,
    MANIFEST_PATH,
]:
    require(
        not path.exists(),
        f"Protocol output already exists: {path}",
    )


stage61b = json.loads(
    STAGE61B_SUMMARY.read_text(
        encoding="utf-8"
    )
)

require(
    stage61b.get("status") == "PASS",
    "Stage 61B is not PASS.",
)


features = load_module(
    "phyguard_features_stage61c",
    FEATURES_PATH,
)

evaluation_common = load_module(
    "phyguard_evaluation_common_stage61c",
    EVALUATION_COMMON_PATH,
)

require(
    hasattr(features, "build_one"),
    "features.py lacks build_one.",
)

require(
    hasattr(
        evaluation_common,
        "single_threshold_rows",
    ),
    (
        "evaluation_common.py lacks "
        "single_threshold_rows."
    ),
)


build_source = inspect.getsource(
    features.build_one
)

threshold_source = inspect.getsource(
    evaluation_common.single_threshold_rows
)

threshold_signature = inspect.signature(
    evaluation_common.single_threshold_rows
)


source_assertions = {
    "phi_baseline_scale_1_5": (
        "phi=lambdaq,s=1.5:"
        in "".join(build_source.split())
    ),
    "adapt_gap_divisor_0_75": (
        "adapt_gap/0.75"
        in build_source.replace(" ", "")
    ),
    "mismatch_penalty_0_35": (
        "-0.35*pos(mcs_drop)"
        in build_source.replace(" ", "")
    ),
    "score_denominator_5": (
        "/5.0"
        in build_source.replace(" ", "")
    ),
    "threshold_default_fsr_0_05": (
        float(
            threshold_signature.parameters[
                "fsr_max"
            ].default
        )
        == 0.05
    ),
    "threshold_default_selected_accuracy_0_90": (
        float(
            threshold_signature.parameters[
                "selected_accuracy_min"
            ].default
        )
        == 0.90
    ),
    "normalized_constraint_penalty_present": (
        "penalty"
        in threshold_source
        and "fsr_max"
        in threshold_source
        and "selected_accuracy_min"
        in threshold_source
    ),
}

require(
    all(
        source_assertions.values()
    ),
    (
        "One or more baseline source assertions failed: "
        f"{source_assertions}"
    ),
)


rng = np.random.default_rng(
    61031
)

equivalence_records = []

for case_id in range(5):
    x = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(80, 10),
    ).astype(
        np.float64
    )

    x[:24] += np.linspace(
        -0.25,
        0.25,
        24,
    )[:, None]

    x[24:] += (
        case_id
        + 1
    ) * 0.03

    original_raw, original_phys = (
        features.build_one(
            x,
            24,
            80,
        )
    )

    rebuilt_raw, rebuilt_phys = (
        parameterized_build_one(
            features,
            x,
            24,
            80,
            phi_scale=1.5,
            adapt_gap_weight=1.0,
            mismatch_mcs_drop_penalty=0.35,
        )
    )

    raw_error = float(
        np.max(
            np.abs(
                original_raw
                - rebuilt_raw
            )
        )
    )

    physical_error = float(
        np.max(
            np.abs(
                original_phys
                - rebuilt_phys
            )
        )
    )

    equivalence_records.append(
        {
            "case_id": case_id,
            "raw_max_absolute_error": raw_error,
            "physical_max_absolute_error": physical_error,
        }
    )


maximum_raw_error = max(
    record[
        "raw_max_absolute_error"
    ]
    for record in equivalence_records
)

maximum_physical_error = max(
    record[
        "physical_max_absolute_error"
    ]
    for record in equivalence_records
)

require(
    maximum_raw_error <= 1e-12,
    (
        "Parameterized raw feature builder does not "
        f"reproduce baseline: {maximum_raw_error}"
    ),
)

require(
    maximum_physical_error <= 1e-12,
    (
        "Parameterized physical feature builder does not "
        f"reproduce baseline: {maximum_physical_error}"
    ),
)


formula_variants = []

baseline_formula = {
    "variant_id": "DEFAULT",
    "factor": "default",
    "phi_scale": 1.5,
    "adapt_gap_weight": 1.0,
    "mismatch_mcs_drop_penalty": 0.35,
}

formula_variants.append(
    baseline_formula
)

for value in [
    1.0,
    1.25,
    2.0,
    2.5,
]:
    formula_variants.append(
        {
            "variant_id": (
                "PHI_SCALE_"
                + str(value).replace(".", "p")
            ),
            "factor": "phi_scale",
            "phi_scale": value,
            "adapt_gap_weight": 1.0,
            "mismatch_mcs_drop_penalty": 0.35,
        }
    )

for value in [
    0.5,
    0.75,
    1.25,
    1.5,
]:
    formula_variants.append(
        {
            "variant_id": (
                "ADAPT_GAP_WEIGHT_"
                + str(value).replace(".", "p")
            ),
            "factor": "adapt_gap_weight",
            "phi_scale": 1.5,
            "adapt_gap_weight": value,
            "mismatch_mcs_drop_penalty": 0.35,
        }
    )

for value in [
    0.0,
    0.175,
    0.525,
    0.70,
]:
    formula_variants.append(
        {
            "variant_id": (
                "MCS_DROP_PENALTY_"
                + str(value).replace(".", "p")
            ),
            "factor": "mismatch_mcs_drop_penalty",
            "phi_scale": 1.5,
            "adapt_gap_weight": 1.0,
            "mismatch_mcs_drop_penalty": value,
        }
    )


threshold_policies = []

for fsr_max in [
    0.025,
    0.05,
    0.10,
]:
    for selected_accuracy_min in [
        0.85,
        0.90,
        0.95,
    ]:
        threshold_policies.append(
            {
                "policy_id": (
                    "FSR_"
                    + str(fsr_max).replace(".", "p")
                    + "__SA_"
                    + str(
                        selected_accuracy_min
                    ).replace(".", "p")
                ),
                "fsr_max": fsr_max,
                "selected_accuracy_min": (
                    selected_accuracy_min
                ),
                "is_default": (
                    fsr_max == 0.05
                    and selected_accuracy_min
                    == 0.90
                ),
            }
        )


protocol = {
    "schema": (
        "phyguard.physics_sensitivity_protocol.v1"
    ),
    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "status": (
        "LOCKED_BEFORE_PHYSICS_SENSITIVITY_EXECUTION"
    ),
    "checkpoint": (
        "PhyGuard_R0_Reconstruction_Checkpoint2"
    ),
    "scientific_question": (
        "How sensitive is the revision-time locked "
        "PhyGuard reconstruction to the actual "
        "physics-score shape, implicit evidence weight, "
        "explicit mismatch penalty, and selective "
        "operating-policy constraints present in source?"
    ),
    "actual_source_interfaces": {
        "physics_score_shape": {
            "parameter": "phi_scale",
            "source_expression": (
                "exp(-abs(q)/s)"
            ),
            "baseline": 1.5,
        },
        "implicit_evidence_weight": {
            "parameter": "adapt_gap_weight",
            "source_expression": (
                "adapt_gap_z = "
                "adapt_gap_weight * adapt_gap / 0.75"
            ),
            "baseline": 1.0,
            "interpretation": (
                "A transparent multiplicative "
                "reparameterization of the existing "
                "adapt_gap/0.75 scaling."
            ),
        },
        "explicit_penalty": {
            "parameter": (
                "mismatch_mcs_drop_penalty"
            ),
            "source_expression": (
                "-gamma * pos(mcs_drop)"
            ),
            "baseline": 0.35,
        },
        "selection_policy": {
            "fsr_max_baseline": 0.05,
            "selected_accuracy_min_baseline": 0.90,
            "fallback_penalty": (
                "normalized violation of both "
                "constraints"
            ),
        },
    },
    "excluded_noninterfaces": {
        "simulator_beta_0_18": (
            "Excluded because it is a data-simulator "
            "parameter rather than a PhyGuard score, "
            "weight, penalty, or inference parameter."
        ),
        "global_physical_block_scale": (
            "Not selected as a primary factor because "
            "the gate includes StandardScaler and the "
            "ExtraTrees resolver is invariant to "
            "monotone feature rescaling; a global block "
            "multiplier would not constitute an "
            "informative sensitivity factor."
        ),
    },
    "design": {
        "formula_design": (
            "one_factor_at_a_time"
        ),
        "formula_variants": formula_variants,
        "formula_variant_count": len(
            formula_variants
        ),
        "repeat_seeds": [
            20260705,
            20260706,
            20260707,
            20260708,
            20260709,
        ],
        "training_task_count": (
            len(
                formula_variants
            )
            * 5
        ),
        "threshold_policy_grid": (
            threshold_policies
        ),
        "threshold_policy_count": len(
            threshold_policies
        ),
        "threshold_policy_scope": (
            "Applied to the DEFAULT formula variant "
            "without refitting the models; validation "
            "predictions select tau and locked test "
            "predictions report the operating point."
        ),
    },
    "primary_metrics": [
        "selected_accuracy",
        "coverage",
        "false_specific_rate",
        "physical_exact_match",
        "control_abstention",
    ],
    "secondary_metrics": [
        "macro_f1",
        "balanced_accuracy",
    ],
    "reporting": {
        "all_formula_variants_reported": True,
        "all_threshold_policies_reported": True,
        "outcome_based_omission": False,
        "primary_summary": (
            "mean and population standard deviation "
            "over five locked repeats"
        ),
        "paired_uncertainty": (
            "paired original-sample cluster bootstrap "
            "against DEFAULT for each non-default "
            "formula variant"
        ),
        "claim_boundary": (
            "The experiment evaluates robustness of "
            "the revision-time locked reconstruction. "
            "It does not establish that the unavailable "
            "original submission code used tunable "
            "weights."
        ),
    },
    "methodological_boundary": {
        "models_fitted": False,
        "predictions_generated": False,
        "performance_metrics_computed": False,
        "parameters_changed_in_source": False,
        "test_results_viewed": False,
    },
    "formal_hardware": (
        "NVIDIA GeForce RTX 3080 10 GB"
    ),
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

ARTIFACT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

equivalence_payload = {
    "status": "PASS",
    "case_count": len(
        equivalence_records
    ),
    "records": equivalence_records,
    "maximum_raw_absolute_error": (
        maximum_raw_error
    ),
    "maximum_physical_absolute_error": (
        maximum_physical_error
    ),
    "models_fitted": False,
    "predictions_generated": False,
}

EQUIVALENCE_PATH.write_text(
    json.dumps(
        equivalence_payload,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

SOURCE_ASSERTIONS_PATH.write_text(
    json.dumps(
        {
            "status": "PASS",
            "assertions": source_assertions,
            "features_sha256": sha256_file(
                FEATURES_PATH
            ),
            "evaluation_common_sha256": (
                sha256_file(
                    EVALUATION_COMMON_PATH
                )
            ),
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

summary = {
    "schema": (
        "phyguard.physics_sensitivity_protocol_summary.v1"
    ),
    "status": "PASS",
    "protocol_status": protocol["status"],
    "formula_variant_count": len(
        formula_variants
    ),
    "training_task_count": (
        len(
            formula_variants
        )
        * 5
    ),
    "threshold_policy_count": len(
        threshold_policies
    ),
    "baseline_equivalence_max_error": max(
        maximum_raw_error,
        maximum_physical_error,
    ),
    "actual_factors": [
        "phi_scale",
        "adapt_gap_weight",
        "mismatch_mcs_drop_penalty",
        "fsr_max",
        "selected_accuracy_min",
    ],
    "models_fitted": False,
    "predictions_generated": False,
    "performance_metrics_computed": False,
    "parameters_changed_in_source": False,
    "test_results_viewed": False,
    "formal_hardware": (
        "NVIDIA GeForce RTX 3080 10 GB"
    ),
    "protocol_path": str(
        PROTOCOL_PATH
    ),
}

SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


files = []

for path in sorted(
    ARTIFACT_ROOT.rglob("*")
):
    if path.is_file():
        files.append(
            {
                "path": str(
                    path.relative_to(
                        ROOT
                    )
                ),
                "size_bytes": (
                    path.stat().st_size
                ),
                "sha256": sha256_file(
                    path
                ),
            }
        )

files.append(
    {
        "path": str(
            PROTOCOL_PATH.relative_to(
                ROOT
            )
        ),
        "size_bytes": (
            PROTOCOL_PATH.stat().st_size
        ),
        "sha256": sha256_file(
            PROTOCOL_PATH
        ),
    }
)

manifest = {
    "schema": (
        "phyguard.physics_sensitivity_protocol_manifest.v1"
    ),
    "status": "PASS",
    "protocol_status": protocol["status"],
    "features_sha256": sha256_file(
        FEATURES_PATH
    ),
    "evaluation_common_sha256": (
        sha256_file(
            EVALUATION_COMMON_PATH
        )
    ),
    "temporal_baseline_freeze_sha256": (
        sha256_file(
            TEMPORAL_FREEZE
        )
    ),
    "files": files,
    "methodological_boundary": protocol[
        "methodological_boundary"
    ],
    "formal_hardware": (
        "NVIDIA GeForce RTX 3080 10 GB"
    ),
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
print(
    "protocol_status:",
    protocol["status"],
)
print(
    "checkpoint: "
    "PhyGuard_R0_Reconstruction_Checkpoint2"
)
print(
    "formal_hardware: "
    "NVIDIA GeForce RTX 3080 10 GB"
)
print("models_fitted: False")
print("predictions_generated: False")
print("performance_metrics_computed: False")
print("parameters_changed_in_source: False")
print("test_results_viewed: False")
print()
print(
    "baseline_equivalence_max_error:",
    max(
        maximum_raw_error,
        maximum_physical_error,
    ),
)
print(
    "formula_variant_count:",
    len(
        formula_variants
    ),
)
print(
    "training_task_count:",
    len(
        formula_variants
    )
    * 5,
)
print(
    "threshold_policy_count:",
    len(
        threshold_policies
    ),
)
print()
print("LOCKED FACTORS")
print(
    " - phi_scale:",
    [
        1.0,
        1.25,
        1.5,
        2.0,
        2.5,
    ],
)
print(
    " - adapt_gap_weight:",
    [
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
    ],
)
print(
    " - mismatch_mcs_drop_penalty:",
    [
        0.0,
        0.175,
        0.35,
        0.525,
        0.70,
    ],
)
print(
    " - fsr_max:",
    [
        0.025,
        0.05,
        0.10,
    ],
)
print(
    " - selected_accuracy_min:",
    [
        0.85,
        0.90,
        0.95,
    ],
)
print()
print("protocol:", PROTOCOL_PATH)
print("summary:", SUMMARY_PATH)
print(
    "baseline_equivalence:",
    EQUIVALENCE_PATH,
)
print(
    "source_assertions:",
    SOURCE_ASSERTIONS_PATH,
)
print("manifest:", MANIFEST_PATH)
print()
print(
    "PHYSICS_SENSITIVITY_PROTOCOL_V1_PASS"
)
