#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
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

SCRIPT_ROOT = R0 / "scripts"
DATA_ROOT = R0 / "data" / "full"
FEATURE_PATH = (
    R0
    / "results"
    / "full"
    / "features"
    / "features.npz"
)

EVALUATION_ROOT = (
    R0
    / "results"
    / "full"
    / "evaluation_suite"
)

METADATA_PATH = DATA_ROOT / "metadata.csv"
METRICS_PATH = EVALUATION_ROOT / "per_repeat_all_methods.csv"
PREDICTIONS_PATH = (
    EVALUATION_ROOT
    / "predictions"
    / "all_test_predictions.csv"
)

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "physics_sensitivity_protocol_v1.json"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "physics_sensitivity_fixed_model_results_v1"
)

TASK_ROOT = OUTPUT_ROOT / "tasks"
PREDICTION_ROOT = OUTPUT_ROOT / "predictions"

PROGRESS_PATH = OUTPUT_ROOT / "progress.json"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
PER_REPEAT_PATH = OUTPUT_ROOT / "per_repeat_metrics.csv"
AGGREGATE_PATH = OUTPUT_ROOT / "aggregate_metrics.csv"
ALL_PREDICTIONS_PATH = OUTPUT_ROOT / "all_test_predictions.csv"
THRESHOLD_POLICY_REPEAT_PATH = (
    OUTPUT_ROOT
    / "threshold_policy_per_repeat_metrics.csv"
)
THRESHOLD_POLICY_AGGREGATE_PATH = (
    OUTPUT_ROOT
    / "threshold_policy_aggregate_metrics.csv"
)
DEFAULT_REPRODUCTION_PATH = (
    OUTPUT_ROOT
    / "default_reproduction_audit.json"
)
COMPATIBILITY_PATH = OUTPUT_ROOT / "compatibility_audit.json"

AMENDMENT_ROOT = (
    ROOT
    / "artifacts"
    / "physics_sensitivity_fixed_model_protocol_amendment_v1"
)

AMENDMENT_SUMMARY = AMENDMENT_ROOT / "summary.json"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "physics_sensitivity_fixed_model_results_v1.json"
)

PHYS = np.array(
    [
        "interference",
        "blockage",
        "mobility",
        "adaptation_mismatch",
    ],
    dtype=object,
)

ABSTAIN = "ABSTAIN"

METRICS = [
    "selected_accuracy",
    "coverage",
    "false_specific_rate",
    "macro_f1",
    "balanced_accuracy",
    "physical_exact_match",
    "control_abstention",
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


def load_module(name: str, path: Path):
    import importlib.util

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


def stable_sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)

    positive = value >= 0

    result[positive] = 1.0 / (
        1.0
        + np.exp(
            -value[positive]
        )
    )

    exponential = np.exp(
        value[~positive]
    )

    result[~positive] = exponential / (
        1.0
        + exponential
    )

    return result


def manual_binary_logistic_probability(
    gate,
    x: np.ndarray,
) -> np.ndarray:
    require(
        hasattr(gate, "steps"),
        "Frozen gate is not a sklearn Pipeline.",
    )

    transformed = x

    for _, transformer in gate.steps[:-1]:
        transformed = transformer.transform(
            transformed
        )

    estimator = gate.steps[-1][1]

    require(
        hasattr(estimator, "coef_")
        and hasattr(estimator, "intercept_")
        and hasattr(estimator, "classes_"),
        "Frozen gate lacks fitted logistic coefficients.",
    )

    classes = np.asarray(
        estimator.classes_
    )

    require(
        classes.shape == (2,),
        f"Frozen gate is not binary: {classes}",
    )

    coefficients = np.asarray(
        estimator.coef_,
        dtype=np.float64,
    )

    intercept = np.asarray(
        estimator.intercept_,
        dtype=np.float64,
    )

    require(
        coefficients.shape[0] == 1,
        (
            "Expected binary logistic coefficient shape (1,F); "
            f"found {coefficients.shape}."
        ),
    )

    decision = (
        np.asarray(
            transformed,
            dtype=np.float64,
        )
        @ coefficients[0]
        + intercept[0]
    )

    probability_of_classes_1 = stable_sigmoid(
        decision
    )

    positive_position = np.flatnonzero(
        classes == 1
    )

    require(
        len(positive_position) == 1,
        (
            "Frozen gate class labels do not contain the "
            f"positive class 1: {classes}"
        ),
    )

    if int(positive_position[0]) == 1:
        return probability_of_classes_1

    return 1.0 - probability_of_classes_1


def manual_forest_probabilities(
    resolver,
    x: np.ndarray,
) -> np.ndarray:
    require(
        hasattr(resolver, "predict_proba")
        and hasattr(resolver, "classes_"),
        "Frozen resolver lacks a fitted probability interface.",
    )

    probabilities = np.asarray(
        resolver.predict_proba(x),
        dtype=np.float64,
    )

    require(
        probabilities.shape
        == (len(x), len(resolver.classes_)),
        (
            "Frozen resolver probability shape is inconsistent "
            f"with resolver.classes_: {probabilities.shape} vs "
            f"{len(resolver.classes_)} classes."
        ),
    )

    require(
        np.isfinite(probabilities).all(),
        "Frozen resolver produced a non-finite probability.",
    )

    row_sum_error = float(
        np.max(
            np.abs(
                probabilities.sum(axis=1)
                - 1.0
            )
        )
    )

    require(
        row_sum_error <= 1e-12,
        (
            "Frozen resolver probabilities do not sum to one: "
            f"max error={row_sum_error}"
        ),
    )

    return probabilities

def frozen_two_stage_probabilities(
    gate,
    resolver,
    physical: np.ndarray,
    combined: np.ndarray,
):
    gate_probability = (
        manual_binary_logistic_probability(
            gate,
            physical,
        )
    )

    mechanism_probability = (
        manual_forest_probabilities(
            resolver,
            combined,
        )
    )

    mechanism_confidence = (
        mechanism_probability.max(
            axis=1
        )
    )

    positions = mechanism_probability.argmax(
        axis=1
    )

    mechanism_label = np.asarray(
        resolver.classes_,
        dtype=object,
    )[positions]

    return (
        gate_probability,
        mechanism_confidence,
        mechanism_label,
    )


def recompute_physical(
    physical_baseline: np.ndarray,
    *,
    phi_scale: float,
    adapt_gap_weight: float,
    mismatch_mcs_drop_penalty: float,
) -> np.ndarray:
    underlying = physical_baseline[
        :,
        :12,
    ].astype(
        np.float64,
        copy=True,
    )

    rsrp_drop = underlying[:, 0]
    sinr_drop = underlying[:, 2]
    evm_rise = underlying[:, 3]
    ber_rise = underlying[:, 4]
    cqi_drop = underlying[:, 6]
    mcs_drop = underlying[:, 7]
    goodput_drop = underlying[:, 8]
    rho_drop = underlying[:, 9]
    gap_rise = underlying[:, 10]

    adapt_gap_z = (
        underlying[:, 11]
        * float(
            adapt_gap_weight
        )
    )

    underlying[:, 11] = adapt_gap_z

    def pos(value):
        return np.maximum(
            value,
            0.0,
        )

    def phi(value):
        return np.exp(
            -np.abs(value)
            / float(
                phi_scale
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
        - float(
            mismatch_mcs_drop_penalty
        )
        * pos(mcs_drop)
    ) / 5.0

    p_non = (
        pos(goodput_drop)
        + phi(rsrp_drop)
        + phi(sinr_drop)
        + phi(evm_rise)
        + phi(ber_rise)
    ) / 5.0

    scores = np.column_stack(
        [
            p_int,
            p_blk,
            p_mob,
            p_mis,
            p_non,
        ]
    )

    sorted_scores = np.sort(
        scores[:, :4],
        axis=1,
    )

    physical = np.column_stack(
        [
            underlying,
            scores,
            scores[:, :4].max(
                axis=1
            ),
            sorted_scores[:, -1]
            - sorted_scores[:, -2],
        ]
    )

    require(
        physical.shape == (1080, 19),
        f"Unexpected physical shape: {physical.shape}",
    )

    require(
        np.isfinite(
            physical
        ).all(),
        "Non-finite physical feature.",
    )

    return physical.astype(
        np.float32,
        copy=False,
    )


def extended_metrics(
    evaluation_common,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    result = evaluation_common.selective_metrics(
        truth,
        prediction,
    )

    truth = np.asarray(
        truth,
        dtype=object,
    )

    prediction = np.asarray(
        prediction,
        dtype=object,
    )

    physical = np.isin(
        truth,
        PHYS,
    )

    control = ~physical

    result[
        "physical_exact_match"
    ] = (
        float(
            np.mean(
                prediction[physical]
                == truth[physical]
            )
        )
        if np.any(physical)
        else 0.0
    )

    result[
        "control_abstention"
    ] = (
        float(
            np.mean(
                prediction[control]
                == ABSTAIN
            )
        )
        if np.any(control)
        else 0.0
    )

    return {
        key: float(value)
        for key, value in result.items()
    }


def aggregate_metrics(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows = []

    for keys, group in frame.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(
            zip(
                group_columns,
                keys,
            )
        )

        row[
            "repeat_count"
        ] = int(
            len(group)
        )

        for metric in METRICS:
            row[
                f"{metric}_mean"
            ] = float(
                group[
                    metric
                ].mean()
            )

            row[
                f"{metric}_std"
            ] = float(
                group[
                    metric
                ].std(
                    ddof=0
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def task_identifier(
    variant_id: str,
    repeat: int,
) -> str:
    return (
        f"{variant_id}"
        f"__repeat_{repeat}"
    )


def write_progress(
    *,
    status: str,
    completed: int,
    total: int,
    started_at: float,
    current_task: str | None,
) -> None:
    elapsed = max(
        0.0,
        time.time()
        - started_at,
    )

    rate = (
        completed
        / elapsed
        if elapsed > 0
        else 0.0
    )

    remaining = (
        (total - completed)
        / rate
        if rate > 0
        else None
    )

    payload = {
        "status": status,
        "completed_tasks": completed,
        "total_tasks": total,
        "fraction_complete": (
            completed
            / total
            if total
            else 0.0
        ),
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": (
            remaining
        ),
        "current_task": current_task,
        "updated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    PROGRESS_PATH.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    started_at = time.time()

    for path in [
        SCRIPT_ROOT / "evaluation_common.py",
        FEATURE_PATH,
        METADATA_PATH,
        METRICS_PATH,
        PREDICTIONS_PATH,
        PROTOCOL_PATH,
    ]:
        require(
            path.exists(),
            f"Required input is missing: {path}",
        )

    for repeat in range(5):
        for path in [
            EVALUATION_ROOT
            / "splits"
            / f"repeat_{repeat}.npz",

            EVALUATION_ROOT
            / "models"
            / f"repeat_{repeat}_phyguard.joblib",
        ]:
            require(
                path.exists(),
                f"Missing frozen repeat artifact: {path}",
            )

    require(
        not SUMMARY_PATH.exists(),
        "Final fixed-model sensitivity result already exists.",
    )

    protocol = json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    require(
        protocol.get("status")
        == "LOCKED_BEFORE_PHYSICS_SENSITIVITY_EXECUTION",
        "Original sensitivity protocol is not locked.",
    )

    variants = protocol[
        "design"
    ][
        "formula_variants"
    ]

    threshold_policies = protocol[
        "design"
    ][
        "threshold_policy_grid"
    ]

    require(
        len(variants) == 13,
        "Expected 13 formula variants.",
    )

    require(
        len(threshold_policies) == 9,
        "Expected nine threshold policies.",
    )

    evaluation_common = load_module(
        "phyguard_evaluation_common_stage61d4",
        SCRIPT_ROOT / "evaluation_common.py",
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    truth = metadata[
        "label"
    ].astype(
        str
    ).to_numpy(
        dtype=object
    )

    with np.load(
        FEATURE_PATH,
        allow_pickle=False,
    ) as archive:
        raw = archive[
            "raw"
        ].copy()

        physical_baseline = archive[
            "physical"
        ].copy()

        combined_baseline = archive[
            "combined"
        ].copy()

    require(
        raw.dtype
        == physical_baseline.dtype
        == combined_baseline.dtype
        == np.float32,
        (
            "Frozen feature archive must remain float32; "
            f"found {raw.dtype}, {physical_baseline.dtype}, "
            f"{combined_baseline.dtype}."
        ),
    )

    source_metrics = pd.read_csv(
        METRICS_PATH
    )

    source_metrics = source_metrics[
        source_metrics[
            "method"
        ].astype(
            str
        )
        == "PhyGuard"
    ].sort_values(
        "repeat"
    )

    source_predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    source_predictions = source_predictions[
        source_predictions[
            "method"
        ].astype(
            str
        )
        == "PhyGuard"
    ].copy()

    require(
        len(source_metrics) == 5,
        "Frozen metric table lacks five PhyGuard rows.",
    )

    require(
        len(source_predictions) == 1080,
        "Frozen prediction table lacks 1080 PhyGuard rows.",
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    TASK_ROOT.mkdir()
    PREDICTION_ROOT.mkdir()

    AMENDMENT_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    amendment = {
        "schema": (
            "phyguard.physics_sensitivity_fixed_model_protocol_amendment.v1"
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "LOCKED_BEFORE_NONDEFAULT_FIXED_MODEL_EXECUTION"
        ),
        "reason": (
            "Fresh refitting in the current environment did not "
            "reproduce the frozen Checkpoint2 reference, and the "
            "serialized logistic gate is not directly compatible "
            "with the current sklearn predict_proba path. Because "
            "no non-default variant had executed, the sensitivity "
            "design is amended to keep the five frozen learned "
            "models fixed and perturb only the source-defined "
            "physics formulas and validation policies."
        ),
        "primary_question": (
            "How sensitive are the frozen PhyGuard decision "
            "functions and selective operating points to reasonable "
            "changes in the source-defined physics scores, implicit "
            "evidence scaling, explicit mismatch penalty, and "
            "validation constraints?"
        ),
        "training_tasks": 0,
        "fixed_model_inference_tasks": 65,
        "threshold_policy_evaluations": 45,
        "default_reproduction_required_before_nondefault": True,
        "manual_probability_path": {
            "gate": (
                "Stored StandardScaler transformation followed by "
                "binary logistic sigmoid computed from stored "
                "coef_ and intercept_."
            ),
            "resolver": (
                "Frozen ExtraTreesClassifier.predict_proba; "
                "member-tree classes are internal numeric positions."
            ),
        },
        "source_models_modified": False,
        "source_features_modified": False,
        "source_results_modified": False,
        "outcome_based_omission": False,
        "all_13_variants_reported": True,
        "all_9_threshold_policies_reported": True,
        "claim_boundary": (
            "This is fixed-model formula and policy sensitivity for "
            "the revision-time locked reconstruction. It is not a "
            "claim that the unavailable original submission code "
            "contained trainable physics weights."
        ),
        "formal_hardware": (
            "NVIDIA GeForce RTX 3080 10 GB"
        ),
    }

    AMENDMENT_SUMMARY.write_text(
        json.dumps(
            amendment,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded_models = {}
    compatibility_records = []

    for repeat in range(5):
        model_path = (
            EVALUATION_ROOT
            / "models"
            / f"repeat_{repeat}_phyguard.joblib"
        )

        with warnings.catch_warnings(
            record=True
        ) as caught:
            warnings.simplefilter(
                "always"
            )

            bundle = joblib.load(
                model_path
            )

        gate = bundle[
            "gate"
        ]

        resolver = bundle[
            "resolver"
        ]

        logistic = gate.steps[
            -1
        ][
            1
        ]

        compatibility_records.append(
            {
                "repeat": repeat,
                "model_path": str(
                    model_path
                ),
                "model_sha256": sha256_file(
                    model_path
                ),
                "gate_class": (
                    type(gate).__name__
                ),
                "logistic_class": (
                    type(logistic).__name__
                ),
                "logistic_has_multi_class_attribute": bool(
                    hasattr(
                        logistic,
                        "multi_class",
                    )
                ),
                "resolver_class": (
                    type(
                        resolver
                    ).__name__
                ),
                "tree_count": int(
                    len(
                        resolver.estimators_
                    )
                ),
                "load_warnings": [
                    {
                        "category": (
                            warning.category.__name__
                        ),
                        "message": str(
                            warning.message
                        ),
                    }
                    for warning in caught
                ],
                "manual_probability_path_used": True,
            }
        )

        loaded_models[
            repeat
        ] = (
            gate,
            resolver,
        )

    COMPATIBILITY_PATH.write_text(
        json.dumps(
            {
                "status": "PASS",
                "records": compatibility_records,
                "source_models_modified": False,
                "manual_probability_path_used": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    total_tasks = (
        len(
            variants
        )
        * 5
    )

    completed = 0

    write_progress(
        status="RUNNING",
        completed=0,
        total=total_tasks,
        started_at=started_at,
        current_task=None,
    )

    task_records = []
    prediction_frames = []

    print(
        "FIXED-MODEL PHYSICS SENSITIVITY START",
        flush=True,
    )

    print(
        "tasks:",
        total_tasks,
        flush=True,
    )

    for variant in variants:
        variant_id = str(
            variant[
                "variant_id"
            ]
        )

        if variant_id == "DEFAULT":
            physical = (
                physical_baseline.copy()
            )

            combined = (
                combined_baseline.copy()
            )

        else:
            physical = recompute_physical(
                physical_baseline,
                phi_scale=float(
                    variant[
                        "phi_scale"
                    ]
                ),
                adapt_gap_weight=float(
                    variant[
                        "adapt_gap_weight"
                    ]
                ),
                mismatch_mcs_drop_penalty=float(
                    variant[
                        "mismatch_mcs_drop_penalty"
                    ]
                ),
            )

            combined = np.concatenate(
                [
                    raw,
                    physical,
                ],
                axis=1,
            ).astype(
                np.float32,
                copy=False,
            )

        require(
            combined.shape
            == (1080, 79),
            (
                f"{variant_id}: invalid combined shape "
                f"{combined.shape}"
            ),
        )

        for repeat in range(5):
            identifier = task_identifier(
                variant_id,
                repeat,
            )

            task_path = (
                TASK_ROOT
                / f"{identifier}.json"
            )

            prediction_path = (
                PREDICTION_ROOT
                / f"{identifier}.csv"
            )

            if (
                task_path.exists()
                and prediction_path.exists()
            ):
                record = json.loads(
                    task_path.read_text(
                        encoding="utf-8"
                    )
                )

                task_records.append(
                    record
                )

                prediction_frames.append(
                    pd.read_csv(
                        prediction_path
                    )
                )

                completed += 1

                continue

            write_progress(
                status="RUNNING",
                completed=completed,
                total=total_tasks,
                started_at=started_at,
                current_task=identifier,
            )

            with np.load(
                EVALUATION_ROOT
                / "splits"
                / f"repeat_{repeat}.npz",
                allow_pickle=False,
            ) as split_archive:
                validation = split_archive[
                    "val"
                ].astype(
                    np.int64
                )

                test = split_archive[
                    "test"
                ].astype(
                    np.int64
                )

            gate, resolver = (
                loaded_models[
                    repeat
                ]
            )

            gv, cv, lv = (
                frozen_two_stage_probabilities(
                    gate,
                    resolver,
                    physical[
                        validation
                    ],
                    combined[
                        validation
                    ],
                )
            )

            best, _ = (
                evaluation_common.operating_point_rows(
                    truth[
                        validation
                    ],
                    lv,
                    gv,
                    cv,
                    fsr_max=0.05,
                    selected_accuracy_min=0.90,
                )
            )

            gt, ct, lt = (
                frozen_two_stage_probabilities(
                    gate,
                    resolver,
                    physical[
                        test
                    ],
                    combined[
                        test
                    ],
                )
            )

            prediction = np.where(
                (
                    gt
                    >= float(
                        best[
                            "tau_g"
                        ]
                    )
                )
                & (
                    ct
                    >= float(
                        best[
                            "tau_c"
                        ]
                    )
                ),
                lt,
                ABSTAIN,
            )

            metrics = extended_metrics(
                evaluation_common,
                truth[
                    test
                ],
                prediction,
            )

            prediction_frame = pd.DataFrame(
                {
                    "variant_id": variant_id,
                    "factor": str(
                        variant[
                            "factor"
                        ]
                    ),
                    "repeat": repeat,
                    "index": test,
                    "truth": truth[
                        test
                    ],
                    "prediction": prediction,
                    "gate_confidence": gt,
                    "mechanism_confidence": ct,
                    "confidence": gt * ct,
                    "tau_g": float(
                        best[
                            "tau_g"
                        ]
                    ),
                    "tau_c": float(
                        best[
                            "tau_c"
                        ]
                    ),
                }
            )

            prediction_frame.to_csv(
                prediction_path,
                index=False,
            )

            record = {
                "status": "PASS",
                "variant_id": variant_id,
                "factor": str(
                    variant[
                        "factor"
                    ]
                ),
                "phi_scale": float(
                    variant[
                        "phi_scale"
                    ]
                ),
                "adapt_gap_weight": float(
                    variant[
                        "adapt_gap_weight"
                    ]
                ),
                "mismatch_mcs_drop_penalty": float(
                    variant[
                        "mismatch_mcs_drop_penalty"
                    ]
                ),
                "repeat": repeat,
                "validation_count": int(
                    len(
                        validation
                    )
                ),
                "test_count": int(
                    len(
                        test
                    )
                ),
                "tau_g": float(
                    best[
                        "tau_g"
                    ]
                ),
                "tau_c": float(
                    best[
                        "tau_c"
                    ]
                ),
                "calibration_feasible": bool(
                    best[
                        "feasible"
                    ]
                ),
                "calibration_penalty": float(
                    best[
                        "penalty"
                    ]
                ),
                **metrics,
                "prediction_path": str(
                    prediction_path
                ),
                "prediction_sha256": sha256_file(
                    prediction_path
                ),
                "source_model_path": str(
                    EVALUATION_ROOT
                    / "models"
                    / f"repeat_{repeat}_phyguard.joblib"
                ),
            }

            task_path.write_text(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            task_records.append(
                record
            )

            prediction_frames.append(
                prediction_frame
            )

            completed += 1

            elapsed = (
                time.time()
                - started_at
            )

            rate = (
                completed
                / elapsed
                if elapsed > 0
                else 0.0
            )

            remaining = (
                (
                    total_tasks
                    - completed
                )
                / rate
                if rate > 0
                else 0.0
            )

            print(
                f"[{completed:02d}/{total_tasks}] "
                f"{identifier} PASS "
                f"elapsed={elapsed:.1f}s "
                f"eta={remaining:.1f}s",
                flush=True,
            )

            write_progress(
                status="RUNNING",
                completed=completed,
                total=total_tasks,
                started_at=started_at,
                current_task=identifier,
            )

        if variant_id == "DEFAULT":
            default_records = pd.DataFrame(
                [
                    record
                    for record in task_records
                    if record[
                        "variant_id"
                    ]
                    == "DEFAULT"
                ]
            ).sort_values(
                "repeat"
            )

            default_predictions = pd.concat(
                [
                    frame
                    for frame in prediction_frames
                    if (
                        frame[
                            "variant_id"
                        ].iloc[
                            0
                        ]
                        == "DEFAULT"
                    )
                ],
                ignore_index=True,
            )

            metric_differences = {}

            for metric in [
                "selected_accuracy",
                "coverage",
                "false_specific_rate",
                "macro_f1",
                "balanced_accuracy",
            ]:
                difference = float(
                    np.max(
                        np.abs(
                            source_metrics[
                                metric
                            ].to_numpy(
                                dtype=float
                            )
                            - default_records[
                                metric
                            ].to_numpy(
                                dtype=float
                            )
                        )
                    )
                )

                metric_differences[
                    metric
                ] = difference

                require(
                    difference
                    <= 1e-12,
                    (
                        "DEFAULT fixed-model metric reproduction "
                        f"failed for {metric}: {difference}"
                    ),
                )

            for threshold in [
                "tau_g",
                "tau_c",
            ]:
                difference = float(
                    np.max(
                        np.abs(
                            source_metrics[
                                threshold
                            ].to_numpy(
                                dtype=float
                            )
                            - default_records[
                                threshold
                            ].to_numpy(
                                dtype=float
                            )
                        )
                    )
                )

                require(
                    difference
                    <= 1e-12,
                    (
                        "DEFAULT fixed-model threshold "
                        f"reproduction failed for {threshold}: "
                        f"{difference}"
                    ),
                )

            source_alignment = (
                source_predictions[
                    [
                        "repeat",
                        "index",
                        "truth",
                        "prediction",
                    ]
                ]
                .sort_values(
                    [
                        "repeat",
                        "index",
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            generated_alignment = (
                default_predictions[
                    [
                        "repeat",
                        "index",
                        "truth",
                        "prediction",
                    ]
                ]
                .sort_values(
                    [
                        "repeat",
                        "index",
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            prediction_match = bool(
                source_alignment.equals(
                    generated_alignment
                )
            )

            require(
                prediction_match,
                (
                    "DEFAULT fixed-model predictions do not "
                    "exactly reproduce all 1080 frozen rows."
                ),
            )

            reproduction = {
                "status": "PASS",
                "metric_max_absolute_differences": metric_differences,
                "tau_g_match": True,
                "tau_c_match": True,
                "prediction_match": True,
                "prediction_row_count": 1080,
                "manual_probability_path_used": True,
                "source_models_modified": False,
            }

            DEFAULT_REPRODUCTION_PATH.write_text(
                json.dumps(
                    reproduction,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                "DEFAULT_FIXED_MODEL_REPRODUCTION_PASS",
                flush=True,
            )

    per_repeat = pd.DataFrame(
        task_records
    ).sort_values(
        [
            "variant_id",
            "repeat",
        ]
    )

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    ).sort_values(
        [
            "variant_id",
            "repeat",
            "index",
        ]
    )

    require(
        len(
            per_repeat
        )
        == 65,
        (
            "Expected 65 fixed-model sensitivity rows; "
            f"found {len(per_repeat)}."
        ),
    )

    require(
        len(
            all_predictions
        )
        == 65
        * 216,
        (
            "Unexpected prediction row count: "
            f"{len(all_predictions)}"
        ),
    )

    per_repeat.to_csv(
        PER_REPEAT_PATH,
        index=False,
    )

    aggregate = aggregate_metrics(
        per_repeat,
        [
            "variant_id",
            "factor",
            "phi_scale",
            "adapt_gap_weight",
            "mismatch_mcs_drop_penalty",
        ],
    )

    aggregate.to_csv(
        AGGREGATE_PATH,
        index=False,
    )

    all_predictions.to_csv(
        ALL_PREDICTIONS_PATH,
        index=False,
    )

    policy_rows = []

    for repeat in range(5):
        with np.load(
            EVALUATION_ROOT
            / "splits"
            / f"repeat_{repeat}.npz",
            allow_pickle=False,
        ) as split_archive:
            validation = split_archive[
                "val"
            ].astype(
                np.int64
            )

            test = split_archive[
                "test"
            ].astype(
                np.int64
            )

        gate, resolver = loaded_models[
            repeat
        ]

        gv, cv, lv = frozen_two_stage_probabilities(
            gate,
            resolver,
            physical_baseline[
                validation
            ],
            combined_baseline[
                validation
            ],
        )

        gt, ct, lt = frozen_two_stage_probabilities(
            gate,
            resolver,
            physical_baseline[
                test
            ],
            combined_baseline[
                test
            ],
        )

        for policy in threshold_policies:
            best, _ = (
                evaluation_common.operating_point_rows(
                    truth[
                        validation
                    ],
                    lv,
                    gv,
                    cv,
                    fsr_max=float(
                        policy[
                            "fsr_max"
                        ]
                    ),
                    selected_accuracy_min=float(
                        policy[
                            "selected_accuracy_min"
                        ]
                    ),
                )
            )

            prediction = np.where(
                (
                    gt
                    >= float(
                        best[
                            "tau_g"
                        ]
                    )
                )
                & (
                    ct
                    >= float(
                        best[
                            "tau_c"
                        ]
                    )
                ),
                lt,
                ABSTAIN,
            )

            metrics = extended_metrics(
                evaluation_common,
                truth[
                    test
                ],
                prediction,
            )

            policy_rows.append(
                {
                    "policy_id": str(
                        policy[
                            "policy_id"
                        ]
                    ),
                    "fsr_max": float(
                        policy[
                            "fsr_max"
                        ]
                    ),
                    "selected_accuracy_min": float(
                        policy[
                            "selected_accuracy_min"
                        ]
                    ),
                    "is_default": bool(
                        policy[
                            "is_default"
                        ]
                    ),
                    "repeat": repeat,
                    "tau_g": float(
                        best[
                            "tau_g"
                        ]
                    ),
                    "tau_c": float(
                        best[
                            "tau_c"
                        ]
                    ),
                    "calibration_feasible": bool(
                        best[
                            "feasible"
                        ]
                    ),
                    "calibration_penalty": float(
                        best[
                            "penalty"
                        ]
                    ),
                    **metrics,
                }
            )

    policy_repeat = pd.DataFrame(
        policy_rows
    ).sort_values(
        [
            "policy_id",
            "repeat",
        ]
    )

    require(
        len(
            policy_repeat
        )
        == 45,
        (
            "Expected 45 policy rows; "
            f"found {len(policy_repeat)}."
        ),
    )

    policy_repeat.to_csv(
        THRESHOLD_POLICY_REPEAT_PATH,
        index=False,
    )

    policy_aggregate = aggregate_metrics(
        policy_repeat,
        [
            "policy_id",
            "fsr_max",
            "selected_accuracy_min",
            "is_default",
        ],
    )

    policy_aggregate.to_csv(
        THRESHOLD_POLICY_AGGREGATE_PATH,
        index=False,
    )

    summary = {
        "schema": (
            "phyguard.physics_sensitivity_fixed_model_results.v1"
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "checkpoint": (
            "PhyGuard_R0_Reconstruction_Checkpoint2"
        ),
        "formula_variant_count": 13,
        "fixed_model_inference_task_count": 65,
        "training_task_count": 0,
        "threshold_policy_count": 9,
        "threshold_policy_evaluation_count": 45,
        "default_reproduction": json.loads(
            DEFAULT_REPRODUCTION_PATH.read_text(
                encoding="utf-8"
            )
        ),
        "source_models_modified": False,
        "source_features_modified": False,
        "source_results_modified": False,
        "test_used_for_threshold_selection": False,
        "outcome_based_omission": False,
        "interpretation": (
            "Fixed-model sensitivity of the revision-time locked "
            "reconstruction to source-defined physics formulas and "
            "validation policies."
        ),
        "formal_hardware": (
            "NVIDIA GeForce RTX 3080 10 GB"
        ),
        "files": {
            "per_repeat_metrics": str(
                PER_REPEAT_PATH
            ),
            "aggregate_metrics": str(
                AGGREGATE_PATH
            ),
            "all_test_predictions": str(
                ALL_PREDICTIONS_PATH
            ),
            "threshold_policy_per_repeat": str(
                THRESHOLD_POLICY_REPEAT_PATH
            ),
            "threshold_policy_aggregate": str(
                THRESHOLD_POLICY_AGGREGATE_PATH
            ),
            "compatibility_audit": str(
                COMPATIBILITY_PATH
            ),
            "protocol_amendment": str(
                AMENDMENT_SUMMARY
            ),
        },
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest_files = []

    for path in sorted(
        OUTPUT_ROOT.rglob("*")
    ):
        if path.is_file():
            manifest_files.append(
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

    manifest_files.append(
        {
            "path": str(
                AMENDMENT_SUMMARY.relative_to(
                    ROOT
                )
            ),
            "size_bytes": (
                AMENDMENT_SUMMARY.stat().st_size
            ),
            "sha256": sha256_file(
                AMENDMENT_SUMMARY
            ),
        }
    )

    manifest = {
        "schema": (
            "phyguard.physics_sensitivity_fixed_model_results_manifest.v1"
        ),
        "status": "PASS",
        "protocol_sha256": sha256_file(
            PROTOCOL_PATH
        ),
        "feature_archive_sha256": sha256_file(
            FEATURE_PATH
        ),
        "source_metrics_sha256": sha256_file(
            METRICS_PATH
        ),
        "source_predictions_sha256": sha256_file(
            PREDICTIONS_PATH
        ),
        "fixed_model_inference_task_count": 65,
        "training_task_count": 0,
        "threshold_policy_evaluation_count": 45,
        "files": manifest_files,
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

    write_progress(
        status="PASS",
        completed=total_tasks,
        total=total_tasks,
        started_at=started_at,
        current_task=None,
    )

    print()
    print("status: PASS")
    print("formula_variant_count: 13")
    print(
        "fixed_model_inference_task_count: 65"
    )
    print("training_task_count: 0")
    print("threshold_policy_count: 9")
    print(
        "threshold_policy_evaluation_count: 45"
    )
    print(
        "default_prediction_match: True"
    )
    print(
        "source_models_modified: False"
    )
    print(
        "test_used_for_threshold_selection: False"
    )
    print(
        "formal_hardware: NVIDIA GeForce RTX 3080 10 GB"
    )
    print("summary:", SUMMARY_PATH)
    print("aggregate_metrics:", AGGREGATE_PATH)
    print(
        "threshold_policy_aggregate:",
        THRESHOLD_POLICY_AGGREGATE_PATH,
    )
    print("manifest:", MANIFEST_PATH)
    print()
    print(
        "PHYSICS_SENSITIVITY_FIXED_MODEL_RESULTS_V1_PASS"
    )


if __name__ == "__main__":
    main()
