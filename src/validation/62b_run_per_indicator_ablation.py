#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
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

SOURCE_DATA_DIR = R0 / "data" / "full"
SOURCE_FEATURE_DIR = R0 / "results" / "full" / "features"
SOURCE_REFERENCE_EVAL_DIR = (
    R0
    / "results"
    / "full"
    / "evaluation_suite"
)

FEATURE_BUILDER_PATH = R0 / "scripts" / "features.py"
EVALUATE_SUITE_PATH = R0 / "scripts" / "evaluate_suite.py"
EVALUATION_COMMON_PATH = R0 / "scripts" / "evaluation_common.py"

LOCKED_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal825_locked_reconstruction_final.zip"
)

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "per_indicator_ablation_protocol_v1.json"
)

PROTOCOL_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "per_indicator_ablation_protocol_v1.json"
)

AUDIT_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "per_indicator_ablation_implementation_audit_v1.json"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "per_indicator_ablation_results_v1"
)

WORK_ROOT = (
    ROOT
    / "work"
    / "per_indicator_ablation_v1"
)

RESULT_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "per_indicator_ablation_results_v1.json"
)

EXPECTED = {
    "locked_archive_sha256":
        "5d08fc3b8265789819b0f49c1f93a9c79c8fe44ce7ec4aec8f1fa603a08c241b",

    "protocol_status":
        "LOCKED_BEFORE_PER_INDICATOR_ABLATION_EXECUTION",

    "sample_count_sionna":
        825,

    "repeat_count":
        5,

    "source_raw_dimension":
        60,

    "source_physical_dimension":
        19,

    "source_combined_dimension":
        79,

    "packet_error_kpi_index":
        5,

    "bootstrap_replicates":
        10000,

    "source_bootstrap_seed":
        62011,

    "sionna_bootstrap_seed":
        62021,

    "retention_margin":
        -0.05,

    "evaluate_seed":
        20260705,
}

VARIANTS = [
    "FULL",
    "NO_PER_RAW",
    "NO_PER_SCORE",
    "NO_ERROR_INDICATOR",
]

PHYSICAL_LABELS = {
    "interference",
    "blockage",
    "mobility",
    "adaptation_mismatch",
}

ABSTAIN = "ABSTAIN"


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


def json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return json_safe(
            value.tolist()
        )

    if isinstance(value, np.generic):
        return json_safe(
            value.item()
        )

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None

    return value


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


def unpack_build_one_result(result):
    if isinstance(result, tuple) and len(result) >= 2:
        raw = np.asarray(
            result[0],
            dtype=np.float64,
        ).reshape(-1)

        physical = np.asarray(
            result[1],
            dtype=np.float64,
        ).reshape(-1)

        return raw, physical

    if isinstance(result, dict):
        raw_key = next(
            (
                key
                for key in [
                    "raw",
                    "raw_features",
                ]
                if key in result
            ),
            None,
        )

        physical_key = next(
            (
                key
                for key in [
                    "physical",
                    "phys",
                    "physical_features",
                ]
                if key in result
            ),
            None,
        )

        require(
            raw_key is not None
            and physical_key is not None,
            "build_one dictionary result lacks raw/physical keys.",
        )

        return (
            np.asarray(
                result[raw_key],
                dtype=np.float64,
            ).reshape(-1),
            np.asarray(
                result[physical_key],
                dtype=np.float64,
            ).reshape(-1),
        )

    raise RuntimeError(
        "Unsupported build_one return type."
    )


def call_build_one(
    build_one,
    sequence: np.ndarray,
):
    attempts = [
        (
            "x_start_end",
            lambda:
                build_one(
                    sequence,
                    24,
                    56,
                ),
        ),
        (
            "x_start_end_eps",
            lambda:
                build_one(
                    sequence,
                    24,
                    56,
                    1e-6,
                ),
        ),
        (
            "x_tuple",
            lambda:
                build_one(
                    sequence,
                    (24, 56),
                ),
        ),
        (
            "x_only",
            lambda:
                build_one(
                    sequence
                ),
        ),
    ]

    failures = []

    for name, function in attempts:
        try:
            result = function()
            raw, physical = unpack_build_one_result(
                result
            )

            if (
                raw.shape
                == (
                    EXPECTED[
                        "source_raw_dimension"
                    ],
                )
                and physical.shape
                == (
                    EXPECTED[
                        "source_physical_dimension"
                    ],
                )
            ):
                return raw, physical, name

            failures.append(
                (
                    name,
                    (
                        f"unexpected shapes raw={raw.shape}, "
                        f"physical={physical.shape}"
                    ),
                )
            )
        except Exception as error:
            failures.append(
                (
                    name,
                    repr(error),
                )
            )

    raise RuntimeError(
        "Could not invoke build_one. Attempts:\n"
        + "\n".join(
            f"{name}: {message}"
            for name, message in failures
        )
    )


def make_probe_sequence() -> np.ndarray:
    t = np.arange(
        80,
        dtype=np.float64,
    )

    x = np.column_stack(
        [
            -82.0
            + 0.8
            * np.sin(
                2.0
                * np.pi
                * t
                / 31.0
            ),

            -72.0
            + 0.6
            * np.cos(
                2.0
                * np.pi
                * t
                / 29.0
            ),

            11.0
            + 0.7
            * np.sin(
                2.0
                * np.pi
                * t
                / 23.0
            ),

            0.12
            + 0.01
            * np.cos(
                2.0
                * np.pi
                * t
                / 19.0
            ),

            0.018
            + 0.003
            * np.sin(
                2.0
                * np.pi
                * t
                / 17.0
            ),

            0.16
            + 0.02
            * np.cos(
                2.0
                * np.pi
                * t
                / 13.0
            ),

            8.0
            + 0.4
            * np.sin(
                2.0
                * np.pi
                * t
                / 37.0
            ),

            1.0
            + 0.1
            * np.cos(
                2.0
                * np.pi
                * t
                / 41.0
            ),

            0.55
            + 0.04
            * np.sin(
                2.0
                * np.pi
                * t
                / 27.0
            ),

            0.91
            + 0.02
            * np.cos(
                2.0
                * np.pi
                * t
                / 21.0
            ),
        ]
    )

    require(
        x.shape == (80, 10),
        "Probe sequence shape mismatch.",
    )

    return x


def discover_packet_error_feature_masks(
    build_one,
):
    baseline = make_probe_sequence()

    baseline_raw, baseline_physical, call_mode = (
        call_build_one(
            build_one,
            baseline,
        )
    )

    perturbations = []

    candidate_raise = baseline.copy()
    candidate_raise[
        24:56,
        EXPECTED[
            "packet_error_kpi_index"
        ],
    ] = np.clip(
        candidate_raise[
            24:56,
            EXPECTED[
                "packet_error_kpi_index"
            ],
        ]
        + 0.31,
        0.0,
        1.0,
    )
    perturbations.append(
        candidate_raise
    )

    reference_raise = baseline.copy()
    reference_raise[
        :24,
        EXPECTED[
            "packet_error_kpi_index"
        ],
    ] = np.clip(
        reference_raise[
            :24,
            EXPECTED[
                "packet_error_kpi_index"
            ],
        ]
        + 0.27,
        0.0,
        1.0,
    )
    perturbations.append(
        reference_raise
    )

    candidate_wave = baseline.copy()
    candidate_wave[
        24:56,
        EXPECTED[
            "packet_error_kpi_index"
        ],
    ] = np.clip(
        0.05
        + 0.55
        * (
            np.arange(
                32,
                dtype=np.float64,
            )
            % 3
            == 0
        ),
        0.0,
        1.0,
    )
    perturbations.append(
        candidate_wave
    )

    full_ramp = baseline.copy()
    full_ramp[
        :,
        EXPECTED[
            "packet_error_kpi_index"
        ],
    ] = np.linspace(
        0.01,
        0.88,
        80,
    )
    perturbations.append(
        full_ramp
    )

    raw_mask = np.zeros(
        EXPECTED[
            "source_raw_dimension"
        ],
        dtype=bool,
    )

    physical_mask = np.zeros(
        EXPECTED[
            "source_physical_dimension"
        ],
        dtype=bool,
    )

    maximum_raw_differences = np.zeros(
        EXPECTED[
            "source_raw_dimension"
        ],
        dtype=np.float64,
    )

    maximum_physical_differences = np.zeros(
        EXPECTED[
            "source_physical_dimension"
        ],
        dtype=np.float64,
    )

    for perturbed in perturbations:
        raw, physical, mode = call_build_one(
            build_one,
            perturbed,
        )

        require(
            mode == call_mode,
            "build_one call mode changed across probes.",
        )

        raw_difference = np.abs(
            raw
            - baseline_raw
        )

        physical_difference = np.abs(
            physical
            - baseline_physical
        )

        maximum_raw_differences = np.maximum(
            maximum_raw_differences,
            raw_difference,
        )

        maximum_physical_differences = np.maximum(
            maximum_physical_differences,
            physical_difference,
        )

        raw_mask |= (
            raw_difference
            > 1e-9
        )

        physical_mask |= (
            physical_difference
            > 1e-9
        )

    raw_indices = np.flatnonzero(
        raw_mask
    ).astype(int)

    physical_indices = np.flatnonzero(
        physical_mask
    ).astype(int)

    require(
        len(raw_indices) == 6,
        (
            "Expected six raw summary features for the "
            "packet-error KPI, found "
            f"{len(raw_indices)}: "
            f"{raw_indices.tolist()}"
        ),
    )

    require(
        1
        <= len(physical_indices)
        < EXPECTED[
            "source_physical_dimension"
        ],
        (
            "Unexpected packet-error-dependent physical "
            f"feature count: {len(physical_indices)} "
            f"indices={physical_indices.tolist()}"
        ),
    )

    return {
        "build_one_signature":
            str(
                inspect.signature(
                    build_one
                )
            ),

        "call_mode":
            call_mode,

        "raw_indices":
            raw_indices.tolist(),

        "physical_indices":
            physical_indices.tolist(),

        "maximum_raw_differences":
            maximum_raw_differences.tolist(),

        "maximum_physical_differences":
            maximum_physical_differences.tolist(),
    }


def build_variant_features(
    raw: np.ndarray,
    physical: np.ndarray,
    variant: str,
    raw_drop_indices: list[int],
    physical_zero_indices: list[int],
):
    require(
        variant in VARIANTS,
        f"Unknown variant: {variant}",
    )

    raw_variant = np.asarray(
        raw,
        dtype=np.float64,
    ).copy()

    physical_variant = np.asarray(
        physical,
        dtype=np.float64,
    ).copy()

    if variant in {
        "NO_PER_RAW",
        "NO_ERROR_INDICATOR",
    }:
        keep = np.ones(
            raw_variant.shape[1],
            dtype=bool,
        )

        keep[
            np.asarray(
                raw_drop_indices,
                dtype=int,
            )
        ] = False

        raw_variant = raw_variant[
            :,
            keep,
        ]

    if variant in {
        "NO_PER_SCORE",
        "NO_ERROR_INDICATOR",
    }:
        physical_variant[
            :,
            np.asarray(
                physical_zero_indices,
                dtype=int,
            ),
        ] = 0.0

    combined_variant = np.concatenate(
        [
            raw_variant,
            physical_variant,
        ],
        axis=1,
    )

    return (
        raw_variant.astype(
            np.float32
        ),
        physical_variant.astype(
            np.float32
        ),
        combined_variant.astype(
            np.float32
        ),
    )


def run_evaluate_suite(
    feature_dir: Path,
    output_dir: Path,
):
    command = [
        sys.executable,
        str(
            EVALUATE_SUITE_PATH
        ),
        "--data-dir",
        str(
            SOURCE_DATA_DIR
        ),
        "--feature-dir",
        str(
            feature_dir
        ),
        "--out-dir",
        str(
            output_dir
        ),
        "--repeats",
        str(
            EXPECTED[
                "repeat_count"
            ]
        ),
        "--seed",
        str(
            EXPECTED[
                "evaluate_seed"
            ]
        ),
    ]

    environment = os.environ.copy()

    existing_pythonpath = environment.get(
        "PYTHONPATH",
        "",
    )

    environment["PYTHONPATH"] = (
        str(
            EVALUATE_SUITE_PATH.parent
        )
        + (
            os.pathsep
            + existing_pythonpath
            if existing_pythonpath
            else ""
        )
    )

    log_path = (
        output_dir.parent
        / f"{output_dir.name}_evaluate_suite.log"
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        completed = subprocess.run(
            command,
            cwd=str(
                EVALUATE_SUITE_PATH.parent
            ),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )

    require(
        completed.returncode == 0,
        (
            "evaluate_suite failed for "
            f"{feature_dir.name}; see {log_path}"
        ),
    )

    required_outputs = [
        output_dir
        / "per_repeat_all_methods.csv",

        output_dir
        / "predictions"
        / "all_test_predictions.csv",

        output_dir
        / "threshold_search_all_methods.csv",
    ]

    for path in required_outputs:
        require(
            path.exists(),
            (
                "evaluate_suite output is missing: "
                f"{path}"
            ),
        )

    for repeat in range(
        EXPECTED[
            "repeat_count"
        ]
    ):
        model_path = (
            output_dir
            / "models"
            / f"repeat_{repeat}_phyguard.joblib"
        )

        require(
            model_path.exists(),
            f"Model output is missing: {model_path}",
        )

    return log_path


def method_mask(
    series: pd.Series,
    target: str = "PhyGuard",
):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        == target.lower()
    )


def compare_source_full_parity(
    generated_dir: Path,
):
    reference_metrics_path = (
        SOURCE_REFERENCE_EVAL_DIR
        / "per_repeat_all_methods.csv"
    )

    generated_metrics_path = (
        generated_dir
        / "per_repeat_all_methods.csv"
    )

    reference_predictions_path = (
        SOURCE_REFERENCE_EVAL_DIR
        / "predictions"
        / "all_test_predictions.csv"
    )

    generated_predictions_path = (
        generated_dir
        / "predictions"
        / "all_test_predictions.csv"
    )

    for path in [
        reference_metrics_path,
        generated_metrics_path,
        reference_predictions_path,
        generated_predictions_path,
    ]:
        require(
            path.exists(),
            f"Parity file is missing: {path}",
        )

    reference_metrics = pd.read_csv(
        reference_metrics_path
    )

    generated_metrics = pd.read_csv(
        generated_metrics_path
    )

    reference_metrics = reference_metrics[
        method_mask(
            reference_metrics[
                "method"
            ]
        )
    ].copy()

    generated_metrics = generated_metrics[
        method_mask(
            generated_metrics[
                "method"
            ]
        )
    ].copy()

    reference_metrics = reference_metrics.sort_values(
        "repeat"
    ).reset_index(
        drop=True
    )

    generated_metrics = generated_metrics.sort_values(
        "repeat"
    ).reset_index(
        drop=True
    )

    require(
        len(reference_metrics)
        == EXPECTED[
            "repeat_count"
        ],
        "Reference PhyGuard repeat count mismatch.",
    )

    require(
        len(generated_metrics)
        == EXPECTED[
            "repeat_count"
        ],
        "Generated FULL repeat count mismatch.",
    )

    metric_columns = [
        "selected_accuracy",
        "coverage",
        "false_specific_rate",
        "macro_f1",
        "balanced_accuracy",
        "tau_g",
        "tau_c",
    ]

    maximum_metric_difference = 0.0

    for column in metric_columns:
        require(
            column in reference_metrics.columns
            and column in generated_metrics.columns,
            f"Parity metric column missing: {column}",
        )

        difference = np.nanmax(
            np.abs(
                reference_metrics[
                    column
                ].to_numpy(
                    dtype=np.float64
                )
                - generated_metrics[
                    column
                ].to_numpy(
                    dtype=np.float64
                )
            )
        )

        maximum_metric_difference = max(
            maximum_metric_difference,
            float(difference),
        )

    require(
        maximum_metric_difference
        <= 1e-12,
        (
            "FULL source metrics did not reproduce the "
            "locked evaluation. Maximum difference="
            f"{maximum_metric_difference}"
        ),
    )

    reference_predictions = pd.read_csv(
        reference_predictions_path
    )

    generated_predictions = pd.read_csv(
        generated_predictions_path
    )

    reference_predictions = reference_predictions[
        method_mask(
            reference_predictions[
                "method"
            ]
        )
    ].copy()

    generated_predictions = generated_predictions[
        method_mask(
            generated_predictions[
                "method"
            ]
        )
    ].copy()

    key_columns = [
        "repeat",
        "index",
    ]

    for column in key_columns:
        require(
            column in reference_predictions.columns
            and column in generated_predictions.columns,
            f"Prediction parity key missing: {column}",
        )

    comparison_columns = [
        column
        for column in [
            "truth",
            "prediction",
        ]
        if (
            column
            in reference_predictions.columns
            and column
            in generated_predictions.columns
        )
    ]

    left = reference_predictions[
        key_columns
        + comparison_columns
    ].copy()

    right = generated_predictions[
        key_columns
        + comparison_columns
    ].copy()

    merged = left.merge(
        right,
        on=key_columns,
        how="outer",
        suffixes=(
            "_reference",
            "_generated",
        ),
        indicator=True,
    )

    require(
        bool(
            (
                merged["_merge"]
                == "both"
            ).all()
        ),
        "FULL source prediction keys do not match.",
    )

    mismatch_count = 0

    for column in comparison_columns:
        mismatch_count += int(
            (
                merged[
                    f"{column}_reference"
                ].astype(str)
                != merged[
                    f"{column}_generated"
                ].astype(str)
            ).sum()
        )

    require(
        mismatch_count == 0,
        (
            "FULL source predictions did not reproduce "
            f"exactly. mismatch_count={mismatch_count}"
        ),
    )

    return {
        "maximum_metric_difference":
            maximum_metric_difference,

        "prediction_mismatch_count":
            mismatch_count,

        "prediction_count":
            len(
                generated_predictions
            ),
    }


def extract_sionna_locked_assets(
    destination: Path,
):
    members = {
        "features.npz":
            "results/sionna_formal_test840_locked_phyguard/features.npz",

        "metadata.csv":
            "results/sionna_formal_test840_locked_phyguard/metadata.csv",

        "predictions.csv":
            "results/sionna_formal_test840_locked_phyguard/predictions.csv",

        "per_repeat_metrics.csv":
            "results/sionna_formal_test840_locked_phyguard/per_repeat_metrics.csv",

        "summary.json":
            "results/sionna_formal_test840_locked_phyguard/summary.json",
    }

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        LOCKED_ARCHIVE,
        "r",
    ) as archive:
        available = set(
            archive.namelist()
        )

        for target_name, member in members.items():
            require(
                member in available,
                (
                    "Locked archive member missing: "
                    f"{member}"
                ),
            )

            target = (
                destination
                / target_name
            )

            target.write_bytes(
                archive.read(
                    member
                )
            )

    return {
        name:
            destination
            / name
        for name in members
    }


def selective_metrics_local(
    truth,
    prediction,
):
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
        list(
            PHYSICAL_LABELS
        ),
    )

    control = ~physical

    emitted = (
        prediction
        != ABSTAIN
    )

    selected_physical = (
        physical
        & emitted
    )

    selected_count = int(
        selected_physical.sum()
    )

    selected_correct = int(
        (
            selected_physical
            & (
                prediction
                == truth
            )
        ).sum()
    )

    physical_count = int(
        physical.sum()
    )

    control_count = int(
        control.sum()
    )

    false_specific_count = int(
        (
            control
            & emitted
        ).sum()
    )

    exact_physical_count = int(
        (
            physical
            & (
                prediction
                == truth
            )
        ).sum()
    )

    selected_accuracy = (
        selected_correct
        / selected_count
        if selected_count > 0
        else math.nan
    )

    physical_selection_rate = (
        selected_count
        / physical_count
        if physical_count > 0
        else math.nan
    )

    false_specific_rate = (
        false_specific_count
        / control_count
        if control_count > 0
        else math.nan
    )

    physical_exact_match = (
        exact_physical_count
        / physical_count
        if physical_count > 0
        else math.nan
    )

    control_abstention = (
        1.0
        - false_specific_rate
        if math.isfinite(
            false_specific_rate
        )
        else math.nan
    )

    return {
        "selected_accuracy":
            float(
                selected_accuracy
            ),

        "physical_selection_rate":
            float(
                physical_selection_rate
            ),

        "coverage":
            float(
                physical_selection_rate
            ),

        "false_specific_rate":
            float(
                false_specific_rate
            ),

        "physical_exact_match":
            float(
                physical_exact_match
            ),

        "control_abstention":
            float(
                control_abstention
            ),

        "selected_physical_count":
            selected_count,

        "physical_count":
            physical_count,

        "control_count":
            control_count,
    }


def build_sionna_predictions(
    variant: str,
    source_eval_dir: Path,
    raw: np.ndarray,
    physical: np.ndarray,
    combined: np.ndarray,
    labels: np.ndarray,
    two_stage_probabilities,
):
    metrics_path = (
        source_eval_dir
        / "per_repeat_all_methods.csv"
    )

    metrics_table = pd.read_csv(
        metrics_path
    )

    metrics_table = metrics_table[
        method_mask(
            metrics_table[
                "method"
            ]
        )
    ].copy()

    metrics_table = metrics_table.sort_values(
        "repeat"
    )

    require(
        len(metrics_table)
        == EXPECTED[
            "repeat_count"
        ],
        (
            "Source threshold row count mismatch "
            f"for {variant}."
        ),
    )

    rows = []
    repeat_metrics = []

    for _, metric_row in metrics_table.iterrows():
        repeat = int(
            metric_row[
                "repeat"
            ]
        )

        model_path = (
            source_eval_dir
            / "models"
            / f"repeat_{repeat}_phyguard.joblib"
        )

        bundle = joblib.load(
            model_path
        )

        require(
            isinstance(
                bundle,
                dict,
            )
            and "gate" in bundle
            and "resolver" in bundle,
            (
                "Unexpected model bundle format: "
                f"{model_path}"
            ),
        )

        gate_probability, mechanism_confidence, mechanism_label = (
            two_stage_probabilities(
                bundle["gate"],
                bundle["resolver"],
                raw,
                physical,
                combined,
                True,
            )
        )

        tau_g = float(
            metric_row[
                "tau_g"
            ]
        )

        tau_c = float(
            metric_row[
                "tau_c"
            ]
        )

        prediction = np.where(
            (
                gate_probability
                >= tau_g
            )
            & (
                mechanism_confidence
                >= tau_c
            ),
            mechanism_label,
            ABSTAIN,
        )

        metrics = selective_metrics_local(
            labels,
            prediction,
        )

        repeat_metrics.append(
            {
                "system":
                    "sionna",

                "variant":
                    variant,

                "repeat":
                    repeat,

                "tau_g":
                    tau_g,

                "tau_c":
                    tau_c,

                **metrics,
            }
        )

        for index in range(
            len(labels)
        ):
            rows.append(
                {
                    "system":
                        "sionna",

                    "variant":
                        variant,

                    "repeat":
                        repeat,

                    "index":
                        index,

                    "truth":
                        str(
                            labels[
                                index
                            ]
                        ),

                    "prediction":
                        str(
                            prediction[
                                index
                            ]
                        ),

                    "gate_confidence":
                        float(
                            gate_probability[
                                index
                            ]
                        ),

                    "mechanism_confidence":
                        float(
                            mechanism_confidence[
                                index
                            ]
                        ),
                }
            )

    return (
        pd.DataFrame(
            rows
        ),
        pd.DataFrame(
            repeat_metrics
        ),
    )


def compare_sionna_full_parity(
    generated: pd.DataFrame,
    locked_predictions_path: Path,
):
    locked = pd.read_csv(
        locked_predictions_path
    )

    if "protocol" in locked.columns:
        protocol_values = (
            locked[
                "protocol"
            ]
            .astype(str)
            .str.lower()
        )

        source_fixed = locked[
            protocol_values.str.contains(
                "source_fixed",
                regex=False,
            )
        ].copy()

        if len(source_fixed) > 0:
            locked = source_fixed

    require(
        len(locked)
        == (
            EXPECTED[
                "sample_count_sionna"
            ]
            * EXPECTED[
                "repeat_count"
            ]
        ),
        (
            "Locked Sionna prediction count mismatch: "
            f"{len(locked)}"
        ),
    )

    if "index" not in locked.columns:
        if "sample_index" in locked.columns:
            locked = locked.rename(
                columns={
                    "sample_index":
                        "index"
                }
            )
        else:
            locked["index"] = (
                locked.groupby(
                    "repeat"
                ).cumcount()
            )

    if "prediction" not in locked.columns:
        candidate = next(
            (
                column
                for column in [
                    "pred",
                    "predicted_label",
                ]
                if column in locked.columns
            ),
            None,
        )

        require(
            candidate is not None,
            "Locked prediction label column is missing.",
        )

        locked = locked.rename(
            columns={
                candidate:
                    "prediction"
            }
        )

    merged = locked[
        [
            "repeat",
            "index",
            "prediction",
        ]
    ].merge(
        generated[
            [
                "repeat",
                "index",
                "prediction",
            ]
        ],
        on=[
            "repeat",
            "index",
        ],
        how="outer",
        suffixes=(
            "_locked",
            "_generated",
        ),
        indicator=True,
    )

    require(
        bool(
            (
                merged["_merge"]
                == "both"
            ).all()
        ),
        "FULL Sionna prediction keys do not match.",
    )

    mismatches = int(
        (
            merged[
                "prediction_locked"
            ].astype(str)
            != merged[
                "prediction_generated"
            ].astype(str)
        ).sum()
    )

    require(
        mismatches == 0,
        (
            "FULL Sionna predictions do not reproduce "
            f"the locked result: mismatches={mismatches}"
        ),
    )

    return {
        "prediction_count":
            len(
                generated
            ),

        "prediction_mismatch_count":
            mismatches,
    }


def source_prediction_table(
    variant: str,
    evaluation_dir: Path,
):
    path = (
        evaluation_dir
        / "predictions"
        / "all_test_predictions.csv"
    )

    frame = pd.read_csv(
        path
    )

    frame = frame[
        method_mask(
            frame[
                "method"
            ]
        )
    ].copy()

    required = [
        "repeat",
        "index",
        "truth",
        "prediction",
    ]

    for column in required:
        require(
            column in frame.columns,
            (
                "Source prediction column missing: "
                f"{column}"
            ),
        )

    frame.insert(
        0,
        "system",
        "source",
    )

    frame.insert(
        1,
        "variant",
        variant,
    )

    return frame


def source_repeat_metrics(
    variant: str,
    evaluation_dir: Path,
):
    path = (
        evaluation_dir
        / "per_repeat_all_methods.csv"
    )

    frame = pd.read_csv(
        path
    )

    frame = frame[
        method_mask(
            frame[
                "method"
            ]
        )
    ].copy()

    frame.insert(
        0,
        "system",
        "source",
    )

    frame.insert(
        1,
        "variant",
        variant,
    )

    return frame


def cluster_sufficient_statistics(
    frame: pd.DataFrame,
):
    frame = frame.copy()

    frame["truth"] = frame[
        "truth"
    ].astype(str)

    frame["prediction"] = frame[
        "prediction"
    ].astype(str)

    physical = frame[
        "truth"
    ].isin(
        PHYSICAL_LABELS
    )

    emitted = (
        frame[
            "prediction"
        ]
        != ABSTAIN
    )

    frame["_denominator"] = (
        physical
        & emitted
    ).astype(
        np.float64
    )

    frame["_numerator"] = (
        physical
        & emitted
        & (
            frame[
                "prediction"
            ]
            == frame[
                "truth"
            ]
        )
    ).astype(
        np.float64
    )

    grouped = (
        frame.groupby(
            "index",
            sort=True,
        )[
            [
                "_numerator",
                "_denominator",
            ]
        ]
        .sum()
    )

    return grouped


def bootstrap_primary_delta(
    full_frame: pd.DataFrame,
    variant_frame: pd.DataFrame,
    seed: int,
    replicates: int,
):
    full_stats = (
        cluster_sufficient_statistics(
            full_frame
        )
    )

    variant_stats = (
        cluster_sufficient_statistics(
            variant_frame
        )
    )

    cluster_ids = sorted(
        set(
            full_stats.index
        )
        | set(
            variant_stats.index
        )
    )

    full_stats = full_stats.reindex(
        cluster_ids,
        fill_value=0.0,
    )

    variant_stats = variant_stats.reindex(
        cluster_ids,
        fill_value=0.0,
    )

    full_numerator = full_stats[
        "_numerator"
    ].to_numpy(
        dtype=np.float64
    )

    full_denominator = full_stats[
        "_denominator"
    ].to_numpy(
        dtype=np.float64
    )

    variant_numerator = variant_stats[
        "_numerator"
    ].to_numpy(
        dtype=np.float64
    )

    variant_denominator = variant_stats[
        "_denominator"
    ].to_numpy(
        dtype=np.float64
    )

    number_of_clusters = len(
        cluster_ids
    )

    require(
        number_of_clusters > 0,
        "No clusters available for bootstrap.",
    )

    rng = np.random.default_rng(
        seed
    )

    differences = np.empty(
        replicates,
        dtype=np.float64,
    )

    valid = np.zeros(
        replicates,
        dtype=bool,
    )

    for replicate in range(
        replicates
    ):
        sample = rng.integers(
            0,
            number_of_clusters,
            size=number_of_clusters,
        )

        weights = np.bincount(
            sample,
            minlength=number_of_clusters,
        ).astype(
            np.float64
        )

        full_den = float(
            weights
            @ full_denominator
        )

        variant_den = float(
            weights
            @ variant_denominator
        )

        if (
            full_den <= 0.0
            or variant_den <= 0.0
        ):
            differences[
                replicate
            ] = math.nan
            continue

        full_metric = float(
            (
                weights
                @ full_numerator
            )
            / full_den
        )

        variant_metric = float(
            (
                weights
                @ variant_numerator
            )
            / variant_den
        )

        differences[
            replicate
        ] = (
            variant_metric
            - full_metric
        )

        valid[
            replicate
        ] = True

    finite = differences[
        valid
    ]

    require(
        len(finite)
        >= int(
            replicates
            * 0.99
        ),
        (
            "Too many invalid bootstrap replicates: "
            f"{len(finite)}/{replicates}"
        ),
    )

    return {
        "replicates_requested":
            replicates,

        "replicates_valid":
            int(
                len(finite)
            ),

        "mean":
            float(
                np.mean(
                    finite
                )
            ),

        "lower_95":
            float(
                np.quantile(
                    finite,
                    0.025,
                )
            ),

        "upper_95":
            float(
                np.quantile(
                    finite,
                    0.975,
                )
            ),

        "distribution":
            finite,
    }


def per_repeat_primary(
    frame: pd.DataFrame,
):
    rows = []

    for repeat, group in frame.groupby(
        "repeat",
        sort=True,
    ):
        metrics = selective_metrics_local(
            group[
                "truth"
            ].to_numpy(
                dtype=object
            ),
            group[
                "prediction"
            ].to_numpy(
                dtype=object
            ),
        )

        rows.append(
            {
                "repeat":
                    int(
                        repeat
                    ),

                "selected_accuracy":
                    metrics[
                        "selected_accuracy"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


def paired_repeat_delta(
    full_frame: pd.DataFrame,
    variant_frame: pd.DataFrame,
):
    full = per_repeat_primary(
        full_frame
    ).rename(
        columns={
            "selected_accuracy":
                "full_selected_accuracy"
        }
    )

    variant = per_repeat_primary(
        variant_frame
    ).rename(
        columns={
            "selected_accuracy":
                "variant_selected_accuracy"
        }
    )

    paired = full.merge(
        variant,
        on="repeat",
        how="inner",
        validate="one_to_one",
    )

    require(
        len(paired)
        == EXPECTED[
            "repeat_count"
        ],
        "Paired repeat count mismatch.",
    )

    paired[
        "delta"
    ] = (
        paired[
            "variant_selected_accuracy"
        ]
        - paired[
            "full_selected_accuracy"
        ]
    )

    return paired


def compute_factorial_effects(
    system_frames: dict[str, pd.DataFrame],
):
    metrics = {
        variant:
            selective_metrics_local(
                frame[
                    "truth"
                ].to_numpy(
                    dtype=object
                ),
                frame[
                    "prediction"
                ].to_numpy(
                    dtype=object
                ),
            )[
                "selected_accuracy"
            ]
        for variant, frame in system_frames.items()
    }

    full = metrics[
        "FULL"
    ]

    no_raw = metrics[
        "NO_PER_RAW"
    ]

    no_score = metrics[
        "NO_PER_SCORE"
    ]

    neither = metrics[
        "NO_ERROR_INDICATOR"
    ]

    raw_presence_effect = (
        0.5
        * (
            full
            + no_score
        )
        - 0.5
        * (
            no_raw
            + neither
        )
    )

    score_presence_effect = (
        0.5
        * (
            full
            + no_raw
        )
        - 0.5
        * (
            no_score
            + neither
        )
    )

    interaction = (
        full
        - no_raw
        - no_score
        + neither
    )

    return {
        "raw_path_presence_main_effect":
            float(
                raw_presence_effect
            ),

        "score_path_presence_main_effect":
            float(
                score_presence_effect
            ),

        "interaction_difference_in_differences":
            float(
                interaction
            ),
    }


def classify_result(
    retention: dict[str, dict[str, bool]],
):
    source_pattern = tuple(
        retention[
            variant
        ][
            "source"
        ]
        for variant in VARIANTS[
            1:
        ]
    )

    sionna_pattern = tuple(
        retention[
            variant
        ][
            "sionna"
        ]
        for variant in VARIANTS[
            1:
        ]
    )

    no_error_both = (
        retention[
            "NO_ERROR_INDICATOR"
        ][
            "source"
        ]
        and retention[
            "NO_ERROR_INDICATOR"
        ][
            "sionna"
        ]
    )

    no_raw_both = (
        retention[
            "NO_PER_RAW"
        ][
            "source"
        ]
        and retention[
            "NO_PER_RAW"
        ][
            "sionna"
        ]
    )

    no_score_both = (
        retention[
            "NO_PER_SCORE"
        ][
            "source"
        ]
        and retention[
            "NO_PER_SCORE"
        ][
            "sionna"
        ]
    )

    if no_error_both:
        return (
            "PACKET_BLOCK_ERROR_INDICATOR_NOT_ESSENTIAL"
        )

    if (
        source_pattern
        != sionna_pattern
    ):
        return (
            "MIXED_OR_INCONCLUSIVE"
        )

    if (
        no_raw_both
        != no_score_both
    ):
        return (
            "PATH_SPECIFIC_DEPENDENCE"
        )

    if (
        no_raw_both
        and no_score_both
    ):
        return (
            "DISTRIBUTED_DEPENDENCE"
        )

    if (
        not no_raw_both
        and not no_score_both
    ):
        return (
            "STRONG_INDICATOR_DEPENDENCE"
        )

    return (
        "MIXED_OR_INCONCLUSIVE"
    )


# ------------------------------------------------------------------
# Preflight
# ------------------------------------------------------------------

required_paths = [
    SOURCE_DATA_DIR,
    SOURCE_FEATURE_DIR / "features.npz",
    SOURCE_REFERENCE_EVAL_DIR,
    FEATURE_BUILDER_PATH,
    EVALUATE_SUITE_PATH,
    EVALUATION_COMMON_PATH,
    LOCKED_ARCHIVE,
    PROTOCOL_PATH,
    PROTOCOL_MANIFEST_PATH,
    AUDIT_MANIFEST_PATH,
]

for path in required_paths:
    require(
        path.exists(),
        f"Required path is missing: {path}",
    )

require(
    sha256_file(
        LOCKED_ARCHIVE
    )
    == EXPECTED[
        "locked_archive_sha256"
    ],
    "Locked reconstruction archive checksum mismatch.",
)

protocol = json.loads(
    PROTOCOL_PATH.read_text(
        encoding="utf-8"
    )
)

protocol_manifest = json.loads(
    PROTOCOL_MANIFEST_PATH.read_text(
        encoding="utf-8"
    )
)

require(
    protocol.get("status")
    == EXPECTED[
        "protocol_status"
    ],
    "Stage 62A protocol status mismatch.",
)

require(
    protocol_manifest.get(
        "protocol_sha256"
    )
    == sha256_file(
        PROTOCOL_PATH
    ),
    "Stage 62A protocol checksum mismatch.",
)

require(
    not OUTPUT_ROOT.exists(),
    f"Result output already exists: {OUTPUT_ROOT}",
)

require(
    not RESULT_MANIFEST_PATH.exists(),
    f"Result manifest already exists: {RESULT_MANIFEST_PATH}",
)

if WORK_ROOT.exists():
    shutil.rmtree(
        WORK_ROOT
    )

WORK_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)


# ------------------------------------------------------------------
# Exact runtime implementation audit and mask discovery
# ------------------------------------------------------------------

feature_module = load_module(
    "phyguard_features_stage62b",
    FEATURE_BUILDER_PATH,
)

require(
    hasattr(
        feature_module,
        "build_one",
    ),
    "features.py does not expose build_one.",
)

feature_mask = (
    discover_packet_error_feature_masks(
        feature_module.build_one
    )
)

raw_drop_indices = feature_mask[
    "raw_indices"
]

physical_zero_indices = feature_mask[
    "physical_indices"
]

(
    OUTPUT_ROOT
    / "feature_mask.json"
).write_text(
    json.dumps(
        feature_mask,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ),
    encoding="utf-8",
)

print(
    "FEATURE MASK",
    flush=True,
)
print(
    "build_one_signature:",
    feature_mask[
        "build_one_signature"
    ],
    flush=True,
)
print(
    "call_mode:",
    feature_mask[
        "call_mode"
    ],
    flush=True,
)
print(
    "raw_drop_indices:",
    raw_drop_indices,
    flush=True,
)
print(
    "physical_zero_indices:",
    physical_zero_indices,
    flush=True,
)


# ------------------------------------------------------------------
# Load original feature matrices and build the four variants
# ------------------------------------------------------------------

source_features = np.load(
    SOURCE_FEATURE_DIR
    / "features.npz"
)

source_raw = np.asarray(
    source_features[
        "raw"
    ]
)

source_physical = np.asarray(
    source_features[
        "physical"
    ]
)

require(
    source_raw.shape[1]
    == EXPECTED[
        "source_raw_dimension"
    ],
    "Source raw feature dimension mismatch.",
)

require(
    source_physical.shape[1]
    == EXPECTED[
        "source_physical_dimension"
    ],
    "Source physical feature dimension mismatch.",
)

sionna_input_dir = (
    WORK_ROOT
    / "sionna_locked_inputs"
)

sionna_assets = (
    extract_sionna_locked_assets(
        sionna_input_dir
    )
)

sionna_features = np.load(
    sionna_assets[
        "features.npz"
    ]
)

sionna_raw = np.asarray(
    sionna_features[
        "raw"
    ]
)

sionna_physical = np.asarray(
    sionna_features[
        "physical"
    ]
)

sionna_metadata = pd.read_csv(
    sionna_assets[
        "metadata.csv"
    ]
)

require(
    len(
        sionna_metadata
    )
    == EXPECTED[
        "sample_count_sionna"
    ],
    "Sionna metadata sample count mismatch.",
)

require(
    sionna_raw.shape
    == (
        EXPECTED[
            "sample_count_sionna"
        ],
        EXPECTED[
            "source_raw_dimension"
        ],
    ),
    "Sionna raw feature shape mismatch.",
)

require(
    sionna_physical.shape
    == (
        EXPECTED[
            "sample_count_sionna"
        ],
        EXPECTED[
            "source_physical_dimension"
        ],
    ),
    "Sionna physical feature shape mismatch.",
)

label_column = next(
    (
        column
        for column in [
            "label",
            "dataset_label",
            "truth",
        ]
        if column in sionna_metadata.columns
    ),
    None,
)

require(
    label_column is not None,
    "Sionna metadata label column is missing.",
)

sionna_labels = (
    sionna_metadata[
        label_column
    ]
    .astype(str)
    .to_numpy(
        dtype=object
    )
)

variant_shapes = {}
variant_source_feature_dirs = {}
variant_sionna_features = {}

for variant in VARIANTS:
    (
        source_variant_raw,
        source_variant_physical,
        source_variant_combined,
    ) = build_variant_features(
        source_raw,
        source_physical,
        variant,
        raw_drop_indices,
        physical_zero_indices,
    )

    source_variant_dir = (
        WORK_ROOT
        / "source_features"
        / variant
    )

    source_variant_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    np.savez_compressed(
        source_variant_dir
        / "features.npz",
        raw=source_variant_raw,
        physical=source_variant_physical,
        combined=source_variant_combined,
    )

    variant_source_feature_dirs[
        variant
    ] = source_variant_dir

    (
        sionna_variant_raw,
        sionna_variant_physical,
        sionna_variant_combined,
    ) = build_variant_features(
        sionna_raw,
        sionna_physical,
        variant,
        raw_drop_indices,
        physical_zero_indices,
    )

    variant_sionna_features[
        variant
    ] = (
        sionna_variant_raw,
        sionna_variant_physical,
        sionna_variant_combined,
    )

    variant_shapes[
        variant
    ] = {
        "raw_dimension":
            int(
                source_variant_raw.shape[
                    1
                ]
            ),

        "physical_dimension":
            int(
                source_variant_physical.shape[
                    1
                ]
            ),

        "combined_dimension":
            int(
                source_variant_combined.shape[
                    1
                ]
            ),
    }


# ------------------------------------------------------------------
# Source-domain refitting and validation-only threshold selection
# ------------------------------------------------------------------

source_evaluation_dirs = {}
source_logs = {}

for variant in VARIANTS:
    print(
        f"SOURCE TRAINING START {variant}",
        flush=True,
    )

    evaluation_dir = (
        WORK_ROOT
        / "source_evaluation"
        / variant
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    log_path = run_evaluate_suite(
        variant_source_feature_dirs[
            variant
        ],
        evaluation_dir,
    )

    source_evaluation_dirs[
        variant
    ] = evaluation_dir

    source_logs[
        variant
    ] = log_path

    print(
        f"SOURCE TRAINING PASS {variant}",
        flush=True,
    )


source_full_parity = (
    compare_source_full_parity(
        source_evaluation_dirs[
            "FULL"
        ]
    )
)

print(
    "SOURCE FULL PARITY PASS",
    source_full_parity,
    flush=True,
)


# ------------------------------------------------------------------
# Sionna inference using each variant's source-trained models/thresholds
# ------------------------------------------------------------------

evaluate_suite_module = load_module(
    "phyguard_evaluate_suite_stage62b",
    EVALUATE_SUITE_PATH,
)

require(
    hasattr(
        evaluate_suite_module,
        "two_stage_probabilities",
    ),
    "evaluate_suite.py lacks two_stage_probabilities.",
)

all_source_predictions = []
all_source_repeat_metrics = []
all_sionna_predictions = []
all_sionna_repeat_metrics = []

system_frames = {
    "source": {},
    "sionna": {},
}

for variant in VARIANTS:
    source_predictions = (
        source_prediction_table(
            variant,
            source_evaluation_dirs[
                variant
            ],
        )
    )

    source_metrics = (
        source_repeat_metrics(
            variant,
            source_evaluation_dirs[
                variant
            ],
        )
    )

    all_source_predictions.append(
        source_predictions
    )

    all_source_repeat_metrics.append(
        source_metrics
    )

    system_frames[
        "source"
    ][
        variant
    ] = source_predictions

    (
        variant_raw,
        variant_physical,
        variant_combined,
    ) = variant_sionna_features[
        variant
    ]

    (
        sionna_predictions,
        sionna_metrics,
    ) = build_sionna_predictions(
        variant,
        source_evaluation_dirs[
            variant
        ],
        variant_raw,
        variant_physical,
        variant_combined,
        sionna_labels,
        evaluate_suite_module.two_stage_probabilities,
    )

    all_sionna_predictions.append(
        sionna_predictions
    )

    all_sionna_repeat_metrics.append(
        sionna_metrics
    )

    system_frames[
        "sionna"
    ][
        variant
    ] = sionna_predictions


sionna_full_parity = (
    compare_sionna_full_parity(
        system_frames[
            "sionna"
        ][
            "FULL"
        ],
        sionna_assets[
            "predictions.csv"
        ],
    )
)

print(
    "SIONNA FULL PARITY PASS",
    sionna_full_parity,
    flush=True,
)


source_predictions_all = pd.concat(
    all_source_predictions,
    ignore_index=True,
)

source_repeat_metrics_all = pd.concat(
    all_source_repeat_metrics,
    ignore_index=True,
)

sionna_predictions_all = pd.concat(
    all_sionna_predictions,
    ignore_index=True,
)

sionna_repeat_metrics_all = pd.concat(
    all_sionna_repeat_metrics,
    ignore_index=True,
)

source_predictions_all.to_csv(
    OUTPUT_ROOT
    / "source_predictions_all_variants.csv",
    index=False,
)

source_repeat_metrics_all.to_csv(
    OUTPUT_ROOT
    / "source_per_repeat_metrics_all_variants.csv",
    index=False,
)

sionna_predictions_all.to_csv(
    OUTPUT_ROOT
    / "sionna_predictions_all_variants.csv",
    index=False,
)

sionna_repeat_metrics_all.to_csv(
    OUTPUT_ROOT
    / "sionna_per_repeat_metrics_all_variants.csv",
    index=False,
)


# ------------------------------------------------------------------
# Paired deltas, bootstrap intervals, and factorial effects
# ------------------------------------------------------------------

delta_rows = []
paired_repeat_rows = []
bootstrap_distributions = {}
retention = {
    variant: {}
    for variant in VARIANTS[
        1:
    ]
}

for system in [
    "source",
    "sionna",
]:
    full_frame = system_frames[
        system
    ][
        "FULL"
    ]

    seed = (
        EXPECTED[
            "source_bootstrap_seed"
        ]
        if system == "source"
        else EXPECTED[
            "sionna_bootstrap_seed"
        ]
    )

    full_metric = selective_metrics_local(
        full_frame[
            "truth"
        ].to_numpy(
            dtype=object
        ),
        full_frame[
            "prediction"
        ].to_numpy(
            dtype=object
        ),
    )[
        "selected_accuracy"
    ]

    for variant_index, variant in enumerate(
        VARIANTS[
            1:
        ],
        start=1,
    ):
        variant_frame = system_frames[
            system
        ][
            variant
        ]

        variant_metric = selective_metrics_local(
            variant_frame[
                "truth"
            ].to_numpy(
                dtype=object
            ),
            variant_frame[
                "prediction"
            ].to_numpy(
                dtype=object
            ),
        )[
            "selected_accuracy"
        ]

        bootstrap = bootstrap_primary_delta(
            full_frame,
            variant_frame,
            seed=(
                seed
                + variant_index
                * 101
            ),
            replicates=EXPECTED[
                "bootstrap_replicates"
            ],
        )

        bootstrap_distributions[
            f"{system}__{variant}"
        ] = bootstrap.pop(
            "distribution"
        )

        retained = (
            bootstrap[
                "lower_95"
            ]
            > EXPECTED[
                "retention_margin"
            ]
        )

        retention[
            variant
        ][
            system
        ] = bool(
            retained
        )

        paired = paired_repeat_delta(
            full_frame,
            variant_frame,
        )

        for _, row in paired.iterrows():
            paired_repeat_rows.append(
                {
                    "system":
                        system,

                    "variant":
                        variant,

                    "repeat":
                        int(
                            row[
                                "repeat"
                            ]
                        ),

                    "full_selected_accuracy":
                        float(
                            row[
                                "full_selected_accuracy"
                            ]
                        ),

                    "variant_selected_accuracy":
                        float(
                            row[
                                "variant_selected_accuracy"
                            ]
                        ),

                    "delta":
                        float(
                            row[
                                "delta"
                            ]
                        ),
                }
            )

        delta_rows.append(
            {
                "system":
                    system,

                "variant":
                    variant,

                "full_selected_accuracy":
                    float(
                        full_metric
                    ),

                "variant_selected_accuracy":
                    float(
                        variant_metric
                    ),

                "absolute_delta":
                    float(
                        variant_metric
                        - full_metric
                    ),

                "paired_repeat_delta_mean":
                    float(
                        paired[
                            "delta"
                        ].mean()
                    ),

                "paired_repeat_delta_std":
                    float(
                        paired[
                            "delta"
                        ].std(
                            ddof=0
                        )
                    ),

                "bootstrap_mean":
                    bootstrap[
                        "mean"
                    ],

                "bootstrap_lower_95":
                    bootstrap[
                        "lower_95"
                    ],

                "bootstrap_upper_95":
                    bootstrap[
                        "upper_95"
                    ],

                "retention_margin":
                    EXPECTED[
                        "retention_margin"
                    ],

                "retained":
                    bool(
                        retained
                    ),

                "bootstrap_replicates_valid":
                    bootstrap[
                        "replicates_valid"
                    ],
            }
        )


delta_table = pd.DataFrame(
    delta_rows
)

paired_repeat_table = pd.DataFrame(
    paired_repeat_rows
)

delta_table.to_csv(
    OUTPUT_ROOT
    / "paired_primary_deltas.csv",
    index=False,
)

paired_repeat_table.to_csv(
    OUTPUT_ROOT
    / "paired_repeat_deltas.csv",
    index=False,
)

np.savez_compressed(
    OUTPUT_ROOT
    / "paired_bootstrap_distributions.npz",
    **bootstrap_distributions,
)


factorial_rows = []

for system in [
    "source",
    "sionna",
]:
    effects = compute_factorial_effects(
        system_frames[
            system
        ]
    )

    factorial_rows.append(
        {
            "system":
                system,

            **effects,
        }
    )

pd.DataFrame(
    factorial_rows
).to_csv(
    OUTPUT_ROOT
    / "factorial_effects.csv",
    index=False,
)


# ------------------------------------------------------------------
# Aggregate variant metrics and final classification
# ------------------------------------------------------------------

aggregate_rows = []

for system in [
    "source",
    "sionna",
]:
    for variant in VARIANTS:
        frame = system_frames[
            system
        ][
            variant
        ]

        metrics = selective_metrics_local(
            frame[
                "truth"
            ].to_numpy(
                dtype=object
            ),
            frame[
                "prediction"
            ].to_numpy(
                dtype=object
            ),
        )

        aggregate_rows.append(
            {
                "system":
                    system,

                "variant":
                    variant,

                **metrics,

                **variant_shapes[
                    variant
                ],
            }
        )

aggregate_table = pd.DataFrame(
    aggregate_rows
)

aggregate_table.to_csv(
    OUTPUT_ROOT
    / "aggregate_variant_metrics.csv",
    index=False,
)


classification = classify_result(
    retention
)

summary = {
    "schema":
        "phyguard.per_indicator_ablation_results.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "PASS",

    "protocol_status":
        protocol[
            "status"
        ],

    "variants":
        VARIANTS,

    "feature_mask":
        feature_mask,

    "variant_shapes":
        variant_shapes,

    "source_full_parity":
        source_full_parity,

    "sionna_full_parity":
        sionna_full_parity,

    "retention_margin":
        EXPECTED[
            "retention_margin"
        ],

    "bootstrap_replicates":
        EXPECTED[
            "bootstrap_replicates"
        ],

    "retention":
        retention,

    "classification":
        classification,

    "claim_boundary": {
        "ber_retained_in_all_variants":
            True,

        "evaluated_indicator":
            (
                "source per_proxy / Sionna bler_post_ldpc"
            ),

        "not_a_calibration_equivalence_test":
            True,

        "no_test_threshold_tuning":
            True,

        "source_models_refit_for_each_variant":
            True,

        "sionna_models_trained_on_sionna":
            False,
    },

    "files": {
        "aggregate_variant_metrics":
            "aggregate_variant_metrics.csv",

        "paired_primary_deltas":
            "paired_primary_deltas.csv",

        "paired_repeat_deltas":
            "paired_repeat_deltas.csv",

        "factorial_effects":
            "factorial_effects.csv",

        "source_predictions":
            "source_predictions_all_variants.csv",

        "sionna_predictions":
            "sionna_predictions_all_variants.csv",

        "bootstrap_distributions":
            "paired_bootstrap_distributions.npz",
    },
}

(
    OUTPUT_ROOT
    / "summary.json"
).write_text(
    json.dumps(
        json_safe(
            summary
        ),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
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
        "phyguard.per_indicator_ablation_results_manifest.v1",

    "status":
        "PASS",

    "protocol_sha256":
        sha256_file(
            PROTOCOL_PATH
        ),

    "implementation_audit_manifest_sha256":
        sha256_file(
            AUDIT_MANIFEST_PATH
        ),

    "feature_builder_sha256":
        sha256_file(
            FEATURE_BUILDER_PATH
        ),

    "evaluate_suite_sha256":
        sha256_file(
            EVALUATE_SUITE_PATH
        ),

    "evaluation_common_sha256":
        sha256_file(
            EVALUATION_COMMON_PATH
        ),

    "locked_archive_sha256":
        sha256_file(
            LOCKED_ARCHIVE
        ),

    "classification":
        classification,

    "files":
        manifest_files,
}

RESULT_MANIFEST_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_MANIFEST_PATH.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("AGGREGATE VARIANT METRICS")
print(
    aggregate_table.to_string(
        index=False
    ),
    flush=True,
)

print()
print("PAIRED PRIMARY DELTAS")
print(
    delta_table.to_string(
        index=False
    ),
    flush=True,
)

print()
print("FACTORIAL EFFECTS")
print(
    pd.DataFrame(
        factorial_rows
    ).to_string(
        index=False
    ),
    flush=True,
)

print()
print("RETENTION")
print(
    json.dumps(
        retention,
        ensure_ascii=False,
        indent=2,
    ),
    flush=True,
)

print()
print(
    "classification:",
    classification,
    flush=True,
)

print(
    "summary:",
    OUTPUT_ROOT
    / "summary.json",
    flush=True,
)

print(
    "manifest:",
    RESULT_MANIFEST_PATH,
    flush=True,
)

print()
print(
    "PER_INDICATOR_ABLATION_RESULTS_V1_PASS",
    flush=True,
)
