import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/phyguard_revision")

RESULT_ROOT = (
    ROOT
    / "results"
    / "sionna_formal_test840_locked_phyguard"
)

PREDICTIONS_PATH = (
    RESULT_ROOT / "predictions.csv"
)

PER_REPEAT_PATH = (
    RESULT_ROOT / "per_repeat_metrics.csv"
)

METADATA_PATH = (
    RESULT_ROOT / "metadata.csv"
)

INFERENCE_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_locked_reconstruction_inference.json"
)

EVALUATION_COMMON_PATH = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
    / "scripts"
    / "evaluation_common.py"
)

OUTPUT_SUMMARY = (
    RESULT_ROOT
    / "statistical_analysis.json"
)

OUTPUT_REPEAT_CI = (
    RESULT_ROOT
    / "repeat_level_confidence_intervals.csv"
)

OUTPUT_BOOTSTRAP_CI = (
    RESULT_ROOT
    / "sample_cluster_bootstrap_confidence_intervals.csv"
)

OUTPUT_STRATIFIED = (
    RESULT_ROOT
    / "stratified_metrics.csv"
)

OUTPUT_CONFUSION_COUNTS = (
    RESULT_ROOT
    / "physical_outcome_confusion_counts.csv"
)

OUTPUT_CONFUSION_NORMALIZED = (
    RESULT_ROOT
    / "physical_outcome_confusion_row_normalized.csv"
)

OUTPUT_SELECTED_CONFUSION = (
    RESULT_ROOT
    / "selected_physical_confusion_counts.csv"
)

OUTPUT_BOOTSTRAP_DRAWS = (
    RESULT_ROOT
    / "sample_cluster_bootstrap_draws.npz"
)

OUTPUT_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal825_locked_reconstruction_statistics.json"
)

EXPECTED_PREDICTION_COUNT = 4125
EXPECTED_SAMPLE_COUNT = 825
EXPECTED_REPEAT_COUNT = 5

BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20260730

T_CRITICAL_95_DF4 = 2.7764451051977987

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

KEY_METRICS = [
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


def import_module(path, name):
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


def bool_series(series):
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


def clean_float(value):
    value = float(value)

    if not np.isfinite(value):
        return None

    return value


def exact_metrics(
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

    values = {
        key: clean_float(value)
        for key, value in selective_metrics(
            truth,
            prediction,
        ).items()
    }

    values.update(
        {
            "physical_exact_diagnosis_rate":
                float(
                    np.mean(
                        prediction[
                            physical_mask
                        ]
                        == truth[
                            physical_mask
                        ]
                    )
                ),

            "physical_selection_rate":
                float(
                    np.mean(
                        selected[
                            physical_mask
                        ]
                    )
                ),

            "control_abstention_rate":
                float(
                    np.mean(
                        prediction[
                            control_mask
                        ]
                        == abstain
                    )
                ),

            "normal_abstention_rate":
                float(
                    np.mean(
                        prediction[
                            normal_mask
                        ]
                        == abstain
                    )
                ),

            "nonphysical_abstention_rate":
                float(
                    np.mean(
                        prediction[
                            nonphysical_mask
                        ]
                        == abstain
                    )
                ),
        }
    )

    return values


def percentile_ci(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    finite = values[
        np.isfinite(values)
    ]

    if len(finite) == 0:
        return {
            "lower_95": None,
            "upper_95": None,
        }

    return {
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
    }


def cluster_ci_for_column(
    group,
    value_column,
    rng,
):
    pivot = group.pivot(
        index="sample_id",
        columns="repeat",
        values=value_column,
    )

    if pivot.isna().any().any():
        raise RuntimeError(
            f"Missing repeat values for "
            f"{value_column}."
        )

    sample_values = (
        pivot.to_numpy(
            dtype=float
        ).mean(axis=1)
    )

    sample_count = len(
        sample_values
    )

    indices = rng.integers(
        0,
        sample_count,
        size=(
            BOOTSTRAP_REPETITIONS,
            sample_count,
        ),
    )

    draws = sample_values[
        indices
    ].mean(axis=1)

    interval = percentile_ci(
        draws
    )

    return {
        "point_estimate":
            float(
                sample_values.mean()
            ),

        "cluster_count":
            int(sample_count),

        **interval,
    }


for required in (
    PREDICTIONS_PATH,
    PER_REPEAT_PATH,
    METADATA_PATH,
    INFERENCE_MANIFEST_PATH,
    EVALUATION_COMMON_PATH,
):
    if not required.exists():
        raise FileNotFoundError(required)


predictions = pd.read_csv(
    PREDICTIONS_PATH
)

per_repeat = pd.read_csv(
    PER_REPEAT_PATH
)

metadata = pd.read_csv(
    METADATA_PATH
)

inference_manifest = json.loads(
    INFERENCE_MANIFEST_PATH.read_text(
        encoding="utf-8"
    )
)


if inference_manifest.get("status") != "PASS":
    raise RuntimeError(
        "Locked inference manifest is not PASS."
    )

if (
    inference_manifest.get(
        "prediction_count"
    )
    != EXPECTED_PREDICTION_COUNT
):
    raise RuntimeError(
        "Inference prediction count mismatch."
    )

if len(predictions) != EXPECTED_PREDICTION_COUNT:
    raise RuntimeError(
        f"Prediction rows={len(predictions)}, "
        f"expected {EXPECTED_PREDICTION_COUNT}."
    )

if predictions["sample_id"].nunique() != EXPECTED_SAMPLE_COUNT:
    raise RuntimeError(
        "Unique formal sample count is not 825."
    )

if predictions["repeat"].nunique() != EXPECTED_REPEAT_COUNT:
    raise RuntimeError(
        "Repeat count is not five."
    )

if len(per_repeat) != EXPECTED_REPEAT_COUNT:
    raise RuntimeError(
        "Per-repeat metric row count is not five."
    )

if len(metadata) != EXPECTED_SAMPLE_COUNT:
    raise RuntimeError(
        "Metadata row count is not 825."
    )


predictions["selected"] = bool_series(
    predictions["selected"]
)

predictions["correct_exact"] = bool_series(
    predictions["correct_exact"]
)

predictions["task_success"] = bool_series(
    predictions["task_success"]
)


evaluation_module = import_module(
    EVALUATION_COMMON_PATH,
    "phyguard_formal825_statistics_common",
)

selective_metrics = (
    evaluation_module.selective_metrics
)

ABSTAIN = str(
    evaluation_module.ABSTAIN
)


expected_sample_ids = sorted(
    predictions[
        predictions["repeat"] == 0
    ]["sample_id"].tolist()
)

if len(expected_sample_ids) != 825:
    raise RuntimeError(
        "Repeat zero does not contain 825 samples."
    )


aligned = {}

for repeat in range(
    EXPECTED_REPEAT_COUNT
):
    frame = (
        predictions[
            predictions["repeat"]
            == repeat
        ]
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    if frame["sample_id"].tolist() != expected_sample_ids:
        raise RuntimeError(
            f"Repeat {repeat} sample alignment mismatch."
        )

    aligned[repeat] = {
        "truth":
            frame["truth"]
            .to_numpy(dtype=object),

        "prediction":
            frame["prediction"]
            .to_numpy(dtype=object),
    }


recomputed_repeat_rows = []

for repeat in range(
    EXPECTED_REPEAT_COUNT
):
    values = exact_metrics(
        aligned[repeat]["truth"],
        aligned[repeat]["prediction"],
        ABSTAIN,
        selective_metrics,
    )

    recomputed_repeat_rows.append(
        {
            "repeat": repeat,
            **values,
        }
    )


recomputed_repeat = pd.DataFrame(
    recomputed_repeat_rows
)

stored_repeat = (
    per_repeat
    .sort_values("repeat")
    .reset_index(drop=True)
)

recomputed_repeat = (
    recomputed_repeat
    .sort_values("repeat")
    .reset_index(drop=True)
)


for metric in KEY_METRICS:
    stored = pd.to_numeric(
        stored_repeat[metric],
        errors="coerce",
    ).to_numpy(dtype=float)

    recomputed = pd.to_numeric(
        recomputed_repeat[metric],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.allclose(
        stored,
        recomputed,
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise RuntimeError(
            f"Stored/recomputed metric mismatch: "
            f"{metric}"
        )


repeat_ci_rows = []

for metric in KEY_METRICS:
    values = pd.to_numeric(
        stored_repeat[metric],
        errors="coerce",
    ).to_numpy(dtype=float)

    values = values[
        np.isfinite(values)
    ]

    mean = float(
        np.mean(values)
    )

    std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    standard_error = (
        std
        / np.sqrt(
            len(values)
        )
    )

    margin = (
        T_CRITICAL_95_DF4
        * standard_error
    )

    repeat_ci_rows.append(
        {
            "metric": metric,
            "repeat_count":
                int(len(values)),
            "mean": mean,
            "standard_deviation":
                std,
            "standard_error":
                standard_error,
            "t_critical_df4":
                T_CRITICAL_95_DF4,
            "lower_95":
                max(
                    0.0,
                    mean - margin,
                ),
            "upper_95":
                min(
                    1.0,
                    mean + margin,
                ),
            "interpretation":
                "Across five locked reconstruction repeats",
        }
    )


repeat_ci = pd.DataFrame(
    repeat_ci_rows
)

repeat_ci.to_csv(
    OUTPUT_REPEAT_CI,
    index=False,
)


rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

bootstrap_draws = {
    metric: np.empty(
        BOOTSTRAP_REPETITIONS,
        dtype=np.float64,
    )
    for metric in KEY_METRICS
}


sample_count = len(
    expected_sample_ids
)

print(
    "Starting sample-cluster bootstrap:",
    BOOTSTRAP_REPETITIONS,
    "replications",
)


for bootstrap_index in range(
    BOOTSTRAP_REPETITIONS
):
    sampled_indices = rng.integers(
        0,
        sample_count,
        size=sample_count,
    )

    repeat_values = {
        metric: []
        for metric in KEY_METRICS
    }

    for repeat in range(
        EXPECTED_REPEAT_COUNT
    ):
        values = exact_metrics(
            aligned[repeat][
                "truth"
            ][sampled_indices],

            aligned[repeat][
                "prediction"
            ][sampled_indices],

            ABSTAIN,
            selective_metrics,
        )

        for metric in KEY_METRICS:
            repeat_values[
                metric
            ].append(
                values[metric]
            )

    for metric in KEY_METRICS:
        bootstrap_draws[
            metric
        ][bootstrap_index] = (
            np.mean(
                repeat_values[
                    metric
                ]
            )
        )

    if (
        bootstrap_index + 1
    ) % 1000 == 0:
        print(
            "bootstrap_completed:",
            bootstrap_index + 1,
            "/",
            BOOTSTRAP_REPETITIONS,
        )


bootstrap_ci_rows = []

for metric in KEY_METRICS:
    point_estimate = float(
        stored_repeat[
            metric
        ].mean()
    )

    interval = percentile_ci(
        bootstrap_draws[
            metric
        ]
    )

    bootstrap_ci_rows.append(
        {
            "metric": metric,
            "point_estimate":
                point_estimate,
            "bootstrap_repetitions":
                BOOTSTRAP_REPETITIONS,
            "bootstrap_seed":
                BOOTSTRAP_SEED,
            "cluster_unit":
                "sample_id",
            "repeat_predictions_kept_together":
                True,
            **interval,
        }
    )


bootstrap_ci = pd.DataFrame(
    bootstrap_ci_rows
)

bootstrap_ci.to_csv(
    OUTPUT_BOOTSTRAP_CI,
    index=False,
)


np.savez_compressed(
    OUTPUT_BOOTSTRAP_DRAWS,
    **bootstrap_draws,
)


stratified_rng = (
    np.random.default_rng(
        BOOTSTRAP_SEED + 1
    )
)

stratified_rows = []


def append_groups(
    dimension,
    group_columns,
):
    grouped = predictions.groupby(
        group_columns,
        dropna=False,
        sort=True,
    )

    for keys, group in grouped:
        if not isinstance(
            keys,
            tuple,
        ):
            keys = (keys,)

        key_map = {
            column: value
            for column, value
            in zip(
                group_columns,
                keys,
            )
        }

        selected_count = int(
            group[
                "selected"
            ].sum()
        )

        selected_accuracy = None

        if selected_count > 0:
            selected_accuracy = float(
                group.loc[
                    group["selected"],
                    "correct_exact",
                ].mean()
            )

        selection_ci = (
            cluster_ci_for_column(
                group,
                "selected",
                stratified_rng,
            )
        )

        task_ci = (
            cluster_ci_for_column(
                group,
                "task_success",
                stratified_rng,
            )
        )

        exact_ci = (
            cluster_ci_for_column(
                group,
                "correct_exact",
                stratified_rng,
            )
        )

        row = {
            "dimension":
                dimension,

            "group":
                "|".join(
                    str(key_map[column])
                    for column
                    in group_columns
                ),

            "sample_count":
                int(
                    group[
                        "sample_id"
                    ].nunique()
                ),

            "prediction_count":
                int(len(group)),

            "selected_count":
                selected_count,

            "abstained_count":
                int(
                    len(group)
                    - selected_count
                ),

            "selection_rate":
                float(
                    group[
                        "selected"
                    ].mean()
                ),

            "selection_rate_ci_low":
                selection_ci[
                    "lower_95"
                ],

            "selection_rate_ci_high":
                selection_ci[
                    "upper_95"
                ],

            "abstention_rate":
                float(
                    1.0
                    - group[
                        "selected"
                    ].mean()
                ),

            "selected_accuracy":
                selected_accuracy,

            "exact_label_rate":
                float(
                    group[
                        "correct_exact"
                    ].mean()
                ),

            "exact_label_rate_ci_low":
                exact_ci[
                    "lower_95"
                ],

            "exact_label_rate_ci_high":
                exact_ci[
                    "upper_95"
                ],

            "task_success_rate":
                float(
                    group[
                        "task_success"
                    ].mean()
                ),

            "task_success_rate_ci_low":
                task_ci[
                    "lower_95"
                ],

            "task_success_rate_ci_high":
                task_ci[
                    "upper_95"
                ],

            "mean_gate_probability":
                float(
                    group[
                        "gate_probability"
                    ].mean()
                ),

            "mean_mechanism_confidence":
                float(
                    group[
                        "mechanism_confidence"
                    ].mean()
                ),
        }

        for column, value in (
            key_map.items()
        ):
            row[column] = value

        stratified_rows.append(
            row
        )


append_groups(
    "label",
    ["truth"],
)

append_groups(
    "severity",
    ["severity"],
)

append_groups(
    "channel",
    ["channel_model"],
)

append_groups(
    "label_x_severity",
    [
        "truth",
        "severity",
    ],
)

append_groups(
    "label_x_channel",
    [
        "truth",
        "channel_model",
    ],
)

append_groups(
    "severity_x_channel",
    [
        "severity",
        "channel_model",
    ],
)


stratified = pd.DataFrame(
    stratified_rows
)

stratified.to_csv(
    OUTPUT_STRATIFIED,
    index=False,
)


physical_predictions = predictions[
    predictions["truth"].isin(
        PHYSICAL_CLASSES
    )
].copy()


outcome_columns = (
    PHYSICAL_CLASSES
    + [ABSTAIN]
)

confusion_counts = pd.crosstab(
    physical_predictions["truth"],
    physical_predictions["prediction"],
)

confusion_counts = (
    confusion_counts
    .reindex(
        index=PHYSICAL_CLASSES,
        columns=outcome_columns,
        fill_value=0,
    )
)

confusion_counts.to_csv(
    OUTPUT_CONFUSION_COUNTS
)


row_denominator = (
    confusion_counts.sum(
        axis=1
    )
    .replace(
        0,
        np.nan,
    )
)

confusion_normalized = (
    confusion_counts.div(
        row_denominator,
        axis=0,
    )
)

confusion_normalized.to_csv(
    OUTPUT_CONFUSION_NORMALIZED
)


selected_physical = (
    physical_predictions[
        physical_predictions[
            "selected"
        ]
    ]
)

selected_confusion = pd.crosstab(
    selected_physical["truth"],
    selected_physical["prediction"],
)

selected_confusion = (
    selected_confusion
    .reindex(
        index=PHYSICAL_CLASSES,
        columns=PHYSICAL_CLASSES,
        fill_value=0,
    )
)

selected_confusion.to_csv(
    OUTPUT_SELECTED_CONFUSION
)


physical_severity = (
    stratified[
        (
            stratified[
                "dimension"
            ]
            == "severity"
        )
        & (
            stratified[
                "severity"
            ].isin(
                [
                    "mild",
                    "moderate",
                    "severe",
                ]
            )
        )
    ]
    .copy()
)


statistics_summary = {
    "schema":
        "phyguard.sionna.formal825."
        "locked_reconstruction.statistics.v1",

    "status":
        "PASS",

    "analysis_scope": {
        "sample_count":
            EXPECTED_SAMPLE_COUNT,
        "repeat_count":
            EXPECTED_REPEAT_COUNT,
        "prediction_count":
            EXPECTED_PREDICTION_COUNT,
        "bootstrap_repetitions":
            BOOTSTRAP_REPETITIONS,
        "bootstrap_seed":
            BOOTSTRAP_SEED,
        "bootstrap_cluster_unit":
            "sample_id",
    },

    "repeat_level_confidence_intervals":
        repeat_ci.to_dict(
            orient="records"
        ),

    "sample_cluster_bootstrap_intervals":
        bootstrap_ci.to_dict(
            orient="records"
        ),

    "label_summary":
        stratified[
            stratified[
                "dimension"
            ]
            == "label"
        ].to_dict(
            orient="records"
        ),

    "physical_severity_summary":
        physical_severity.to_dict(
            orient="records"
        ),

    "channel_summary":
        stratified[
            stratified[
                "dimension"
            ]
            == "channel"
        ].to_dict(
            orient="records"
        ),

    "interpretation_boundary": {
        "diagnosis_conditioned_on_locked_candidate_interval":
            True,

        "end_to_end_localization_claim":
            False,

        "model_training":
            False,

        "threshold_tuning":
            False,

        "threshold_calibration":
            False,

        "model_selection":
            False,

        "post_test_pipeline_modification":
            False,

        "controls_count_success_as_abstention":
            True,

        "control_exact_label_rate_not_applicable":
            True,
    },
}

OUTPUT_SUMMARY.write_text(
    json.dumps(
        statistics_summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


output_files = [
    OUTPUT_SUMMARY,
    OUTPUT_REPEAT_CI,
    OUTPUT_BOOTSTRAP_CI,
    OUTPUT_STRATIFIED,
    OUTPUT_CONFUSION_COUNTS,
    OUTPUT_CONFUSION_NORMALIZED,
    OUTPUT_SELECTED_CONFUSION,
    OUTPUT_BOOTSTRAP_DRAWS,
]


manifest = {
    "schema":
        "phyguard.sionna.formal825."
        "locked_reconstruction.statistics_manifest.v1",

    "status":
        "PASS",

    "source_files": {
        "predictions": {
            "path":
                str(
                    PREDICTIONS_PATH
                ),
            "sha256":
                sha256_file(
                    PREDICTIONS_PATH
                ),
        },

        "per_repeat_metrics": {
            "path":
                str(
                    PER_REPEAT_PATH
                ),
            "sha256":
                sha256_file(
                    PER_REPEAT_PATH
                ),
        },

        "metadata": {
            "path":
                str(
                    METADATA_PATH
                ),
            "sha256":
                sha256_file(
                    METADATA_PATH
                ),
        },

        "inference_manifest": {
            "path":
                str(
                    INFERENCE_MANIFEST_PATH
                ),
            "sha256":
                sha256_file(
                    INFERENCE_MANIFEST_PATH
                ),
        },
    },

    "integrity": {
        "stored_repeat_metrics_recomputed_exactly":
            True,

        "sample_alignment_across_repeats":
            True,

        "sample_count":
            EXPECTED_SAMPLE_COUNT,

        "repeat_count":
            EXPECTED_REPEAT_COUNT,

        "prediction_count":
            EXPECTED_PREDICTION_COUNT,
    },

    "bootstrap": {
        "method":
            "nonparametric sample-cluster percentile bootstrap",

        "cluster_unit":
            "sample_id",

        "repeat_predictions_kept_together":
            True,

        "repetitions":
            BOOTSTRAP_REPETITIONS,

        "seed":
            BOOTSTRAP_SEED,
    },

    "methodological_boundary": {
        "training_performed":
            False,

        "threshold_tuning_performed":
            False,

        "calibration_performed":
            False,

        "model_selection_performed":
            False,

        "new_predictions_generated":
            False,

        "locked_predictions_modified":
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

OUTPUT_MANIFEST.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print("status: PASS")
print("sample_count: 825")
print("repeat_count: 5")
print("prediction_count: 4125")
print(
    "bootstrap_repetitions:",
    BOOTSTRAP_REPETITIONS,
)
print(
    "stored_repeat_metrics_recomputed_exactly:",
    True,
)
print(
    "sample_alignment_across_repeats:",
    True,
)


print("\nREPEAT-LEVEL 95% t INTERVALS")

for row in repeat_ci.to_dict(
    orient="records"
):
    print(
        row["metric"],
        "mean=",
        f"{row['mean']:.8f}",
        "lower=",
        f"{row['lower_95']:.8f}",
        "upper=",
        f"{row['upper_95']:.8f}",
    )


print("\nSAMPLE-CLUSTER BOOTSTRAP 95% INTERVALS")

for row in bootstrap_ci.to_dict(
    orient="records"
):
    print(
        row["metric"],
        "estimate=",
        f"{row['point_estimate']:.8f}",
        "lower=",
        f"{row['lower_95']:.8f}",
        "upper=",
        f"{row['upper_95']:.8f}",
    )


print("\nLABEL SUMMARY")

label_view = stratified[
    stratified["dimension"]
    == "label"
][
    [
        "truth",
        "sample_count",
        "selection_rate",
        "selected_accuracy",
        "exact_label_rate",
        "task_success_rate",
        "task_success_rate_ci_low",
        "task_success_rate_ci_high",
    ]
]

print(
    label_view.to_string(
        index=False
    )
)


print("\nSEVERITY SUMMARY")

severity_view = stratified[
    stratified["dimension"]
    == "severity"
][
    [
        "severity",
        "sample_count",
        "selection_rate",
        "selected_accuracy",
        "exact_label_rate",
        "task_success_rate",
        "task_success_rate_ci_low",
        "task_success_rate_ci_high",
    ]
]

print(
    severity_view.to_string(
        index=False
    )
)


print("\nCHANNEL SUMMARY")

channel_view = stratified[
    stratified["dimension"]
    == "channel"
][
    [
        "channel_model",
        "sample_count",
        "selection_rate",
        "selected_accuracy",
        "exact_label_rate",
        "task_success_rate",
        "task_success_rate_ci_low",
        "task_success_rate_ci_high",
    ]
]

print(
    channel_view.to_string(
        index=False
    )
)


print("\nPHYSICAL OUTCOME CONFUSION COUNTS")

print(
    confusion_counts.to_string()
)


print("\nsummary:", OUTPUT_SUMMARY)
print(
    "repeat_ci:",
    OUTPUT_REPEAT_CI,
)
print(
    "bootstrap_ci:",
    OUTPUT_BOOTSTRAP_CI,
)
print(
    "stratified_metrics:",
    OUTPUT_STRATIFIED,
)
print(
    "manifest:",
    OUTPUT_MANIFEST,
)

print(
    "\nSIONNA_FORMAL825_LOCKED_"
    "RECONSTRUCTION_STATISTICS_PASS"
)
