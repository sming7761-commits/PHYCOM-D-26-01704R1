#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/phyguard_revision")

C2_ROOT = (
    ROOT
    / "artifacts"
    / "temporal_baseline_metric_repair_audit_v1"
)

C2_SUMMARY = C2_ROOT / "summary.json"
BASELINE_REPEAT_PATH = C2_ROOT / "corrected_per_repeat_metrics.csv"
BASELINE_PREDICTIONS_PATH = C2_ROOT / "corrected_predictions.csv"

C3_ROOT = (
    ROOT
    / "artifacts"
    / "phyguard_temporal_comparison_audit_v1"
)

C3_SUMMARY = C3_ROOT / "summary.json"
PHYGUARD_REPEAT_PATH = C3_ROOT / "phyguard_per_repeat_metrics.csv"

R0 = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
)

SUITE_PREDICTIONS_PATH = (
    R0
    / "results"
    / "full"
    / "evaluation_suite"
    / "predictions"
    / "all_test_predictions.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "phyguard_temporal_paired_uncertainty_v1"
)

BOOTSTRAP_PATH = OUTPUT_ROOT / "paired_cluster_bootstrap.csv"
REPEAT_DIFFERENCE_PATH = OUTPUT_ROOT / "paired_repeat_differences.csv"
POINT_PATH = OUTPUT_ROOT / "paired_point_estimates.csv"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
MANUSCRIPT_PATH = OUTPUT_ROOT / "MANUSCRIPT_READY_RESULTS.md"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "phyguard_temporal_paired_uncertainty_v1.json"
)

MODEL_ORDER = [
    "ROCKET",
    "MINIROCKET",
    "MULTIROCKET",
    "HYDRA",
    "MULTIROCKET_HYDRA",
    "ARSENAL",
    "FCN1D",
    "RESNET1D",
    "INCEPTIONTIME_LITE",
    "LITE1D",
    "TCN",
    "BIGRU",
    "BILSTM",
    "TINY_TRANSFORMER",
]

PHYSICAL_LABELS = {
    "interference",
    "blockage",
    "mobility",
    "adaptation_mismatch",
}

CONTROL_LABELS = {
    "normal",
    "nonphysical_goodput",
}

METRICS = [
    "selected_accuracy",
    "coverage",
    "false_specific_rate",
    "physical_exact_match",
    "control_abstention",
]

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 64041
BATCH_SIZE = 250


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    result = np.full(
        np.broadcast_shapes(
            numerator.shape,
            denominator.shape,
        ),
        np.nan,
        dtype=float,
    )

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator > 0,
    )

    return result


def prediction_sufficient_statistics(
    frame: pd.DataFrame,
    prediction_column: str,
    abstain_token: str,
    repeats: list[int],
    indices: list[int],
) -> np.ndarray:
    repeat_position = {
        repeat: position
        for position, repeat in enumerate(repeats)
    }

    index_position = {
        index: position
        for position, index in enumerate(indices)
    }

    statistics = np.zeros(
        (
            len(repeats),
            len(indices),
            7,
        ),
        dtype=np.float64,
    )

    for row in frame[
        [
            "repeat",
            "index",
            "truth",
            prediction_column,
        ]
    ].itertuples(index=False):
        repeat = int(row.repeat)
        index = int(row.index)
        truth = str(row.truth)
        prediction = str(
            getattr(row, prediction_column)
        )

        r = repeat_position[repeat]
        i = index_position[index]

        selected = prediction != abstain_token
        physical = truth in PHYSICAL_LABELS
        control = truth in CONTROL_LABELS
        correct = prediction == truth

        statistics[r, i, 0] += float(
            selected and correct
        )
        statistics[r, i, 1] += float(selected)
        statistics[r, i, 2] += float(
            physical and selected
        )
        statistics[r, i, 3] += float(physical)
        statistics[r, i, 4] += float(
            control and selected
        )
        statistics[r, i, 5] += float(control)
        statistics[r, i, 6] += float(
            physical and correct
        )

    return statistics


def metrics_from_sums(
    sums: np.ndarray,
) -> dict[str, np.ndarray]:
    # Locked evaluation semantics: selected accuracy is conditional
    # accuracy among selected physical cases. When no physical case
    # is selected, the helper returns 0.0 rather than NaN.
    selected_accuracy = np.zeros_like(
        sums[..., 6],
        dtype=float,
    )

    np.divide(
        sums[..., 6],
        sums[..., 2],
        out=selected_accuracy,
        where=sums[..., 2] > 0,
    )

    coverage = safe_ratio(
        sums[..., 2],
        sums[..., 3],
    )

    false_specific_rate = safe_ratio(
        sums[..., 4],
        sums[..., 5],
    )

    physical_exact_match = safe_ratio(
        sums[..., 6],
        sums[..., 3],
    )

    control_abstention = (
        1.0
        - false_specific_rate
    )

    return {
        "selected_accuracy":
            selected_accuracy,

        "coverage":
            coverage,

        "false_specific_rate":
            false_specific_rate,

        "physical_exact_match":
            physical_exact_match,

        "control_abstention":
            control_abstention,
    }


def mean_across_repeats(
    metric_values: np.ndarray,
) -> np.ndarray:
    with np.errstate(
        invalid="ignore",
    ):
        return np.nanmean(
            metric_values,
            axis=-1,
        )


def exact_signflip_p(
    differences: np.ndarray,
) -> float:
    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(
            differences
        )
    ]

    require(
        len(differences) > 0,
        "No finite repeat differences for sign-flip test.",
    )

    observed = abs(
        float(
            np.mean(
                differences
            )
        )
    )

    values = []

    for signs in itertools.product(
        [-1.0, 1.0],
        repeat=len(
            differences
        ),
    ):
        values.append(
            abs(
                float(
                    np.mean(
                        differences
                        * np.asarray(
                            signs,
                            dtype=float,
                        )
                    )
                )
            )
        )

    return float(
        np.mean(
            np.asarray(
                values
            )
            >= (
                observed
                - 1e-15
            )
        )
    )


def bootstrap_classification(
    metric: str,
    ci_low: float,
    ci_high: float,
) -> str:
    if metric in {
        "selected_accuracy",
        "coverage",
        "physical_exact_match",
        "control_abstention",
    }:
        if ci_low > 0:
            return "BASELINE_HIGHER"
        if ci_high < 0:
            return "PHYGUARD_HIGHER"
        return "INCONCLUSIVE"

    if metric == "false_specific_rate":
        if ci_low > 0:
            return "PHYGUARD_LOWER_RISK"
        if ci_high < 0:
            return "BASELINE_LOWER_RISK"
        return "INCONCLUSIVE"

    raise KeyError(metric)


for path in [
    C2_SUMMARY,
    BASELINE_REPEAT_PATH,
    BASELINE_PREDICTIONS_PATH,
    C3_SUMMARY,
    PHYGUARD_REPEAT_PATH,
    SUITE_PREDICTIONS_PATH,
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
        f"Output already exists: {path}",
    )


c2_summary = json.loads(
    C2_SUMMARY.read_text(
        encoding="utf-8"
    )
)

c3_summary = json.loads(
    C3_SUMMARY.read_text(
        encoding="utf-8"
    )
)

require(
    c2_summary.get("status") == "PASS",
    "Stage 64C2 status is not PASS.",
)

require(
    c3_summary.get("status") == "PASS",
    "Stage 64C3 status is not PASS.",
)

abstain_token = str(
    c2_summary[
        "detected_abstain_token"
    ]
)

phyguard_method = str(
    c3_summary[
        "phyguard_method_identified"
    ]
)

baseline_repeat = pd.read_csv(
    BASELINE_REPEAT_PATH
)

baseline_predictions = pd.read_csv(
    BASELINE_PREDICTIONS_PATH
)

phyguard_repeat = pd.read_csv(
    PHYGUARD_REPEAT_PATH
)

suite_predictions = pd.read_csv(
    SUITE_PREDICTIONS_PATH
)

phyguard_predictions = suite_predictions[
    suite_predictions[
        "method"
    ].astype(str)
    == phyguard_method
].copy()

require(
    len(
        phyguard_predictions
    )
    == 1080,
    (
        "Unexpected PhyGuard prediction count: "
        f"{len(phyguard_predictions)}"
    ),
)

require(
    set(
        baseline_predictions[
            "model_id"
        ].astype(str)
    )
    == set(
        MODEL_ORDER
    ),
    "Corrected baseline model set mismatch.",
)

repeats = sorted(
    int(value)
    for value in phyguard_predictions[
        "repeat"
    ].unique()
)

indices = sorted(
    int(value)
    for value in pd.concat(
        [
            phyguard_predictions[
                "index"
            ],
            baseline_predictions[
                "index"
            ],
        ],
        ignore_index=True,
    ).unique()
)

require(
    repeats == [
        0,
        1,
        2,
        3,
        4,
    ],
    f"Unexpected repeats: {repeats}",
)

test_rows_by_repeat = {
    int(key): int(value)
    for key, value in phyguard_predictions.groupby(
        "repeat"
    ).size().to_dict().items()
}

require(
    len(phyguard_predictions) == 1080,
    (
        "Expected 1080 aligned test observations across five "
        f"repeats; found {len(phyguard_predictions)}."
    ),
)

require(
    test_rows_by_repeat
    == {
        0: 216,
        1: 216,
        2: 216,
        3: 216,
        4: 216,
    },
    (
        "Expected 216 test observations per repeat; found "
        f"{test_rows_by_repeat}."
    ),
)

require(
    len(indices) == 726,
    (
        "Expected 726 unique original-sample clusters in the "
        "union of the five repeated test sets; found "
        f"{len(indices)}."
    ),
)


phyguard_statistics = prediction_sufficient_statistics(
    phyguard_predictions,
    "prediction",
    abstain_token,
    repeats,
    indices,
)

phyguard_point_metrics = metrics_from_sums(
    phyguard_statistics.sum(
        axis=1
    )
)

phyguard_point_means = {
    metric:
        float(
            np.nanmean(
                values
            )
        )
    for metric, values in
    phyguard_point_metrics.items()
}

for metric in METRICS:
    table_value = float(
        phyguard_repeat[
            metric
        ].mean()
    )

    require(
        abs(
            phyguard_point_means[
                metric
            ]
            - table_value
        )
        <= 1e-12,
        (
            f"PhyGuard point reproduction failed for {metric}: "
            f"{phyguard_point_means[metric]} vs {table_value}"
        ),
    )


rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

bootstrap_weights = np.empty(
    (
        BOOTSTRAP_REPLICATES,
        len(
            indices
        ),
    ),
    dtype=np.int16,
)

probabilities = np.full(
    len(
        indices
    ),
    1.0
    / len(
        indices
    ),
    dtype=np.float64,
)

for start in range(
    0,
    BOOTSTRAP_REPLICATES,
    BATCH_SIZE,
):
    stop = min(
        BOOTSTRAP_REPLICATES,
        start
        + BATCH_SIZE,
    )

    bootstrap_weights[
        start:
        stop
    ] = rng.multinomial(
        len(
            indices
        ),
        probabilities,
        size=stop
        - start,
    ).astype(
        np.int16
    )


point_rows = []
repeat_difference_rows = []
bootstrap_rows = []
model_summaries = {}


for model_id in MODEL_ORDER:
    baseline_model_predictions = baseline_predictions[
        baseline_predictions[
            "model_id"
        ].astype(str)
        == model_id
    ].copy()

    require(
        len(
            baseline_model_predictions
        )
        == 1080,
        (
            f"Unexpected prediction count for {model_id}: "
            f"{len(baseline_model_predictions)}"
        ),
    )

    alignment = baseline_model_predictions[
        [
            "repeat",
            "index",
            "truth",
        ]
    ].merge(
        phyguard_predictions[
            [
                "repeat",
                "index",
                "truth",
            ]
        ],
        on=[
            "repeat",
            "index",
        ],
        how="outer",
        suffixes=(
            "_baseline",
            "_phyguard",
        ),
        indicator=True,
        validate="one_to_one",
    )

    require(
        len(
            alignment
        )
        == 1080
        and (
            alignment[
                "_merge"
            ]
            == "both"
        ).all()
        and (
            alignment[
                "truth_baseline"
            ].astype(str)
            == alignment[
                "truth_phyguard"
            ].astype(str)
        ).all(),
        f"Prediction alignment failed for {model_id}.",
    )

    baseline_statistics = prediction_sufficient_statistics(
        baseline_model_predictions,
        "corrected_prediction",
        abstain_token,
        repeats,
        indices,
    )

    baseline_point_metrics = metrics_from_sums(
        baseline_statistics.sum(
            axis=1
        )
    )

    baseline_point_means = {
        metric:
            float(
                np.nanmean(
                    values
                )
            )
        for metric, values in
        baseline_point_metrics.items()
    }

    model_repeat_table = baseline_repeat[
        baseline_repeat[
            "model_id"
        ].astype(str)
        == model_id
    ].sort_values(
        "repeat"
    )

    require(
        len(
            model_repeat_table
        )
        == 5,
        f"Missing per-repeat metrics for {model_id}.",
    )

    for metric in METRICS:
        table_value = float(
            model_repeat_table[
                metric
            ].mean()
        )

        require(
            abs(
                baseline_point_means[
                    metric
                ]
                - table_value
            )
            <= 1e-12,
            (
                f"Baseline point reproduction failed for "
                f"{model_id}/{metric}: "
                f"{baseline_point_means[metric]} vs {table_value}"
            ),
        )

    baseline_bootstrap_sums = np.einsum(
        "bi,ric->brc",
        bootstrap_weights,
        baseline_statistics,
        optimize=True,
    )

    phyguard_bootstrap_sums = np.einsum(
        "bi,ric->brc",
        bootstrap_weights,
        phyguard_statistics,
        optimize=True,
    )

    baseline_bootstrap_metrics = metrics_from_sums(
        baseline_bootstrap_sums
    )

    phyguard_bootstrap_metrics = metrics_from_sums(
        phyguard_bootstrap_sums
    )

    model_summary = {}

    for metric in METRICS:
        baseline_repeat_values = np.asarray(
            baseline_point_metrics[
                metric
            ],
            dtype=float,
        )

        phyguard_repeat_values = np.asarray(
            phyguard_point_metrics[
                metric
            ],
            dtype=float,
        )

        repeat_differences = (
            baseline_repeat_values
            - phyguard_repeat_values
        )

        for repeat, (
            baseline_value,
            phyguard_value,
            difference,
        ) in enumerate(
            zip(
                baseline_repeat_values,
                phyguard_repeat_values,
                repeat_differences,
            )
        ):
            repeat_difference_rows.append(
                {
                    "model_id":
                        model_id,

                    "metric":
                        metric,

                    "repeat":
                        repeat,

                    "baseline_value":
                        float(
                            baseline_value
                        ),

                    "phyguard_value":
                        float(
                            phyguard_value
                        ),

                    "baseline_minus_phyguard":
                        float(
                            difference
                        ),
                }
            )

        point_delta = float(
            np.nanmean(
                repeat_differences
            )
        )

        baseline_bootstrap_mean = mean_across_repeats(
            baseline_bootstrap_metrics[
                metric
            ]
        )

        phyguard_bootstrap_mean = mean_across_repeats(
            phyguard_bootstrap_metrics[
                metric
            ]
        )

        bootstrap_delta = (
            baseline_bootstrap_mean
            - phyguard_bootstrap_mean
        )

        finite = np.isfinite(
            bootstrap_delta
        )

        valid = bootstrap_delta[
            finite
        ]

        require(
            len(valid)
            >= 9000,
            (
                f"Too few valid bootstrap replicates for "
                f"{model_id}/{metric}: {len(valid)}"
            ),
        )

        ci_low = float(
            np.quantile(
                valid,
                0.025,
            )
        )

        ci_high = float(
            np.quantile(
                valid,
                0.975,
            )
        )

        bootstrap_p = float(
            min(
                1.0,
                2.0
                * min(
                    np.mean(
                        valid
                        <= 0
                    ),
                    np.mean(
                        valid
                        >= 0
                    ),
                ),
            )
        )

        signflip_p = exact_signflip_p(
            repeat_differences
        )

        classification = bootstrap_classification(
            metric,
            ci_low,
            ci_high,
        )

        point_rows.append(
            {
                "model_id":
                    model_id,

                "metric":
                    metric,

                "baseline_mean":
                    baseline_point_means[
                        metric
                    ],

                "phyguard_mean":
                    phyguard_point_means[
                        metric
                    ],

                "baseline_minus_phyguard":
                    point_delta,
            }
        )

        bootstrap_rows.append(
            {
                "model_id":
                    model_id,

                "metric":
                    metric,

                "baseline_minus_phyguard":
                    point_delta,

                "cluster_bootstrap_ci_low":
                    ci_low,

                "cluster_bootstrap_ci_high":
                    ci_high,

                "cluster_bootstrap_two_sided_p":
                    bootstrap_p,

                "exact_repeat_signflip_p":
                    signflip_p,

                "valid_bootstrap_replicates":
                    int(
                        len(
                            valid
                        )
                    ),

                "classification":
                    classification,
            }
        )

        model_summary[
            metric
        ] = {
            "delta":
                point_delta,

            "ci":
                [
                    ci_low,
                    ci_high,
                ],

            "classification":
                classification,

            "bootstrap_p":
                bootstrap_p,

            "signflip_p":
                signflip_p,
        }

    model_summaries[
        model_id
    ] = model_summary


point_frame = pd.DataFrame(
    point_rows
)

repeat_difference_frame = pd.DataFrame(
    repeat_difference_rows
)

bootstrap_frame = pd.DataFrame(
    bootstrap_rows
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

point_frame.to_csv(
    POINT_PATH,
    index=False,
)

repeat_difference_frame.to_csv(
    REPEAT_DIFFERENCE_PATH,
    index=False,
)

bootstrap_frame.to_csv(
    BOOTSTRAP_PATH,
    index=False,
)


higher_selected_accuracy = []

accuracy_safety_tradeoffs = []

phyguard_selected_accuracy_advantages = []

for model_id in MODEL_ORDER:
    selected = model_summaries[
        model_id
    ][
        "selected_accuracy"
    ]

    fsr = model_summaries[
        model_id
    ][
        "false_specific_rate"
    ]

    control = model_summaries[
        model_id
    ][
        "control_abstention"
    ]

    if selected[
        "delta"
    ] > 0:
        higher_selected_accuracy.append(
            model_id
        )

    if (
        selected[
            "delta"
        ]
        > 0
        and fsr[
            "delta"
        ]
        > 0
        and control[
            "delta"
        ]
        < 0
    ):
        accuracy_safety_tradeoffs.append(
            model_id
        )

    if selected[
        "classification"
    ] == "PHYGUARD_HIGHER":
        phyguard_selected_accuracy_advantages.append(
            model_id
        )


summary = {
    "schema":
        "phyguard.temporal_paired_uncertainty.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "PASS",

    "formal_hardware":
        "NVIDIA GeForce RTX 3080 10 GB",

    "bootstrap": {
        "method":
            (
                "Paired cluster bootstrap over original sample "
                "index, retaining all occurrences of each index "
                "across the five repeats."
            ),

        "replicates":
            BOOTSTRAP_REPLICATES,

        "seed":
            BOOTSTRAP_SEED,

        "cluster_count":
            len(
                indices
            ),
    },

    "phyguard_means":
        phyguard_point_means,

    "models_with_higher_point_selected_accuracy":
        higher_selected_accuracy,

    "models_with_accuracy_safety_tradeoff":
        accuracy_safety_tradeoffs,

    "models_with_statistically_supported_phyguard_selected_accuracy_advantage":
        phyguard_selected_accuracy_advantages,

    "model_results":
        model_summaries,

    "interpretation_boundary":
        (
            "A positive baseline-minus-PhyGuard selected-accuracy "
            "delta indicates higher conditional accuracy for the "
            "baseline. A positive false-specific-rate delta and "
            "negative control-abstention delta indicate a safety "
            "penalty relative to PhyGuard. Claims must preserve "
            "this multi-objective interpretation."
        ),
}

SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ),
    encoding="utf-8",
)


lines = [
    "# Paired uncertainty analysis",
    "",
    (
        "All 14 baselines and PhyGuard were compared on exactly "
        "aligned repeat/index/truth records."
    ),
    "",
    (
        f"PhyGuard means: selected accuracy "
        f"{phyguard_point_means['selected_accuracy']:.4f}, "
        f"coverage {phyguard_point_means['coverage']:.4f}, "
        f"false-specific rate "
        f"{phyguard_point_means['false_specific_rate']:.4f}, "
        f"physical exact match "
        f"{phyguard_point_means['physical_exact_match']:.4f}, "
        f"and control abstention "
        f"{phyguard_point_means['control_abstention']:.4f}."
    ),
    "",
    "## Models with a point selected-accuracy gain",
    "",
]

for model_id in higher_selected_accuracy:
    selected = model_summaries[
        model_id
    ][
        "selected_accuracy"
    ]

    fsr = model_summaries[
        model_id
    ][
        "false_specific_rate"
    ]

    control = model_summaries[
        model_id
    ][
        "control_abstention"
    ]

    lines.append(
        (
            f"- {model_id}: selected-accuracy delta "
            f"{selected['delta']:+.4f} "
            f"[{selected['ci'][0]:+.4f}, "
            f"{selected['ci'][1]:+.4f}]; "
            f"false-specific-rate delta "
            f"{fsr['delta']:+.4f} "
            f"[{fsr['ci'][0]:+.4f}, "
            f"{fsr['ci'][1]:+.4f}]; "
            f"control-abstention delta "
            f"{control['delta']:+.4f} "
            f"[{control['ci'][0]:+.4f}, "
            f"{control['ci'][1]:+.4f}]."
        )
    )

lines.extend(
    [
        "",
        "## Reporting boundary",
        "",
        (
            "No baseline may be described as uniformly superior "
            "unless it improves selected accuracy and coverage "
            "while not worsening false-specific rate, physical "
            "exact match, or control abstention."
        ),
        "",
    ]
)

MANUSCRIPT_PATH.write_text(
    "\n".join(
        lines
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
        "phyguard.temporal_paired_uncertainty_manifest.v1",

    "status":
        "PASS",

    "c2_summary_sha256":
        sha256_file(
            C2_SUMMARY
        ),

    "c3_summary_sha256":
        sha256_file(
            C3_SUMMARY
        ),

    "baseline_predictions_sha256":
        sha256_file(
            BASELINE_PREDICTIONS_PATH
        ),

    "suite_predictions_sha256":
        sha256_file(
            SUITE_PREDICTIONS_PATH
        ),

    "files":
        files,

    "formal_hardware":
        "NVIDIA GeForce RTX 3080 10 GB",
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


display = bootstrap_frame[
    bootstrap_frame[
        "metric"
    ].isin(
        [
            "selected_accuracy",
            "false_specific_rate",
            "control_abstention",
        ]
    )
].copy()

print("status: PASS")
print(
    "bootstrap_replicates:",
    BOOTSTRAP_REPLICATES,
)
print(
    "cluster_count:",
    len(
        indices
    ),
)
print(
    "formal_hardware: NVIDIA GeForce RTX 3080 10 GB"
)
print()
print("PAIRED UNCERTAINTY RESULTS")
print(
    display[
        [
            "model_id",
            "metric",
            "baseline_minus_phyguard",
            "cluster_bootstrap_ci_low",
            "cluster_bootstrap_ci_high",
            "cluster_bootstrap_two_sided_p",
            "exact_repeat_signflip_p",
            "classification",
        ]
    ].to_string(
        index=False
    )
)
print()
print(
    "models_with_higher_point_selected_accuracy:",
    higher_selected_accuracy,
)
print(
    "models_with_accuracy_safety_tradeoff:",
    accuracy_safety_tradeoffs,
)
print(
    "models_with_supported_phyguard_selected_accuracy_advantage:",
    phyguard_selected_accuracy_advantages,
)
print()
print("summary:", SUMMARY_PATH)
print("bootstrap_table:", BOOTSTRAP_PATH)
print("manuscript_ready:", MANUSCRIPT_PATH)
print("manifest:", MANIFEST_PATH)
print()
print(
    "PHYGUARD_TEMPORAL_PAIRED_UNCERTAINTY_V1_PASS"
)
