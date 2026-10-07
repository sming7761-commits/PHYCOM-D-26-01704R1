#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/phyguard_revision")

INPUT_ROOT = (
    ROOT
    / "artifacts"
    / "physics_sensitivity_fixed_model_results_v1"
)

PER_REPEAT_PATH = INPUT_ROOT / "per_repeat_metrics.csv"
AGGREGATE_PATH = INPUT_ROOT / "aggregate_metrics.csv"
PREDICTIONS_PATH = INPUT_ROOT / "all_test_predictions.csv"
POLICY_AGGREGATE_PATH = (
    INPUT_ROOT
    / "threshold_policy_aggregate_metrics.csv"
)
SOURCE_SUMMARY_PATH = INPUT_ROOT / "summary.json"

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "physics_sensitivity_uncertainty_v1"
)

SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
PROGRESS_PATH = OUTPUT_ROOT / "progress.json"
PAIRED_PATH = OUTPUT_ROOT / "paired_formula_uncertainty.csv"
FACTOR_RANGE_PATH = OUTPUT_ROOT / "factor_operating_ranges.csv"
FORMULA_ALL_PATH = OUTPUT_ROOT / "supplementary_all_13_formula_configs.csv"
POLICY_ALL_PATH = OUTPUT_ROOT / "supplementary_all_9_threshold_policies.csv"
POLICY_MAIN_PATH = OUTPUT_ROOT / "main_table_threshold_policy_sa_0p90.csv"
MAIN_FACTOR_PATH = OUTPUT_ROOT / "main_table_formula_factor_ranges.csv"
MANUSCRIPT_RESULTS_PATH = OUTPUT_ROOT / "manuscript_results_draft.txt"
REVIEWER_RESPONSE_PATH = OUTPUT_ROOT / "reviewer_response_draft.txt"
CLAIM_BOUNDARY_PATH = OUTPUT_ROOT / "claim_boundary.json"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "physics_sensitivity_uncertainty_v1.json"
)

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260731

PHYSICAL_LABELS = np.array(
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
    import hashlib

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def weighted_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=object)
    prediction = np.asarray(prediction, dtype=object)
    weights = np.asarray(weights, dtype=np.float64)

    require(
        truth.shape == prediction.shape == weights.shape,
        "Truth, prediction, and weight shapes differ.",
    )

    physical = np.isin(truth, PHYSICAL_LABELS)
    control = ~physical
    emitted = prediction != ABSTAIN
    selected_physical = physical & emitted
    correct = prediction == truth

    physical_weight = float(weights[physical].sum())
    control_weight = float(weights[control].sum())
    selected_weight = float(weights[selected_physical].sum())

    selected_accuracy = (
        float(weights[selected_physical & correct].sum())
        / selected_weight
        if selected_weight > 0
        else 0.0
    )

    coverage = (
        float(weights[selected_physical].sum())
        / physical_weight
        if physical_weight > 0
        else 0.0
    )

    false_specific_rate = (
        float(weights[control & emitted].sum())
        / control_weight
        if control_weight > 0
        else 0.0
    )

    physical_exact_match = (
        float(weights[physical & correct].sum())
        / physical_weight
        if physical_weight > 0
        else 0.0
    )

    control_abstention = (
        float(weights[control & (~emitted)].sum())
        / control_weight
        if control_weight > 0
        else 0.0
    )

    recalls = []
    f1_values = []

    # This matches evaluation_common.selective_metrics:
    # classification quality is computed on the physical subset,
    # with ABSTAIN counted as a non-matching prediction.
    for label in PHYSICAL_LABELS:
        truth_label = physical & (truth == label)
        predicted_label = physical & (prediction == label)

        tp = float(
            weights[
                truth_label
                & predicted_label
            ].sum()
        )

        fn = float(
            weights[
                truth_label
                & (~predicted_label)
            ].sum()
        )

        fp = float(
            weights[
                physical
                & (truth != label)
                & (prediction == label)
            ].sum()
        )

        recall_denominator = tp + fn

        recalls.append(
            tp / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1_denominator = (
            2.0 * tp
            + fp
            + fn
        )

        f1_values.append(
            2.0 * tp / f1_denominator
            if f1_denominator > 0
            else 0.0
        )

    return {
        "selected_accuracy": selected_accuracy,
        "coverage": coverage,
        "false_specific_rate": false_specific_rate,
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recalls)),
        "physical_exact_match": physical_exact_match,
        "control_abstention": control_abstention,
    }


def holm_adjust(values: list[float]) -> list[float]:
    count = len(values)
    order = np.argsort(values)
    adjusted = np.empty(count, dtype=np.float64)

    running = 0.0

    for rank, position in enumerate(order):
        candidate = (
            (count - rank)
            * float(values[position])
        )

        running = max(
            running,
            candidate,
        )

        adjusted[position] = min(
            1.0,
            running,
        )

    return adjusted.tolist()


def exact_signflip_p(
    differences: np.ndarray,
) -> float:
    differences = np.asarray(
        differences,
        dtype=np.float64,
    )

    observed = abs(
        float(
            differences.mean()
        )
    )

    transformed = []

    for signs in product(
        [-1.0, 1.0],
        repeat=len(
            differences
        ),
    ):
        transformed.append(
            abs(
                float(
                    np.mean(
                        differences
                        * np.asarray(
                            signs,
                            dtype=np.float64,
                        )
                    )
                )
            )
        )

    transformed = np.asarray(
        transformed,
        dtype=np.float64,
    )

    return float(
        np.mean(
            transformed
            >= observed
            - 1e-15
        )
    )


def write_progress(
    completed: int,
    total: int,
    started_at: float,
    status: str,
) -> None:
    elapsed = max(
        0.0,
        time.time()
        - started_at,
    )

    rate = (
        completed / elapsed
        if elapsed > 0
        else 0.0
    )

    remaining = (
        (total - completed) / rate
        if rate > 0
        else None
    )

    payload = {
        "status": status,
        "completed_replicates": completed,
        "total_replicates": total,
        "fraction_complete": (
            completed / total
            if total
            else 0.0
        ),
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": remaining,
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


for path in [
    PER_REPEAT_PATH,
    AGGREGATE_PATH,
    PREDICTIONS_PATH,
    POLICY_AGGREGATE_PATH,
    SOURCE_SUMMARY_PATH,
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


started_at = time.time()

per_repeat = pd.read_csv(
    PER_REPEAT_PATH
)

aggregate = pd.read_csv(
    AGGREGATE_PATH
)

predictions = pd.read_csv(
    PREDICTIONS_PATH
)

policy = pd.read_csv(
    POLICY_AGGREGATE_PATH
)

source_summary = json.loads(
    SOURCE_SUMMARY_PATH.read_text(
        encoding="utf-8"
    )
)

require(
    source_summary.get("status") == "PASS",
    "Fixed-model sensitivity source summary is not PASS.",
)

require(
    source_summary.get(
        "default_reproduction",
        {},
    ).get(
        "prediction_match"
    )
    is True,
    "DEFAULT prediction reproduction did not pass.",
)

require(
    len(per_repeat) == 65,
    f"Expected 65 per-repeat rows; found {len(per_repeat)}.",
)

require(
    len(aggregate) == 13,
    f"Expected 13 aggregate rows; found {len(aggregate)}.",
)

require(
    len(policy) == 9,
    f"Expected nine policy rows; found {len(policy)}.",
)

require(
    len(predictions) == 65 * 216,
    (
        "Expected 14,040 prediction rows; "
        f"found {len(predictions)}."
    ),
)

variants = (
    aggregate[
        "variant_id"
    ].astype(
        str
    ).tolist()
)

require(
    "DEFAULT" in variants,
    "DEFAULT variant is missing.",
)

nondefault_variants = [
    variant
    for variant in variants
    if variant != "DEFAULT"
]

require(
    len(nondefault_variants) == 12,
    (
        "Expected 12 non-default formula variants; "
        f"found {len(nondefault_variants)}."
    ),
)

repeats = sorted(
    predictions[
        "repeat"
    ].unique().tolist()
)

require(
    repeats == [0, 1, 2, 3, 4],
    f"Unexpected repeat set: {repeats}",
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

default_predictions = predictions[
    predictions[
        "variant_id"
    ].astype(
        str
    )
    == "DEFAULT"
].copy()

cluster_ids = np.sort(
    default_predictions[
        "index"
    ].unique()
)

cluster_count = int(
    len(
        cluster_ids
    )
)

cluster_position = {
    int(cluster_id): position
    for position, cluster_id in enumerate(
        cluster_ids
    )
}


# Build aligned arrays and verify that the weighted implementation
# exactly reproduces every existing unweighted task metric.
aligned: dict[str, dict[int, dict[str, np.ndarray]]] = {}

for variant in variants:
    aligned[
        variant
    ] = {}

    for repeat in repeats:
        frame = predictions[
            (
                predictions[
                    "variant_id"
                ].astype(
                    str
                )
                == variant
            )
            & (
                predictions[
                    "repeat"
                ]
                == repeat
            )
        ].sort_values(
            "index"
        )

        require(
            len(frame) == 216,
            (
                f"{variant}, repeat {repeat}: "
                f"expected 216 predictions; found {len(frame)}."
            ),
        )

        reference = default_predictions[
            default_predictions[
                "repeat"
            ]
            == repeat
        ].sort_values(
            "index"
        )

        require(
            np.array_equal(
                frame[
                    "index"
                ].to_numpy(),
                reference[
                    "index"
                ].to_numpy(),
            ),
            (
                f"{variant}, repeat {repeat}: "
                "test indices do not align with DEFAULT."
            ),
        )

        require(
            np.array_equal(
                frame[
                    "truth"
                ].astype(
                    str
                ).to_numpy(),
                reference[
                    "truth"
                ].astype(
                    str
                ).to_numpy(),
            ),
            (
                f"{variant}, repeat {repeat}: "
                "truth labels do not align with DEFAULT."
            ),
        )

        indices = frame[
            "index"
        ].to_numpy(
            dtype=np.int64
        )

        positions = np.asarray(
            [
                cluster_position[
                    int(
                        index
                    )
                ]
                for index in indices
            ],
            dtype=np.int64,
        )

        truth = frame[
            "truth"
        ].astype(
            str
        ).to_numpy(
            dtype=object
        )

        prediction = frame[
            "prediction"
        ].astype(
            str
        ).to_numpy(
            dtype=object
        )

        aligned[
            variant
        ][
            repeat
        ] = {
            "cluster_positions": positions,
            "truth": truth,
            "prediction": prediction,
        }

        calculated = weighted_metrics(
            truth,
            prediction,
            np.ones(
                len(
                    truth
                ),
                dtype=np.float64,
            ),
        )

        reported = per_repeat[
            (
                per_repeat[
                    "variant_id"
                ].astype(
                    str
                )
                == variant
            )
            & (
                per_repeat[
                    "repeat"
                ]
                == repeat
            )
        ]

        require(
            len(reported) == 1,
            (
                f"{variant}, repeat {repeat}: "
                "per-repeat metric row is missing."
            ),
        )

        row = reported.iloc[0]

        for metric in METRICS:
            difference = abs(
                calculated[
                    metric
                ]
                - float(
                    row[
                        metric
                    ]
                )
            )

            require(
                difference <= 1e-12,
                (
                    "Weighted metric implementation does not "
                    f"reproduce {variant}, repeat {repeat}, "
                    f"{metric}: difference={difference}"
                ),
            )


# Per-repeat point differences and exact sign-flip p-values.
point_difference_rows = []

default_per_repeat = (
    per_repeat[
        per_repeat[
            "variant_id"
        ].astype(
            str
        )
        == "DEFAULT"
    ]
    .sort_values(
        "repeat"
    )
    .reset_index(
        drop=True
    )
)

for variant in nondefault_variants:
    variant_rows = (
        per_repeat[
            per_repeat[
                "variant_id"
            ].astype(
                str
            )
            == variant
        ]
        .sort_values(
            "repeat"
        )
        .reset_index(
            drop=True
        )
    )

    require(
        len(
            variant_rows
        )
        == 5,
        f"{variant}: expected five repeat rows.",
    )

    aggregate_row = aggregate[
        aggregate[
            "variant_id"
        ].astype(
            str
        )
        == variant
    ].iloc[
        0
    ]

    for metric in METRICS:
        differences = (
            variant_rows[
                metric
            ].to_numpy(
                dtype=np.float64
            )
            - default_per_repeat[
                metric
            ].to_numpy(
                dtype=np.float64
            )
        )

        point_difference_rows.append(
            {
                "variant_id": variant,
                "factor": str(
                    aggregate_row[
                        "factor"
                    ]
                ),
                "metric": metric,
                "point_difference": float(
                    differences.mean()
                ),
                "repeat_difference_min": float(
                    differences.min()
                ),
                "repeat_difference_max": float(
                    differences.max()
                ),
                "exact_signflip_p": exact_signflip_p(
                    differences
                ),
            }
        )


point_difference = pd.DataFrame(
    point_difference_rows
)

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

bootstrap_values = {
    (
        variant,
        metric,
    ): np.empty(
        BOOTSTRAP_REPLICATES,
        dtype=np.float64,
    )
    for variant in nondefault_variants
    for metric in METRICS
}

write_progress(
    completed=0,
    total=BOOTSTRAP_REPLICATES,
    started_at=started_at,
    status="RUNNING",
)

for replicate in range(
    BOOTSTRAP_REPLICATES
):
    sampled_positions = rng.integers(
        0,
        cluster_count,
        size=cluster_count,
    )

    counts = np.bincount(
        sampled_positions,
        minlength=cluster_count,
    ).astype(
        np.float64
    )

    default_metric_by_repeat = {
        metric: []
        for metric in METRICS
    }

    for repeat in repeats:
        data = aligned[
            "DEFAULT"
        ][
            repeat
        ]

        weights = counts[
            data[
                "cluster_positions"
            ]
        ]

        metrics = weighted_metrics(
            data[
                "truth"
            ],
            data[
                "prediction"
            ],
            weights,
        )

        for metric in METRICS:
            default_metric_by_repeat[
                metric
            ].append(
                metrics[
                    metric
                ]
            )

    default_mean = {
        metric: float(
            np.mean(
                default_metric_by_repeat[
                    metric
                ]
            )
        )
        for metric in METRICS
    }

    for variant in nondefault_variants:
        variant_metric_by_repeat = {
            metric: []
            for metric in METRICS
        }

        for repeat in repeats:
            data = aligned[
                variant
            ][
                repeat
            ]

            weights = counts[
                data[
                    "cluster_positions"
                ]
            ]

            metrics = weighted_metrics(
                data[
                    "truth"
                ],
                data[
                    "prediction"
                ],
                weights,
            )

            for metric in METRICS:
                variant_metric_by_repeat[
                    metric
                ].append(
                    metrics[
                        metric
                    ]
                )

        for metric in METRICS:
            bootstrap_values[
                (
                    variant,
                    metric,
                )
            ][
                replicate
            ] = (
                float(
                    np.mean(
                        variant_metric_by_repeat[
                            metric
                        ]
                    )
                )
                - default_mean[
                    metric
                ]
            )

    if (
        replicate + 1
    ) % 250 == 0 or (
        replicate + 1
    ) == BOOTSTRAP_REPLICATES:
        write_progress(
            completed=replicate + 1,
            total=BOOTSTRAP_REPLICATES,
            started_at=started_at,
            status="RUNNING",
        )

        elapsed = time.time() - started_at
        rate = (
            (replicate + 1)
            / elapsed
            if elapsed > 0
            else 0.0
        )

        remaining = (
            (
                BOOTSTRAP_REPLICATES
                - replicate
                - 1
            )
            / rate
            if rate > 0
            else 0.0
        )

        print(
            f"[{replicate + 1:05d}/"
            f"{BOOTSTRAP_REPLICATES}] "
            f"bootstrap PASS "
            f"elapsed={elapsed:.1f}s "
            f"eta={remaining:.1f}s",
            flush=True,
        )


uncertainty_rows = []

for _, row in point_difference.iterrows():
    variant = str(
        row[
            "variant_id"
        ]
    )

    metric = str(
        row[
            "metric"
        ]
    )

    values = bootstrap_values[
        (
            variant,
            metric,
        )
    ]

    lower, upper = np.quantile(
        values,
        [
            0.025,
            0.975,
        ],
    )

    count_nonpositive = int(
        np.sum(
            values <= 0.0
        )
    )

    count_nonnegative = int(
        np.sum(
            values >= 0.0
        )
    )

    bootstrap_p = min(
        1.0,
        2.0
        * min(
            (
                count_nonpositive
                + 1
            )
            / (
                BOOTSTRAP_REPLICATES
                + 1
            ),
            (
                count_nonnegative
                + 1
            )
            / (
                BOOTSTRAP_REPLICATES
                + 1
            ),
        ),
    )

    uncertainty_rows.append(
        {
            **row.to_dict(),
            "bootstrap_ci_lower": float(
                lower
            ),
            "bootstrap_ci_upper": float(
                upper
            ),
            "bootstrap_two_sided_p": float(
                bootstrap_p
            ),
            "bootstrap_replicates": (
                BOOTSTRAP_REPLICATES
            ),
            "cluster_count": cluster_count,
        }
    )


uncertainty = pd.DataFrame(
    uncertainty_rows
)

uncertainty[
    "bootstrap_holm_p"
] = np.nan

uncertainty[
    "signflip_holm_p"
] = np.nan

for metric in METRICS:
    mask = uncertainty[
        "metric"
    ] == metric

    uncertainty.loc[
        mask,
        "bootstrap_holm_p",
    ] = holm_adjust(
        uncertainty.loc[
            mask,
            "bootstrap_two_sided_p",
        ].astype(
            float
        ).tolist()
    )

    uncertainty.loc[
        mask,
        "signflip_holm_p",
    ] = holm_adjust(
        uncertainty.loc[
            mask,
            "exact_signflip_p",
        ].astype(
            float
        ).tolist()
    )


uncertainty[
    "bootstrap_ci_excludes_zero"
] = (
    (
        uncertainty[
            "bootstrap_ci_lower"
        ]
        > 0.0
    )
    | (
        uncertainty[
            "bootstrap_ci_upper"
        ]
        < 0.0
    )
)

uncertainty[
    "confirmatory_significance_claim_allowed"
] = False

uncertainty.to_csv(
    PAIRED_PATH,
    index=False,
)


# Compact factor ranges, including DEFAULT as the central locked value.
factor_specs = [
    (
        "phi_scale",
        "phi_scale",
        [
            "PHI_SCALE_1p0",
            "PHI_SCALE_1p25",
            "DEFAULT",
            "PHI_SCALE_2p0",
            "PHI_SCALE_2p5",
        ],
    ),
    (
        "adapt_gap_weight",
        "adapt_gap_weight",
        [
            "ADAPT_GAP_WEIGHT_0p5",
            "ADAPT_GAP_WEIGHT_0p75",
            "DEFAULT",
            "ADAPT_GAP_WEIGHT_1p25",
            "ADAPT_GAP_WEIGHT_1p5",
        ],
    ),
    (
        "mismatch_mcs_drop_penalty",
        "mismatch_mcs_drop_penalty",
        [
            "MCS_DROP_PENALTY_0p0",
            "MCS_DROP_PENALTY_0p175",
            "DEFAULT",
            "MCS_DROP_PENALTY_0p525",
            "MCS_DROP_PENALTY_0p7",
        ],
    ),
]

factor_rows = []

default_row = aggregate[
    aggregate[
        "variant_id"
    ].astype(
        str
    )
    == "DEFAULT"
].iloc[
    0
]

for factor, value_column, selected_ids in factor_specs:
    frame = aggregate[
        aggregate[
            "variant_id"
        ].astype(
            str
        ).isin(
            selected_ids
        )
    ].copy()

    require(
        len(
            frame
        )
        == 5,
        (
            f"{factor}: expected five sweep points "
            f"including DEFAULT; found {len(frame)}."
        ),
    )

    row = {
        "factor": factor,
        "tested_values": ", ".join(
            str(
                value
            )
            for value in sorted(
                frame[
                    value_column
                ].astype(
                    float
                ).unique()
            )
        ),
        "configuration_count": 5,
    }

    for metric in METRICS:
        column = f"{metric}_mean"

        values = frame[
            column
        ].astype(
            float
        )

        deltas = (
            values
            - float(
                default_row[
                    column
                ]
            )
        )

        row[
            f"{metric}_min"
        ] = float(
            values.min()
        )

        row[
            f"{metric}_max"
        ] = float(
            values.max()
        )

        row[
            f"{metric}_delta_min"
        ] = float(
            deltas.min()
        )

        row[
            f"{metric}_delta_max"
        ] = float(
            deltas.max()
        )

        row[
            f"{metric}_max_abs_delta"
        ] = float(
            np.max(
                np.abs(
                    deltas
                )
            )
        )

    factor_rows.append(
        row
    )


factor_ranges = pd.DataFrame(
    factor_rows
)

factor_ranges.to_csv(
    FACTOR_RANGE_PATH,
    index=False,
)

factor_ranges.to_csv(
    MAIN_FACTOR_PATH,
    index=False,
)


# Complete supplementary tables and a predeclared three-point
# threshold-policy main table at SA_min=0.90.
aggregate.to_csv(
    FORMULA_ALL_PATH,
    index=False,
)

policy.to_csv(
    POLICY_ALL_PATH,
    index=False,
)

policy_main = policy[
    np.isclose(
        policy[
            "selected_accuracy_min"
        ].astype(
            float
        ),
        0.90,
    )
].sort_values(
    "fsr_max"
)

require(
    len(
        policy_main
    )
    == 3,
    (
        "Expected three SA_min=0.90 policy rows; "
        f"found {len(policy_main)}."
    ),
)

policy_main.to_csv(
    POLICY_MAIN_PATH,
    index=False,
)


# Overall point envelopes relative to DEFAULT.
overall_delta = aggregate.copy()

for metric in METRICS:
    overall_delta[
        f"delta_{metric}"
    ] = (
        overall_delta[
            f"{metric}_mean"
        ].astype(
            float
        )
        - float(
            default_row[
                f"{metric}_mean"
            ]
        )
    )

nondefault_delta = overall_delta[
    overall_delta[
        "variant_id"
    ].astype(
        str
    )
    != "DEFAULT"
]

overall_envelope = {
    metric: {
        "delta_min": float(
            nondefault_delta[
                f"delta_{metric}"
            ].min()
        ),
        "delta_max": float(
            nondefault_delta[
                f"delta_{metric}"
            ].max()
        ),
        "max_abs_delta": float(
            nondefault_delta[
                f"delta_{metric}"
            ].abs().max()
        ),
    }
    for metric in METRICS
}


strict_policy = policy_main[
    np.isclose(
        policy_main[
            "fsr_max"
        ],
        0.025,
    )
].iloc[
    0
]

default_policy = policy_main[
    np.isclose(
        policy_main[
            "fsr_max"
        ],
        0.05,
    )
].iloc[
    0
]

relaxed_policy = policy_main[
    np.isclose(
        policy_main[
            "fsr_max"
        ],
        0.10,
    )
].iloc[
    0
]


results_text = f"""Physics-score and selection-policy sensitivity (draft)

The analysis kept the five frozen PhyGuard decision functions fixed and perturbed only the source-defined physics formulas and validation policies. The DEFAULT path reproduced all 1,080 frozen test predictions exactly. Across the 12 non-default formula configurations, the mean selected-accuracy change relative to DEFAULT ranged from {overall_envelope['selected_accuracy']['delta_min']:+.4f} to {overall_envelope['selected_accuracy']['delta_max']:+.4f}; coverage changed from {overall_envelope['coverage']['delta_min']:+.4f} to {overall_envelope['coverage']['delta_max']:+.4f}; and false-specific rate changed from {overall_envelope['false_specific_rate']['delta_min']:+.4f} to {overall_envelope['false_specific_rate']['delta_max']:+.4f}. The corresponding macro-F1 envelope was {overall_envelope['macro_f1']['delta_min']:+.4f} to {overall_envelope['macro_f1']['delta_max']:+.4f}.

The phi-scale sweep was comparatively stable: selected accuracy remained between {factor_ranges.loc[factor_ranges['factor']=='phi_scale','selected_accuracy_min'].iloc[0]:.4f} and {factor_ranges.loc[factor_ranges['factor']=='phi_scale','selected_accuracy_max'].iloc[0]:.4f}, coverage between {factor_ranges.loc[factor_ranges['factor']=='phi_scale','coverage_min'].iloc[0]:.4f} and {factor_ranges.loc[factor_ranges['factor']=='phi_scale','coverage_max'].iloc[0]:.4f}, and false-specific rate between {factor_ranges.loc[factor_ranges['factor']=='phi_scale','false_specific_rate_min'].iloc[0]:.4f} and {factor_ranges.loc[factor_ranges['factor']=='phi_scale','false_specific_rate_max'].iloc[0]:.4f}. Changing the adaptation-gap evidence weight produced an interpretable risk–coverage shift: lower weights increased selected accuracy and control abstention but reduced physical coverage, whereas higher weights increased coverage while increasing false-specific output. The mismatch-penalty sweep showed the same broad operating-point behavior, with no evidence that the DEFAULT result depended on a single numerically fragile setting.

The validation-policy sweep further exposed the intended selective trade-off. At selected_accuracy_min=0.90, tightening fsr_max from 0.05 to 0.025 changed test selected accuracy from {default_policy['selected_accuracy_mean']:.4f} to {strict_policy['selected_accuracy_mean']:.4f}, coverage from {default_policy['coverage_mean']:.4f} to {strict_policy['coverage_mean']:.4f}, and false-specific rate from {default_policy['false_specific_rate_mean']:.4f} to {strict_policy['false_specific_rate_mean']:.4f}. Relaxing fsr_max to 0.10 increased coverage to {relaxed_policy['coverage_mean']:.4f} and physical exact match to {relaxed_policy['physical_exact_match_mean']:.4f}, while increasing false-specific rate to {relaxed_policy['false_specific_rate_mean']:.4f}. These are validation-selected operating points; the validation constraint is not claimed as a deterministic upper bound on held-out test risk.

Paired uncertainty used 10,000 cluster-bootstrap replicates over {cluster_count} unique original sequence indices, preserving repeated appearances of an index across the five locked splits. Exact five-repeat sign-flip probabilities are reported only as a robustness check, whose minimum attainable non-zero two-sided value is 0.0625. Holm adjustment was applied separately within each metric. Because this analysis evaluates a revision-time locked reconstruction, the results support robustness and operating-profile interpretation rather than a claim that the unavailable original submission contained trainable physics weights.
"""

MANUSCRIPT_RESULTS_PATH.write_text(
    results_text,
    encoding="utf-8",
)


reviewer_text = f"""Response draft — physics-score, weight, penalty, and selection-policy sensitivity

We thank the reviewer for requesting a clearer assessment of the dependence on physics-score design and selection parameters. We added a fixed-model sensitivity analysis around the revision-time locked reconstruction. The five frozen PhyGuard decision functions and the five saved train/validation/test partitions were kept unchanged. We varied the phi scale, the adaptation-gap evidence weight, and the explicit MCS-drop penalty over the complete protocol grid, giving 13 formula configurations including DEFAULT. We also evaluated all nine combinations of validation false-specific-rate limits and selected-accuracy requirements.

Before any non-default result was accepted, the compatibility path was required to reproduce the five DEFAULT operating points and all 1,080 frozen test predictions exactly. This check passed. Across the 12 non-default formula settings, selected accuracy changed by {overall_envelope['selected_accuracy']['delta_min']:+.4f} to {overall_envelope['selected_accuracy']['delta_max']:+.4f}, coverage by {overall_envelope['coverage']['delta_min']:+.4f} to {overall_envelope['coverage']['delta_max']:+.4f}, and false-specific rate by {overall_envelope['false_specific_rate']['delta_min']:+.4f} to {overall_envelope['false_specific_rate']['delta_max']:+.4f}. The adaptation-gap sweep exhibited the expected risk–coverage movement rather than an abrupt collapse, while the phi-scale sweep remained comparatively stable.

The policy sweep also confirmed that the reported DEFAULT point is one operating point rather than a universal optimum. A stricter validation risk limit reduced held-out coverage, whereas a relaxed limit increased coverage and physical exact match at the cost of more false-specific outputs. We report all configurations and paired cluster-bootstrap uncertainty; no outcome-based configuration was omitted. We have also clarified that validation constraints guide threshold selection but do not constitute deterministic guarantees on the held-out test set.
"""

REVIEWER_RESPONSE_PATH.write_text(
    reviewer_text,
    encoding="utf-8",
)


claim_boundary = {
    "schema": (
        "phyguard.physics_sensitivity_claim_boundary.v1"
    ),
    "status": "PASS",
    "fixed_model_analysis": True,
    "revision_time_locked_reconstruction": True,
    "original_submission_binary_replay_claimed": False,
    "trainable_physics_weights_claimed": False,
    "test_used_for_threshold_selection": False,
    "validation_constraint_claimed_as_test_guarantee": False,
    "all_formula_configs_reported": True,
    "all_threshold_policies_reported": True,
    "confirmatory_significance_claim_allowed": False,
    "paired_uncertainty_role": (
        "robustness and uncertainty characterization"
    ),
}

CLAIM_BOUNDARY_PATH.write_text(
    json.dumps(
        claim_boundary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


supported_positive = uncertainty[
    (
        uncertainty[
            "bootstrap_ci_lower"
        ]
        > 0.0
    )
]

supported_negative = uncertainty[
    (
        uncertainty[
            "bootstrap_ci_upper"
        ]
        < 0.0
    )
]

summary = {
    "schema": (
        "phyguard.physics_sensitivity_uncertainty.v1"
    ),
    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "status": "PASS",
    "formula_variant_count": 13,
    "nondefault_variant_count": 12,
    "repeat_count": 5,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "cluster_count": cluster_count,
    "paired_comparison_count": int(
        len(
            uncertainty
        )
    ),
    "metric_count": len(
        METRICS
    ),
    "exact_signflip_minimum_nonzero_two_sided_p": (
        0.0625
    ),
    "holm_scope": (
        "separately across 12 formula variants within each metric"
    ),
    "default_prediction_match": True,
    "overall_nondefault_delta_envelope": overall_envelope,
    "bootstrap_ci_excludes_zero_positive_count": int(
        len(
            supported_positive
        )
    ),
    "bootstrap_ci_excludes_zero_negative_count": int(
        len(
            supported_negative
        )
    ),
    "confirmatory_significance_claim_allowed": False,
    "test_used_for_threshold_selection": False,
    "source_models_modified": False,
    "all_formula_configs_reported": True,
    "all_threshold_policies_reported": True,
    "formal_hardware": (
        "NVIDIA GeForce RTX 3080 10 GB"
    ),
    "files": {
        "paired_formula_uncertainty": str(
            PAIRED_PATH
        ),
        "factor_operating_ranges": str(
            FACTOR_RANGE_PATH
        ),
        "main_table_formula_factor_ranges": str(
            MAIN_FACTOR_PATH
        ),
        "main_table_threshold_policy_sa_0p90": str(
            POLICY_MAIN_PATH
        ),
        "supplementary_all_13_formula_configs": str(
            FORMULA_ALL_PATH
        ),
        "supplementary_all_9_threshold_policies": str(
            POLICY_ALL_PATH
        ),
        "manuscript_results_draft": str(
            MANUSCRIPT_RESULTS_PATH
        ),
        "reviewer_response_draft": str(
            REVIEWER_RESPONSE_PATH
        ),
        "claim_boundary": str(
            CLAIM_BOUNDARY_PATH
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

manifest = {
    "schema": (
        "phyguard.physics_sensitivity_uncertainty_manifest.v1"
    ),
    "status": "PASS",
    "input_sha256": {
        "per_repeat": sha256_file(
            PER_REPEAT_PATH
        ),
        "aggregate": sha256_file(
            AGGREGATE_PATH
        ),
        "predictions": sha256_file(
            PREDICTIONS_PATH
        ),
        "policy_aggregate": sha256_file(
            POLICY_AGGREGATE_PATH
        ),
        "source_summary": sha256_file(
            SOURCE_SUMMARY_PATH
        ),
    },
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "cluster_count": cluster_count,
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
    completed=BOOTSTRAP_REPLICATES,
    total=BOOTSTRAP_REPLICATES,
    started_at=started_at,
    status="PASS",
)

print()
print("status: PASS")
print("formula_variant_count: 13")
print("nondefault_variant_count: 12")
print("repeat_count: 5")
print(
    "bootstrap_replicates:",
    BOOTSTRAP_REPLICATES,
)
print("cluster_count:", cluster_count)
print(
    "paired_comparison_count:",
    len(
        uncertainty
    ),
)
print(
    "default_prediction_match: True"
)
print(
    "confirmatory_significance_claim_allowed: False"
)
print(
    "test_used_for_threshold_selection: False"
)
print(
    "formal_hardware: NVIDIA GeForce RTX 3080 10 GB"
)
print("summary:", SUMMARY_PATH)
print("paired_uncertainty:", PAIRED_PATH)
print("factor_ranges:", FACTOR_RANGE_PATH)
print("policy_main_table:", POLICY_MAIN_PATH)
print("manuscript_draft:", MANUSCRIPT_RESULTS_PATH)
print("reviewer_draft:", REVIEWER_RESPONSE_PATH)
print("manifest:", MANIFEST_PATH)
print()
print(
    "PHYSICS_SENSITIVITY_UNCERTAINTY_V1_PASS"
)
