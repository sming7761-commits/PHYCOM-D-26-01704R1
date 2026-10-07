#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
import zipfile
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

C3_ROOT = (
    ROOT
    / "artifacts"
    / "phyguard_temporal_comparison_audit_v1"
)

C4_ROOT = (
    ROOT
    / "artifacts"
    / "phyguard_temporal_paired_uncertainty_v1"
)

C2_SUMMARY = C2_ROOT / "summary.json"
C3_SUMMARY = C3_ROOT / "summary.json"
C4_SUMMARY = C4_ROOT / "summary.json"

BASELINE_REPEAT_PATH = (
    C2_ROOT
    / "corrected_per_repeat_metrics.csv"
)

BASELINE_AGGREGATE_PATH = (
    C2_ROOT
    / "corrected_aggregate_metrics.csv"
)

PHYGUARD_AGGREGATE_PATH = (
    C3_ROOT
    / "phyguard_aggregate_metrics.csv"
)

PAIRWISE_PATH = (
    C3_ROOT
    / "pairwise_comparison.csv"
)

BOOTSTRAP_PATH = (
    C4_ROOT
    / "paired_cluster_bootstrap.csv"
)

RESULTS_ROOT = (
    ROOT
    / "artifacts"
    / "temporal_baseline_results_v1"
)

PROTOCOL_V1 = (
    ROOT
    / "configs"
    / "temporal_baseline_protocol_v1.json"
)

PROTOCOL_V11 = (
    ROOT
    / "configs"
    / "temporal_baseline_protocol_v1_1.json"
)

MANUSCRIPT_ROOT = (
    ROOT
    / "artifacts"
    / "temporal_baseline_manuscript_assets_v1"
)

MAIN_TABLE_PATH = (
    MANUSCRIPT_ROOT
    / "main_table_representative_models.csv"
)

SUPPLEMENT_TABLE_PATH = (
    MANUSCRIPT_ROOT
    / "supplementary_all_14_models.csv"
)

KEY_UNCERTAINTY_PATH = (
    MANUSCRIPT_ROOT
    / "key_paired_uncertainty.csv"
)

FULL_UNCERTAINTY_PATH = (
    MANUSCRIPT_ROOT
    / "all_paired_uncertainty_with_holm.csv"
)

MANUSCRIPT_READY_PATH = (
    MANUSCRIPT_ROOT
    / "MANUSCRIPT_READY_RESULTS.md"
)

REVIEWER_RESPONSE_PATH = (
    MANUSCRIPT_ROOT
    / "REVIEWER_RESPONSE_TEMPORAL_BASELINES.md"
)

DECISIONS_PATH = (
    MANUSCRIPT_ROOT
    / "REPORTING_DECISIONS.json"
)

FREEZE_SUMMARY_PATH = (
    MANUSCRIPT_ROOT
    / "FREEZE_SUMMARY.json"
)

ARCHIVE_PATH = (
    ROOT
    / "results"
    / "phyguard_temporal_baselines_final.zip"
)

SHA_PATH = Path(
    str(
        ARCHIVE_PATH
    )
    + ".sha256"
)

FREEZE_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "phyguard_temporal_baselines_final.json"
)

MAIN_MODEL_ORDER = [
    "PhyGuard",
    "MINIROCKET",
    "MULTIROCKET_HYDRA",
    "FCN1D",
    "INCEPTIONTIME_LITE",
    "TCN",
    "BILSTM",
    "TINY_TRANSFORMER",
]

ALL_MODEL_ORDER = [
    "PhyGuard",
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

KEY_GAIN_MODELS = [
    "MINIROCKET",
    "MULTIROCKET",
    "MULTIROCKET_HYDRA",
    "TCN",
]

METRIC_LABELS = {
    "selected_accuracy":
        "Selected accuracy",

    "coverage":
        "Coverage",

    "false_specific_rate":
        "False-specific rate",

    "physical_exact_match":
        "Physical exact match",

    "control_abstention":
        "Control abstention",
}

REQUIRED_STAGE_PATHS = [
    ROOT
    / "artifacts"
    / "temporal_baseline_environment_v1",

    ROOT
    / "artifacts"
    / "temporal_baseline_environment_hotfix_v1",

    ROOT
    / "artifacts"
    / "temporal_baseline_protocol_v1",

    ROOT
    / "artifacts"
    / "temporal_baseline_protocol_v1_1",

    ROOT
    / "artifacts"
    / "temporal_baseline_api_metric_audit_v1",

    ROOT
    / "artifacts"
    / "temporal_baseline_preflight_v1",

    RESULTS_ROOT,

    C2_ROOT,

    C3_ROOT,

    ROOT
    / "artifacts"
    / "phyguard_temporal_paired_uncertainty_protocol_amendment_v1",

    ROOT
    / "artifacts"
    / "phyguard_temporal_paired_uncertainty_metric_semantics_amendment_v1",

    ROOT
    / "artifacts"
    / "phyguard_temporal_paired_uncertainty_zero_selection_amendment_v1",

    C4_ROOT,
]

SCRIPT_NAMES = [
    "64b1_guard_14_model_protocol.py",
    "64b1_verify_14_model_environment.py",
    "64c0_audit_temporal_model_apis_and_metrics.py",
    "64c1_freeze_temporal_protocol_v1_1.py",
    "64c1_preflight_temporal_training.py",
    "64c1_run_temporal_baselines.py",
    "64c1_check_temporal_progress.py",
    "64c2_audit_and_recompute_selective_metrics.py",
    "64c3_compare_phyguard_with_temporal_baselines.py",
    "64c4_paired_uncertainty_analysis.py",
    "64c4a_patch_observed_cluster_union.py",
    "64c4b_patch_selected_accuracy_semantics.py",
    "64c4c_patch_zero_selection_convention.py",
    "64c5_freeze_temporal_baselines.py",
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


def holm_adjust(
    p_values: pd.Series,
) -> pd.Series:
    values = p_values.to_numpy(
        dtype=float
    )

    order = np.argsort(
        values
    )

    adjusted = np.empty_like(
        values
    )

    running_maximum = 0.0
    count = len(
        values
    )

    for rank, position in enumerate(
        order
    ):
        multiplier = count - rank

        candidate = min(
            1.0,
            multiplier
            * values[
                position
            ],
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted[
            position
        ] = running_maximum

    return pd.Series(
        adjusted,
        index=p_values.index,
    )


def mean_std_string(
    mean: float,
    standard_deviation: float,
) -> str:
    return (
        f"{mean:.4f} ± "
        f"{standard_deviation:.4f}"
    )


def build_method_table(
    phyguard: pd.DataFrame,
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    phyguard_row = phyguard.iloc[
        0
    ]

    rows.append(
        {
            "model_id":
                "PhyGuard",

            **{
                f"{metric}_mean":
                    float(
                        phyguard_row[
                            f"{metric}_mean"
                        ]
                    )

                for metric in METRIC_LABELS
            },

            **{
                f"{metric}_std":
                    float(
                        phyguard_row[
                            f"{metric}_std"
                        ]
                    )

                for metric in METRIC_LABELS
            },
        }
    )

    for _, row in baselines.iterrows():
        output = {
            "model_id":
                str(
                    row[
                        "model_id"
                    ]
                ),
        }

        for metric in METRIC_LABELS:
            output[
                f"{metric}_mean"
            ] = float(
                row[
                    f"{metric}_mean"
                ]
            )

            output[
                f"{metric}_std"
            ] = float(
                row[
                    f"{metric}_std"
                ]
            )

        rows.append(
            output
        )

    frame = pd.DataFrame(
        rows
    )

    frame[
        "model_id"
    ] = pd.Categorical(
        frame[
            "model_id"
        ],
        categories=ALL_MODEL_ORDER,
        ordered=True,
    )

    frame = (
        frame
        .sort_values(
            "model_id"
        )
        .reset_index(
            drop=True
        )
    )

    frame[
        "model_id"
    ] = frame[
        "model_id"
    ].astype(
        str
    )

    for metric in METRIC_LABELS:
        frame[
            METRIC_LABELS[
                metric
            ]
        ] = [
            mean_std_string(
                mean,
                standard_deviation,
            )
            for mean, standard_deviation
            in zip(
                frame[
                    f"{metric}_mean"
                ],
                frame[
                    f"{metric}_std"
                ],
            )
        ]

    return frame


for path in [
    C2_SUMMARY,
    C3_SUMMARY,
    C4_SUMMARY,
    BASELINE_REPEAT_PATH,
    BASELINE_AGGREGATE_PATH,
    PHYGUARD_AGGREGATE_PATH,
    PAIRWISE_PATH,
    BOOTSTRAP_PATH,
    PROTOCOL_V1,
    PROTOCOL_V11,
    *REQUIRED_STAGE_PATHS,
]:
    require(
        path.exists(),
        f"Required input is missing: {path}",
    )

for path in [
    MANUSCRIPT_ROOT,
    ARCHIVE_PATH,
    SHA_PATH,
    FREEZE_MANIFEST_PATH,
]:
    require(
        not path.exists(),
        f"Freeze output already exists: {path}",
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

c4_summary = json.loads(
    C4_SUMMARY.read_text(
        encoding="utf-8"
    )
)

require(
    c2_summary.get(
        "status"
    )
    == "PASS",
    "Stage 64C2 is not PASS.",
)

require(
    c3_summary.get(
        "status"
    )
    == "PASS",
    "Stage 64C3 is not PASS.",
)

require(
    c4_summary.get(
        "status"
    )
    == "PASS",
    "Stage 64C4 is not PASS.",
)

require(
    c2_summary.get(
        "task_count"
    )
    == 70,
    "Stage 64C2 task count is not 70.",
)

require(
    c4_summary.get(
        "bootstrap",
        {}
    ).get(
        "replicates"
    )
    == 10000,
    "Stage 64C4 bootstrap count is not 10000.",
)

require(
    c4_summary.get(
        "bootstrap",
        {}
    ).get(
        "cluster_count"
    )
    == 726,
    "Stage 64C4 cluster count is not 726.",
)

require(
    c3_summary.get(
        "baseline_pareto_dominators_core_three"
    )
    == [],
    "A baseline Pareto dominator exists for the core three metrics.",
)

require(
    c3_summary.get(
        "baseline_pareto_dominators_full_five"
    )
    == [],
    "A baseline Pareto dominator exists for the full five metrics.",
)


baseline_repeat = pd.read_csv(
    BASELINE_REPEAT_PATH
)

baseline_aggregate = pd.read_csv(
    BASELINE_AGGREGATE_PATH
)

phyguard_aggregate = pd.read_csv(
    PHYGUARD_AGGREGATE_PATH
)

pairwise = pd.read_csv(
    PAIRWISE_PATH
)

bootstrap = pd.read_csv(
    BOOTSTRAP_PATH
)

require(
    len(
        baseline_repeat
    )
    == 70,
    "Corrected per-repeat table does not have 70 rows.",
)

require(
    set(
        baseline_aggregate[
            "model_id"
        ].astype(
            str
        )
    )
    == set(
        ALL_MODEL_ORDER[
            1:
        ]
    ),
    "The 14-model aggregate set is incomplete.",
)

require(
    len(
        phyguard_aggregate
    )
    == 1,
    "PhyGuard aggregate table must have exactly one row.",
)

require(
    len(
        bootstrap
    )
    == 14
    * 5,
    "Paired uncertainty table must have 70 rows.",
)


bootstrap = bootstrap.copy()

bootstrap[
    "holm_adjusted_cluster_bootstrap_p"
] = np.nan

for metric, group in bootstrap.groupby(
    "metric",
    sort=False,
):
    adjusted = holm_adjust(
        group[
            "cluster_bootstrap_two_sided_p"
        ]
    )

    bootstrap.loc[
        group.index,
        "holm_adjusted_cluster_bootstrap_p",
    ] = adjusted


all_methods = build_method_table(
    phyguard_aggregate,
    baseline_aggregate,
)

main_table = all_methods[
    all_methods[
        "model_id"
    ].isin(
        MAIN_MODEL_ORDER
    )
].copy()

main_table[
    "model_id"
] = pd.Categorical(
    main_table[
        "model_id"
    ],
    categories=MAIN_MODEL_ORDER,
    ordered=True,
)

main_table = (
    main_table
    .sort_values(
        "model_id"
    )
    .reset_index(
        drop=True
    )
)

main_table[
    "model_id"
] = main_table[
    "model_id"
].astype(
    str
)

display_columns = [
    "model_id",
    "Selected accuracy",
    "Coverage",
    "False-specific rate",
    "Physical exact match",
    "Control abstention",
]

MANUSCRIPT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

main_table[
    display_columns
].to_csv(
    MAIN_TABLE_PATH,
    index=False,
)

all_methods[
    display_columns
].to_csv(
    SUPPLEMENT_TABLE_PATH,
    index=False,
)

bootstrap.to_csv(
    FULL_UNCERTAINTY_PATH,
    index=False,
)


key_rows = bootstrap[
    bootstrap[
        "model_id"
    ].astype(
        str
    ).isin(
        KEY_GAIN_MODELS
    )
    & bootstrap[
        "metric"
    ].isin(
        [
            "selected_accuracy",
            "false_specific_rate",
            "control_abstention",
        ]
    )
].copy()

key_rows[
    "model_id"
] = pd.Categorical(
    key_rows[
        "model_id"
    ],
    categories=KEY_GAIN_MODELS,
    ordered=True,
)

key_rows[
    "metric"
] = pd.Categorical(
    key_rows[
        "metric"
    ],
    categories=[
        "selected_accuracy",
        "false_specific_rate",
        "control_abstention",
    ],
    ordered=True,
)

key_rows = key_rows.sort_values(
    [
        "model_id",
        "metric",
    ]
)

key_rows[
    "model_id"
] = key_rows[
    "model_id"
].astype(
    str
)

key_rows[
    "metric"
] = key_rows[
    "metric"
].astype(
    str
)

key_rows.to_csv(
    KEY_UNCERTAINTY_PATH,
    index=False,
)


phyguard_row = phyguard_aggregate.iloc[
    0
]

key_lines = []

for model_id in KEY_GAIN_MODELS:
    model_rows = key_rows[
        key_rows[
            "model_id"
        ]
        == model_id
    ]

    selected = model_rows[
        model_rows[
            "metric"
        ]
        == "selected_accuracy"
    ].iloc[
        0
    ]

    fsr = model_rows[
        model_rows[
            "metric"
        ]
        == "false_specific_rate"
    ].iloc[
        0
    ]

    abstention = model_rows[
        model_rows[
            "metric"
        ]
        == "control_abstention"
    ].iloc[
        0
    ]

    key_lines.append(
        (
            f"- **{model_id}**: selected-accuracy difference "
            f"{selected['baseline_minus_phyguard']:+.4f} "
            f"(95% CI "
            f"{selected['cluster_bootstrap_ci_low']:+.4f} to "
            f"{selected['cluster_bootstrap_ci_high']:+.4f}); "
            f"false-specific-rate difference "
            f"{fsr['baseline_minus_phyguard']:+.4f} "
            f"(95% CI "
            f"{fsr['cluster_bootstrap_ci_low']:+.4f} to "
            f"{fsr['cluster_bootstrap_ci_high']:+.4f}, "
            f"Holm-adjusted p="
            f"{fsr['holm_adjusted_cluster_bootstrap_p']:.4f}); "
            f"control-abstention difference "
            f"{abstention['baseline_minus_phyguard']:+.4f} "
            f"(95% CI "
            f"{abstention['cluster_bootstrap_ci_low']:+.4f} to "
            f"{abstention['cluster_bootstrap_ci_high']:+.4f})."
        )
    )


manuscript_text = [
    "# Modern temporal baseline results",
    "",
    (
        "We compared PhyGuard against fourteen modern temporal "
        "classifiers under the same five stratified repeats, with "
        "648/216/216 train/validation/test sequences per repeat."
    ),
    "",
    (
        f"PhyGuard achieved selected accuracy "
        f"{phyguard_row['selected_accuracy_mean']:.4f} ± "
        f"{phyguard_row['selected_accuracy_std']:.4f}, coverage "
        f"{phyguard_row['coverage_mean']:.4f} ± "
        f"{phyguard_row['coverage_std']:.4f}, false-specific rate "
        f"{phyguard_row['false_specific_rate_mean']:.4f} ± "
        f"{phyguard_row['false_specific_rate_std']:.4f}, physical "
        f"exact match "
        f"{phyguard_row['physical_exact_match_mean']:.4f} ± "
        f"{phyguard_row['physical_exact_match_std']:.4f}, and "
        f"control abstention "
        f"{phyguard_row['control_abstention_mean']:.4f} ± "
        f"{phyguard_row['control_abstention_std']:.4f}."
    ),
    "",
    (
        "MiniRocket, MultiRocket, MultiRocket-Hydra, and TCN had "
        "small positive point differences in selected accuracy, "
        "but every 95% paired cluster-bootstrap interval crossed "
        "zero. Each also increased false-specific risk and reduced "
        "control abstention relative to PhyGuard."
    ),
    "",
    *key_lines,
    "",
    (
        "No temporal baseline Pareto-dominated PhyGuard on either "
        "the selected-accuracy/coverage/false-specific-rate triplet "
        "or the extended five-metric set including physical exact "
        "match and control abstention."
    ),
    "",
    (
        "The main manuscript table uses the representative model "
        "subset fixed in the pre-execution protocol. Results for all "
        "fourteen baselines are retained in the supplementary table."
    ),
    "",
    "## Statistical boundary",
    "",
    (
        "The paired bootstrap used 10,000 replicates over the 726 "
        "unique original-sample clusters appearing across the five "
        "repeated test sets. The five-repeat exact sign-flip test "
        "has a minimum attainable nonzero two-sided p-value of "
        "0.0625 and is therefore reported as a small-sample "
        "robustness check rather than the primary inferential test."
    ),
    "",
]

MANUSCRIPT_READY_PATH.write_text(
    "\n".join(
        manuscript_text
    ),
    encoding="utf-8",
)


reviewer_text = [
    "# Draft response: modern temporal baselines",
    "",
    "**Response.** We thank the reviewer for requesting comparisons "
    "with modern temporal models. We added fourteen baselines: "
    "ROCKET, MiniRocket, MultiRocket, Hydra, MultiRocket-Hydra, "
    "Arsenal, FCN, ResNet1D, InceptionTime-Lite, LITE, TCN, "
    "BiGRU, BiLSTM, and a compact Transformer. All methods used "
    "the same 80×10 KPI input, the same five stratified repeats, "
    "and validation-only threshold selection.",
    "",
    (
        f"PhyGuard obtained selected accuracy "
        f"{phyguard_row['selected_accuracy_mean']:.4f}, coverage "
        f"{phyguard_row['coverage_mean']:.4f}, false-specific rate "
        f"{phyguard_row['false_specific_rate_mean']:.4f}, and "
        f"control abstention "
        f"{phyguard_row['control_abstention_mean']:.4f}."
    ),
    "",
    (
        "The four baselines with higher point selected accuracy "
        "(MiniRocket, MultiRocket, MultiRocket-Hydra, and TCN) did "
        "not show a supported selected-accuracy advantage because "
        "their paired 95% bootstrap intervals all included zero. "
        "In contrast, each incurred a materially higher "
        "false-specific rate and lower control abstention. The "
        "Holm-adjusted paired-bootstrap results preserve the "
        "false-specific-risk disadvantage for all four models."
    ),
    "",
    (
        "These results clarify that PhyGuard's contribution is not "
        "unconditional classification dominance. Its advantage is "
        "a safer selective operating point that retains high "
        "diagnostic accuracy while suppressing unsupported specific "
        "diagnoses on controls. No baseline Pareto-dominated "
        "PhyGuard across the prespecified core or extended metric "
        "sets."
    ),
    "",
]

REVIEWER_RESPONSE_PATH.write_text(
    "\n".join(
        reviewer_text
    ),
    encoding="utf-8",
)


decisions = {
    "schema":
        "phyguard.temporal_baseline_reporting_decisions.v1",

    "status":
        "FROZEN",

    "main_table_models":
        MAIN_MODEL_ORDER,

    "main_table_basis":
        (
            "Representative subset fixed in the temporal-baseline "
            "protocol before model execution."
        ),

    "supplementary_models":
        ALL_MODEL_ORDER,

    "outcome_based_model_omission":
        False,

    "primary_comparison_metrics": [
        "selected_accuracy",
        "coverage",
        "false_specific_rate",
        "physical_exact_match",
        "control_abstention",
    ],

    "primary_uncertainty_method":
        (
            "10,000-replicate paired cluster bootstrap over 726 "
            "unique original sample indices."
        ),

    "multiplicity_adjustment":
        (
            "Holm adjustment across the 14 model comparisons "
            "separately within each metric."
        ),

    "exact_signflip_role":
        (
            "Small-sample robustness check; with five repeats the "
            "minimum nonzero two-sided p-value is 0.0625."
        ),

    "formal_hardware":
        "NVIDIA GeForce RTX 3080 10 GB",

    "claim_boundary":
        (
            "Do not claim universal accuracy superiority. Claim "
            "that no baseline Pareto-dominates PhyGuard and that "
            "point accuracy gains of the four strongest competitors "
            "trade off against higher false-specific risk and lower "
            "control abstention."
        ),
}

DECISIONS_PATH.write_text(
    json.dumps(
        decisions,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


freeze_summary = {
    "schema":
        "phyguard.temporal_baselines_final_freeze.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "PASS",

    "model_count":
        14,

    "repeat_count":
        5,

    "task_count":
        70,

    "test_rows_per_repeat":
        216,

    "aligned_prediction_rows":
        1080,

    "unique_bootstrap_clusters":
        726,

    "bootstrap_replicates":
        10000,

    "phyguard_metrics": {
        metric:
            {
                "mean":
                    float(
                        phyguard_row[
                            f"{metric}_mean"
                        ]
                    ),

                "std":
                    float(
                        phyguard_row[
                            f"{metric}_std"
                        ]
                    ),
            }

        for metric in METRIC_LABELS
    },

    "point_selected_accuracy_gain_models":
        KEY_GAIN_MODELS,

    "supported_baseline_selected_accuracy_advantage_models":
        [],

    "pareto_dominators_core_three":
        [],

    "pareto_dominators_full_five":
        [],

    "main_table_models":
        MAIN_MODEL_ORDER,

    "formal_hardware":
        "NVIDIA GeForce RTX 3080 10 GB",
}

FREEZE_SUMMARY_PATH.write_text(
    json.dumps(
        freeze_summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


archive_sources = [
    PROTOCOL_V1,
    PROTOCOL_V11,
    *REQUIRED_STAGE_PATHS,
    MANUSCRIPT_ROOT,
]

for script_name in SCRIPT_NAMES:
    script_path = (
        ROOT
        / "scripts"
        / script_name
    )

    require(
        script_path.exists(),
        f"Required script is missing: {script_path}",
    )

    archive_sources.append(
        script_path
    )


with tempfile.TemporaryDirectory(
    prefix="phyguard_temporal_freeze_"
) as temporary_directory:
    staging_root = (
        Path(
            temporary_directory
        )
        / "phyguard_temporal_baselines_final"
    )

    staging_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    copied_paths = []

    for source in archive_sources:
        relative = source.relative_to(
            ROOT
        )

        target = staging_root / relative

        if source.is_dir():
            shutil.copytree(
                source,
                target,
            )
        else:
            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                source,
                target,
            )

        copied_paths.append(
            relative
        )

    checksum_rows = []

    for path in sorted(
        staging_root.rglob("*")
    ):
        if path.is_file():
            checksum_rows.append(
                {
                    "path":
                        str(
                            path.relative_to(
                                staging_root
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

    checksum_path = (
        staging_root
        / "SHA256SUMS.txt"
    )

    checksum_path.write_text(
        "\n".join(
            (
                f"{row['sha256']}  "
                f"{row['path']}"
            )
            for row in checksum_rows
        )
        + "\n",
        encoding="utf-8",
    )

    archive_summary = {
        "schema":
            "phyguard.temporal_baselines_archive_manifest.v1",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "PASS",

        "copied_top_level_sources": [
            str(
                path
            )
            for path in copied_paths
        ],

        "file_count_before_archive_manifest":
            len(
                checksum_rows
            ),

        "formal_hardware":
            "NVIDIA GeForce RTX 3080 10 GB",
    }

    archive_manifest_path = (
        staging_root
        / "ARCHIVE_MANIFEST.json"
    )

    archive_manifest_path.write_text(
        json.dumps(
            archive_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    ARCHIVE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        ARCHIVE_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(
            staging_root.rglob("*")
        ):
            if path.is_file():
                archive.write(
                    path,
                    Path(
                        staging_root.name
                    )
                    / path.relative_to(
                        staging_root
                    ),
                )


archive_sha = sha256_file(
    ARCHIVE_PATH
)

SHA_PATH.write_text(
    (
        f"{archive_sha}  "
        f"{ARCHIVE_PATH.name}\n"
    ),
    encoding="utf-8",
)

freeze_manifest = {
    "schema":
        "phyguard.temporal_baselines_final_manifest.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "PASS",

    "archive_path":
        str(
            ARCHIVE_PATH
        ),

    "archive_size_bytes":
        ARCHIVE_PATH.stat().st_size,

    "archive_sha256":
        archive_sha,

    "sha256_file":
        str(
            SHA_PATH
        ),

    "manuscript_assets":
        str(
            MANUSCRIPT_ROOT
        ),

    "model_count":
        14,

    "task_count":
        70,

    "bootstrap_replicates":
        10000,

    "cluster_count":
        726,

    "formal_hardware":
        "NVIDIA GeForce RTX 3080 10 GB",
}

FREEZE_MANIFEST_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

FREEZE_MANIFEST_PATH.write_text(
    json.dumps(
        freeze_manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print("status: PASS")
print("model_count: 14")
print("repeat_count: 5")
print("task_count: 70")
print("bootstrap_replicates: 10000")
print("cluster_count: 726")
print(
    "supported_baseline_selected_accuracy_advantage_models: []"
)
print("pareto_dominators_core_three: []")
print("pareto_dominators_full_five: []")
print(
    "formal_hardware: NVIDIA GeForce RTX 3080 10 GB"
)
print("main_table:", MAIN_TABLE_PATH)
print(
    "supplementary_table:",
    SUPPLEMENT_TABLE_PATH,
)
print(
    "key_uncertainty:",
    KEY_UNCERTAINTY_PATH,
)
print(
    "manuscript_ready:",
    MANUSCRIPT_READY_PATH,
)
print(
    "reviewer_response:",
    REVIEWER_RESPONSE_PATH,
)
print("archive:", ARCHIVE_PATH)
print(
    "archive_size_bytes:",
    ARCHIVE_PATH.stat().st_size,
)
print(
    "archive_sha256:",
    archive_sha,
)
print("sha256_file:", SHA_PATH)
print(
    "freeze_manifest:",
    FREEZE_MANIFEST_PATH,
)
print()
print(
    "TEMPORAL_BASELINE_FINAL_FREEZE_PASS"
)
