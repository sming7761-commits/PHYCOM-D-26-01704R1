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

import pandas as pd


ROOT = Path("/root/phyguard_revision")

RESULT_ROOT = (
    ROOT
    / "artifacts"
    / "per_indicator_ablation_results_v1"
)

PROTOCOL_ROOT = (
    ROOT
    / "artifacts"
    / "per_indicator_ablation_protocol_v1"
)

AUDIT_ROOT = (
    ROOT
    / "artifacts"
    / "per_indicator_ablation_implementation_audit_v1"
)

PROTOCOL_CONFIG = (
    ROOT
    / "configs"
    / "per_indicator_ablation_protocol_v1.json"
)

PROTOCOL_MANIFEST = (
    ROOT
    / "manifests"
    / "per_indicator_ablation_protocol_v1.json"
)

AUDIT_MANIFEST = (
    ROOT
    / "manifests"
    / "per_indicator_ablation_implementation_audit_v1.json"
)

RESULT_MANIFEST = (
    ROOT
    / "manifests"
    / "per_indicator_ablation_results_v1.json"
)

RUN_SCRIPT = (
    ROOT
    / "scripts"
    / "62b_run_per_indicator_ablation.py"
)

RUN_LOG = (
    ROOT
    / "logs"
    / "62b_run_per_indicator_ablation.log"
)

OUTPUT_ZIP = (
    ROOT
    / "results"
    / "phyguard_per_indicator_ablation_v1_final.zip"
)

OUTPUT_SHA = OUTPUT_ZIP.with_suffix(
    ".zip.sha256"
)

EXPECTED_RESULT_FILES = [
    "feature_mask.json",
    "aggregate_variant_metrics.csv",
    "paired_primary_deltas.csv",
    "paired_repeat_deltas.csv",
    "factorial_effects.csv",
    "source_predictions_all_variants.csv",
    "source_per_repeat_metrics_all_variants.csv",
    "sionna_predictions_all_variants.csv",
    "sionna_per_repeat_metrics_all_variants.csv",
    "paired_bootstrap_distributions.npz",
    "summary.json",
]

EXPECTED = {
    "classification":
        "PACKET_BLOCK_ERROR_INDICATOR_NOT_ESSENTIAL",

    "source_full_selected_accuracy":
        0.943239,

    "source_no_error_selected_accuracy":
        0.958403,

    "source_no_error_delta":
        0.015164,

    "source_no_error_ci_low":
        0.004866,

    "source_no_error_ci_high":
        0.027498,

    "sionna_full_selected_accuracy":
        0.600342,

    "sionna_no_error_selected_accuracy":
        0.602969,

    "sionna_no_error_delta":
        0.002628,

    "sionna_no_error_ci_low":
        -0.010625,

    "sionna_no_error_ci_high":
        0.015760,

    "source_full_prediction_count":
        1080,

    "sionna_full_prediction_count":
        4125,

    "raw_drop_indices":
        [30, 31, 32, 33, 34, 35],

    "physical_zero_indices":
        [5],

    "retention_margin":
        -0.05,

    "bootstrap_replicates":
        10000,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(
    actual: float,
    expected: float,
    tolerance: float = 5e-7,
) -> bool:
    return abs(actual - expected) <= tolerance


required_paths = [
    RESULT_ROOT,
    PROTOCOL_ROOT,
    AUDIT_ROOT,
    PROTOCOL_CONFIG,
    PROTOCOL_MANIFEST,
    AUDIT_MANIFEST,
    RESULT_MANIFEST,
    RUN_SCRIPT,
    RUN_LOG,
]

for path in required_paths:
    require(
        path.exists(),
        f"Required path is missing: {path}",
    )

for filename in EXPECTED_RESULT_FILES:
    path = RESULT_ROOT / filename
    require(
        path.exists(),
        f"Expected result file is missing: {path}",
    )

require(
    not OUTPUT_ZIP.exists(),
    f"Final archive already exists: {OUTPUT_ZIP}",
)

require(
    not OUTPUT_SHA.exists(),
    f"Final SHA file already exists: {OUTPUT_SHA}",
)


log_text = RUN_LOG.read_text(
    encoding="utf-8",
    errors="replace",
)

require(
    "PER_INDICATOR_ABLATION_RESULTS_V1_PASS"
    in log_text,
    "Stage 62B PASS marker is missing from the run log.",
)

require(
    "SOURCE FULL PARITY PASS"
    in log_text,
    "Source FULL parity marker is missing.",
)

require(
    "SIONNA FULL PARITY PASS"
    in log_text,
    "Sionna FULL parity marker is missing.",
)


summary = json.loads(
    (
        RESULT_ROOT
        / "summary.json"
    ).read_text(
        encoding="utf-8"
    )
)

feature_mask = json.loads(
    (
        RESULT_ROOT
        / "feature_mask.json"
    ).read_text(
        encoding="utf-8"
    )
)

aggregate = pd.read_csv(
    RESULT_ROOT
    / "aggregate_variant_metrics.csv"
)

deltas = pd.read_csv(
    RESULT_ROOT
    / "paired_primary_deltas.csv"
)

factorial = pd.read_csv(
    RESULT_ROOT
    / "factorial_effects.csv"
)

require(
    summary.get("status") == "PASS",
    "Result summary status is not PASS.",
)

require(
    summary.get("classification")
    == EXPECTED["classification"],
    "Result classification mismatch.",
)

require(
    summary.get("retention_margin")
    == EXPECTED["retention_margin"],
    "Retention margin mismatch.",
)

require(
    summary.get("bootstrap_replicates")
    == EXPECTED["bootstrap_replicates"],
    "Bootstrap replicate count mismatch.",
)

require(
    feature_mask.get("raw_indices")
    == EXPECTED["raw_drop_indices"],
    "Raw feature-mask indices mismatch.",
)

require(
    feature_mask.get("physical_indices")
    == EXPECTED["physical_zero_indices"],
    "Physical feature-mask indices mismatch.",
)

source_parity = summary[
    "source_full_parity"
]

sionna_parity = summary[
    "sionna_full_parity"
]

require(
    close(
        float(
            source_parity[
                "maximum_metric_difference"
            ]
        ),
        0.0,
        tolerance=1e-15,
    ),
    "Source FULL metric parity is not exact.",
)

require(
    int(
        source_parity[
            "prediction_mismatch_count"
        ]
    )
    == 0,
    "Source FULL prediction parity failed.",
)

require(
    int(
        source_parity[
            "prediction_count"
        ]
    )
    == EXPECTED[
        "source_full_prediction_count"
    ],
    "Unexpected source FULL prediction count.",
)

require(
    int(
        sionna_parity[
            "prediction_mismatch_count"
        ]
    )
    == 0,
    "Sionna FULL prediction parity failed.",
)

require(
    int(
        sionna_parity[
            "prediction_count"
        ]
    )
    == EXPECTED[
        "sionna_full_prediction_count"
    ],
    "Unexpected Sionna FULL prediction count.",
)


for variant in [
    "NO_PER_RAW",
    "NO_PER_SCORE",
    "NO_ERROR_INDICATOR",
]:
    require(
        summary[
            "retention"
        ][
            variant
        ][
            "source"
        ]
        is True,
        f"Source retention failed for {variant}.",
    )

    require(
        summary[
            "retention"
        ][
            variant
        ][
            "sionna"
        ]
        is True,
        f"Sionna retention failed for {variant}.",
    )


def get_aggregate(
    system: str,
    variant: str,
) -> pd.Series:
    rows = aggregate[
        (
            aggregate["system"]
            == system
        )
        & (
            aggregate["variant"]
            == variant
        )
    ]

    require(
        len(rows) == 1,
        (
            "Expected exactly one aggregate row for "
            f"{system}/{variant}; found {len(rows)}."
        ),
    )

    return rows.iloc[0]


def get_delta(
    system: str,
    variant: str,
) -> pd.Series:
    rows = deltas[
        (
            deltas["system"]
            == system
        )
        & (
            deltas["variant"]
            == variant
        )
    ]

    require(
        len(rows) == 1,
        (
            "Expected exactly one delta row for "
            f"{system}/{variant}; found {len(rows)}."
        ),
    )

    return rows.iloc[0]


source_full = get_aggregate(
    "source",
    "FULL",
)

source_no_error = get_aggregate(
    "source",
    "NO_ERROR_INDICATOR",
)

sionna_full = get_aggregate(
    "sionna",
    "FULL",
)

sionna_no_error = get_aggregate(
    "sionna",
    "NO_ERROR_INDICATOR",
)

source_no_error_delta = get_delta(
    "source",
    "NO_ERROR_INDICATOR",
)

sionna_no_error_delta = get_delta(
    "sionna",
    "NO_ERROR_INDICATOR",
)


checks = [
    (
        float(
            source_full[
                "selected_accuracy"
            ]
        ),
        EXPECTED[
            "source_full_selected_accuracy"
        ],
        "source FULL selected accuracy",
    ),
    (
        float(
            source_no_error[
                "selected_accuracy"
            ]
        ),
        EXPECTED[
            "source_no_error_selected_accuracy"
        ],
        "source NO_ERROR selected accuracy",
    ),
    (
        float(
            source_no_error_delta[
                "absolute_delta"
            ]
        ),
        EXPECTED[
            "source_no_error_delta"
        ],
        "source NO_ERROR delta",
    ),
    (
        float(
            source_no_error_delta[
                "bootstrap_lower_95"
            ]
        ),
        EXPECTED[
            "source_no_error_ci_low"
        ],
        "source NO_ERROR CI low",
    ),
    (
        float(
            source_no_error_delta[
                "bootstrap_upper_95"
            ]
        ),
        EXPECTED[
            "source_no_error_ci_high"
        ],
        "source NO_ERROR CI high",
    ),
    (
        float(
            sionna_full[
                "selected_accuracy"
            ]
        ),
        EXPECTED[
            "sionna_full_selected_accuracy"
        ],
        "Sionna FULL selected accuracy",
    ),
    (
        float(
            sionna_no_error[
                "selected_accuracy"
            ]
        ),
        EXPECTED[
            "sionna_no_error_selected_accuracy"
        ],
        "Sionna NO_ERROR selected accuracy",
    ),
    (
        float(
            sionna_no_error_delta[
                "absolute_delta"
            ]
        ),
        EXPECTED[
            "sionna_no_error_delta"
        ],
        "Sionna NO_ERROR delta",
    ),
    (
        float(
            sionna_no_error_delta[
                "bootstrap_lower_95"
            ]
        ),
        EXPECTED[
            "sionna_no_error_ci_low"
        ],
        "Sionna NO_ERROR CI low",
    ),
    (
        float(
            sionna_no_error_delta[
                "bootstrap_upper_95"
            ]
        ),
        EXPECTED[
            "sionna_no_error_ci_high"
        ],
        "Sionna NO_ERROR CI high",
    ),
]

for actual, expected, label in checks:
    require(
        close(actual, expected),
        (
            f"Unexpected {label}: "
            f"expected={expected}, actual={actual}"
        ),
    )


sionna_operating_profile = {
    "full_physical_selection_rate":
        float(
            sionna_full[
                "physical_selection_rate"
            ]
        ),

    "no_error_physical_selection_rate":
        float(
            sionna_no_error[
                "physical_selection_rate"
            ]
        ),

    "physical_selection_delta":
        float(
            sionna_no_error[
                "physical_selection_rate"
            ]
            - sionna_full[
                "physical_selection_rate"
            ]
        ),

    "full_false_specific_rate":
        float(
            sionna_full[
                "false_specific_rate"
            ]
        ),

    "no_error_false_specific_rate":
        float(
            sionna_no_error[
                "false_specific_rate"
            ]
        ),

    "false_specific_delta":
        float(
            sionna_no_error[
                "false_specific_rate"
            ]
            - sionna_full[
                "false_specific_rate"
            ]
        ),

    "full_physical_exact_match":
        float(
            sionna_full[
                "physical_exact_match"
            ]
        ),

    "no_error_physical_exact_match":
        float(
            sionna_no_error[
                "physical_exact_match"
            ]
        ),

    "physical_exact_match_delta":
        float(
            sionna_no_error[
                "physical_exact_match"
            ]
            - sionna_full[
                "physical_exact_match"
            ]
        ),

    "full_control_abstention":
        float(
            sionna_full[
                "control_abstention"
            ]
        ),

    "no_error_control_abstention":
        float(
            sionna_no_error[
                "control_abstention"
            ]
        ),

    "control_abstention_delta":
        float(
            sionna_no_error[
                "control_abstention"
            ]
            - sionna_full[
                "control_abstention"
            ]
        ),
}


freeze_summary = {
    "schema":
        "phyguard.per_indicator_ablation_freeze.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "FROZEN",

    "classification":
        EXPECTED[
            "classification"
        ],

    "feature_mask": {
        "raw_drop_indices":
            EXPECTED[
                "raw_drop_indices"
            ],

        "physical_zero_indices":
            EXPECTED[
                "physical_zero_indices"
            ],
    },

    "source": {
        "full_selected_accuracy":
            float(
                source_full[
                    "selected_accuracy"
                ]
            ),

        "no_error_selected_accuracy":
            float(
                source_no_error[
                    "selected_accuracy"
                ]
            ),

        "no_error_absolute_delta":
            float(
                source_no_error_delta[
                    "absolute_delta"
                ]
            ),

        "no_error_bootstrap_ci95": [
            float(
                source_no_error_delta[
                    "bootstrap_lower_95"
                ]
            ),
            float(
                source_no_error_delta[
                    "bootstrap_upper_95"
                ]
            ),
        ],

        "no_error_retained":
            bool(
                source_no_error_delta[
                    "retained"
                ]
            ),
    },

    "sionna": {
        "full_selected_accuracy":
            float(
                sionna_full[
                    "selected_accuracy"
                ]
            ),

        "no_error_selected_accuracy":
            float(
                sionna_no_error[
                    "selected_accuracy"
                ]
            ),

        "no_error_absolute_delta":
            float(
                sionna_no_error_delta[
                    "absolute_delta"
                ]
            ),

        "no_error_bootstrap_ci95": [
            float(
                sionna_no_error_delta[
                    "bootstrap_lower_95"
                ]
            ),
            float(
                sionna_no_error_delta[
                    "bootstrap_upper_95"
                ]
            ),
        ],

        "no_error_retained":
            bool(
                sionna_no_error_delta[
                    "retained"
                ]
            ),

        "operating_profile":
            sionna_operating_profile,
    },

    "interpretation_boundary": {
        "supported":
            (
                "The packet/block-error indicator is not "
                "essential for retaining the preregistered "
                "primary selected-accuracy endpoint within "
                "the -0.05 practical margin in either system."
            ),

        "not_supported":
            (
                "The result does not show that the indicator "
                "is useless or physically meaningless."
            ),

        "secondary_tradeoff":
            (
                "Removing the indicator changes the Sionna "
                "coverage/selectivity operating profile, "
                "including lower physical selection and "
                "physical exact match and a higher "
                "false-specific rate."
            ),

        "ber_boundary":
            (
                "BER/ber_pre_ldpc remained present in all "
                "variants. The experiment removes only "
                "per_proxy/bler_post_ldpc."
            ),

        "aggregation_boundary":
            (
                "The preregistered retention decision is based "
                "on the paired sample-cluster bootstrap. "
                "Per-repeat mean deltas remain descriptive and "
                "may differ from pooled selected-accuracy "
                "deltas when coverage denominators vary."
            ),
    },
}


manuscript_text = f"""# Packet/block-error indicator ablation

## Frozen result

We conducted a preregistered 2×2 factorial ablation separating the raw
packet/block-error feature from its physics-score contribution. The source-domain
indicator was the revision-time locked 48-bit uncoded packet-error proxy
(`per_proxy`), and the coded Sionna counterpart was post-LDPC BLER
(`bler_post_ldpc`). BER remained available in every variant.

Removing the packet/block-error indicator from both paths did not reduce the
primary selected-accuracy endpoint beyond the preregistered -5 percentage-point
retention margin. In the source reconstruction, selected accuracy changed from
{float(source_full['selected_accuracy']):.3f} to
{float(source_no_error['selected_accuracy']):.3f}, a paired pooled difference of
{float(source_no_error_delta['absolute_delta']):+.3f}
(95% sample-cluster bootstrap interval
[{float(source_no_error_delta['bootstrap_lower_95']):+.3f},
 {float(source_no_error_delta['bootstrap_upper_95']):+.3f}]).
In the frozen Sionna evaluation, it changed from
{float(sionna_full['selected_accuracy']):.3f} to
{float(sionna_no_error['selected_accuracy']):.3f}, a difference of
{float(sionna_no_error_delta['absolute_delta']):+.3f}
(95% interval
[{float(sionna_no_error_delta['bootstrap_lower_95']):+.3f},
 {float(sionna_no_error_delta['bootstrap_upper_95']):+.3f}]).

The correct interpretation is that the packet/block-error indicator was **not
essential for retaining selected accuracy**, not that it had no operational
value. In Sionna, removing it reduced physical selection from
{sionna_operating_profile['full_physical_selection_rate']:.3f} to
{sionna_operating_profile['no_error_physical_selection_rate']:.3f}, reduced
physical exact match from
{sionna_operating_profile['full_physical_exact_match']:.3f} to
{sionna_operating_profile['no_error_physical_exact_match']:.3f}, and increased
the false-specific rate from
{sionna_operating_profile['full_false_specific_rate']:.3f} to
{sionna_operating_profile['no_error_false_specific_rate']:.3f}. The indicator
therefore affects the coverage/selectivity operating profile even though the
primary selected-accuracy retention criterion remains satisfied.

## Reviewer-response wording

We added a preregistered 2×2 ablation that independently removed the raw
packet/block-error feature and its physics-score contribution. All three
ablations retained selected accuracy within the prespecified -5 percentage-point
margin in both the reconstruction and frozen Sionna evaluation. The combined
removal changed selected accuracy by
{100.0 * float(source_no_error_delta['absolute_delta']):+.2f} percentage points
in the reconstruction and
{100.0 * float(sionna_no_error_delta['absolute_delta']):+.2f} percentage points
in Sionna. This shows that the 48-bit proxy is not a single point of failure for
the primary endpoint. We nevertheless retain a limited interpretation because
the Sionna operating profile changed: physical selection and physical exact
match decreased, while false-specific output increased. BER was retained in
every variant, so this experiment specifically evaluates dependence on the
packet/block-error indicator rather than removing the entire error-evidence
family.
"""


with tempfile.TemporaryDirectory(
    prefix="stage62c_",
    dir=str(ROOT / "results"),
) as temporary_directory:
    stage = Path(temporary_directory)

    for directory in [
        "artifacts",
        "configs",
        "manifests",
        "scripts",
        "logs",
    ]:
        (
            stage
            / directory
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    shutil.copytree(
        RESULT_ROOT,
        stage
        / "artifacts"
        / RESULT_ROOT.name,
    )

    shutil.copytree(
        PROTOCOL_ROOT,
        stage
        / "artifacts"
        / PROTOCOL_ROOT.name,
    )

    shutil.copytree(
        AUDIT_ROOT,
        stage
        / "artifacts"
        / AUDIT_ROOT.name,
    )

    for source, destination in [
        (
            PROTOCOL_CONFIG,
            stage
            / "configs"
            / PROTOCOL_CONFIG.name,
        ),
        (
            PROTOCOL_MANIFEST,
            stage
            / "manifests"
            / PROTOCOL_MANIFEST.name,
        ),
        (
            AUDIT_MANIFEST,
            stage
            / "manifests"
            / AUDIT_MANIFEST.name,
        ),
        (
            RESULT_MANIFEST,
            stage
            / "manifests"
            / RESULT_MANIFEST.name,
        ),
        (
            RUN_SCRIPT,
            stage
            / "scripts"
            / RUN_SCRIPT.name,
        ),
        (
            RUN_LOG,
            stage
            / "logs"
            / RUN_LOG.name,
        ),
    ]:
        shutil.copy2(
            source,
            destination,
        )

    (
        stage
        / "FREEZE_SUMMARY.json"
    ).write_text(
        json.dumps(
            freeze_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        stage
        / "MANUSCRIPT_READY_RESULTS.md"
    ).write_text(
        manuscript_text,
        encoding="utf-8",
    )

    checksum_lines = []

    for path in sorted(
        stage.rglob("*")
    ):
        if (
            path.is_file()
            and path.name
            != "SHA256SUMS.txt"
        ):
            checksum_lines.append(
                f"{sha256_file(path)}  "
                f"{path.relative_to(stage).as_posix()}"
            )

    (
        stage
        / "SHA256SUMS.txt"
    ).write_text(
        "\n".join(
            checksum_lines
        )
        + "\n",
        encoding="utf-8",
    )

    OUTPUT_ZIP.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        OUTPUT_ZIP,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(
            stage.rglob("*")
        ):
            if path.is_file():
                archive.write(
                    path,
                    path.relative_to(stage),
                )


archive_sha256 = sha256_file(
    OUTPUT_ZIP
)

OUTPUT_SHA.write_text(
    f"{archive_sha256}  {OUTPUT_ZIP.name}\n",
    encoding="utf-8",
)


print(
    "classification:",
    EXPECTED[
        "classification"
    ],
)
print(
    "source_full_selected_accuracy:",
    format(
        float(
            source_full[
                "selected_accuracy"
            ]
        ),
        ".6f",
    ),
)
print(
    "source_no_error_selected_accuracy:",
    format(
        float(
            source_no_error[
                "selected_accuracy"
            ]
        ),
        ".6f",
    ),
)
print(
    "source_no_error_delta_ci95:",
    (
        format(
            float(
                source_no_error_delta[
                    "absolute_delta"
                ]
            ),
            ".6f",
        ),
        format(
            float(
                source_no_error_delta[
                    "bootstrap_lower_95"
                ]
            ),
            ".6f",
        ),
        format(
            float(
                source_no_error_delta[
                    "bootstrap_upper_95"
                ]
            ),
            ".6f",
        ),
    ),
)
print(
    "sionna_full_selected_accuracy:",
    format(
        float(
            sionna_full[
                "selected_accuracy"
            ]
        ),
        ".6f",
    ),
)
print(
    "sionna_no_error_selected_accuracy:",
    format(
        float(
            sionna_no_error[
                "selected_accuracy"
            ]
        ),
        ".6f",
    ),
)
print(
    "sionna_no_error_delta_ci95:",
    (
        format(
            float(
                sionna_no_error_delta[
                    "absolute_delta"
                ]
            ),
            ".6f",
        ),
        format(
            float(
                sionna_no_error_delta[
                    "bootstrap_lower_95"
                ]
            ),
            ".6f",
        ),
        format(
            float(
                sionna_no_error_delta[
                    "bootstrap_upper_95"
                ]
            ),
            ".6f",
        ),
    ),
)
print(
    "sionna_operating_profile:",
    sionna_operating_profile,
)
print(
    "archive:",
    OUTPUT_ZIP,
)
print(
    "archive_size_bytes:",
    OUTPUT_ZIP.stat().st_size,
)
print(
    "archive_sha256:",
    archive_sha256,
)
print(
    "sha_file:",
    OUTPUT_SHA,
)
print()
print(
    "PER_INDICATOR_ABLATION_V1_FINAL_FREEZE_PASS"
)
