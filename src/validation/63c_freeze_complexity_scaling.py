#!/usr/bin/env python3
from __future__ import annotations

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
    / "complexity_scaling_results_v1"
)

PROTOCOL_ROOT = (
    ROOT
    / "artifacts"
    / "complexity_scaling_protocol_v1"
)

PROTOCOL_CONFIG = (
    ROOT
    / "configs"
    / "complexity_scaling_protocol_v1.json"
)

PROTOCOL_MANIFEST = (
    ROOT
    / "manifests"
    / "complexity_scaling_protocol_v1.json"
)

RESULT_MANIFEST = (
    ROOT
    / "manifests"
    / "complexity_scaling_results_v1.json"
)

EXECUTION_SCRIPT = (
    ROOT
    / "scripts"
    / "63b_run_complexity_scaling.py"
)

EXECUTION_LOG = (
    ROOT
    / "logs"
    / "63b_run_complexity_scaling.log"
)

HOTFIX_REPORT = (
    ROOT
    / "artifacts"
    / "complexity_scaling_protocol_v1"
    / "stage63b_pipeline_topology_hotfix_v2_report.json"
)

OUTPUT_ZIP = (
    ROOT
    / "results"
    / "phyguard_complexity_scaling_v1_final.zip"
)

OUTPUT_SHA = OUTPUT_ZIP.with_suffix(
    ".zip.sha256"
)

EXPECTED_RESULT_FILES = [
    "theoretical_complexity.json",
    "theoretical_complexity.md",
    "runtime_environment.json",
    "model_topology.json",
    "t_scaling.csv",
    "k_scaling.csv",
    "inference_batch_scaling.csv",
    "summary.json",
]

EXPECTED = {
    "theoretical_total_time":
        "O(TK + F_g + MD)",

    "simplified_total_time":
        "O(TK + MD)",

    "gate_input_dimension":
        19,

    "resolver_input_dimension":
        79,

    "tree_count":
        200,

    "total_tree_nodes":
        34730,

    "mean_tree_depth":
        13.855,

    "maximum_tree_depth":
        18,

    "serialized_model_bytes":
        3417386,

    "t_slope":
        0.0763717805577184,

    "t_r2":
        0.8318892585422626,

    "k_slope":
        0.33358629663670925,

    "k_r2":
        0.9561826884972218,

    "batch_slope":
        0.13929764746610698,

    "batch_r2":
        0.6712547885686255,

    "t_40_median_us":
        3132.7795,

    "t_1280_median_us":
        4155.3270,

    "batch_1_median_ms":
        21.360898,

    "batch_2048_median_ms":
        71.004687,

    "batch_1_per_interval_us":
        21360.898000,

    "batch_2048_per_interval_us":
        34.670257,

    "batch_2048_throughput":
        28843.166570,
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
    tolerance: float = 5e-6,
) -> bool:
    return abs(actual - expected) <= tolerance


required_paths = [
    RESULT_ROOT,
    PROTOCOL_ROOT,
    PROTOCOL_CONFIG,
    PROTOCOL_MANIFEST,
    RESULT_MANIFEST,
    EXECUTION_SCRIPT,
    EXECUTION_LOG,
    HOTFIX_REPORT,
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

log_text = EXECUTION_LOG.read_text(
    encoding="utf-8",
    errors="replace",
)

require(
    "COMPLEXITY_SCALING_RESULTS_V1_PASS"
    in log_text,
    "Stage 63B PASS marker is missing from the log.",
)

summary = json.loads(
    (
        RESULT_ROOT
        / "summary.json"
    ).read_text(
        encoding="utf-8"
    )
)

topology = json.loads(
    (
        RESULT_ROOT
        / "model_topology.json"
    ).read_text(
        encoding="utf-8"
    )
)

t_table = pd.read_csv(
    RESULT_ROOT
    / "t_scaling.csv"
)

k_table = pd.read_csv(
    RESULT_ROOT
    / "k_scaling.csv"
)

inference_table = pd.read_csv(
    RESULT_ROOT
    / "inference_batch_scaling.csv"
)

require(
    summary.get("status") == "PASS",
    "Complexity summary status is not PASS.",
)

require(
    summary.get("theoretical_total_time")
    == EXPECTED["theoretical_total_time"],
    "Theoretical complexity mismatch.",
)

require(
    summary.get("simplified_total_time")
    == EXPECTED["simplified_total_time"],
    "Simplified complexity mismatch.",
)

for key in [
    "gate_input_dimension",
    "resolver_input_dimension",
    "tree_count",
    "total_tree_nodes",
    "maximum_tree_depth",
    "serialized_model_bytes",
]:
    require(
        int(topology[key])
        == int(EXPECTED[key]),
        f"Topology mismatch for {key}.",
    )

require(
    close(
        float(topology["mean_tree_depth"]),
        EXPECTED["mean_tree_depth"],
    ),
    "Mean tree depth mismatch.",
)

fits = {
    "t":
        summary[
            "t_scaling_log_log_fit"
        ],

    "k":
        summary[
            "k_scaling_log_log_fit"
        ],

    "batch":
        summary[
            "inference_batch_log_log_fit"
        ],
}

for label, expected_slope, expected_r2 in [
    (
        "t",
        EXPECTED["t_slope"],
        EXPECTED["t_r2"],
    ),
    (
        "k",
        EXPECTED["k_slope"],
        EXPECTED["k_r2"],
    ),
    (
        "batch",
        EXPECTED["batch_slope"],
        EXPECTED["batch_r2"],
    ),
]:
    require(
        close(
            float(
                fits[label]["slope"]
            ),
            expected_slope,
            tolerance=1e-10,
        ),
        f"{label} slope mismatch.",
    )

    require(
        close(
            float(
                fits[label]["r_squared"]
            ),
            expected_r2,
            tolerance=1e-10,
        ),
        f"{label} R-squared mismatch.",
    )


def one_row(
    table: pd.DataFrame,
    column: str,
    value: int,
) -> pd.Series:
    rows = table[
        table[column] == value
    ]

    require(
        len(rows) == 1,
        (
            f"Expected one row for "
            f"{column}={value}; found {len(rows)}."
        ),
    )

    return rows.iloc[0]


t40 = one_row(
    t_table,
    "T",
    40,
)

t1280 = one_row(
    t_table,
    "T",
    1280,
)

batch1 = one_row(
    inference_table,
    "batch_size",
    1,
)

batch2048 = one_row(
    inference_table,
    "batch_size",
    2048,
)

checks = [
    (
        float(t40["median_us"]),
        EXPECTED["t_40_median_us"],
        "T=40 median_us",
    ),
    (
        float(t1280["median_us"]),
        EXPECTED["t_1280_median_us"],
        "T=1280 median_us",
    ),
    (
        float(
            batch1[
                "median_batch_ms"
            ]
        ),
        EXPECTED[
            "batch_1_median_ms"
        ],
        "batch=1 median_batch_ms",
    ),
    (
        float(
            batch2048[
                "median_batch_ms"
            ]
        ),
        EXPECTED[
            "batch_2048_median_ms"
        ],
        "batch=2048 median_batch_ms",
    ),
    (
        float(
            batch1[
                "median_per_interval_us"
            ]
        ),
        EXPECTED[
            "batch_1_per_interval_us"
        ],
        "batch=1 per-interval latency",
    ),
    (
        float(
            batch2048[
                "median_per_interval_us"
            ]
        ),
        EXPECTED[
            "batch_2048_per_interval_us"
        ],
        "batch=2048 per-interval latency",
    ),
    (
        float(
            batch2048[
                "intervals_per_second"
            ]
        ),
        EXPECTED[
            "batch_2048_throughput"
        ],
        "batch=2048 throughput",
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


freeze_summary = {
    "schema":
        "phyguard.complexity_scaling_freeze.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "FROZEN",

    "theoretical_complexity": {
        "full":
            EXPECTED[
                "theoretical_total_time"
            ],

        "simplified":
            EXPECTED[
                "simplified_total_time"
            ],

        "memory":
            "O(TK) input working memory plus model storage",
    },

    "model_topology":
        topology,

    "empirical_scaling": {
        "t_scaling": {
            "log_log_slope":
                float(
                    fits["t"]["slope"]
                ),

            "r_squared":
                float(
                    fits["t"]["r_squared"]
                ),

            "median_us_T40":
                float(
                    t40["median_us"]
                ),

            "median_us_T1280":
                float(
                    t1280["median_us"]
                ),

            "interpretation":
                (
                    "The observed slope is sublinear over the "
                    "measured finite range because fixed Python, "
                    "quantile, and model-call overhead dominates "
                    "short sequences. It is corroborative timing "
                    "evidence, not a replacement for the "
                    "operation-count complexity."
                ),
        },

        "k_scaling": {
            "log_log_slope":
                float(
                    fits["k"]["slope"]
                ),

            "r_squared":
                float(
                    fits["k"]["r_squared"]
                ),

            "interpretation":
                (
                    "This is a transparent raw-statistics "
                    "microkernel. It isolates K scaling and is "
                    "not the exact ten-KPI physical-rule builder."
                ),
        },

        "inference_scaling": {
            "log_log_slope":
                float(
                    fits["batch"]["slope"]
                ),

            "r_squared":
                float(
                    fits["batch"]["r_squared"]
                ),

            "batch_1_median_ms":
                float(
                    batch1[
                        "median_batch_ms"
                    ]
                ),

            "batch_2048_median_ms":
                float(
                    batch2048[
                        "median_batch_ms"
                    ]
                ),

            "batch_1_per_interval_us":
                float(
                    batch1[
                        "median_per_interval_us"
                    ]
                ),

            "batch_2048_per_interval_us":
                float(
                    batch2048[
                        "median_per_interval_us"
                    ]
                ),

            "batch_2048_intervals_per_second":
                float(
                    batch2048[
                        "intervals_per_second"
                    ]
                ),
        },
    },

    "claim_boundaries": {
        "diagnosis_conditioned_on_candidate_interval":
            True,

        "upstream_localization_excluded":
            True,

        "absolute_latency_hardware_specific":
            True,

        "single_thread_cpu_only":
            True,

        "empirical_slope_not_asymptotic_proof":
            True,

        "no_gpu_claim":
            True,
    },
}


manuscript_text = f"""# Complexity and scalability

For a candidate interval with `T` time points and `K` KPI channels, evidence
extraction requires `O(TK)` time. The logistic gate adds `O(F_g)`, while the
Extra Trees resolver adds `O(MD)`, where `M` is the number of trees and `D` is
the mean traversal depth. The total diagnosis-stage complexity is therefore
`O(TK + F_g + MD)`, simplified as `O(TK + MD)` for fixed feature dimensions.

The frozen repeat-0 model used a 19-dimensional logistic gate and a
79-dimensional Extra Trees resolver with {int(topology['tree_count'])} trees,
{int(topology['total_tree_nodes'])} total nodes, mean depth
{float(topology['mean_tree_depth']):.3f}, maximum depth
{int(topology['maximum_tree_depth'])}, and a serialized size of
{int(topology['serialized_model_bytes']) / (1024 * 1024):.2f} MiB.

On the measured single-thread CPU, exact evidence extraction increased from a
median of {float(t40['median_us']) / 1000.0:.3f} ms at `T=40` to
{float(t1280['median_us']) / 1000.0:.3f} ms at `T=1280`. The finite-range
log-log slope was {float(fits['t']['slope']):.3f}; this sublinear empirical
slope reflects dominant fixed Python and statistical-function overhead over
the tested range and is not used as an asymptotic-complexity claim.

Single-interval frozen-model inference required a median of
{float(batch1['median_batch_ms']):.3f} ms. Batched inference reduced amortized
latency to {float(batch2048['median_per_interval_us']):.2f} microseconds per
interval at batch size 2048, corresponding to
{float(batch2048['intervals_per_second']):.0f} intervals/s. These absolute
latencies are implementation- and hardware-specific.

## Reviewer-response wording

We added both an operation-count analysis and a reproducible single-thread CPU
benchmark. Diagnosis conditioned on an identified candidate interval has time
complexity `O(TK + F_g + MD)` and input working memory `O(TK)`. The frozen
resolver contains {int(topology['tree_count'])} trees with mean depth
{float(topology['mean_tree_depth']):.3f}, and the complete repeat-0 model
occupies {int(topology['serialized_model_bytes']) / (1024 * 1024):.2f} MiB.
Across `T=40` to `T=1280`, exact evidence-extraction latency increased from
{float(t40['median_us']) / 1000.0:.3f} to
{float(t1280['median_us']) / 1000.0:.3f} ms. Frozen-model inference was
{float(batch1['median_batch_ms']):.3f} ms for one interval and achieved
{float(batch2048['intervals_per_second']):.0f} intervals/s at batch size 2048.
We explicitly limit these latency values to the measured CPU/software
environment and exclude upstream anomaly localization from the diagnosis-stage
benchmark.
"""


with tempfile.TemporaryDirectory(
    prefix="stage63c_",
    dir=str(ROOT / "results"),
) as temporary_directory:
    stage = Path(
        temporary_directory
    )

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
            RESULT_MANIFEST,
            stage
            / "manifests"
            / RESULT_MANIFEST.name,
        ),
        (
            EXECUTION_SCRIPT,
            stage
            / "scripts"
            / EXECUTION_SCRIPT.name,
        ),
        (
            EXECUTION_LOG,
            stage
            / "logs"
            / EXECUTION_LOG.name,
        ),
        (
            HOTFIX_REPORT,
            stage
            / "artifacts"
            / PROTOCOL_ROOT.name
            / HOTFIX_REPORT.name,
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
            and path.name != "SHA256SUMS.txt"
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
    "theoretical_total_time:",
    EXPECTED[
        "theoretical_total_time"
    ],
)
print(
    "simplified_total_time:",
    EXPECTED[
        "simplified_total_time"
    ],
)
print(
    "gate_input_dimension:",
    topology[
        "gate_input_dimension"
    ],
)
print(
    "resolver_input_dimension:",
    topology[
        "resolver_input_dimension"
    ],
)
print(
    "tree_count:",
    topology[
        "tree_count"
    ],
)
print(
    "total_tree_nodes:",
    topology[
        "total_tree_nodes"
    ],
)
print(
    "model_size_bytes:",
    topology[
        "serialized_model_bytes"
    ],
)
print(
    "T40_to_T1280_median_us:",
    (
        float(
            t40["median_us"]
        ),
        float(
            t1280["median_us"]
        ),
    ),
)
print(
    "batch1_median_ms:",
    float(
        batch1[
            "median_batch_ms"
        ]
    ),
)
print(
    "batch2048_per_interval_us:",
    float(
        batch2048[
            "median_per_interval_us"
        ]
    ),
)
print(
    "batch2048_throughput:",
    float(
        batch2048[
            "intervals_per_second"
        ]
    ),
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
    "COMPLEXITY_SCALING_V1_FINAL_FREEZE_PASS"
)
