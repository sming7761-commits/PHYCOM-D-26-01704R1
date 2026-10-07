#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

LOCKED_RECONSTRUCTION_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal825_locked_reconstruction_final.zip"
)

PER_ABLATION_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_per_indicator_ablation_v1_final.zip"
)

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "complexity_scaling_protocol_v1.json"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "complexity_scaling_protocol_v1"
)

SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
READY_PATH = OUTPUT_ROOT / "PROTOCOL_READY.md"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "complexity_scaling_protocol_v1.json"
)

EXPECTED_SHA256 = {
    str(LOCKED_RECONSTRUCTION_ARCHIVE):
        "5d08fc3b8265789819b0f49c1f93a9c79c8fe44ce7ec4aec8f1fa603a08c241b",

    str(PER_ABLATION_ARCHIVE):
        "c49c4465dead8422348d03d80b76420ed2396cca9663d9bbfa2301e2e8550555",
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


for path in [
    LOCKED_RECONSTRUCTION_ARCHIVE,
    PER_ABLATION_ARCHIVE,
]:
    require(
        path.exists(),
        f"Required frozen asset is missing: {path}",
    )

    require(
        sha256_file(path)
        == EXPECTED_SHA256[str(path)],
        f"Frozen asset checksum mismatch: {path}",
    )

for path in [
    PROTOCOL_PATH,
    OUTPUT_ROOT,
    MANIFEST_PATH,
]:
    require(
        not path.exists(),
        f"Protocol output already exists: {path}",
    )


status = (
    "LOCKED_BEFORE_COMPLEXITY_SCALING_EXECUTION"
)

protocol = {
    "schema":
        "phyguard.complexity_scaling_protocol.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        status,

    "purpose":
        (
            "Provide a formal asymptotic complexity statement "
            "and a reproducible single-thread CPU scaling study "
            "for evidence extraction and frozen-model inference."
        ),

    "theoretical_contract": {
        "symbols": {
            "T":
                "number of time points in the candidate sequence",

            "K":
                "number of KPI channels",

            "F_g":
                "gate feature dimension",

            "M":
                "number of trees in the Extra Trees resolver",

            "D":
                "mean tree traversal depth",

            "N":
                "number of candidate intervals in an inference batch",
        },

        "evidence_extraction_time":
            "O(TK)",

        "gate_inference_time_per_interval":
            "O(F_g)",

        "resolver_inference_time_per_interval":
            "O(MD)",

        "total_time_per_interval":
            "O(TK + F_g + MD)",

        "simplified_total_time":
            "O(TK + MD)",

        "input_working_memory":
            "O(TK)",

        "evidence_vector_memory":
            "O(F)",

        "model_storage":
            (
                "O(F_g) for the logistic gate plus "
                "O(total resolver tree nodes) for Extra Trees."
            ),

        "scope_boundary":
            (
                "The complexity claim applies to diagnosis "
                "conditioned on an already identified candidate "
                "interval; upstream anomaly localization is outside "
                "this benchmark."
            ),
    },

    "runtime_environment": {
        "device":
            "CPU only",

        "threading":
            "single thread",

        "thread_environment_variables": {
            "OMP_NUM_THREADS":
                "1",

            "MKL_NUM_THREADS":
                "1",

            "OPENBLAS_NUM_THREADS":
                "1",

            "NUMEXPR_NUM_THREADS":
                "1",
        },

        "threadpoolctl_limit":
            1,

        "timer":
            "time.perf_counter_ns",

        "warmup_calls_per_point":
            5,

        "measurement_blocks":
            5,

        "calls_per_block":
            40,

        "reported_time_statistics": [
            "median",
            "p95",
            "mean",
            "standard deviation",
        ],

        "memory_measurement":
            (
                "tracemalloc peak Python allocation; report "
                "input-array bytes separately because native "
                "allocator memory is not fully captured."
            ),
    },

    "t_scaling": {
        "kernel":
            (
                "Exact revision-time evidence builder "
                "features.py::build_one."
            ),

        "fixed_k":
            10,

        "t_values": [
            40,
            80,
            160,
            320,
            640,
            1280,
        ],

        "candidate_interval":
            (
                "start=floor(0.30T), end=ceil(0.70T), "
                "with deterministic synthetic KPI trajectories."
            ),

        "required_output_dimensions": {
            "raw":
                60,

            "physical":
                19,
        },

        "slope":
            (
                "OLS slope of log(median runtime) on log(T)."
            ),
    },

    "k_scaling": {
        "kernel":
            (
                "Transparent six-statistics-per-channel raw "
                "evidence microkernel, used only to isolate K "
                "scaling because the exact physical feature "
                "builder is defined for ten named KPIs."
            ),

        "fixed_t":
            80,

        "k_values": [
            5,
            10,
            20,
            40,
            80,
        ],

        "statistics_per_channel": [
            "mean",
            "standard deviation",
            "median",
            "median absolute deviation",
            "5th percentile",
            "95th percentile",
        ],

        "slope":
            (
                "OLS slope of log(median runtime) on log(K)."
            ),

        "claim_boundary":
            (
                "K-scaling is corroborative for the raw "
                "statistics branch and is not presented as an "
                "exact benchmark of ten-KPI physical rules."
            ),
    },

    "inference_scaling": {
        "model":
            (
                "Repeat-0 frozen FULL logistic gate plus "
                "Extra Trees resolver."
            ),

        "batch_sizes": [
            1,
            8,
            32,
            128,
            512,
            2048,
        ],

        "warmup_calls_per_point":
            5,

        "measurement_blocks":
            5,

        "calls_per_block":
            20,

        "reported_metrics": [
            "batch latency median and p95",
            "per-interval latency",
            "intervals per second",
        ],

        "topology_reporting": [
            "gate input dimension",
            "resolver input dimension",
            "number of trees",
            "total tree nodes",
            "mean tree depth",
            "maximum tree depth",
            "serialized model bytes",
        ],
    },

    "empirical_interpretation": {
        "primary":
            (
                "The asymptotic statement is established by "
                "operation structure, not by timing regression."
            ),

        "corroborative":
            (
                "Empirical slopes, throughput, and memory are "
                "reported as implementation-specific evidence."
            ),

        "no_hardware_generalization":
            (
                "Absolute latency is specific to the measured "
                "CPU/software environment and must not be "
                "generalized to all deployments."
            ),

        "no_gpu_claim":
            True,

        "no_end_to_end_localization_claim":
            True,
    },

    "mandatory_outputs": [
        "theoretical_complexity.json",
        "theoretical_complexity.md",
        "runtime_environment.json",
        "model_topology.json",
        "t_scaling.csv",
        "k_scaling.csv",
        "inference_batch_scaling.csv",
        "summary.json",
        "manifest with SHA256 hashes",
    ],

    "prohibited_posthoc_actions": [
        "changing T values after viewing timings",
        "changing K values after viewing timings",
        "dropping slow points",
        "removing warmup only for selected points",
        "using GPU timing",
        "using more than one CPU thread",
        "reporting only the best timing block",
        "claiming K-scaling microkernel is the exact physical-rule implementation",
        "claiming absolute latency is hardware independent",
        "including upstream detector localization in the measured diagnosis latency",
    ],

    "methodological_boundary": {
        "timings_measured":
            False,

        "kpi_values_opened":
            False,

        "models_executed":
            False,

        "performance_metrics_computed":
            False,

        "protocol_locked_before_execution":
            True,
    },
}

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

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

summary = {
    "schema":
        "phyguard.complexity_scaling_protocol_summary.v1",

    "status":
        status,

    "theoretical_total_time":
        "O(TK + F_g + MD)",

    "simplified_total_time":
        "O(TK + MD)",

    "t_values":
        protocol[
            "t_scaling"
        ][
            "t_values"
        ],

    "k_values":
        protocol[
            "k_scaling"
        ][
            "k_values"
        ],

    "inference_batch_sizes":
        protocol[
            "inference_scaling"
        ][
            "batch_sizes"
        ],

    "measurement_blocks":
        5,

    "single_thread":
        True,

    "protocol_path":
        str(
            PROTOCOL_PATH.relative_to(
                ROOT
            )
        ),

    "protocol_sha256":
        sha256_file(
            PROTOCOL_PATH
        ),

    "methodological_boundary":
        protocol[
            "methodological_boundary"
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

ready_text = """# Stage 63A — Complexity and scaling protocol

Status: **LOCKED BEFORE EXECUTION**

The diagnosis-stage time complexity is reported as:

- evidence extraction: `O(TK)`
- logistic gate: `O(F_g)`
- Extra Trees resolver: `O(MD)`
- total per interval: `O(TK + F_g + MD)`
- simplified: `O(TK + MD)`

The empirical study is single-thread CPU only. It measures exact evidence-builder
scaling over `T = 40...1280`, a transparent raw-statistics microkernel over
`K = 5...80`, and frozen-model inference over batch sizes `1...2048`.

Absolute latency is hardware-specific. The benchmark is conditioned on a locked
candidate interval and excludes upstream anomaly localization.
"""

READY_PATH.write_text(
    ready_text,
    encoding="utf-8",
)

manifest = {
    "schema":
        "phyguard.complexity_scaling_protocol_manifest.v1",

    "status":
        status,

    "protocol_sha256":
        sha256_file(
            PROTOCOL_PATH
        ),

    "summary_sha256":
        sha256_file(
            SUMMARY_PATH
        ),

    "ready_text_sha256":
        sha256_file(
            READY_PATH
        ),

    "locked_assets": {
        str(
            LOCKED_RECONSTRUCTION_ARCHIVE.relative_to(
                ROOT
            )
        ):
            EXPECTED_SHA256[
                str(
                    LOCKED_RECONSTRUCTION_ARCHIVE
                )
            ],

        str(
            PER_ABLATION_ARCHIVE.relative_to(
                ROOT
            )
        ):
            EXPECTED_SHA256[
                str(
                    PER_ABLATION_ARCHIVE
                )
            ],
    },

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

print("status:", status)
print("theoretical_total_time: O(TK + F_g + MD)")
print("simplified_total_time: O(TK + MD)")
print("T_values:", protocol["t_scaling"]["t_values"])
print("K_values:", protocol["k_scaling"]["k_values"])
print(
    "batch_sizes:",
    protocol[
        "inference_scaling"
    ][
        "batch_sizes"
    ],
)
print("single_thread: True")
print("timings_measured: False")
print("models_executed: False")
print("protocol:", PROTOCOL_PATH)
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)
print()
print(
    "COMPLEXITY_SCALING_PROTOCOL_V1_PASS"
)
