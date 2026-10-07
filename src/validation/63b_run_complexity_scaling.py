#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits


ROOT = Path("/root/phyguard_revision")

R0 = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
)

FEATURE_BUILDER_PATH = (
    R0
    / "scripts"
    / "features.py"
)

EVALUATE_SUITE_PATH = (
    R0
    / "scripts"
    / "evaluate_suite.py"
)

SOURCE_FEATURES_PATH = (
    R0
    / "results"
    / "full"
    / "features"
    / "features.npz"
)

MODEL_PATH = (
    R0
    / "results"
    / "full"
    / "evaluation_suite"
    / "models"
    / "repeat_0_phyguard.joblib"
)

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "complexity_scaling_protocol_v1.json"
)

PROTOCOL_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "complexity_scaling_protocol_v1.json"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "complexity_scaling_results_v1"
)

RESULT_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "complexity_scaling_results_v1.json"
)


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


def synthetic_sequence(
    t_length: int,
    k_count: int = 10,
) -> np.ndarray:
    t = np.arange(
        t_length,
        dtype=np.float64,
    )

    base_columns = [
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

    if k_count <= 10:
        return np.column_stack(
            base_columns[
                :k_count
            ]
        )

    columns = list(
        base_columns
    )

    for channel in range(
        10,
        k_count,
    ):
        period = (
            11
            + (
                channel
                % 23
            )
        )

        columns.append(
            0.1
            * channel
            + np.sin(
                2.0
                * np.pi
                * t
                / period
            )
        )

    return np.column_stack(
        columns
    )


def raw_statistics_microkernel(
    x: np.ndarray,
) -> np.ndarray:
    mean = np.mean(
        x,
        axis=0,
    )

    standard_deviation = np.std(
        x,
        axis=0,
        ddof=0,
    )

    median = np.median(
        x,
        axis=0,
    )

    median_absolute_deviation = np.median(
        np.abs(
            x
            - median
        ),
        axis=0,
    )

    q05 = np.quantile(
        x,
        0.05,
        axis=0,
    )

    q95 = np.quantile(
        x,
        0.95,
        axis=0,
    )

    return np.concatenate(
        [
            mean,
            standard_deviation,
            median,
            median_absolute_deviation,
            q05,
            q95,
        ]
    )


def unpack_build_one(result):
    if isinstance(result, tuple) and len(result) >= 2:
        return (
            np.asarray(
                result[0],
                dtype=np.float64,
            ).reshape(-1),

            np.asarray(
                result[1],
                dtype=np.float64,
            ).reshape(-1),
        )

    if isinstance(result, dict):
        raw_key = next(
            key
            for key in [
                "raw",
                "raw_features",
            ]
            if key in result
        )

        physical_key = next(
            key
            for key in [
                "physical",
                "phys",
                "physical_features",
            ]
            if key in result
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
    x: np.ndarray,
):
    t_length = x.shape[0]

    start = max(
        4,
        int(
            math.floor(
                0.30
                * t_length
            )
        ),
    )

    end = min(
        t_length
        - 4,
        int(
            math.ceil(
                0.70
                * t_length
            )
        ),
    )

    require(
        end > start,
        "Invalid candidate interval.",
    )

    raw, physical = unpack_build_one(
        build_one(
            x,
            start,
            end,
        )
    )

    require(
        raw.shape == (60,),
        f"Unexpected raw dimension: {raw.shape}",
    )

    require(
        physical.shape == (19,),
        f"Unexpected physical dimension: {physical.shape}",
    )

    return raw, physical


def benchmark_callable(
    function,
    warmup_calls: int,
    blocks: int,
    calls_per_block: int,
):
    for _ in range(
        warmup_calls
    ):
        function()

    observations_ns = []
    block_medians_ns = []

    for _ in range(
        blocks
    ):
        block = []

        for _ in range(
            calls_per_block
        ):
            start = time.perf_counter_ns()
            function()
            elapsed = (
                time.perf_counter_ns()
                - start
            )

            block.append(
                float(
                    elapsed
                )
            )

        observations_ns.extend(
            block
        )

        block_medians_ns.append(
            float(
                np.median(
                    block
                )
            )
        )

    observations_ns = np.asarray(
        observations_ns,
        dtype=np.float64,
    )

    return {
        "observation_count":
            int(
                len(
                    observations_ns
                )
            ),

        "median_ns":
            float(
                np.median(
                    observations_ns
                )
            ),

        "p95_ns":
            float(
                np.quantile(
                    observations_ns,
                    0.95,
                )
            ),

        "mean_ns":
            float(
                np.mean(
                    observations_ns
                )
            ),

        "std_ns":
            float(
                np.std(
                    observations_ns,
                    ddof=0,
                )
            ),

        "block_median_ns":
            [
                float(value)
                for value in block_medians_ns
            ],
    }


def peak_python_allocation(
    function,
    repetitions: int = 5,
):
    peaks = []

    for _ in range(
        repetitions
    ):
        tracemalloc.start()
        function()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peaks.append(
            int(
                peak
            )
        )

    return {
        "median_peak_bytes":
            int(
                np.median(
                    peaks
                )
            ),

        "maximum_peak_bytes":
            int(
                np.max(
                    peaks
                )
            ),

        "all_peak_bytes":
            peaks,
    }


def log_log_fit(
    x_values,
    y_values,
):
    x = np.log(
        np.asarray(
            x_values,
            dtype=np.float64,
        )
    )

    y = np.log(
        np.asarray(
            y_values,
            dtype=np.float64,
        )
    )

    coefficients = np.polyfit(
        x,
        y,
        deg=1,
    )

    prediction = np.polyval(
        coefficients,
        x,
    )

    residual = np.sum(
        (
            y
            - prediction
        )
        ** 2
    )

    total = np.sum(
        (
            y
            - np.mean(
                y
            )
        )
        ** 2
    )

    r_squared = (
        1.0
        - residual
        / total
        if total > 0
        else math.nan
    )

    return {
        "slope":
            float(
                coefficients[
                    0
                ]
            ),

        "intercept":
            float(
                coefficients[
                    1
                ]
            ),

        "r_squared":
            float(
                r_squared
            ),
    }


# STAGE63B_PIPELINE_TOPOLOGY_HOTFIX_V2
def final_estimator(model):
    """Return the final fitted estimator from a Pipeline."""
    steps = getattr(
        model,
        "steps",
        None,
    )

    if steps:
        return steps[-1][1]

    named_steps = getattr(
        model,
        "named_steps",
        None,
    )

    if named_steps:
        return list(
            named_steps.values()
        )[-1]

    return model


def fitted_input_dimension(
    outer_model,
    final_model,
):
    for candidate in [
        outer_model,
        final_model,
    ]:
        value = getattr(
            candidate,
            "n_features_in_",
            None,
        )

        if value is not None:
            return int(value)

    coefficient = getattr(
        final_model,
        "coef_",
        None,
    )

    if coefficient is not None:
        coefficient = np.asarray(
            coefficient
        )

        require(
            coefficient.ndim >= 2,
            "Unexpected coefficient shape.",
        )

        return int(
            coefficient.shape[-1]
        )

    raise RuntimeError(
        "Could not determine fitted input dimension."
    )


def linear_parameter_count(estimator):
    count = 0

    coefficient = getattr(
        estimator,
        "coef_",
        None,
    )

    intercept = getattr(
        estimator,
        "intercept_",
        None,
    )

    if coefficient is not None:
        count += int(
            np.asarray(
                coefficient
            ).size
        )

    if intercept is not None:
        count += int(
            np.asarray(
                intercept
            ).size
        )

    return count


def model_topology(
    model_bundle,
):
    require(
        isinstance(
            model_bundle,
            dict,
        )
        and "gate" in model_bundle
        and "resolver" in model_bundle,
        "Unexpected model bundle format.",
    )

    gate_outer = model_bundle[
        "gate"
    ]

    resolver_outer = model_bundle[
        "resolver"
    ]

    gate = final_estimator(
        gate_outer
    )

    resolver = final_estimator(
        resolver_outer
    )

    estimators = list(
        getattr(
            resolver,
            "estimators_",
            [],
        )
    )

    require(
        len(estimators) > 0,
        "Resolver has no fitted estimators.",
    )

    node_counts = [
        int(
            estimator.tree_.node_count
        )
        for estimator in estimators
    ]

    depths = [
        int(
            estimator.tree_.max_depth
        )
        for estimator in estimators
    ]

    gate_dimension = fitted_input_dimension(
        gate_outer,
        gate,
    )

    resolver_dimension = fitted_input_dimension(
        resolver_outer,
        resolver,
    )

    return {
        "gate_outer_class":
            gate_outer.__class__.__name__,

        "gate_class":
            gate.__class__.__name__,

        "gate_pipeline_steps":
            [
                str(name)
                for name, _
                in getattr(
                    gate_outer,
                    "steps",
                    [],
                )
            ],

        "gate_input_dimension":
            gate_dimension,

        "gate_coefficient_count":
            linear_parameter_count(
                gate
            ),

        "resolver_outer_class":
            resolver_outer.__class__.__name__,

        "resolver_class":
            resolver.__class__.__name__,

        "resolver_pipeline_steps":
            [
                str(name)
                for name, _
                in getattr(
                    resolver_outer,
                    "steps",
                    [],
                )
            ],

        "resolver_input_dimension":
            resolver_dimension,

        "tree_count":
            len(
                estimators
            ),

        "total_tree_nodes":
            int(
                np.sum(
                    node_counts
                )
            ),

        "mean_tree_nodes":
            float(
                np.mean(
                    node_counts
                )
            ),

        "mean_tree_depth":
            float(
                np.mean(
                    depths
                )
            ),

        "median_tree_depth":
            float(
                np.median(
                    depths
                )
            ),

        "maximum_tree_depth":
            int(
                np.max(
                    depths
                )
            ),

        "serialized_model_bytes":
            MODEL_PATH.stat().st_size,
    }


required_paths = [
    FEATURE_BUILDER_PATH,
    EVALUATE_SUITE_PATH,
    SOURCE_FEATURES_PATH,
    MODEL_PATH,
    PROTOCOL_PATH,
    PROTOCOL_MANIFEST_PATH,
]

for path in required_paths:
    require(
        path.exists(),
        f"Required path is missing: {path}",
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
    == "LOCKED_BEFORE_COMPLEXITY_SCALING_EXECUTION",
    "Protocol status mismatch.",
)

require(
    protocol_manifest.get(
        "protocol_sha256"
    )
    == sha256_file(
        PROTOCOL_PATH
    ),
    "Protocol checksum mismatch.",
)

require(
    not OUTPUT_ROOT.exists(),
    f"Result output already exists: {OUTPUT_ROOT}",
)

require(
    not RESULT_MANIFEST_PATH.exists(),
    f"Result manifest already exists: {RESULT_MANIFEST_PATH}",
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

feature_module = load_module(
    "phyguard_features_stage63",
    FEATURE_BUILDER_PATH,
)

evaluate_module = load_module(
    "phyguard_evaluate_stage63",
    EVALUATE_SUITE_PATH,
)

require(
    hasattr(
        feature_module,
        "build_one",
    ),
    "features.py lacks build_one.",
)

require(
    hasattr(
        evaluate_module,
        "two_stage_probabilities",
    ),
    "evaluate_suite.py lacks two_stage_probabilities.",
)

model_bundle = joblib.load(
    MODEL_PATH
)

topology = model_topology(
    model_bundle
)

features_archive = np.load(
    SOURCE_FEATURES_PATH
)

raw_features = np.asarray(
    features_archive[
        "raw"
    ],
    dtype=np.float64,
)

physical_features = np.asarray(
    features_archive[
        "physical"
    ],
    dtype=np.float64,
)

combined_features = np.asarray(
    features_archive[
        "combined"
    ],
    dtype=np.float64,
)

require(
    raw_features.shape[
        1
    ]
    == 60,
    "Raw source feature dimension mismatch.",
)

require(
    physical_features.shape[
        1
    ]
    == 19,
    "Physical source feature dimension mismatch.",
)

require(
    combined_features.shape[
        1
    ]
    == 79,
    "Combined source feature dimension mismatch.",
)


runtime_environment = {
    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat()
        if False
        else None,

    "platform":
        platform.platform(),

    "python":
        sys.version,

    "numpy":
        np.__version__,

    "scipy":
        scipy.__version__,

    "sklearn":
        sklearn.__version__,

    "processor":
        platform.processor(),

    "cpu_count_logical":
        os.cpu_count(),

    "thread_environment": {
        key:
            os.environ.get(
                key
            )
        for key in [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]
    },

    "threadpool_info":
        threadpool_info(),

    "timer":
        "time.perf_counter_ns",

    "feature_builder_sha256":
        sha256_file(
            FEATURE_BUILDER_PATH
        ),

    "evaluate_suite_sha256":
        sha256_file(
            EVALUATE_SUITE_PATH
        ),

    "model_sha256":
        sha256_file(
            MODEL_PATH
        ),
}

(
    OUTPUT_ROOT
    / "runtime_environment.json"
).write_text(
    json.dumps(
        json_safe(
            runtime_environment
        ),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ),
    encoding="utf-8",
)

(
    OUTPUT_ROOT
    / "model_topology.json"
).write_text(
    json.dumps(
        topology,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ),
    encoding="utf-8",
)


theoretical = protocol[
    "theoretical_contract"
]

(
    OUTPUT_ROOT
    / "theoretical_complexity.json"
).write_text(
    json.dumps(
        theoretical,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

theory_md = f"""# PhyGuard diagnosis-stage complexity

Let `T` be sequence length, `K` the number of KPI channels, `F_g` the logistic
gate input dimension, `M` the number of resolver trees, and `D` the mean tree
traversal depth.

- Evidence extraction: **O(TK)**
- Logistic gate per interval: **O(F_g)**
- Extra Trees resolver per interval: **O(MD)**
- Total diagnosis time per interval: **O(TK + F_g + MD)**
- Simplified total: **O(TK + MD)**

For the frozen repeat-0 model:

- gate dimension: {topology['gate_input_dimension']}
- resolver dimension: {topology['resolver_input_dimension']}
- trees: {topology['tree_count']}
- total tree nodes: {topology['total_tree_nodes']}
- mean tree depth: {topology['mean_tree_depth']:.3f}
- maximum tree depth: {topology['maximum_tree_depth']}
- serialized model size: {topology['serialized_model_bytes']} bytes

The benchmark is conditioned on a locked candidate interval and excludes
upstream anomaly localization. Absolute runtime is hardware-specific.
"""

(
    OUTPUT_ROOT
    / "theoretical_complexity.md"
).write_text(
    theory_md,
    encoding="utf-8",
)


t_rows = []

with threadpool_limits(
    limits=1
):
    for t_length in protocol[
        "t_scaling"
    ][
        "t_values"
    ]:
        sequence = synthetic_sequence(
            int(
                t_length
            ),
            10,
        )

        function = lambda sequence=sequence: call_build_one(
            feature_module.build_one,
            sequence,
        )

        timing = benchmark_callable(
            function,
            warmup_calls=protocol[
                "runtime_environment"
            ][
                "warmup_calls_per_point"
            ],
            blocks=protocol[
                "runtime_environment"
            ][
                "measurement_blocks"
            ],
            calls_per_block=protocol[
                "runtime_environment"
            ][
                "calls_per_block"
            ],
        )

        memory = peak_python_allocation(
            function,
            repetitions=5,
        )

        row = {
            "T":
                int(
                    t_length
                ),

            "K":
                10,

            "input_bytes":
                int(
                    sequence.nbytes
                ),

            **timing,

            **memory,

            "median_us":
                timing[
                    "median_ns"
                ]
                / 1000.0,

            "p95_us":
                timing[
                    "p95_ns"
                ]
                / 1000.0,
        }

        t_rows.append(
            row
        )

        print(
            "T_SCALING",
            row,
            flush=True,
        )


k_rows = []

with threadpool_limits(
    limits=1
):
    for k_count in protocol[
        "k_scaling"
    ][
        "k_values"
    ]:
        sequence = synthetic_sequence(
            80,
            int(
                k_count
            ),
        )

        function = lambda sequence=sequence: raw_statistics_microkernel(
            sequence
        )

        output = function()

        require(
            output.shape
            == (
                6
                * int(
                    k_count
                ),
            ),
            "K-scaling output dimension mismatch.",
        )

        timing = benchmark_callable(
            function,
            warmup_calls=protocol[
                "runtime_environment"
            ][
                "warmup_calls_per_point"
            ],
            blocks=protocol[
                "runtime_environment"
            ][
                "measurement_blocks"
            ],
            calls_per_block=protocol[
                "runtime_environment"
            ][
                "calls_per_block"
            ],
        )

        memory = peak_python_allocation(
            function,
            repetitions=5,
        )

        row = {
            "T":
                80,

            "K":
                int(
                    k_count
                ),

            "output_dimension":
                int(
                    output.size
                ),

            "input_bytes":
                int(
                    sequence.nbytes
                ),

            **timing,

            **memory,

            "median_us":
                timing[
                    "median_ns"
                ]
                / 1000.0,

            "p95_us":
                timing[
                    "p95_ns"
                ]
                / 1000.0,
        }

        k_rows.append(
            row
        )

        print(
            "K_SCALING",
            row,
            flush=True,
        )


inference_rows = []

with threadpool_limits(
    limits=1
):
    for batch_size in protocol[
        "inference_scaling"
    ][
        "batch_sizes"
    ]:
        indices = np.arange(
            int(
                batch_size
            )
        ) % raw_features.shape[
            0
        ]

        raw_batch = raw_features[
            indices
        ]

        physical_batch = physical_features[
            indices
        ]

        combined_batch = combined_features[
            indices
        ]

        function = lambda: evaluate_module.two_stage_probabilities(
            model_bundle[
                "gate"
            ],
            model_bundle[
                "resolver"
            ],
            raw_batch,
            physical_batch,
            combined_batch,
            True,
        )

        function()

        timing = benchmark_callable(
            function,
            warmup_calls=protocol[
                "inference_scaling"
            ][
                "warmup_calls_per_point"
            ],
            blocks=protocol[
                "inference_scaling"
            ][
                "measurement_blocks"
            ],
            calls_per_block=protocol[
                "inference_scaling"
            ][
                "calls_per_block"
            ],
        )

        memory = peak_python_allocation(
            function,
            repetitions=5,
        )

        median_seconds = (
            timing[
                "median_ns"
            ]
            / 1e9
        )

        row = {
            "batch_size":
                int(
                    batch_size
                ),

            **timing,

            **memory,

            "median_batch_ms":
                timing[
                    "median_ns"
                ]
                / 1e6,

            "p95_batch_ms":
                timing[
                    "p95_ns"
                ]
                / 1e6,

            "median_per_interval_us":
                timing[
                    "median_ns"
                ]
                / 1000.0
                / int(
                    batch_size
                ),

            "intervals_per_second":
                (
                    int(
                        batch_size
                    )
                    / median_seconds
                ),
        }

        inference_rows.append(
            row
        )

        print(
            "INFERENCE_SCALING",
            row,
            flush=True,
        )


t_table = pd.DataFrame(
    t_rows
)

k_table = pd.DataFrame(
    k_rows
)

inference_table = pd.DataFrame(
    inference_rows
)

t_table.to_csv(
    OUTPUT_ROOT
    / "t_scaling.csv",
    index=False,
)

k_table.to_csv(
    OUTPUT_ROOT
    / "k_scaling.csv",
    index=False,
)

inference_table.to_csv(
    OUTPUT_ROOT
    / "inference_batch_scaling.csv",
    index=False,
)


t_fit = log_log_fit(
    t_table[
        "T"
    ],
    t_table[
        "median_ns"
    ],
)

k_fit = log_log_fit(
    k_table[
        "K"
    ],
    k_table[
        "median_ns"
    ],
)

inference_fit = log_log_fit(
    inference_table[
        "batch_size"
    ],
    inference_table[
        "median_ns"
    ],
)


summary = {
    "schema":
        "phyguard.complexity_scaling_results.v1",

    "status":
        "PASS",

    "theoretical_total_time":
        "O(TK + F_g + MD)",

    "simplified_total_time":
        "O(TK + MD)",

    "model_topology":
        topology,

    "t_scaling_log_log_fit":
        t_fit,

    "k_scaling_log_log_fit":
        k_fit,

    "inference_batch_log_log_fit":
        inference_fit,

    "t_scaling_fastest_median_us":
        float(
            t_table[
                "median_us"
            ].min()
        ),

    "t_scaling_slowest_median_us":
        float(
            t_table[
                "median_us"
            ].max()
        ),

    "inference_batch_1_median_us":
        float(
            inference_table.loc[
                inference_table[
                    "batch_size"
                ]
                == 1,
                "median_per_interval_us",
            ].iloc[
                0
            ]
        ),

    "inference_batch_2048_throughput":
        float(
            inference_table.loc[
                inference_table[
                    "batch_size"
                ]
                == 2048,
                "intervals_per_second",
            ].iloc[
                0
            ]
        ),

    "interpretation": {
        "theory_is_primary":
            True,

        "timing_is_corroborative":
            True,

        "absolute_latency_hardware_specific":
            True,

        "k_scaling_microkernel_not_exact_physical_rules":
            True,

        "upstream_localization_excluded":
            True,
    },

    "files": {
        "theoretical_complexity":
            "theoretical_complexity.json",

        "theoretical_complexity_text":
            "theoretical_complexity.md",

        "runtime_environment":
            "runtime_environment.json",

        "model_topology":
            "model_topology.json",

        "t_scaling":
            "t_scaling.csv",

        "k_scaling":
            "k_scaling.csv",

        "inference_batch_scaling":
            "inference_batch_scaling.csv",
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
        "phyguard.complexity_scaling_results_manifest.v1",

    "status":
        "PASS",

    "protocol_sha256":
        sha256_file(
            PROTOCOL_PATH
        ),

    "feature_builder_sha256":
        sha256_file(
            FEATURE_BUILDER_PATH
        ),

    "evaluate_suite_sha256":
        sha256_file(
            EVALUATE_SUITE_PATH
        ),

    "model_sha256":
        sha256_file(
            MODEL_PATH
        ),

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
print("MODEL TOPOLOGY")
print(
    json.dumps(
        topology,
        ensure_ascii=False,
        indent=2,
    ),
    flush=True,
)

print()
print("SCALING FITS")
print(
    "T_fit:",
    t_fit,
    flush=True,
)
print(
    "K_fit:",
    k_fit,
    flush=True,
)
print(
    "inference_batch_fit:",
    inference_fit,
    flush=True,
)

print()
print("T SCALING TABLE")
print(
    t_table[
        [
            "T",
            "median_us",
            "p95_us",
            "median_peak_bytes",
            "input_bytes",
        ]
    ].to_string(
        index=False
    ),
    flush=True,
)

print()
print("K SCALING TABLE")
print(
    k_table[
        [
            "K",
            "median_us",
            "p95_us",
            "median_peak_bytes",
            "input_bytes",
        ]
    ].to_string(
        index=False
    ),
    flush=True,
)

print()
print("INFERENCE TABLE")
print(
    inference_table[
        [
            "batch_size",
            "median_batch_ms",
            "p95_batch_ms",
            "median_per_interval_us",
            "intervals_per_second",
            "median_peak_bytes",
        ]
    ].to_string(
        index=False
    ),
    flush=True,
)

print()
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
    "COMPLEXITY_SCALING_RESULTS_V1_PASS",
    flush=True,
)
