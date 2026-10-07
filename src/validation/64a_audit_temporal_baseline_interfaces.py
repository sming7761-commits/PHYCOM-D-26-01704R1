#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/root/phyguard_revision")

R0 = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
)

DATA_ROOT = R0 / "data" / "full"
RESULT_ROOT = R0 / "results" / "full"
SCRIPT_ROOT = R0 / "scripts"

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

COMPLEXITY_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_complexity_scaling_v1_final.zip"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "temporal_baseline_interface_audit_v1"
)

SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
PACKAGE_AUDIT_PATH = OUTPUT_ROOT / "package_environment.json"
NPZ_INVENTORY_PATH = OUTPUT_ROOT / "npz_inventory.csv"
CSV_INVENTORY_PATH = OUTPUT_ROOT / "csv_inventory.csv"
FUNCTION_INVENTORY_PATH = OUTPUT_ROOT / "function_inventory.csv"
CONTEXT_PATH = OUTPUT_ROOT / "implementation_contexts.txt"
PLAN_PATH = OUTPUT_ROOT / "NEXT_PROTOCOL_INPUTS.md"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "temporal_baseline_interface_audit_v1.json"
)

EXPECTED_SHA256 = {
    str(LOCKED_RECONSTRUCTION_ARCHIVE):
        "5d08fc3b8265789819b0f49c1f93a9c79c8fe44ce7ec4aec8f1fa603a08c241b",

    str(PER_ABLATION_ARCHIVE):
        "c49c4465dead8422348d03d80b76420ed2396cca9663d9bbfa2301e2e8550555",

    str(COMPLEXITY_ARCHIVE):
        "cdf7b1a8f3415cde2c7c5f8c0b26c9421e4fc707f80de14d5c227e00afc61e9c",
}

PACKAGE_NAMES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "torch",
    "aeon",
    "sktime",
    "numba",
    "joblib",
    "threadpoolctl",
]

MODEL_IMPORTS = [
    (
        "aeon_minrocket",
        "aeon.classification.convolution_based",
        "MiniRocketClassifier",
    ),
    (
        "aeon_multirocket",
        "aeon.classification.convolution_based",
        "MultiRocketClassifier",
    ),
    (
        "aeon_rocket",
        "aeon.classification.convolution_based",
        "RocketClassifier",
    ),
    (
        "aeon_hydra",
        "aeon.classification.convolution_based",
        "HydraClassifier",
    ),
    (
        "sktime_minrocket",
        "sktime.classification.kernel_based",
        "RocketClassifier",
    ),
]

SOURCE_FILES = [
    SCRIPT_ROOT / "evaluate_suite.py",
    SCRIPT_ROOT / "evaluation_common.py",
    SCRIPT_ROOT / "features.py",
    SCRIPT_ROOT / "train_phyguard.py",
    SCRIPT_ROOT / "evaluate_phyguard.py",
]

CONTEXT_TERMS = [
    "train_test_split",
    "Stratified",
    "repeat",
    "seed",
    "tau_g",
    "tau_c",
    "selected_accuracy",
    "false_specific_rate",
    "physical_selection_rate",
    "macro_f1",
    "balanced_accuracy",
    "predict_proba",
    "class_weight",
    "labels",
    "telemetry",
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


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def import_audit(
    label: str,
    module_name: str,
    class_name: str,
) -> dict[str, object]:
    try:
        module = importlib.import_module(
            module_name
        )

        model_class = getattr(
            module,
            class_name,
        )

        return {
            "label":
                label,

            "module":
                module_name,

            "class_name":
                class_name,

            "available":
                True,

            "resolved_class":
                (
                    f"{model_class.__module__}."
                    f"{model_class.__name__}"
                ),

            "error":
                None,
        }
    except Exception as error:
        return {
            "label":
                label,

            "module":
                module_name,

            "class_name":
                class_name,

            "available":
                False,

            "resolved_class":
                None,

            "error":
                repr(error),
        }


def npz_inventory(path: Path) -> list[dict[str, object]]:
    rows = []

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            for name in archive.files:
                array = archive[name]

                rows.append(
                    {
                        "relative_path":
                            str(
                                path.relative_to(
                                    ROOT
                                )
                            ),

                        "array_name":
                            name,

                        "shape":
                            json.dumps(
                                list(
                                    array.shape
                                )
                            ),

                        "dtype":
                            str(
                                array.dtype
                            ),

                        "size":
                            int(
                                array.size
                            ),

                        "nbytes":
                            int(
                                array.nbytes
                            ),
                    }
                )
    except Exception as error:
        rows.append(
            {
                "relative_path":
                    str(
                        path.relative_to(
                            ROOT
                        )
                    ),

                "array_name":
                    "__ERROR__",

                "shape":
                    "",

                "dtype":
                    "",

                "size":
                    0,

                "nbytes":
                    0,

                "error":
                    repr(error),
            }
        )

    return rows


def csv_inventory(path: Path) -> dict[str, object]:
    row_count = 0
    header = []

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            reader = csv.reader(
                stream
            )

            try:
                header = next(
                    reader
                )
            except StopIteration:
                header = []

            for _ in reader:
                row_count += 1

        return {
            "relative_path":
                str(
                    path.relative_to(
                        ROOT
                    )
                ),

            "row_count":
                row_count,

            "column_count":
                len(
                    header
                ),

            "columns":
                json.dumps(
                    header,
                    ensure_ascii=False,
                ),

            "size_bytes":
                path.stat().st_size,

            "error":
                None,
        }
    except Exception as error:
        return {
            "relative_path":
                str(
                    path.relative_to(
                        ROOT
                    )
                ),

            "row_count":
                None,

            "column_count":
                None,

            "columns":
                None,

            "size_bytes":
                path.stat().st_size,

            "error":
                repr(error),
        }


def function_inventory(
    path: Path,
) -> list[dict[str, object]]:
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    tree = ast.parse(
        text,
        filename=str(path),
    )

    rows = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        arguments = []

        for argument in (
            list(
                node.args.posonlyargs
            )
            + list(
                node.args.args
            )
            + list(
                node.args.kwonlyargs
            )
        ):
            arguments.append(
                argument.arg
            )

        if node.args.vararg is not None:
            arguments.append(
                "*"
                + node.args.vararg.arg
            )

        if node.args.kwarg is not None:
            arguments.append(
                "**"
                + node.args.kwarg.arg
            )

        rows.append(
            {
                "relative_path":
                    str(
                        path.relative_to(
                            ROOT
                        )
                    ),

                "function":
                    node.name,

                "line":
                    int(
                        node.lineno
                    ),

                "arguments":
                    json.dumps(
                        arguments
                    ),

                "relevant":
                    bool(
                        re.search(
                            (
                                r"split|train|predict|prob|threshold|"
                                r"metric|select|repeat|feature|label"
                            ),
                            node.name,
                            flags=re.IGNORECASE,
                        )
                    ),
            }
        )

    return rows


def line_contexts(
    path: Path,
    terms: list[str],
    radius: int = 3,
    maximum_per_term: int = 8,
) -> list[str]:
    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    blocks = []

    for term in terms:
        hits = 0

        for index, line in enumerate(
            lines
        ):
            if term.lower() not in line.lower():
                continue

            start = max(
                0,
                index - radius,
            )

            end = min(
                len(lines),
                index + radius + 1,
            )

            block = [
                (
                    f"\n[{path.relative_to(ROOT)}] "
                    f"term={term} line={index + 1}\n"
                )
            ]

            for cursor in range(
                start,
                end,
            ):
                marker = (
                    ">>"
                    if cursor == index
                    else "  "
                )

                block.append(
                    (
                        f"{marker}{cursor + 1:05d}: "
                        f"{lines[cursor]}\n"
                    )
                )

            blocks.append(
                "".join(
                    block
                )
            )

            hits += 1

            if hits >= maximum_per_term:
                break

    return blocks


for path in [
    LOCKED_RECONSTRUCTION_ARCHIVE,
    PER_ABLATION_ARCHIVE,
    COMPLEXITY_ARCHIVE,
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
    R0,
    DATA_ROOT,
    RESULT_ROOT,
    SCRIPT_ROOT,
]:
    require(
        path.exists(),
        f"Required reconstruction path is missing: {path}",
    )

for path in [
    OUTPUT_ROOT,
    MANIFEST_PATH,
]:
    require(
        not path.exists(),
        f"Audit output already exists: {path}",
    )


package_versions = {
    name:
        package_version(
            name
        )
    for name in PACKAGE_NAMES
}

model_imports = [
    import_audit(
        label,
        module_name,
        class_name,
    )
    for (
        label,
        module_name,
        class_name,
    )
    in MODEL_IMPORTS
]

torch_info = {
    "available":
        False,
}

try:
    import torch

    torch_info = {
        "available":
            True,

        "version":
            torch.__version__,

        "cuda_available":
            bool(
                torch.cuda.is_available()
            ),

        "cuda_version":
            torch.version.cuda,

        "device_count":
            int(
                torch.cuda.device_count()
            ),

        "device_names":
            [
                torch.cuda.get_device_name(
                    index
                )
                for index in range(
                    torch.cuda.device_count()
                )
            ],

        "device_memory_bytes":
            [
                int(
                    torch.cuda.get_device_properties(
                        index
                    ).total_memory
                )
                for index in range(
                    torch.cuda.device_count()
                )
            ],
    }
except Exception as error:
    torch_info = {
        "available":
            False,

        "error":
            repr(error),
    }


npz_rows = []

for path in sorted(
    set(
        list(
            DATA_ROOT.rglob(
                "*.npz"
            )
        )
        + list(
            RESULT_ROOT.rglob(
                "*.npz"
            )
        )
    )
):
    if path.stat().st_size > 1_000_000_000:
        continue

    npz_rows.extend(
        npz_inventory(
            path
        )
    )


csv_rows = []

for path in sorted(
    set(
        list(
            DATA_ROOT.rglob(
                "*.csv"
            )
        )
        + list(
            RESULT_ROOT.rglob(
                "*.csv"
            )
        )
    )
):
    if path.stat().st_size > 200_000_000:
        continue

    csv_rows.append(
        csv_inventory(
            path
        )
    )


function_rows = []
context_blocks = []

for path in SOURCE_FILES:
    if not path.exists():
        continue

    try:
        function_rows.extend(
            function_inventory(
                path
            )
        )
    except Exception as error:
        function_rows.append(
            {
                "relative_path":
                    str(
                        path.relative_to(
                            ROOT
                        )
                    ),

                "function":
                    "__PARSE_ERROR__",

                "line":
                    0,

                "arguments":
                    "[]",

                "relevant":
                    False,

                "error":
                    repr(error),
            }
        )

    context_blocks.extend(
        line_contexts(
            path,
            CONTEXT_TERMS,
        )
    )


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

with NPZ_INVENTORY_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    fieldnames = [
        "relative_path",
        "array_name",
        "shape",
        "dtype",
        "size",
        "nbytes",
        "error",
    ]

    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )

    writer.writeheader()
    writer.writerows(
        npz_rows
    )


with CSV_INVENTORY_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    fieldnames = [
        "relative_path",
        "row_count",
        "column_count",
        "columns",
        "size_bytes",
        "error",
    ]

    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        csv_rows
    )


with FUNCTION_INVENTORY_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    fieldnames = [
        "relative_path",
        "function",
        "line",
        "arguments",
        "relevant",
        "error",
    ]

    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )

    writer.writeheader()
    writer.writerows(
        function_rows
    )


CONTEXT_PATH.write_text(
    "\n".join(
        context_blocks
    ),
    encoding="utf-8",
)


package_audit = {
    "schema":
        "phyguard.temporal_baseline_environment.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "platform":
        platform.platform(),

    "python":
        sys.version,

    "package_versions":
        package_versions,

    "model_imports":
        model_imports,

    "torch":
        torch_info,
}

PACKAGE_AUDIT_PATH.write_text(
    json.dumps(
        package_audit,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


telemetry_rows = [
    row
    for row in npz_rows
    if (
        row.get(
            "array_name"
        )
        in {
            "X",
            "telemetry",
            "kpi_sequence",
            "labels",
            "y",
            "kpi_names",
        }
        or "telemetry" in row.get(
            "relative_path",
            ""
        ).lower()
    )
]

split_candidates = [
    row
    for row in csv_rows
    if any(
        term in row[
            "relative_path"
        ].lower()
        for term in [
            "split",
            "repeat",
            "prediction",
            "threshold",
        ]
    )
]

relevant_functions = [
    row
    for row in function_rows
    if row.get(
        "relevant"
    )
]


summary = {
    "schema":
        "phyguard.temporal_baseline_interface_audit.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "PASS",

    "npz_array_count":
        len(
            npz_rows
        ),

    "csv_file_count":
        len(
            csv_rows
        ),

    "function_count":
        len(
            function_rows
        ),

    "relevant_function_count":
        len(
            relevant_functions
        ),

    "telemetry_and_label_candidates":
        telemetry_rows,

    "split_and_repeat_candidates":
        split_candidates,

    "package_versions":
        package_versions,

    "model_imports":
        model_imports,

    "torch":
        torch_info,

    "recommended_candidate_models": [
        {
            "model":
                "MiniRocket",

            "implementation":
                (
                    "Prefer aeon MiniRocketClassifier when "
                    "available; otherwise install the locked "
                    "dependency before protocol execution."
                ),
        },
        {
            "model":
                "FCN1D",

            "implementation":
                "Custom lightweight PyTorch implementation.",
        },
        {
            "model":
                "BiLSTM",

            "implementation":
                "Custom lightweight PyTorch implementation.",
        },
        {
            "model":
                "TinyTransformer",

            "implementation":
                "Custom lightweight PyTorch implementation.",
        },
        {
            "model":
                "InceptionTimeLite",

            "implementation":
                "Custom lightweight PyTorch implementation.",
        },
    ],

    "methodological_boundary": {
        "models_trained":
            False,

        "predictions_generated":
            False,

        "performance_metrics_computed":
            False,

        "test_thresholds_selected":
            False,

        "source_interfaces_and_metadata_audited":
            True,
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


plan_text = """# Inputs for the modern temporal-baseline protocol

This audit does not train or evaluate any model.

The next protocol should lock:

1. The exact source tensor and label arrays.
2. The five reconstruction repeat memberships.
3. A six-class temporal classification interface, with normal and nonphysical
   outputs mapped to abstention.
4. Validation-only confidence selection using the existing selective-diagnosis
   metric contract.
5. MiniRocket, FCN1D, BiLSTM, TinyTransformer, and InceptionTimeLite.
6. Identical test memberships for every model.
7. Parameter count, training time, inference time, selected accuracy, physical
   selection, false-specific rate, macro-F1, balanced accuracy, physical exact
   match, and control abstention.
8. No test-set model or threshold selection.

Use `function_inventory.csv` and `implementation_contexts.txt` to bind the
execution script to the existing repeat/split and metric implementation rather
than creating new memberships.
"""

PLAN_PATH.write_text(
    plan_text,
    encoding="utf-8",
)


manifest_files = []

for path in sorted(
    OUTPUT_ROOT.rglob(
        "*"
    )
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
        "phyguard.temporal_baseline_interface_audit_manifest.v1",

    "status":
        "PASS",

    "locked_assets":
        {
            str(
                path.relative_to(
                    ROOT
                )
            ):
                EXPECTED_SHA256[
                    str(path)
                ]
            for path in [
                LOCKED_RECONSTRUCTION_ARCHIVE,
                PER_ABLATION_ARCHIVE,
                COMPLEXITY_ARCHIVE,
            ]
        },

    "files":
        manifest_files,

    "methodological_boundary":
        summary[
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


print("status: PASS")
print("python:", sys.version.split()[0])
print("package_versions:", package_versions)
print("torch:", torch_info)
print()
print("MODEL IMPORTS")
for row in model_imports:
    print(
        row["label"],
        "| available=",
        row["available"],
        "| resolved=",
        row["resolved_class"],
        "| error=",
        row["error"],
    )

print()
print("TELEMETRY AND LABEL CANDIDATES")
for row in telemetry_rows[:20]:
    print(
        row.get(
            "relative_path"
        ),
        "::",
        row.get(
            "array_name"
        ),
        "| shape=",
        row.get(
            "shape"
        ),
        "| dtype=",
        row.get(
            "dtype"
        ),
    )

print()
print("SPLIT AND REPEAT CANDIDATES")
for row in split_candidates[:20]:
    print(
        row.get(
            "relative_path"
        ),
        "| rows=",
        row.get(
            "row_count"
        ),
        "| columns=",
        row.get(
            "columns"
        ),
    )

print()
print("RELEVANT FUNCTIONS")
for row in relevant_functions[:30]:
    print(
        row.get(
            "relative_path"
        ),
        "::",
        row.get(
            "function"
        ),
        row.get(
            "arguments"
        ),
        "| line=",
        row.get(
            "line"
        ),
    )

print()
print("summary:", SUMMARY_PATH)
print("package_environment:", PACKAGE_AUDIT_PATH)
print("npz_inventory:", NPZ_INVENTORY_PATH)
print("csv_inventory:", CSV_INVENTORY_PATH)
print("function_inventory:", FUNCTION_INVENTORY_PATH)
print("contexts:", CONTEXT_PATH)
print("manifest:", MANIFEST_PATH)
print()
print(
    "TEMPORAL_BASELINE_INTERFACE_AUDIT_V1_PASS"
)
