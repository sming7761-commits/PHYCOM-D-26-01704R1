#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from aeon.classification.convolution_based import (
    Arsenal,
    HydraClassifier,
    MiniRocketClassifier,
    MultiRocketClassifier,
    MultiRocketHydraClassifier,
    RocketClassifier,
)


ROOT = Path("/root/phyguard_revision")

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "temporal_baseline_environment_v1"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "temporal_baseline_environment_v1.json"
)


def package_version(name: str) -> str:
    return importlib.metadata.version(
        name
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


require(
    not OUTPUT_ROOT.exists(),
    f"Environment output already exists: {OUTPUT_ROOT}",
)

require(
    not MANIFEST_PATH.exists(),
    f"Environment manifest already exists: {MANIFEST_PATH}",
)

classes = [
    RocketClassifier,
    MiniRocketClassifier,
    MultiRocketClassifier,
    HydraClassifier,
    MultiRocketHydraClassifier,
    Arsenal,
]

resolved_classes = [
    (
        f"{model_class.__module__}."
        f"{model_class.__name__}"
    )
    for model_class in classes
]

cuda_available = bool(
    torch.cuda.is_available()
)

require(
    cuda_available,
    "CUDA is not available in the cloned temporal environment.",
)

device = torch.device(
    "cuda:0"
)

probe = torch.randn(
    4,
    10,
    80,
    device=device,
)

layer = torch.nn.Conv1d(
    10,
    16,
    kernel_size=3,
    padding=1,
).to(
    device
)

with torch.no_grad():
    output = layer(
        probe
    )

require(
    tuple(
        output.shape
    )
    == (
        4,
        16,
        80,
    ),
    "CUDA Conv1d smoke shape mismatch.",
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

summary = {
    "schema":
        "phyguard.temporal_baseline_environment.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "PASS",

    "environment":
        "phyguard_temporal180",

    "platform":
        platform.platform(),

    "python":
        sys.version,

    "versions": {
        "numpy":
            np.__version__,

        "pandas":
            pd.__version__,

        "scipy":
            scipy.__version__,

        "scikit_learn":
            sklearn.__version__,

        "torch":
            torch.__version__,

        "aeon":
            package_version(
                "aeon"
            ),

        "numba":
            package_version(
                "numba"
            ),
    },

    "cuda": {
        "available":
            cuda_available,

        "torch_cuda_version":
            torch.version.cuda,

        "device_count":
            int(
                torch.cuda.device_count()
            ),

        "device_name":
            torch.cuda.get_device_name(
                0
            ),

        "device_memory_bytes":
            int(
                torch.cuda.get_device_properties(
                    0
                ).total_memory
            ),

        "conv1d_smoke_shape":
            list(
                output.shape
            ),
    },

    "aeon_classes":
        resolved_classes,

    "pip_check_required":
        True,
}

SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

MANIFEST_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

MANIFEST_PATH.write_text(
    json.dumps(
        {
            "schema":
                "phyguard.temporal_baseline_environment_manifest.v1",

            "status":
                "PASS",

            "summary_path":
                str(
                    SUMMARY_PATH.relative_to(
                        ROOT
                    )
                ),

            "versions":
                summary[
                    "versions"
                ],

            "cuda":
                summary[
                    "cuda"
                ],

            "aeon_classes":
                resolved_classes,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print("environment: phyguard_temporal180")
print("versions:", summary["versions"])
print("cuda:", summary["cuda"])
print("aeon_classes:")
for value in resolved_classes:
    print(" -", value)
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)
print()
print(
    "TEMPORAL_BASELINE_ENVIRONMENT_V1_PASS"
)
