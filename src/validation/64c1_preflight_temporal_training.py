#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "temporal_baseline_protocol_v1_1.json"
)

SPLIT_PATH = (
    ROOT
    / "artifacts"
    / "temporal_baseline_protocol_v1"
    / "split_memberships.csv"
)

R0 = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
)

TELEMETRY_PATH = (
    R0
    / "data"
    / "full"
    / "telemetry.npz"
)

METADATA_PATH = (
    R0
    / "data"
    / "full"
    / "metadata.csv"
)

EVALUATION_COMMON_PATH = (
    R0
    / "scripts"
    / "evaluation_common.py"
)

OUTPUT_PATH = (
    ROOT
    / "artifacts"
    / "temporal_baseline_preflight_v1"
    / "summary.json"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


for path in [
    PROTOCOL_PATH,
    SPLIT_PATH,
    TELEMETRY_PATH,
    METADATA_PATH,
    EVALUATION_COMMON_PATH,
]:
    require(
        path.exists(),
        f"Required input is missing: {path}",
    )

require(
    not OUTPUT_PATH.exists(),
    f"Preflight output already exists: {OUTPUT_PATH}",
)

protocol = json.loads(
    PROTOCOL_PATH.read_text(
        encoding="utf-8"
    )
)

require(
    protocol.get("status")
    == "LOCKED_BEFORE_TEMPORAL_BASELINE_EXECUTION",
    "Protocol v1.1 status mismatch.",
)

require(
    len(
        protocol[
            "model_registry"
        ]
    )
    == 14,
    "Protocol v1.1 does not contain 14 models.",
)

with np.load(
    TELEMETRY_PATH,
    allow_pickle=False,
) as archive:
    X = np.asarray(
        archive[
            "X"
        ]
    )

require(
    X.shape == (1080, 80, 10),
    f"Unexpected telemetry shape: {X.shape}",
)

metadata = pd.read_csv(
    METADATA_PATH
)

splits = pd.read_csv(
    SPLIT_PATH
)

require(
    len(
        metadata
    )
    == 1080,
    "Metadata row count mismatch.",
)

require(
    splits[
        "repeat"
    ].nunique()
    == 5,
    "Split repeat count mismatch.",
)

aeon_models = [
    RocketClassifier(
        n_kernels=5000,
        n_jobs=1,
        random_state=20260705,
    ),
    MiniRocketClassifier(
        n_kernels=9996,
        n_jobs=1,
        random_state=20260705,
    ),
    MultiRocketClassifier(
        n_kernels=6250,
        n_jobs=1,
        random_state=20260705,
    ),
    HydraClassifier(
        n_kernels=8,
        n_groups=64,
        n_jobs=1,
        random_state=20260705,
    ),
    MultiRocketHydraClassifier(
        n_kernels=8,
        n_groups=64,
        n_jobs=1,
        random_state=20260705,
    ),
    Arsenal(
        n_kernels=2000,
        n_estimators=10,
        n_jobs=1,
        random_state=20260705,
    ),
]

for model in aeon_models:
    require(
        hasattr(
            model,
            "fit"
        )
        and hasattr(
            model,
            "predict_proba"
        ),
        f"Aeon model API incomplete: {type(model).__name__}",
    )


class TinyProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            10,
            16,
            kernel_size=3,
            padding=1,
        )
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.head = torch.nn.Linear(
            16,
            6,
        )

    def forward(self, x):
        x = torch.relu(
            self.conv(
                x
            )
        )
        x = self.pool(
            x
        ).squeeze(-1)
        return self.head(
            x
        )


require(
    torch.cuda.is_available(),
    "CUDA is unavailable.",
)

device = torch.device(
    "cuda:0"
)

probe_model = TinyProbe().to(
    device
)

probe_input = torch.randn(
    8,
    10,
    80,
    device=device,
)

with torch.no_grad():
    probe_output = probe_model(
        probe_input
    )

require(
    tuple(
        probe_output.shape
    )
    == (
        8,
        6,
    ),
    "Deep-model forward smoke failed.",
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=False,
)

summary = {
    "status":
        "PASS",

    "formal_hardware":
        "NVIDIA GeForce RTX 3080 10 GB",

    "telemetry_shape":
        list(
            X.shape
        ),

    "metadata_rows":
        len(
            metadata
        ),

    "split_repeats":
        int(
            splits[
                "repeat"
            ].nunique()
        ),

    "aeon_constructor_count":
        len(
            aeon_models
        ),

    "deep_forward_shape":
        list(
            probe_output.shape
        ),

    "models_fitted":
        False,

    "predictions_generated":
        False,

    "test_results_viewed":
        False,
}

OUTPUT_PATH.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    "formal_hardware: NVIDIA GeForce RTX 3080 10 GB"
)
print(
    "telemetry_shape:",
    list(
        X.shape
    ),
)
print(
    "split_repeats:",
    int(
        splits[
            "repeat"
        ].nunique()
    ),
)
print(
    "aeon_constructor_count:",
    len(
        aeon_models
    ),
)
print(
    "deep_forward_shape:",
    list(
        probe_output.shape
    ),
)
print(
    "models_fitted: False"
)
print(
    "test_results_viewed: False"
)
print(
    "summary:",
    OUTPUT_PATH,
)
print()
print(
    "TEMPORAL_BASELINE_PREFLIGHT_V1_PASS"
)
