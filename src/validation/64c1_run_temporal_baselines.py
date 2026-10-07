#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

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

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "temporal_baseline_results_v1"
)

TASK_ROOT = OUTPUT_ROOT / "tasks"
PREDICTION_ROOT = OUTPUT_ROOT / "predictions"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
PROGRESS_PATH = OUTPUT_ROOT / "progress.json"
EVENT_LOG_PATH = OUTPUT_ROOT / "events.jsonl"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "temporal_baseline_results_v1.json"
)

PHYSICAL_LABELS = [
    "interference",
    "blockage",
    "mobility",
    "adaptation_mismatch",
]

CONTROL_LABELS = [
    "normal",
    "nonphysical_goodput",
]

ALL_LABELS = [
    "adaptation_mismatch",
    "blockage",
    "interference",
    "mobility",
    "nonphysical_goodput",
    "normal",
]

AEON_MODEL_IDS = [
    "ROCKET",
    "MINIROCKET",
    "MULTIROCKET",
    "HYDRA",
    "MULTIROCKET_HYDRA",
    "ARSENAL",
]

DEEP_MODEL_IDS = [
    "FCN1D",
    "RESNET1D",
    "INCEPTIONTIME_LITE",
    "LITE1D",
    "TCN",
    "BIGRU",
    "BILSTM",
    "TINY_TRANSFORMER",
]

MODEL_ORDER = AEON_MODEL_IDS + DEEP_MODEL_IDS

BASE_SEED = 20260705
REPEATS = 5
TOTAL_TASKS = len(
    MODEL_ORDER
) * REPEATS


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def iso_now() -> str:
    return utc_now().isoformat()


def atomic_json_write(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(
            path.parent
        ),
        delete=False,
        prefix=path.name + ".",
        suffix=".tmp",
    ) as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

        temporary_path = Path(
            stream.name
        )

    os.replace(
        temporary_path,
        path,
    )


def append_event(
    event: dict,
) -> None:
    EVENT_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with EVENT_LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as stream:
        stream.write(
            json.dumps(
                event,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )


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


def set_seed(seed: int) -> None:
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def format_duration(
    seconds: float | None,
) -> str | None:
    if seconds is None:
        return None

    seconds = max(
        0,
        int(
            round(
                seconds
            )
        ),
    )

    return str(
        timedelta(
            seconds=seconds
        )
    )


@dataclass
class ProgressTracker:
    started_at: datetime
    completed_tasks: int = 0
    current_model: str | None = None
    current_repeat: int | None = None
    current_stage: str | None = None
    current_epoch: int | None = None
    current_max_epochs: int | None = None
    last_completed: str | None = None

    def fraction(self) -> float:
        current_fraction = 0.0

        if (
            self.current_epoch is not None
            and self.current_max_epochs
        ):
            current_fraction = min(
                0.99,
                max(
                    0.0,
                    self.current_epoch
                    / self.current_max_epochs,
                ),
            )

        return min(
            1.0,
            (
                self.completed_tasks
                + current_fraction
            )
            / TOTAL_TASKS,
        )

    def eta_seconds(self) -> float | None:
        fraction = self.fraction()

        if fraction <= 0:
            return None

        elapsed = (
            utc_now()
            - self.started_at
        ).total_seconds()

        return max(
            0.0,
            elapsed
            * (
                1.0
                - fraction
            )
            / fraction,
        )

    def write(self) -> None:
        elapsed_seconds = (
            utc_now()
            - self.started_at
        ).total_seconds()

        eta_seconds = self.eta_seconds()

        payload = {
            "status":
                "RUNNING",

            "formal_hardware":
                "NVIDIA GeForce RTX 3080 10 GB",

            "started_at_utc":
                self.started_at.isoformat(),

            "updated_at_utc":
                iso_now(),

            "total_tasks":
                TOTAL_TASKS,

            "completed_tasks":
                self.completed_tasks,

            "progress_fraction":
                self.fraction(),

            "progress_percent":
                100.0
                * self.fraction(),

            "current_model":
                self.current_model,

            "current_repeat":
                self.current_repeat,

            "current_stage":
                self.current_stage,

            "current_epoch":
                self.current_epoch,

            "current_max_epochs":
                self.current_max_epochs,

            "last_completed":
                self.last_completed,

            "elapsed_seconds":
                elapsed_seconds,

            "elapsed_human":
                format_duration(
                    elapsed_seconds
                ),

            "eta_seconds":
                eta_seconds,

            "eta_human":
                format_duration(
                    eta_seconds
                ),

            "estimated_finish_utc":
                (
                    (
                        utc_now()
                        + timedelta(
                            seconds=eta_seconds
                        )
                    ).isoformat()
                    if eta_seconds is not None
                    else None
                ),

            "initial_estimate_hours":
                [
                    6,
                    12,
                ],
        }

        atomic_json_write(
            PROGRESS_PATH,
            payload,
        )


class FCN1D(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(
                10,
                64,
                kernel_size=8,
                padding="same",
            ),
            torch.nn.BatchNorm1d(
                64
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                0.1
            ),
            torch.nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding="same",
            ),
            torch.nn.BatchNorm1d(
                128
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                0.1
            ),
            torch.nn.Conv1d(
                128,
                64,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.BatchNorm1d(
                64
            ),
            torch.nn.ReLU(),
        )

        self.pool = torch.nn.AdaptiveAvgPool1d(
            1
        )

        self.head = torch.nn.Linear(
            64,
            6,
        )

    def forward(self, x):
        x = self.features(
            x
        )
        x = self.pool(
            x
        ).squeeze(-1)
        return self.head(
            x
        )


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int = 1,
    ):
        super().__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            torch.nn.BatchNorm1d(
                channels
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                0.1
            ),
            torch.nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            torch.nn.BatchNorm1d(
                channels
            ),
        )

        self.activation = torch.nn.ReLU()

    def forward(self, x):
        return self.activation(
            x
            + self.block(
                x
            )
        )


class ResNet1D(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(
                10,
                64,
                kernel_size=7,
                padding=3,
            ),
            torch.nn.BatchNorm1d(
                64
            ),
            torch.nn.ReLU(),
        )

        self.blocks = torch.nn.Sequential(
            ResidualBlock(
                64,
                1,
            ),
            ResidualBlock(
                64,
                2,
            ),
            ResidualBlock(
                64,
                4,
            ),
        )

        self.pool = torch.nn.AdaptiveAvgPool1d(
            1
        )

        self.head = torch.nn.Linear(
            64,
            6,
        )

    def forward(self, x):
        x = self.stem(
            x
        )
        x = self.blocks(
            x
        )
        x = self.pool(
            x
        ).squeeze(-1)
        return self.head(
            x
        )


class InceptionBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int = 32,
    ):
        super().__init__()

        bottleneck_channels = min(
            32,
            in_channels,
        )

        self.bottleneck = torch.nn.Conv1d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
        )

        self.branches = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    bottleneck_channels,
                    filters,
                    kernel_size=kernel,
                    padding="same",
                )
                for kernel in [
                    9,
                    19,
                    39,
                ]
            ]
        )

        self.pool_branch = torch.nn.Sequential(
            torch.nn.MaxPool1d(
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.Conv1d(
                in_channels,
                filters,
                kernel_size=1,
            ),
        )

        self.norm = torch.nn.BatchNorm1d(
            filters
            * 4
        )

        self.activation = torch.nn.ReLU()

    def forward(self, x):
        bottleneck = self.bottleneck(
            x
        )

        outputs = [
            branch(
                bottleneck
            )
            for branch in self.branches
        ]

        outputs.append(
            self.pool_branch(
                x
            )
        )

        return self.activation(
            self.norm(
                torch.cat(
                    outputs,
                    dim=1,
                )
            )
        )


class InceptionTimeLite(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = InceptionBlock(
            10,
            32,
        )

        self.block2 = InceptionBlock(
            128,
            32,
        )

        self.block3 = InceptionBlock(
            128,
            32,
        )

        self.pool = torch.nn.AdaptiveAvgPool1d(
            1
        )

        self.head = torch.nn.Linear(
            128,
            6,
        )

    def forward(self, x):
        x = self.block1(
            x
        )
        x = self.block2(
            x
        )
        x = self.block3(
            x
        )
        x = self.pool(
            x
        ).squeeze(-1)
        return self.head(
            x
        )


class DepthwiseSeparable(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ):
        super().__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding="same",
                groups=in_channels,
            ),
            torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
            ),
            torch.nn.BatchNorm1d(
                out_channels
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                0.1
            ),
        )

    def forward(self, x):
        return self.block(
            x
        )


class LITE1D(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.features = torch.nn.Sequential(
            DepthwiseSeparable(
                10,
                32,
                9,
            ),
            DepthwiseSeparable(
                32,
                64,
                5,
            ),
            DepthwiseSeparable(
                64,
                64,
                3,
            ),
        )

        self.pool = torch.nn.AdaptiveAvgPool1d(
            1
        )

        self.head = torch.nn.Linear(
            64,
            6,
        )

    def forward(self, x):
        x = self.features(
            x
        )
        x = self.pool(
            x
        ).squeeze(-1)
        return self.head(
            x
        )


class TCNBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
    ):
        super().__init__()

        self.block = ResidualBlock(
            channels,
            dilation,
        )

    def forward(self, x):
        return self.block(
            x
        )


class TCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = torch.nn.Conv1d(
            10,
            64,
            kernel_size=1,
        )

        self.blocks = torch.nn.Sequential(
            TCNBlock(
                64,
                1,
            ),
            TCNBlock(
                64,
                2,
            ),
            TCNBlock(
                64,
                4,
            ),
        )

        self.pool = torch.nn.AdaptiveAvgPool1d(
            1
        )

        self.head = torch.nn.Linear(
            64,
            6,
        )

    def forward(self, x):
        x = self.stem(
            x
        )
        x = self.blocks(
            x
        )
        x = self.pool(
            x
        ).squeeze(-1)
        return self.head(
            x
        )


class RecurrentModel(torch.nn.Module):
    def __init__(
        self,
        kind: str,
    ):
        super().__init__()

        recurrent_class = (
            torch.nn.GRU
            if kind == "gru"
            else torch.nn.LSTM
        )

        self.recurrent = recurrent_class(
            input_size=10,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.head = torch.nn.Linear(
            128,
            6,
        )

    def forward(self, x):
        x = x.transpose(
            1,
            2,
        )

        output, _ = self.recurrent(
            x
        )

        return self.head(
            output[
                :,
                -1,
                :,
            ]
        )


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.projection = torch.nn.Linear(
            10,
            64,
        )

        self.position = torch.nn.Parameter(
            torch.zeros(
                1,
                80,
                64,
            )
        )

        layer = torch.nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=2,
        )

        self.norm = torch.nn.LayerNorm(
            64
        )

        self.head = torch.nn.Linear(
            64,
            6,
        )

    def forward(self, x):
        x = x.transpose(
            1,
            2,
        )

        x = self.projection(
            x
        )

        x = x + self.position[
            :,
            :x.shape[
                1
            ],
            :,
        ]

        x = self.encoder(
            x
        )

        x = self.norm(
            x.mean(
                dim=1
            )
        )

        return self.head(
            x
        )


def make_deep_model(
    model_id: str,
) -> torch.nn.Module:
    if model_id == "FCN1D":
        return FCN1D()

    if model_id == "RESNET1D":
        return ResNet1D()

    if model_id == "INCEPTIONTIME_LITE":
        return InceptionTimeLite()

    if model_id == "LITE1D":
        return LITE1D()

    if model_id == "TCN":
        return TCN()

    if model_id == "BIGRU":
        return RecurrentModel(
            "gru"
        )

    if model_id == "BILSTM":
        return RecurrentModel(
            "lstm"
        )

    if model_id == "TINY_TRANSFORMER":
        return TinyTransformer()

    raise KeyError(
        model_id
    )


def make_aeon_model(
    model_id: str,
    seed: int,
):
    if model_id == "ROCKET":
        return RocketClassifier(
            n_kernels=5000,
            n_jobs=1,
            random_state=seed,
        )

    if model_id == "MINIROCKET":
        return MiniRocketClassifier(
            n_kernels=9996,
            n_jobs=1,
            random_state=seed,
        )

    if model_id == "MULTIROCKET":
        return MultiRocketClassifier(
            n_kernels=6250,
            n_jobs=1,
            random_state=seed,
        )

    if model_id == "HYDRA":
        return HydraClassifier(
            n_kernels=8,
            n_groups=64,
            n_jobs=1,
            random_state=seed,
        )

    if model_id == "MULTIROCKET_HYDRA":
        return MultiRocketHydraClassifier(
            n_kernels=8,
            n_groups=64,
            n_jobs=1,
            random_state=seed,
        )

    if model_id == "ARSENAL":
        return Arsenal(
            n_kernels=2000,
            n_estimators=10,
            n_jobs=1,
            random_state=seed,
        )

    raise KeyError(
        model_id
    )


def train_normalization(
    X_train: np.ndarray,
):
    mean = X_train.mean(
        axis=(
            0,
            1,
        ),
        keepdims=True,
    )

    standard_deviation = X_train.std(
        axis=(
            0,
            1,
        ),
        keepdims=True,
    )

    standard_deviation = np.where(
        standard_deviation
        < 1e-6,
        1.0,
        standard_deviation,
    )

    return (
        mean.astype(
            np.float32
        ),
        standard_deviation.astype(
            np.float32
        ),
    )


def normalize(
    X: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> np.ndarray:
    return (
        (
            X
            - mean
        )
        / standard_deviation
    ).astype(
        np.float32
    )


def labels_to_indices(
    labels: np.ndarray,
) -> np.ndarray:
    mapping = {
        label:
            index
        for index, label in enumerate(
            ALL_LABELS
        )
    }

    return np.asarray(
        [
            mapping[
                str(label)
            ]
            for label in labels
        ],
        dtype=np.int64,
    )


def probabilities_to_candidates(
    probabilities: np.ndarray,
    classes: np.ndarray,
):
    classes = np.asarray(
        classes,
        dtype=object,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    require(
        probabilities.ndim == 2,
        "Probability matrix must be 2D.",
    )

    require(
        probabilities.shape[
            1
        ]
        == len(
            classes
        ),
        "Probability/class dimension mismatch.",
    )

    overall_indices = np.argmax(
        probabilities,
        axis=1,
    )

    overall_labels = classes[
        overall_indices
    ].astype(
        object
    )

    physical_indices = np.asarray(
        [
            int(
                np.where(
                    classes == label
                )[
                    0
                ][
                    0
                ]
            )
            for label in PHYSICAL_LABELS
        ],
        dtype=int,
    )

    physical_probabilities = probabilities[
        :,
        physical_indices,
    ]

    max_physical_indices = np.argmax(
        physical_probabilities,
        axis=1,
    )

    max_physical_labels = np.asarray(
        PHYSICAL_LABELS,
        dtype=object,
    )[
        max_physical_indices
    ]

    max_physical_confidence = np.max(
        physical_probabilities,
        axis=1,
    )

    is_overall_physical = np.isin(
        overall_labels,
        PHYSICAL_LABELS,
    )

    candidate_labels = np.where(
        is_overall_physical,
        max_physical_labels,
        "normal",
    ).astype(
        object
    )

    confidence = np.where(
        is_overall_physical,
        max_physical_confidence,
        0.0,
    ).astype(
        np.float64
    )

    return (
        candidate_labels,
        confidence,
        overall_labels,
    )


def apply_threshold(
    candidate_labels: np.ndarray,
    confidence: np.ndarray,
    tau: float,
) -> np.ndarray:
    return np.where(
        confidence
        >= tau,
        candidate_labels,
        "normal",
    ).astype(
        object
    )


def extra_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    physical_mask = np.isin(
        y_true,
        PHYSICAL_LABELS,
    )

    control_mask = np.isin(
        y_true,
        CONTROL_LABELS,
    )

    physical_selection = (
        np.isin(
            y_pred[
                physical_mask
            ],
            PHYSICAL_LABELS,
        )
        if np.any(
            physical_mask
        )
        else np.asarray(
            [],
            dtype=bool,
        )
    )

    physical_exact = (
        y_pred[
            physical_mask
        ]
        == y_true[
            physical_mask
        ]
        if np.any(
            physical_mask
        )
        else np.asarray(
            [],
            dtype=bool,
        )
    )

    control_abstention = (
        ~np.isin(
            y_pred[
                control_mask
            ],
            PHYSICAL_LABELS,
        )
        if np.any(
            control_mask
        )
        else np.asarray(
            [],
            dtype=bool,
        )
    )

    return {
        "physical_selection_rate":
            (
                float(
                    np.mean(
                        physical_selection
                    )
                )
                if physical_selection.size
                else math.nan
            ),

        "physical_exact_match":
            (
                float(
                    np.mean(
                        physical_exact
                    )
                )
                if physical_exact.size
                else math.nan
            ),

        "control_abstention":
            (
                float(
                    np.mean(
                        control_abstention
                    )
                )
                if control_abstention.size
                else math.nan
            ),
    }


def model_parameter_count(
    model,
) -> int | None:
    if isinstance(
        model,
        torch.nn.Module,
    ):
        return int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
            )
        )

    return None


def serialized_size_bytes(
    model,
    suffix: str,
) -> int:
    with tempfile.TemporaryDirectory(
        prefix="model_size_"
    ) as directory:
        path = Path(
            directory
        ) / (
            "model"
            + suffix
        )

        if isinstance(
            model,
            torch.nn.Module,
        ):
            torch.save(
                model.state_dict(),
                path,
            )
        else:
            joblib.dump(
                model,
                path,
                compress=3,
            )

        return int(
            path.stat().st_size
        )


def inference_timing(
    function: Callable[[], np.ndarray],
    repeats: int = 20,
):
    for _ in range(
        3
    ):
        function()

    values = []

    for _ in range(
        repeats
    ):
        start = time.perf_counter()
        function()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        values.append(
            time.perf_counter()
            - start
        )

    return {
        "median_seconds":
            float(
                np.median(
                    values
                )
            ),

        "p95_seconds":
            float(
                np.quantile(
                    values,
                    0.95,
                )
            ),
    }


def train_deep_model(
    model_id: str,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    tracker: ProgressTracker,
):
    set_seed(
        seed
    )

    device = torch.device(
        "cuda:0"
    )

    model = make_deep_model(
        model_id
    ).to(
        device
    )

    train_targets = labels_to_indices(
        y_train
    )

    validation_targets = labels_to_indices(
        y_validation
    )

    train_dataset = TensorDataset(
        torch.from_numpy(
            X_train.transpose(
                0,
                2,
                1,
            )
        ),
        torch.from_numpy(
            train_targets
        ),
    )

    validation_dataset = TensorDataset(
        torch.from_numpy(
            X_validation.transpose(
                0,
                2,
                1,
            )
        ),
        torch.from_numpy(
            validation_targets
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    class_counts = np.bincount(
        train_targets,
        minlength=6,
    ).astype(
        np.float64
    )

    class_weights = (
        len(
            train_targets
        )
        / (
            6.0
            * class_counts
        )
    )

    loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

    best_validation_loss = math.inf
    best_state = None
    best_epoch = 0
    patience_counter = 0
    epoch_history = []

    train_start = time.perf_counter()

    for epoch in range(
        1,
        101,
    ):
        tracker.current_epoch = epoch
        tracker.current_max_epochs = 100
        tracker.current_stage = "deep_training"
        tracker.write()

        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(
                device,
                non_blocking=True,
            )

            batch_y = batch_y.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=True,
            ):
                logits = model(
                    batch_x
                )

                loss = loss_function(
                    logits,
                    batch_y,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            train_loss_sum += (
                float(
                    loss.detach().cpu()
                )
                * len(
                    batch_y
                )
            )

            train_count += len(
                batch_y
            )

        model.eval()
        validation_loss_sum = 0.0
        validation_count = 0

        with torch.no_grad():
            for batch_x, batch_y in validation_loader:
                batch_x = batch_x.to(
                    device,
                    non_blocking=True,
                )

                batch_y = batch_y.to(
                    device,
                    non_blocking=True,
                )

                with torch.amp.autocast(
                    "cuda",
                    enabled=True,
                ):
                    logits = model(
                        batch_x
                    )

                    loss = loss_function(
                        logits,
                        batch_y,
                    )

                validation_loss_sum += (
                    float(
                        loss.detach().cpu()
                    )
                    * len(
                        batch_y
                    )
                )

                validation_count += len(
                    batch_y
                )

        train_loss = (
            train_loss_sum
            / max(
                train_count,
                1,
            )
        )

        validation_loss = (
            validation_loss_sum
            / max(
                validation_count,
                1,
            )
        )

        scheduler.step(
            validation_loss
        )

        epoch_history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "validation_loss":
                    validation_loss,

                "learning_rate":
                    float(
                        optimizer.param_groups[
                            0
                        ][
                            "lr"
                        ]
                    ),
            }
        )

        if validation_loss < (
            best_validation_loss
            - 1e-6
        ):
            best_validation_loss = validation_loss
            best_epoch = epoch
            patience_counter = 0

            best_state = {
                key:
                    value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            patience_counter += 1

        if patience_counter >= 15:
            break

    train_seconds = (
        time.perf_counter()
        - train_start
    )

    require(
        best_state is not None,
        "Deep model did not produce a best checkpoint.",
    )

    model.load_state_dict(
        best_state
    )

    model.to(
        device
    )

    tracker.current_epoch = None
    tracker.current_max_epochs = None

    return (
        model,
        {
            "training_seconds":
                float(
                    train_seconds
                ),

            "best_epoch":
                int(
                    best_epoch
                ),

            "best_validation_loss":
                float(
                    best_validation_loss
                ),

            "epochs_completed":
                len(
                    epoch_history
                ),

            "epoch_history":
                epoch_history,
        },
    )


def predict_deep(
    model: torch.nn.Module,
    X: np.ndarray,
) -> np.ndarray:
    device = torch.device(
        "cuda:0"
    )

    dataset = TensorDataset(
        torch.from_numpy(
            X.transpose(
                0,
                2,
                1,
            )
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    outputs = []

    model.eval()

    with torch.no_grad():
        for (
            batch_x,
        ) in loader:
            batch_x = batch_x.to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                "cuda",
                enabled=True,
            ):
                logits = model(
                    batch_x
                )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            outputs.append(
                probabilities.cpu().numpy()
            )

    return np.concatenate(
        outputs,
        axis=0,
    ).astype(
        np.float64
    )


def task_id(
    model_id: str,
    repeat: int,
) -> str:
    return (
        f"{model_id}__repeat_{repeat}"
    )


def task_done_path(
    model_id: str,
    repeat: int,
) -> Path:
    return (
        TASK_ROOT
        / (
            task_id(
                model_id,
                repeat,
            )
            + ".json"
        )
    )


def prediction_path(
    model_id: str,
    repeat: int,
) -> Path:
    return (
        PREDICTION_ROOT
        / (
            task_id(
                model_id,
                repeat,
            )
            + ".csv"
        )
    )


def completed_task_count() -> int:
    return sum(
        task_done_path(
            model_id,
            repeat,
        ).exists()
        for model_id in MODEL_ORDER
        for repeat in range(
            REPEATS
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    arguments = parser.parse_args()

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
        not MANIFEST_PATH.exists(),
        (
            "Final temporal-baseline manifest already exists: "
            f"{MANIFEST_PATH}"
        ),
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
        "Protocol does not contain 14 models.",
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TASK_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTION_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if PROGRESS_PATH.exists():
        old_progress = json.loads(
            PROGRESS_PATH.read_text(
                encoding="utf-8"
            )
        )

        started_at = datetime.fromisoformat(
            old_progress[
                "started_at_utc"
            ]
        )
    else:
        started_at = utc_now()

    tracker = ProgressTracker(
        started_at=started_at,
        completed_tasks=completed_task_count(),
    )

    tracker.write()

    evaluation_common = load_module(
        "phyguard_evaluation_common_stage64c1",
        EVALUATION_COMMON_PATH,
    )

    with np.load(
        TELEMETRY_PATH,
        allow_pickle=False,
    ) as archive:
        X = np.asarray(
            archive[
                "X"
            ],
            dtype=np.float32,
        )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    labels = metadata[
        "label"
    ].astype(
        str
    ).to_numpy(
        dtype=object
    )

    splits = pd.read_csv(
        SPLIT_PATH
    )

    threshold_grid = np.asarray(
        protocol[
            "task_contract"
        ][
            "threshold_grid"
        ],
        dtype=float,
    )

    results = []

    for model_id in MODEL_ORDER:
        for repeat in range(
            REPEATS
        ):
            done_path = task_done_path(
                model_id,
                repeat,
            )

            if done_path.exists():
                results.append(
                    json.loads(
                        done_path.read_text(
                            encoding="utf-8"
                        )
                    )
                )

                tracker.completed_tasks = completed_task_count()
                tracker.last_completed = task_id(
                    model_id,
                    repeat,
                )
                tracker.write()
                continue

            tracker.current_model = model_id
            tracker.current_repeat = repeat
            tracker.current_stage = "preparing"
            tracker.current_epoch = None
            tracker.current_max_epochs = None
            tracker.completed_tasks = completed_task_count()
            tracker.write()

            append_event(
                {
                    "event":
                        "TASK_START",

                    "time_utc":
                        iso_now(),

                    "model_id":
                        model_id,

                    "repeat":
                        repeat,
                }
            )

            seed = (
                BASE_SEED
                + repeat
            )

            set_seed(
                seed
            )

            repeat_split = splits[
                splits[
                    "repeat"
                ]
                == repeat
            ]

            role_indices = {
                role:
                    repeat_split[
                        repeat_split[
                            "role"
                        ]
                        == role
                    ][
                        "index"
                    ].to_numpy(
                        dtype=int
                    )
                for role in [
                    "train",
                    "validation",
                    "test",
                ]
            }

            mean, standard_deviation = train_normalization(
                X[
                    role_indices[
                        "train"
                    ]
                ]
            )

            X_train = normalize(
                X[
                    role_indices[
                        "train"
                    ]
                ],
                mean,
                standard_deviation,
            )

            X_validation = normalize(
                X[
                    role_indices[
                        "validation"
                    ]
                ],
                mean,
                standard_deviation,
            )

            X_test = normalize(
                X[
                    role_indices[
                        "test"
                    ]
                ],
                mean,
                standard_deviation,
            )

            y_train = labels[
                role_indices[
                    "train"
                ]
            ]

            y_validation = labels[
                role_indices[
                    "validation"
                ]
            ]

            y_test = labels[
                role_indices[
                    "test"
                ]
            ]

            task_start = time.perf_counter()

            deep_metadata = None

            if model_id in AEON_MODEL_IDS:
                tracker.current_stage = "aeon_fit"
                tracker.write()

                model = make_aeon_model(
                    model_id,
                    seed,
                )

                aeon_train = X_train.transpose(
                    0,
                    2,
                    1,
                )

                aeon_validation = X_validation.transpose(
                    0,
                    2,
                    1,
                )

                aeon_test = X_test.transpose(
                    0,
                    2,
                    1,
                )

                fit_start = time.perf_counter()

                model.fit(
                    aeon_train,
                    y_train,
                )

                training_seconds = (
                    time.perf_counter()
                    - fit_start
                )

                tracker.current_stage = "aeon_prediction"
                tracker.write()

                validation_probabilities = model.predict_proba(
                    aeon_validation
                )

                test_probabilities = model.predict_proba(
                    aeon_test
                )

                classes = np.asarray(
                    model.classes_,
                    dtype=object,
                )

                timing = inference_timing(
                    lambda: model.predict_proba(
                        aeon_test
                    ),
                    repeats=10,
                )

                parameter_count = None

                serialized_bytes = serialized_size_bytes(
                    model,
                    ".joblib",
                )
            else:
                tracker.current_stage = "deep_fit"
                tracker.write()

                model, deep_metadata = train_deep_model(
                    model_id,
                    seed,
                    X_train,
                    y_train,
                    X_validation,
                    y_validation,
                    tracker,
                )

                training_seconds = deep_metadata[
                    "training_seconds"
                ]

                tracker.current_stage = "deep_prediction"
                tracker.write()

                validation_probabilities = predict_deep(
                    model,
                    X_validation,
                )

                test_probabilities = predict_deep(
                    model,
                    X_test,
                )

                classes = np.asarray(
                    ALL_LABELS,
                    dtype=object,
                )

                timing = inference_timing(
                    lambda: predict_deep(
                        model,
                        X_test,
                    ),
                    repeats=10,
                )

                parameter_count = model_parameter_count(
                    model
                )

                serialized_bytes = serialized_size_bytes(
                    model,
                    ".pt",
                )

            (
                validation_candidates,
                validation_confidence,
                validation_overall,
            ) = probabilities_to_candidates(
                validation_probabilities,
                classes,
            )

            threshold_best, threshold_rows = (
                evaluation_common.single_threshold_rows(
                    y_validation,
                    validation_candidates,
                    validation_confidence,
                    threshold_grid,
                    0.10,
                    0.80,
                )
            )

            tau = float(
                threshold_best[
                    "tau"
                ]
            )

            (
                test_candidates,
                test_confidence,
                test_overall,
            ) = probabilities_to_candidates(
                test_probabilities,
                classes,
            )

            test_predictions = apply_threshold(
                test_candidates,
                test_confidence,
                tau,
            )

            metrics = evaluation_common.selective_metrics(
                y_test,
                test_predictions,
            )

            metrics.update(
                extra_metrics(
                    y_test,
                    test_predictions,
                )
            )

            prediction_frame = pd.DataFrame(
                {
                    "model_id":
                        model_id,

                    "repeat":
                        repeat,

                    "index":
                        role_indices[
                            "test"
                        ],

                    "truth":
                        y_test,

                    "overall_prediction":
                        test_overall,

                    "candidate_prediction":
                        test_candidates,

                    "prediction":
                        test_predictions,

                    "confidence":
                        test_confidence,

                    "tau":
                        tau,
                }
            )

            for class_index, class_label in enumerate(
                classes
            ):
                prediction_frame[
                    (
                        "probability__"
                        + str(
                            class_label
                        )
                    )
                ] = test_probabilities[
                    :,
                    class_index,
                ]

            prediction_frame.to_csv(
                prediction_path(
                    model_id,
                    repeat,
                ),
                index=False,
            )

            task_seconds = (
                time.perf_counter()
                - task_start
            )

            task_result = {
                "model_id":
                    model_id,

                "repeat":
                    repeat,

                "seed":
                    seed,

                "status":
                    "PASS",

                "tau":
                    tau,

                "validation_threshold_record":
                    threshold_best,

                "metrics":
                    {
                        key:
                            float(
                                value
                            )
                        for key, value in metrics.items()
                    },

                "training_seconds":
                    float(
                        training_seconds
                    ),

                "task_seconds":
                    float(
                        task_seconds
                    ),

                "inference_median_seconds":
                    timing[
                        "median_seconds"
                    ],

                "inference_p95_seconds":
                    timing[
                        "p95_seconds"
                    ],

                "inference_median_ms_per_interval":
                    (
                        1000.0
                        * timing[
                            "median_seconds"
                        ]
                        / len(
                            y_test
                        )
                    ),

                "parameter_count":
                    parameter_count,

                "serialized_model_bytes":
                    serialized_bytes,

                "deep_training":
                    deep_metadata,

                "prediction_path":
                    str(
                        prediction_path(
                            model_id,
                            repeat,
                        ).relative_to(
                            ROOT
                        )
                    ),

                "completed_at_utc":
                    iso_now(),
            }

            atomic_json_write(
                done_path,
                task_result,
            )

            results.append(
                task_result
            )

            tracker.completed_tasks = completed_task_count()
            tracker.last_completed = task_id(
                model_id,
                repeat,
            )
            tracker.current_stage = "task_complete"
            tracker.current_epoch = None
            tracker.current_max_epochs = None
            tracker.write()

            append_event(
                {
                    "event":
                        "TASK_PASS",

                    "time_utc":
                        iso_now(),

                    "model_id":
                        model_id,

                    "repeat":
                        repeat,

                    "task_seconds":
                        task_seconds,

                    "selected_accuracy":
                        float(
                            metrics[
                                "selected_accuracy"
                            ]
                        ),

                    "coverage":
                        float(
                            metrics[
                                "coverage"
                            ]
                        ),

                    "false_specific_rate":
                        float(
                            metrics[
                                "false_specific_rate"
                            ]
                        ),
                }
            )

            print(
                "TASK_PASS",
                model_id,
                repeat,
                "selected_accuracy=",
                format(
                    float(
                        metrics[
                            "selected_accuracy"
                        ]
                    ),
                    ".6f",
                ),
                "coverage=",
                format(
                    float(
                        metrics[
                            "coverage"
                        ]
                    ),
                    ".6f",
                ),
                "false_specific_rate=",
                format(
                    float(
                        metrics[
                            "false_specific_rate"
                        ]
                    ),
                    ".6f",
                ),
                "task_seconds=",
                format(
                    task_seconds,
                    ".1f",
                ),
                flush=True,
            )

            del model
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    require(
        completed_task_count()
        == TOTAL_TASKS,
        (
            "Not all tasks completed: "
            f"{completed_task_count()}/{TOTAL_TASKS}"
        ),
    )

    task_results = [
        json.loads(
            task_done_path(
                model_id,
                repeat,
            ).read_text(
                encoding="utf-8"
            )
        )
        for model_id in MODEL_ORDER
        for repeat in range(
            REPEATS
        )
    ]

    metric_rows = []

    for result in task_results:
        metric_rows.append(
            {
                "model_id":
                    result[
                        "model_id"
                    ],

                "repeat":
                    result[
                        "repeat"
                    ],

                "tau":
                    result[
                        "tau"
                    ],

                **result[
                    "metrics"
                ],

                "training_seconds":
                    result[
                        "training_seconds"
                    ],

                "task_seconds":
                    result[
                        "task_seconds"
                    ],

                "inference_median_ms_per_interval":
                    result[
                        "inference_median_ms_per_interval"
                    ],

                "parameter_count":
                    result[
                        "parameter_count"
                    ],

                "serialized_model_bytes":
                    result[
                        "serialized_model_bytes"
                    ],
            }
        )

    metric_frame = pd.DataFrame(
        metric_rows
    )

    metric_frame.to_csv(
        OUTPUT_ROOT
        / "per_repeat_metrics.csv",
        index=False,
    )

    aggregate_rows = []

    numeric_columns = [
        column
        for column in metric_frame.columns
        if column
        not in {
            "model_id",
            "repeat",
        }
        and pd.api.types.is_numeric_dtype(
            metric_frame[
                column
            ]
        )
    ]

    for model_id, group in metric_frame.groupby(
        "model_id",
        sort=False,
    ):
        row = {
            "model_id":
                model_id,
        }

        for column in numeric_columns:
            values = group[
                column
            ].dropna()

            row[
                column
                + "_mean"
            ] = (
                float(
                    values.mean()
                )
                if len(
                    values
                )
                else math.nan
            )

            row[
                column
                + "_std"
            ] = (
                float(
                    values.std(
                        ddof=0
                    )
                )
                if len(
                    values
                )
                else math.nan
            )

        aggregate_rows.append(
            row
        )

    aggregate_frame = pd.DataFrame(
        aggregate_rows
    )

    aggregate_frame.to_csv(
        OUTPUT_ROOT
        / "aggregate_metrics.csv",
        index=False,
    )

    final_progress = json.loads(
        PROGRESS_PATH.read_text(
            encoding="utf-8"
        )
    )

    final_progress[
        "status"
    ] = "PASS"

    final_progress[
        "completed_tasks"
    ] = TOTAL_TASKS

    final_progress[
        "progress_fraction"
    ] = 1.0

    final_progress[
        "progress_percent"
    ] = 100.0

    final_progress[
        "current_model"
    ] = None

    final_progress[
        "current_repeat"
    ] = None

    final_progress[
        "current_stage"
    ] = "complete"

    final_progress[
        "current_epoch"
    ] = None

    final_progress[
        "current_max_epochs"
    ] = None

    final_progress[
        "eta_seconds"
    ] = 0.0

    final_progress[
        "eta_human"
    ] = "0:00:00"

    final_progress[
        "completed_at_utc"
    ] = iso_now()

    atomic_json_write(
        PROGRESS_PATH,
        final_progress,
    )

    summary = {
        "schema":
            "phyguard.temporal_baseline_results.v1",

        "status":
            "PASS",

        "formal_hardware":
            "NVIDIA GeForce RTX 3080 10 GB",

        "model_count":
            len(
                MODEL_ORDER
            ),

        "repeat_count":
            REPEATS,

        "task_count":
            TOTAL_TASKS,

        "model_ids":
            MODEL_ORDER,

        "aggregate_metrics_path":
            str(
                (
                    OUTPUT_ROOT
                    / "aggregate_metrics.csv"
                ).relative_to(
                    ROOT
                )
            ),

        "per_repeat_metrics_path":
            str(
                (
                    OUTPUT_ROOT
                    / "per_repeat_metrics.csv"
                ).relative_to(
                    ROOT
                )
            ),

        "progress_path":
            str(
                PROGRESS_PATH.relative_to(
                    ROOT
                )
            ),

        "protocol_path":
            str(
                PROTOCOL_PATH.relative_to(
                    ROOT
                )
            ),
    }

    atomic_json_write(
        SUMMARY_PATH,
        summary,
    )

    files = []

    for path in sorted(
        OUTPUT_ROOT.rglob(
            "*"
        )
    ):
        if path.is_file():
            digest = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

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
                        digest,
                }
            )

    manifest = {
        "schema":
            "phyguard.temporal_baseline_results_manifest.v1",

        "status":
            "PASS",

        "formal_hardware":
            "NVIDIA GeForce RTX 3080 10 GB",

        "protocol_sha256":
            hashlib.sha256(
                PROTOCOL_PATH.read_bytes()
            ).hexdigest(),

        "files":
            files,
    }

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_json_write(
        MANIFEST_PATH,
        manifest,
    )

    print()
    print(
        "AGGREGATE METRICS"
    )
    print(
        aggregate_frame.to_string(
            index=False
        )
    )
    print()
    print(
        "summary:",
        SUMMARY_PATH,
    )
    print(
        "manifest:",
        MANIFEST_PATH,
    )
    print()
    print(
        "TEMPORAL_BASELINE_RESULTS_V1_PASS"
    )


if __name__ == "__main__":
    main()
