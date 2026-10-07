#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

PROGRESS_PATH = (
    ROOT
    / "artifacts"
    / "temporal_baseline_results_v1"
    / "progress.json"
)

LOG_PATH = (
    ROOT
    / "logs"
    / "64c1_run_temporal_baselines.log"
)

PID_PATH = (
    ROOT
    / "logs"
    / "64c1_run_temporal_baselines.pid"
)


def print_tail(
    path: Path,
    lines: int = 18,
):
    if not path.exists():
        return

    content = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    print()
    print("RECENT LOG")

    for line in content[
        -lines:
    ]:
        print(
            line
        )


if not PROGRESS_PATH.exists():
    print(
        "STAGE64C1_STATUS=NOT_STARTED"
    )
    sys.exit(
        1
    )

progress = json.loads(
    PROGRESS_PATH.read_text(
        encoding="utf-8"
    )
)

status = progress.get(
    "status",
    "UNKNOWN",
)

if (
    status == "RUNNING"
    and PID_PATH.exists()
):
    try:
        pid = int(
            PID_PATH.read_text().strip()
        )

        os.kill(
            pid,
            0,
        )

        pid_alive = True
    except Exception:
        pid_alive = False
else:
    pid_alive = False

if status == "RUNNING" and not pid_alive:
    status = "EXITED_OR_FAILED"

percent = float(
    progress.get(
        "progress_percent",
        0.0,
    )
)

width = 40

filled = int(
    round(
        width
        * percent
        / 100.0
    )
)

filled = max(
    0,
    min(
        width,
        filled,
    ),
)

bar = (
    "#"
    * filled
    + "-"
    * (
        width
        - filled
    )
)

print(
    f"STAGE64C1_STATUS={status}"
)
print(
    f"[{bar}] {percent:6.2f}%"
)
print(
    "tasks:",
    f"{progress.get('completed_tasks', 0)}/"
    f"{progress.get('total_tasks', 70)}",
)
print(
    "current:",
    progress.get(
        "current_model"
    ),
    "repeat=",
    progress.get(
        "current_repeat"
    ),
    "stage=",
    progress.get(
        "current_stage"
    ),
)

if progress.get(
    "current_epoch"
) is not None:
    print(
        "epoch:",
        (
            f"{progress.get('current_epoch')}/"
            f"{progress.get('current_max_epochs')}"
        ),
    )

print(
    "elapsed:",
    progress.get(
        "elapsed_human"
    ),
)
print(
    "ETA:",
    (
        progress.get(
            "eta_human"
        )
        or "等待首个任务完成后动态估计"
    ),
)
print(
    "estimated_finish_utc:",
    progress.get(
        "estimated_finish_utc"
    ),
)
print(
    "last_completed:",
    progress.get(
        "last_completed"
    ),
)
print(
    "formal_hardware:",
    progress.get(
        "formal_hardware"
    ),
)

print_tail(
    LOG_PATH
)
