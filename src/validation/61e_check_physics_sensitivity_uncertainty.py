#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "physics_sensitivity_uncertainty_v1"
)

PROGRESS_PATH = OUTPUT_ROOT / "progress.json"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

LOG_PATH = (
    ROOT
    / "logs"
    / "61e_physics_sensitivity_uncertainty.log"
)


def format_seconds(value):
    if value is None:
        return "unknown"

    value = max(
        0,
        int(round(value)),
    )

    minutes, seconds = divmod(
        value,
        60,
    )

    if minutes:
        return (
            f"{minutes}m {seconds}s"
        )

    return f"{seconds}s"


if SUMMARY_PATH.exists():
    summary = json.loads(
        SUMMARY_PATH.read_text(
            encoding="utf-8"
        )
    )

    print(
        "status:",
        summary.get(
            "status"
        ),
    )

    print(
        "bootstrap_replicates:",
        summary.get(
            "bootstrap_replicates"
        ),
    )

    print(
        "cluster_count:",
        summary.get(
            "cluster_count"
        ),
    )

    print(
        "paired_comparison_count:",
        summary.get(
            "paired_comparison_count"
        ),
    )

    print(
        "summary:",
        SUMMARY_PATH,
    )

elif PROGRESS_PATH.exists():
    progress = json.loads(
        PROGRESS_PATH.read_text(
            encoding="utf-8"
        )
    )

    print(
        "status:",
        progress.get(
            "status"
        ),
    )

    print(
        "completed:",
        f"{progress.get('completed_replicates')}/"
        f"{progress.get('total_replicates')}",
    )

    print(
        "fraction:",
        f"{100 * progress.get('fraction_complete', 0):.1f}%",
    )

    print(
        "elapsed:",
        format_seconds(
            progress.get(
                "elapsed_seconds"
            )
        ),
    )

    print(
        "estimated_remaining:",
        format_seconds(
            progress.get(
                "estimated_remaining_seconds"
            )
        ),
    )

else:
    print("status: NOT_STARTED")

print("log:", LOG_PATH)

if LOG_PATH.exists():
    print()
    print("LAST 20 LOG LINES")

    lines = LOG_PATH.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    for line in lines[-20:]:
        print(line)
