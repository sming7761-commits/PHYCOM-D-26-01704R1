#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "per_indicator_ablation_protocol_v1.json"
)

PROTOCOL_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "per_indicator_ablation_protocol_v1.json"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "per_indicator_ablation_implementation_audit_v1"
)

SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
CANDIDATE_PATH = OUTPUT_ROOT / "candidate_files.csv"
CONTEXT_PATH = OUTPUT_ROOT / "token_contexts.txt"
ARCHIVE_MEMBER_PATH = OUTPUT_ROOT / "archive_member_candidates.csv"
IMPLEMENTATION_MAP_PATH = OUTPUT_ROOT / "IMPLEMENTATION_MAP.md"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "per_indicator_ablation_implementation_audit_v1.json"
)

SCAN_ROOTS = [
    ROOT / "scripts",
    (
        ROOT
        / "artifacts"
        / "uploaded_phyguard_packages"
        / "PhyGuard_R0_Reconstruction_Checkpoint2"
        / "phyguard_rebuild"
    ),
]

ARCHIVES = [
    (
        ROOT
        / "results"
        / "phyguard_sionna_formal825_locked_reconstruction_final.zip"
    ),
    (
        ROOT
        / "results"
        / "phyguard_sionna_formal_test840_causal_final.zip"
    ),
]

TEXT_SUFFIXES = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".md",
    ".csv",
}

TOKENS = {
    "packet_error_source":
        ["per_proxy"],

    "packet_error_sionna":
        ["bler_post_ldpc"],

    "bit_error_source":
        ["ber"],

    "bit_error_sionna":
        ["ber_pre_ldpc"],

    "selected_accuracy":
        ["selected_accuracy"],

    "gate_model":
        [
            "LogisticRegression",
            "logistic",
            "gate",
        ],

    "resolver_model":
        [
            "ExtraTreesClassifier",
            "ExtraTrees",
            "resolver",
        ],

    "training":
        [
            ".fit(",
            "fit(",
            "train_test_split",
        ],

    "prediction":
        [
            "predict_proba",
            ".predict(",
            "abstain",
        ],

    "physics":
        [
            "physics",
            "physical",
            "consistency",
            "penalty",
            "score",
        ],

    "repeat_bundle":
        [
            "repeat",
            "seed",
            "bundle",
        ],
}

HIGH_VALUE_EXACT = {
    "per_proxy": 12,
    "bler_post_ldpc": 12,
    "selected_accuracy": 10,
    "ExtraTreesClassifier": 8,
    "LogisticRegression": 8,
    "predict_proba": 5,
    "physical_selection_rate": 5,
    "control_abstention": 5,
    "analysis825": 4,
    "repeat_bundle": 4,
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


def safe_read_text(path: Path) -> str:
    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def score_text(text: str) -> tuple[int, dict[str, int]]:
    counts = {}

    for token, weight in HIGH_VALUE_EXACT.items():
        count = text.count(token)
        counts[token] = count

    score = sum(
        counts[token] * weight
        for token, weight in HIGH_VALUE_EXACT.items()
    )

    if ".fit(" in text or "fit(" in text:
        score += 5

    if "predict_proba" in text:
        score += 5

    if (
        "per_proxy" in text
        and "bler_post_ldpc" in text
    ):
        score += 20

    if (
        "selected_accuracy" in text
        and (
            "ExtraTreesClassifier" in text
            or "LogisticRegression" in text
        )
    ):
        score += 15

    return score, counts


def iter_text_files():
    seen = set()

    for scan_root in SCAN_ROOTS:
        if not scan_root.exists():
            continue

        for path in sorted(scan_root.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue

            if path.stat().st_size > 5_000_000:
                continue

            resolved = path.resolve()

            if resolved in seen:
                continue

            seen.add(resolved)
            yield path


def line_contexts(
    text: str,
    token: str,
    radius: int = 4,
    maximum_hits: int = 8,
):
    lines = text.splitlines()
    hits = []

    for index, line in enumerate(lines):
        if token not in line:
            continue

        start = max(0, index - radius)
        end = min(len(lines), index + radius + 1)

        block = []

        for cursor in range(start, end):
            marker = ">>" if cursor == index else "  "
            block.append(
                f"{marker}{cursor + 1:05d}: {lines[cursor]}"
            )

        hits.append(
            "\n".join(block)
        )

        if len(hits) >= maximum_hits:
            break

    return hits


required_paths = [
    PROTOCOL_PATH,
    PROTOCOL_MANIFEST_PATH,
]

for path in required_paths:
    require(
        path.exists(),
        f"Required path is missing: {path}",
    )

protocol = json.loads(
    safe_read_text(PROTOCOL_PATH)
)

protocol_manifest = json.loads(
    safe_read_text(PROTOCOL_MANIFEST_PATH)
)

require(
    protocol.get("status")
    == "LOCKED_BEFORE_PER_INDICATOR_ABLATION_EXECUTION",
    "Protocol status mismatch.",
)

require(
    protocol_manifest.get("protocol_sha256")
    == sha256_file(PROTOCOL_PATH),
    "Protocol checksum mismatch.",
)

require(
    not OUTPUT_ROOT.exists(),
    f"Audit output already exists: {OUTPUT_ROOT}",
)

require(
    not MANIFEST_PATH.exists(),
    f"Audit manifest already exists: {MANIFEST_PATH}",
)


candidate_rows = []
context_sections = []
global_token_counts = Counter()

for path in iter_text_files():
    text = safe_read_text(path)
    score, exact_counts = score_text(text)

    category_hits = {}

    for category, category_tokens in TOKENS.items():
        count = sum(
            text.lower().count(
                token.lower()
            )
            for token in category_tokens
        )
        category_hits[category] = count
        global_token_counts[category] += count

    if score <= 0 and not any(category_hits.values()):
        continue

    relative = str(path.relative_to(ROOT))

    candidate_rows.append(
        {
            "source_type":
                "filesystem",

            "path":
                relative,

            "score":
                score,

            "size_bytes":
                path.stat().st_size,

            **{
                f"category_{key}":
                    value
                for key, value in category_hits.items()
            },

            **{
                f"exact_{key}":
                    value
                for key, value in exact_counts.items()
            },
        }
    )

    important_tokens = [
        "per_proxy",
        "bler_post_ldpc",
        "selected_accuracy",
        "ExtraTreesClassifier",
        "LogisticRegression",
        "predict_proba",
        "physical_selection_rate",
        "control_abstention",
    ]

    selected_contexts = []

    for token in important_tokens:
        if token not in text:
            continue

        hits = line_contexts(
            text,
            token,
        )

        for hit_index, hit in enumerate(hits, start=1):
            selected_contexts.append(
                (
                    f"\n[{relative}] "
                    f"token={token} "
                    f"hit={hit_index}\n"
                    f"{hit}\n"
                )
            )

    if selected_contexts:
        context_sections.extend(
            selected_contexts
        )


archive_rows = []

for archive_path in ARCHIVES:
    if not archive_path.exists():
        continue

    with zipfile.ZipFile(
        archive_path,
        "r",
    ) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue

            suffix = Path(info.filename).suffix.lower()

            if suffix not in TEXT_SUFFIXES:
                continue

            if info.file_size > 5_000_000:
                continue

            try:
                raw = archive.read(info)
                text = raw.decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                continue

            score, exact_counts = score_text(
                text
            )

            if score <= 0:
                continue

            archive_rows.append(
                {
                    "archive":
                        str(
                            archive_path.relative_to(
                                ROOT
                            )
                        ),

                    "member":
                        info.filename,

                    "score":
                        score,

                    "size_bytes":
                        info.file_size,

                    **{
                        f"exact_{key}":
                            value
                        for key, value in exact_counts.items()
                    },
                }
            )


candidate_rows.sort(
    key=lambda row: (
        -int(row["score"]),
        row["path"],
    )
)

archive_rows.sort(
    key=lambda row: (
        -int(row["score"]),
        row["archive"],
        row["member"],
    )
)

require(
    len(candidate_rows) > 0,
    "No filesystem candidates were found.",
)

require(
    global_token_counts[
        "packet_error_source"
    ] > 0,
    "per_proxy was not found.",
)

require(
    global_token_counts[
        "packet_error_sionna"
    ] > 0,
    "bler_post_ldpc was not found.",
)

require(
    global_token_counts[
        "selected_accuracy"
    ] > 0,
    "selected_accuracy was not found.",
)


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

with CANDIDATE_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    fieldnames = list(
        candidate_rows[0].keys()
    )

    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        candidate_rows
    )


if archive_rows:
    with ARCHIVE_MEMBER_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        fieldnames = list(
            archive_rows[0].keys()
        )

        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            archive_rows
        )
else:
    ARCHIVE_MEMBER_PATH.write_text(
        "archive,member,score,size_bytes\n",
        encoding="utf-8",
    )


CONTEXT_PATH.write_text(
    "\n".join(context_sections),
    encoding="utf-8",
)


top_files = candidate_rows[:20]
top_archive_members = archive_rows[:20]

summary = {
    "schema":
        "phyguard.per_indicator_ablation_implementation_audit.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "PASS",

    "protocol_sha256":
        sha256_file(
            PROTOCOL_PATH
        ),

    "filesystem_candidate_count":
        len(candidate_rows),

    "archive_candidate_count":
        len(archive_rows),

    "global_token_counts":
        dict(global_token_counts),

    "top_files":
        [
            {
                "path":
                    row["path"],

                "score":
                    row["score"],

                "per_proxy":
                    row.get(
                        "exact_per_proxy",
                        0,
                    ),

                "bler_post_ldpc":
                    row.get(
                        "exact_bler_post_ldpc",
                        0,
                    ),

                "selected_accuracy":
                    row.get(
                        "exact_selected_accuracy",
                        0,
                    ),

                "ExtraTreesClassifier":
                    row.get(
                        "exact_ExtraTreesClassifier",
                        0,
                    ),

                "LogisticRegression":
                    row.get(
                        "exact_LogisticRegression",
                        0,
                    ),
            }
            for row in top_files
        ],

    "top_archive_members":
        [
            {
                "archive":
                    row["archive"],

                "member":
                    row["member"],

                "score":
                    row["score"],
            }
            for row in top_archive_members
        ],

    "methodological_boundary": {
        "kpi_arrays_opened":
            False,

        "models_trained":
            False,

        "predictions_generated":
            False,

        "metrics_computed":
            False,

        "source_code_and_archive_text_inspected":
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


implementation_map_lines = [
    "# Stage 62B0 implementation map",
    "",
    "Status: **PASS**",
    "",
    "No KPI arrays were opened and no model was trained.",
    "",
    "## Highest-ranked filesystem candidates",
    "",
    "| Rank | Score | Path | per_proxy | bler_post_ldpc | selected_accuracy | ExtraTrees | LogisticRegression |",
    "|---:|---:|---|---:|---:|---:|---:|---:|",
]

for index, row in enumerate(
    top_files[:15],
    start=1,
):
    implementation_map_lines.append(
        "| "
        f"{index} | "
        f"{row['score']} | "
        f"`{row['path']}` | "
        f"{row.get('exact_per_proxy', 0)} | "
        f"{row.get('exact_bler_post_ldpc', 0)} | "
        f"{row.get('exact_selected_accuracy', 0)} | "
        f"{row.get('exact_ExtraTreesClassifier', 0)} | "
        f"{row.get('exact_LogisticRegression', 0)} |"
    )

implementation_map_lines.extend(
    [
        "",
        "## Highest-ranked frozen-archive members",
        "",
        "| Rank | Score | Archive | Member |",
        "|---:|---:|---|---|",
    ]
)

for index, row in enumerate(
    top_archive_members[:15],
    start=1,
):
    implementation_map_lines.append(
        "| "
        f"{index} | "
        f"{row['score']} | "
        f"`{row['archive']}` | "
        f"`{row['member']}` |"
    )

implementation_map_lines.extend(
    [
        "",
        "## Next action",
        "",
        (
            "Use the ranked implementation files and exact line contexts "
            "to build the Stage 62B execution script without guessing "
            "feature order, score-term names, split membership, or "
            "prediction interfaces."
        ),
    ]
)

IMPLEMENTATION_MAP_PATH.write_text(
    "\n".join(
        implementation_map_lines
    )
    + "\n",
    encoding="utf-8",
)


manifest = {
    "schema":
        "phyguard.per_indicator_ablation_implementation_audit_manifest.v1",

    "status":
        "PASS",

    "protocol_sha256":
        sha256_file(
            PROTOCOL_PATH
        ),

    "summary_sha256":
        sha256_file(
            SUMMARY_PATH
        ),

    "candidate_files_sha256":
        sha256_file(
            CANDIDATE_PATH
        ),

    "archive_member_candidates_sha256":
        sha256_file(
            ARCHIVE_MEMBER_PATH
        ),

    "token_contexts_sha256":
        sha256_file(
            CONTEXT_PATH
        ),

    "implementation_map_sha256":
        sha256_file(
            IMPLEMENTATION_MAP_PATH
        ),

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
print(
    "filesystem_candidate_count:",
    len(candidate_rows),
)
print(
    "archive_candidate_count:",
    len(archive_rows),
)
print(
    "global_token_counts:",
    dict(global_token_counts),
)
print()
print("TOP FILES")
for index, row in enumerate(
    top_files[:10],
    start=1,
):
    print(
        f"{index:02d}",
        "| score=",
        row["score"],
        "|",
        row["path"],
        "| per_proxy=",
        row.get(
            "exact_per_proxy",
            0,
        ),
        "| bler_post_ldpc=",
        row.get(
            "exact_bler_post_ldpc",
            0,
        ),
        "| selected_accuracy=",
        row.get(
            "exact_selected_accuracy",
            0,
        ),
    )

print()
print("TOP ARCHIVE MEMBERS")
for index, row in enumerate(
    top_archive_members[:10],
    start=1,
):
    print(
        f"{index:02d}",
        "| score=",
        row["score"],
        "|",
        row["archive"],
        "::",
        row["member"],
    )

print()
print("summary:", SUMMARY_PATH)
print("candidate_files:", CANDIDATE_PATH)
print("archive_candidates:", ARCHIVE_MEMBER_PATH)
print("token_contexts:", CONTEXT_PATH)
print("implementation_map:", IMPLEMENTATION_MAP_PATH)
print("manifest:", MANIFEST_PATH)
print()
print(
    "PER_INDICATOR_ABLATION_"
    "IMPLEMENTATION_AUDIT_V1_PASS"
)
