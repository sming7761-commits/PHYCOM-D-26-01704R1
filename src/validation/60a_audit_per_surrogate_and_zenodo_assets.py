import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "per_surrogate_zenodo_asset_audit_v1"
)

SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

SOURCE_HITS_PATH = (
    OUTPUT_ROOT
    / "per_surrogate_source_hits.txt"
)

ZENODO_FILES_PATH = (
    OUTPUT_ROOT
    / "zenodo_csv_inventory.csv"
)

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "per_surrogate_zenodo_asset_audit_v1.json"
)

SEARCH_ROOT_NAMES = [
    "scripts",
    "configs",
    "src",
    "paper_assets",
    "manuscript",
]

TEXT_EXTENSIONS = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".tex",
    ".md",
    ".txt",
    ".csv",
}

EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    "logs",
    "downloads",
    "quarantine",
}

SEARCH_PATTERN = re.compile(
    r"("
    r"\bPER\b|"
    r"\bBLER\b|"
    r"\bBER\b|"
    r"packet[_ -]?error|"
    r"block[_ -]?error|"
    r"48[_ -]?(?:bit|bits)|"
    r"erfc|"
    r"qfunc|"
    r"q_function|"
    r"surrogate|"
    r"reliability"
    r")",
    flags=re.IGNORECASE,
)

ZENODO_REQUIRED_COLUMNS = {
    "time",
    "phy_mcs",
    "mac_dl_cqi",
    "mac_dl_ri",
    "mac_dl_pmi",
    "mac_ul_buffer",
    "mac_n_prb",
    "rsrq",
    "rsrp",
    "rssi",
    "dl_sinr",
    "se",
    "dl_bler",
    "delay",
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def score_hit(line):
    lower = line.lower()
    score = 0

    if "48" in lower:
        score += 20

    if "surrogate" in lower:
        score += 12

    if "packet" in lower:
        score += 10

    if "erfc" in lower or "qfunc" in lower:
        score += 10

    if "bler" in lower:
        score += 8

    if re.search(r"\bper\b", lower):
        score += 7

    if re.search(r"\bber\b", lower):
        score += 6

    if any(
        token in line
        for token in [
            "=",
            "return",
            "def ",
            "lambda",
        ]
    ):
        score += 4

    return score


if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Stage 60A output already exists; "
        "refusing to overwrite."
    )

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)


# ------------------------------------------------------------
# 1. Search source/config files for the exact PER/BER formula.
# ------------------------------------------------------------

candidate_files = []

for root_name in SEARCH_ROOT_NAMES:
    search_root = ROOT / root_name

    if not search_root.exists():
        continue

    for path in search_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        if any(
            part in EXCLUDED_PARTS
            for part in path.parts
        ):
            continue

        if path.stat().st_size > 10 * 1024 * 1024:
            continue

        candidate_files.append(path)


source_hits = []

for path in sorted(candidate_files):
    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except Exception:
        continue

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        if SEARCH_PATTERN.search(line) is None:
            continue

        context_start = max(
            0,
            line_number - 4,
        )

        context_end = min(
            len(lines),
            line_number + 3,
        )

        context = [
            {
                "line_number":
                    index + 1,

                "text":
                    lines[index],
            }
            for index in range(
                context_start,
                context_end,
            )
        ]

        source_hits.append(
            {
                "score":
                    score_hit(line),

                "relative_path":
                    str(
                        path.relative_to(ROOT)
                    ),

                "match_line_number":
                    line_number,

                "match_text":
                    line,

                "context":
                    context,
            }
        )


source_hits.sort(
    key=lambda row: (
        -row["score"],
        row["relative_path"],
        row["match_line_number"],
    )
)

source_hits = source_hits[:150]


with SOURCE_HITS_PATH.open(
    "w",
    encoding="utf-8",
) as stream:
    for index, hit in enumerate(
        source_hits,
        start=1,
    ):
        stream.write(
            "=" * 96 + "\n"
        )

        stream.write(
            f"HIT {index:03d} "
            f"| score={hit['score']} "
            f"| {hit['relative_path']}:"
            f"{hit['match_line_number']}\n"
        )

        stream.write(
            "=" * 96 + "\n"
        )

        for context_row in hit["context"]:
            marker = (
                ">>>"
                if context_row["line_number"]
                == hit["match_line_number"]
                else "   "
            )

            stream.write(
                f"{marker} "
                f"{context_row['line_number']:05d}: "
                f"{context_row['text']}\n"
            )

        stream.write("\n")


# ------------------------------------------------------------
# 2. Locate the ten local Zenodo KPI CSV files by exact schema.
#    Only the header is opened; no data rows are read.
# ------------------------------------------------------------

zenodo_records = []

ZENODO_CANONICAL_ROOT = (
    ROOT
    / "artifacts"
    / "public_real_measurement_package_structure_audit_v1"
    / "downloads"
    / "zenodo_oran_video_kpi"
)

if not ZENODO_CANONICAL_ROOT.exists():
    raise FileNotFoundError(
        ZENODO_CANONICAL_ROOT
    )

for path in ZENODO_CANONICAL_ROOT.rglob("*.csv"):
    if not path.is_file():
        continue

    if path.stat().st_size > 20 * 1024 * 1024:
        continue

    if "downloads" in path.parts:
        # These small Zenodo CSV downloads are allowed.
        pass

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
    except Exception:
        continue

    if not header:
        continue

    normalized_header = [
        str(value).strip()
        for value in header
    ]

    if set(normalized_header) != ZENODO_REQUIRED_COLUMNS:
        continue

    zenodo_records.append(
        {
            "relative_path":
                str(
                    path.relative_to(ROOT)
                ),

            "filename":
                path.name,

            "size_bytes":
                path.stat().st_size,

            "sha256":
                sha256_file(path),

            "column_count":
                len(normalized_header),

            "columns":
                "|".join(normalized_header),
        }
    )


zenodo_records.sort(
    key=lambda row: row["filename"]
)


if len(zenodo_records) != 10:
    raise RuntimeError(
        "Expected exactly 10 local Zenodo KPI CSV files, "
        f"found {len(zenodo_records)}."
    )


with ZENODO_FILES_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            zenodo_records[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(zenodo_records)


summary = {
    "schema":
        "phyguard.per_surrogate_"
        "zenodo_asset_audit.v1",

    "status":
        "PASS",

    "source_candidate_file_count":
        len(candidate_files),

    "ranked_source_hit_count":
        len(source_hits),

    "zenodo_csv_file_count":
        len(zenodo_records),

    "zenodo_total_size_bytes":
        sum(
            row["size_bytes"]
            for row in zenodo_records
        ),

    "zenodo_schema": [
        "time",
        "phy_mcs",
        "mac_dl_cqi",
        "mac_dl_ri",
        "mac_dl_pmi",
        "mac_ul_buffer",
        "mac_n_prb",
        "rsrq",
        "rsrp",
        "rssi",
        "dl_sinr",
        "se",
        "dl_bler",
        "delay",
    ],

    "files": {
        "source_hits_path":
            str(SOURCE_HITS_PATH),

        "source_hits_sha256":
            sha256_file(SOURCE_HITS_PATH),

        "zenodo_inventory_path":
            str(ZENODO_FILES_PATH),

        "zenodo_inventory_sha256":
            sha256_file(ZENODO_FILES_PATH),
    },

    "methodological_boundary": {
        "zenodo_data_rows_read":
            False,

        "zenodo_values_analyzed":
            False,

        "surrogate_formula_changed":
            False,

        "hypotheses_selected":
            False,

        "models_trained":
            False,

        "performance_computed":
            False,

        "thresholds_selected":
            False,
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


manifest = {
    "schema":
        "phyguard.per_surrogate_"
        "zenodo_asset_audit_manifest.v1",

    "status":
        "PASS",

    "summary_sha256":
        sha256_file(SUMMARY_PATH),

    "source_hits_sha256":
        sha256_file(SOURCE_HITS_PATH),

    "zenodo_inventory_sha256":
        sha256_file(ZENODO_FILES_PATH),

    "zenodo_csv_file_count":
        len(zenodo_records),

    "methodological_boundary":
        summary["methodological_boundary"],
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
    "source_candidate_file_count:",
    len(candidate_files),
)
print(
    "ranked_source_hit_count:",
    len(source_hits),
)
print(
    "zenodo_csv_file_count:",
    len(zenodo_records),
)
print(
    "zenodo_total_size_bytes:",
    summary["zenodo_total_size_bytes"],
)
print("zenodo_data_rows_read: False")
print("zenodo_values_analyzed: False")
print("surrogate_formula_changed: False")
print("models_trained: False")
print("performance_computed: False")
print("thresholds_selected: False")

print("\nTOP SOURCE HITS")
for hit in source_hits[:25]:
    print(
        f"score={hit['score']:02d}",
        "|",
        f"{hit['relative_path']}:"
        f"{hit['match_line_number']}",
        "|",
        hit["match_text"].strip()[:180],
    )

print("\nZENODO CSV FILES")
for row in zenodo_records:
    print(
        row["filename"],
        "| bytes=",
        row["size_bytes"],
        "| sha256=",
        row["sha256"],
    )

print("\nsource_hits:", SOURCE_HITS_PATH)
print("zenodo_inventory:", ZENODO_FILES_PATH)
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)
print(
    "\nPER_SURROGATE_ZENODO_"
    "ASSET_AUDIT_V1_PASS"
)
