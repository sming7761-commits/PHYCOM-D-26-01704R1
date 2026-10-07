import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("/root/phyguard_revision")

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_asset_inventory_v2"
)

ZENODO_INVENTORY_PATH = (
    OUTPUT_ROOT
    / "zenodo_canonical_inventory.csv"
)

SOURCE_INVENTORY_PATH = (
    OUTPUT_ROOT
    / "source_reconstruction_asset_inventory.csv"
)

SIONNA_FILE_INVENTORY_PATH = (
    OUTPUT_ROOT
    / "sionna_candidate_file_inventory.csv"
)

SIONNA_ZIP_INVENTORY_PATH = (
    OUTPUT_ROOT
    / "sionna_archive_member_inventory.csv"
)

SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "cross_system_reliability_asset_inventory_v2.json"
)

ZENODO_ROOT = (
    ROOT
    / "artifacts"
    / "public_real_measurement_package_structure_audit_v1"
    / "downloads"
    / "zenodo_oran_video_kpi"
)

SOURCE_ROOTS = [
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint1"
    / "phyguard_rebuild"
    / "data"
    / "full",

    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint1"
    / "phyguard_rebuild"
    / "results"
    / "full",

    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
    / "data"
    / "full",

    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
    / "results"
    / "full",
]

ZENODO_EXPECTED_COLUMNS = [
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
]

CANDIDATE_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".npy",
    ".npz",
    ".parquet",
    ".pkl",
    ".pickle",
}

PATH_PATTERN = re.compile(
    r"("
    r"sionna|"
    r"formal(?:825|840)?|"
    r"sequence|"
    r"telemetry|"
    r"\bkpi\b|"
    r"sample|"
    r"record|"
    r"frame|"
    r"causal|"
    r"bler|"
    r"ber_pre_ldpc|"
    r"post_ldpc|"
    r"goodput"
    r")",
    flags=re.IGNORECASE,
)

SKIP_PARTS = {
    ".git",
    "__pycache__",
    "quarantine",
    "public_real_measurement_package_structure_audit_v1_failed_20260730T151827Z",
    "public_real_measurement_package_structure_audit_v1_failed_20260730T152920Z",
    "kuleuven_scenario1_locked_download_v1",
    "uploaded_phyguard_packages",
}

EXACT_COLUMN_GROUPS = {
    "sinr": {
        "sinr",
        "sinr_db",
        "sinr_db_meas",
        "dl_sinr",
    },
    "ber": {
        "ber",
        "ber_pre_ldpc",
        "pre_ldpc_ber",
    },
    "per_proxy": {
        "per",
        "per_proxy",
        "per_48",
        "packet_error_proxy",
    },
    "bler": {
        "bler",
        "bler_post_ldpc",
        "post_ldpc_bler",
        "dl_bler",
    },
    "cqi": {
        "cqi",
        "cqi_index",
        "mac_dl_cqi",
    },
    "mcs": {
        "mcs",
        "mcs_index",
        "phy_mcs",
    },
    "goodput": {
        "goodput",
        "goodput_norm",
        "normalized_goodput",
    },
    "spectral_efficiency": {
        "se",
        "spectral_efficiency",
    },
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def normalize(value):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def read_csv_header(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as stream:
        header = next(
            csv.reader(stream),
            None,
        )

    if not header:
        return []

    return [
        normalize(value)
        for value in header
    ]


def exact_column_matches(header):
    header_set = set(header)

    return {
        group: sorted(
            header_set & aliases
        )
        for group, aliases
        in EXACT_COLUMN_GROUPS.items()
    }


def path_score(path_text, suffix):
    lower = path_text.lower()
    score = 0

    weights = {
        "bler_post_ldpc": 80,
        "ber_pre_ldpc": 70,
        "post_ldpc": 60,
        "telemetry": 50,
        "kpi": 45,
        "sequence": 35,
        "formal825": 30,
        "formal840": 30,
        "sionna": 25,
        "goodput": 20,
        "sample": 15,
        "record": 10,
    }

    for token, weight in weights.items():
        if token in lower:
            score += weight

    if suffix in {
        ".npy",
        ".npz",
        ".parquet",
        ".jsonl",
    }:
        score += 20

    if suffix == ".csv":
        score += 10

    return score


def inspect_array_metadata(path):
    if path.suffix.lower() == ".npy":
        try:
            array = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )

            return {
                "array_names":
                    "",

                "array_shapes":
                    "x".join(
                        str(value)
                        for value in array.shape
                    ),

                "array_dtypes":
                    str(array.dtype),
            }
        except Exception as error:
            return {
                "array_names":
                    "",

                "array_shapes":
                    "",

                "array_dtypes":
                    f"ERROR:{type(error).__name__}",
            }

    if path.suffix.lower() == ".npz":
        try:
            with zipfile.ZipFile(
                path,
                mode="r",
            ) as archive:
                names = [
                    item.filename
                    for item in archive.infolist()
                    if not item.is_dir()
                ]

            return {
                "array_names":
                    "|".join(names),

                "array_shapes":
                    "",

                "array_dtypes":
                    "",
            }
        except Exception as error:
            return {
                "array_names":
                    "",

                "array_shapes":
                    "",

                "array_dtypes":
                    f"ERROR:{type(error).__name__}",
            }

    return {
        "array_names": "",
        "array_shapes": "",
        "array_dtypes": "",
    }


for path in [ZENODO_ROOT, *SOURCE_ROOTS]:
    if not path.exists():
        raise FileNotFoundError(path)


if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Stage 60D2 output already exists; "
        "refusing to overwrite."
    )

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)


# ------------------------------------------------------------
# 1. Canonical Zenodo files: exact directory and exact schema.
# ------------------------------------------------------------

zenodo_records = []

for path in sorted(
    ZENODO_ROOT.glob("*.csv")
):
    header = read_csv_header(path)

    if header != ZENODO_EXPECTED_COLUMNS:
        raise RuntimeError(
            f"Unexpected Zenodo schema: {path.name}: {header}"
        )

    matches = exact_column_matches(
        header
    )

    zenodo_records.append(
        {
            "filename":
                path.name,

            "relative_path":
                str(
                    path.relative_to(ROOT)
                ),

            "size_bytes":
                path.stat().st_size,

            "sha256":
                sha256_file(path),

            "column_count":
                len(header),

            "columns":
                "|".join(header),

            "sinr_columns":
                "|".join(matches["sinr"]),

            "bler_columns":
                "|".join(matches["bler"]),

            "cqi_columns":
                "|".join(matches["cqi"]),

            "mcs_columns":
                "|".join(matches["mcs"]),

            "spectral_efficiency_columns":
                "|".join(
                    matches[
                        "spectral_efficiency"
                    ]
                ),

            "data_rows_read":
                False,
        }
    )


if len(zenodo_records) != 10:
    raise RuntimeError(
        f"Expected 10 canonical Zenodo files, "
        f"found {len(zenodo_records)}."
    )


with ZENODO_INVENTORY_PATH.open(
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


# ------------------------------------------------------------
# 2. Source reconstruction full-data assets.
# ------------------------------------------------------------

source_records = []

for source_root in SOURCE_ROOTS:
    checkpoint = (
        "checkpoint1"
        if "Checkpoint1" in str(source_root)
        else "checkpoint2"
    )

    section = (
        "data_full"
        if "/data/full" in str(source_root)
        else "results_full"
    )

    for path in sorted(
        source_root.rglob("*")
    ):
        if not path.is_file():
            continue

        metadata = inspect_array_metadata(
            path
        )

        header = []

        if path.suffix.lower() == ".csv":
            try:
                header = read_csv_header(
                    path
                )
            except Exception:
                header = []

        matches = exact_column_matches(
            header
        )

        source_records.append(
            {
                "checkpoint":
                    checkpoint,

                "section":
                    section,

                "relative_path":
                    str(
                        path.relative_to(ROOT)
                    ),

                "filename":
                    path.name,

                "suffix":
                    path.suffix.lower(),

                "size_bytes":
                    path.stat().st_size,

                "sha256":
                    sha256_file(path),

                "columns":
                    "|".join(header),

                "sinr_columns":
                    "|".join(matches["sinr"]),

                "ber_columns":
                    "|".join(matches["ber"]),

                "per_proxy_columns":
                    "|".join(
                        matches["per_proxy"]
                    ),

                "cqi_columns":
                    "|".join(matches["cqi"]),

                "mcs_columns":
                    "|".join(matches["mcs"]),

                "goodput_columns":
                    "|".join(
                        matches["goodput"]
                    ),

                **metadata,

                "data_values_read":
                    False,
            }
        )


with SOURCE_INVENTORY_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            source_records[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(source_records)


# ------------------------------------------------------------
# 3. Extracted Sionna candidate files.
# ------------------------------------------------------------

sionna_records = []

SEARCH_ROOTS = [
    ROOT / "data",
    ROOT / "results",
    ROOT / "artifacts",
]

for search_root in SEARCH_ROOTS:
    if not search_root.exists():
        continue

    for path in search_root.rglob("*"):
        if not path.is_file():
            continue

        if any(
            part in SKIP_PARTS
            for part in path.parts
        ):
            continue

        suffix = path.suffix.lower()

        if suffix not in CANDIDATE_EXTENSIONS:
            continue

        relative_path = str(
            path.relative_to(ROOT)
        )

        if PATH_PATTERN.search(
            relative_path
        ) is None:
            continue

        score = path_score(
            relative_path,
            suffix,
        )

        header = []

        if suffix == ".csv":
            try:
                header = read_csv_header(
                    path
                )
            except Exception:
                header = []

        matches = exact_column_matches(
            header
        )

        metadata = inspect_array_metadata(
            path
        )

        sionna_records.append(
            {
                "score":
                    score,

                "relative_path":
                    relative_path,

                "filename":
                    path.name,

                "suffix":
                    suffix,

                "size_bytes":
                    path.stat().st_size,

                "sha256":
                    sha256_file(path),

                "columns":
                    "|".join(header),

                "sinr_columns":
                    "|".join(matches["sinr"]),

                "ber_columns":
                    "|".join(matches["ber"]),

                "bler_columns":
                    "|".join(matches["bler"]),

                "cqi_columns":
                    "|".join(matches["cqi"]),

                "mcs_columns":
                    "|".join(matches["mcs"]),

                "goodput_columns":
                    "|".join(
                        matches["goodput"]
                    ),

                **metadata,

                "data_values_read":
                    False,
            }
        )


sionna_records.sort(
    key=lambda row: (
        -row["score"],
        row["relative_path"],
    )
)


with SIONNA_FILE_INVENTORY_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    fieldnames = [
        "score",
        "relative_path",
        "filename",
        "suffix",
        "size_bytes",
        "sha256",
        "columns",
        "sinr_columns",
        "ber_columns",
        "bler_columns",
        "cqi_columns",
        "mcs_columns",
        "goodput_columns",
        "array_names",
        "array_shapes",
        "array_dtypes",
        "data_values_read",
    ]

    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(sionna_records)


# ------------------------------------------------------------
# 4. Sionna ZIP archive members by exact path-name ranking.
# ------------------------------------------------------------

archive_records = []

sionna_archives = sorted(
    (
        list(
            (ROOT / "results").glob(
                "*sionna*.zip"
            )
        )
        if (ROOT / "results").exists()
        else []
    ),
    key=lambda path: path.name,
)


for archive_path in sionna_archives:
    with zipfile.ZipFile(
        archive_path,
        mode="r",
    ) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            member_path = member.filename
            suffix = Path(
                member_path
            ).suffix.lower()

            if suffix not in CANDIDATE_EXTENSIONS:
                continue

            if PATH_PATTERN.search(
                member_path
            ) is None:
                continue

            archive_records.append(
                {
                    "score":
                        path_score(
                            member_path,
                            suffix,
                        ),

                    "archive_name":
                        archive_path.name,

                    "archive_sha256":
                        sha256_file(
                            archive_path
                        ),

                    "member_path":
                        member_path,

                    "member_filename":
                        Path(
                            member_path
                        ).name,

                    "suffix":
                        suffix,

                    "member_size_bytes":
                        member.file_size,

                    "member_compressed_bytes":
                        member.compress_size,

                    "data_values_read":
                        False,
                }
            )


archive_records.sort(
    key=lambda row: (
        -row["score"],
        row["archive_name"],
        row["member_path"],
    )
)


with SIONNA_ZIP_INVENTORY_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    fieldnames = [
        "score",
        "archive_name",
        "archive_sha256",
        "member_path",
        "member_filename",
        "suffix",
        "member_size_bytes",
        "member_compressed_bytes",
        "data_values_read",
    ]

    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(archive_records)


checkpoint_pairs = {}

for row in source_records:
    relative_tail = row["relative_path"].split(
        "phyguard_rebuild/",
        1,
    )[-1]

    checkpoint_pairs.setdefault(
        relative_tail,
        {},
    )[row["checkpoint"]] = row["sha256"]


identical_checkpoint_asset_count = sum(
    record.get("checkpoint1")
    == record.get("checkpoint2")
    and {
        "checkpoint1",
        "checkpoint2",
    }.issubset(record)
    for record in checkpoint_pairs.values()
)


summary = {
    "schema":
        "phyguard.cross_system_"
        "reliability_asset_inventory.v2",

    "status":
        "PASS",

    "canonical_zenodo_csv_count":
        len(zenodo_records),

    "canonical_zenodo_total_size_bytes":
        sum(
            row["size_bytes"]
            for row in zenodo_records
        ),

    "source_reconstruction_asset_count":
        len(source_records),

    "source_checkpoint_pair_count":
        len(checkpoint_pairs),

    "source_identical_checkpoint_asset_count":
        identical_checkpoint_asset_count,

    "extracted_sionna_candidate_count":
        len(sionna_records),

    "sionna_archive_count":
        len(sionna_archives),

    "sionna_archive_candidate_member_count":
        len(archive_records),

    "sionna_candidate_extension_counts":
        dict(
            Counter(
                row["suffix"]
                for row in sionna_records
            )
        ),

    "archive_candidate_extension_counts":
        dict(
            Counter(
                row["suffix"]
                for row in archive_records
            )
        ),

    "methodological_boundary": {
        "zenodo_data_rows_read":
            False,

        "zenodo_values_analyzed":
            False,

        "source_values_analyzed":
            False,

        "sionna_values_analyzed":
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
        "phyguard.cross_system_"
        "reliability_asset_inventory_manifest.v2",

    "status":
        "PASS",

    "summary_sha256":
        sha256_file(SUMMARY_PATH),

    "zenodo_inventory_sha256":
        sha256_file(
            ZENODO_INVENTORY_PATH
        ),

    "source_inventory_sha256":
        sha256_file(
            SOURCE_INVENTORY_PATH
        ),

    "sionna_file_inventory_sha256":
        sha256_file(
            SIONNA_FILE_INVENTORY_PATH
        ),

    "sionna_zip_inventory_sha256":
        sha256_file(
            SIONNA_ZIP_INVENTORY_PATH
        ),

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
    "canonical_zenodo_csv_count:",
    len(zenodo_records),
)
print(
    "canonical_zenodo_total_size_bytes:",
    summary[
        "canonical_zenodo_total_size_bytes"
    ],
)
print(
    "source_reconstruction_asset_count:",
    len(source_records),
)
print(
    "source_identical_checkpoint_asset_count:",
    identical_checkpoint_asset_count,
)
print(
    "extracted_sionna_candidate_count:",
    len(sionna_records),
)
print(
    "sionna_archive_count:",
    len(sionna_archives),
)
print(
    "sionna_archive_candidate_member_count:",
    len(archive_records),
)
print(
    "sionna_candidate_extension_counts:",
    summary[
        "sionna_candidate_extension_counts"
    ],
)
print(
    "archive_candidate_extension_counts:",
    summary[
        "archive_candidate_extension_counts"
    ],
)
print("zenodo_data_rows_read: False")
print("zenodo_values_analyzed: False")
print("source_values_analyzed: False")
print("sionna_values_analyzed: False")
print("performance_computed: False")

print("\nCANONICAL ZENODO FILES")
for row in zenodo_records:
    print(
        row["filename"],
        "| bytes=",
        row["size_bytes"],
        "| sha256=",
        row["sha256"],
        "| SINR=",
        row["sinr_columns"],
        "| BLER=",
        row["bler_columns"],
        "| CQI=",
        row["cqi_columns"],
        "| MCS=",
        row["mcs_columns"],
        "| SE=",
        row[
            "spectral_efficiency_columns"
        ],
    )

print("\nSOURCE RECONSTRUCTION ASSETS")
for row in source_records:
    print(
        row["checkpoint"],
        "|",
        row["section"],
        "|",
        row["relative_path"],
        "|",
        row["suffix"],
        "| bytes=",
        row["size_bytes"],
        "| shape=",
        row["array_shapes"],
        "| arrays=",
        row["array_names"][:180],
        "| columns=",
        row["columns"][:180],
    )

print("\nTOP EXTRACTED SIONNA CANDIDATES")
for row in sionna_records[:120]:
    print(
        f"score={row['score']:03d}",
        "|",
        row["relative_path"],
        "|",
        row["suffix"],
        "| bytes=",
        row["size_bytes"],
        "| shape=",
        row["array_shapes"],
        "| arrays=",
        row["array_names"][:160],
        "| columns=",
        row["columns"][:160],
    )

print("\nTOP SIONNA ARCHIVE MEMBERS")
for row in archive_records[:160]:
    print(
        f"score={row['score']:03d}",
        "|",
        row["archive_name"],
        "::",
        row["member_path"],
        "|",
        row["suffix"],
        "| bytes=",
        row["member_size_bytes"],
    )

print()
print("zenodo_inventory:", ZENODO_INVENTORY_PATH)
print("source_inventory:", SOURCE_INVENTORY_PATH)
print(
    "sionna_file_inventory:",
    SIONNA_FILE_INVENTORY_PATH,
)
print(
    "sionna_zip_inventory:",
    SIONNA_ZIP_INVENTORY_PATH,
)
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)
print(
    "\nCROSS_SYSTEM_RELIABILITY_"
    "ASSET_INVENTORY_V2_PASS"
)
