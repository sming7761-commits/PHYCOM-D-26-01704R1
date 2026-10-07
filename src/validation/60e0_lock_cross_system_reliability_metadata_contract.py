import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/root/phyguard_revision")

INVENTORY_SUMMARY_PATH = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_asset_inventory_v2"
    / "summary.json"
)

INVENTORY_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "cross_system_reliability_asset_inventory_v2.json"
)

ZENODO_INVENTORY_PATH = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_asset_inventory_v2"
    / "zenodo_canonical_inventory.csv"
)

SOURCE_CP1_PATH = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint1"
    / "phyguard_rebuild"
    / "data"
    / "full"
    / "telemetry.npz"
)

SOURCE_CP2_PATH = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
    / "data"
    / "full"
    / "telemetry.npz"
)

SIONNA_DATA_ROOT = (
    ROOT
    / "data"
    / "sionna_formal_test840"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_metadata_contract_v1"
)

SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

SIONNA_IDS_PATH = (
    OUTPUT_ROOT
    / "sionna_analysis825_ids.txt"
)

MEMBERSHIP_SOURCES_PATH = (
    OUTPUT_ROOT
    / "sionna_membership_sources.csv"
)

ASSET_CONTRACT_PATH = (
    OUTPUT_ROOT
    / "asset_contract.csv"
)

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "cross_system_reliability_metadata_contract_v1.json"
)

SAMPLE_ID_PATTERN = re.compile(
    r"\bFT840_[A-Z0-9_]+\b"
)

TEXT_MEMBERSHIP_EXTENSIONS = {
    ".csv",
    ".json",
    ".txt",
    ".jsonl",
}

SEARCH_ROOTS = [
    ROOT / "results",
    ROOT / "artifacts",
    ROOT / "manifests",
    ROOT / "configs",
]

SEARCH_PATH_TOKENS = {
    "causal",
    "825",
    "formal_test840",
    "locked",
    "analysis",
}

CONTROL_LABELS = {
    "normal",
    "nonphysical_goodput",
}

SOURCE_REQUIRED = {
    "sinr": {
        "sinr_db",
        "sinr",
    },
    "ber": {
        "ber",
    },
    "per_proxy": {
        "per_proxy",
    },
    "cqi": {
        "cqi",
    },
    "mcs": {
        "mcs",
    },
    "goodput": {
        "goodput_norm",
        "goodput",
    },
}

SIONNA_REQUIRED_TOKENS = {
    "sinr": [
        "sinr",
    ],
    "ber_pre_ldpc": [
        "ber",
        "pre",
    ],
    "bler_post_ldpc": [
        "bler",
    ],
    "cqi": [
        "cqi",
    ],
    "mcs": [
        "mcs",
    ],
    "goodput": [
        "goodput",
    ],
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


def load_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_csv(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        return list(
            csv.DictReader(stream)
        )


def resolve_source_names(names):
    normalized = [
        normalize(value)
        for value in names
    ]

    resolved = {}

    for role, aliases in SOURCE_REQUIRED.items():
        matches = [
            name
            for name in normalized
            if name in aliases
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Source KPI role {role} "
                f"resolved to {matches}; "
                f"all names={normalized}"
            )

        resolved[role] = matches[0]

    return normalized, resolved


def resolve_sionna_names(names):
    normalized = [
        normalize(value)
        for value in names
    ]

    resolved = {}

    for role, required_tokens in (
        SIONNA_REQUIRED_TOKENS.items()
    ):
        matches = []

        for name in normalized:
            if all(
                token in name
                for token in required_tokens
            ):
                matches.append(name)

        if role == "ber_pre_ldpc":
            matches = [
                name
                for name in matches
                if "bler" not in name
            ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Sionna KPI role {role} "
                f"resolved to {matches}; "
                f"all names={normalized}"
            )

        resolved[role] = matches[0]

    return normalized, resolved


def nearest_causal_ancestor(path):
    for parent in path.parents:
        if (
            "causal" in parent.name.lower()
            or "825" in parent.name.lower()
        ):
            try:
                return str(
                    parent.relative_to(ROOT)
                )
            except ValueError:
                return str(parent)

    return str(
        path.parent.relative_to(ROOT)
    )


for required_path in [
    INVENTORY_SUMMARY_PATH,
    INVENTORY_MANIFEST_PATH,
    ZENODO_INVENTORY_PATH,
    SOURCE_CP1_PATH,
    SOURCE_CP2_PATH,
    SIONNA_DATA_ROOT,
]:
    if not required_path.exists():
        raise FileNotFoundError(
            required_path
        )


inventory_summary = load_json(
    INVENTORY_SUMMARY_PATH
)

inventory_manifest = load_json(
    INVENTORY_MANIFEST_PATH
)

if inventory_summary.get("status") != "PASS":
    raise RuntimeError(
        "Stage 60D2 summary is not PASS."
    )

if inventory_manifest.get("status") != "PASS":
    raise RuntimeError(
        "Stage 60D2 manifest is not PASS."
    )

if (
    inventory_summary.get(
        "canonical_zenodo_csv_count"
    )
    != 10
):
    raise RuntimeError(
        "Canonical Zenodo count is not 10."
    )


if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Stage 60E0 output already exists; "
        "refusing to overwrite."
    )

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)


# ------------------------------------------------------------
# 1. Source reconstruction metadata contract.
# ------------------------------------------------------------

source_hash_1 = sha256_file(
    SOURCE_CP1_PATH
)

source_hash_2 = sha256_file(
    SOURCE_CP2_PATH
)

if source_hash_1 != source_hash_2:
    raise RuntimeError(
        "Checkpoint telemetry files are not byte-identical."
    )


source_records = []

for checkpoint, path in [
    ("checkpoint1", SOURCE_CP1_PATH),
    ("checkpoint2", SOURCE_CP2_PATH),
]:
    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        if set(archive.files) != {
            "X",
            "kpi_names",
        }:
            raise RuntimeError(
                f"{checkpoint}: unexpected NPZ keys "
                f"{archive.files}"
            )

        source_shape = tuple(
            int(value)
            for value in archive["X"].shape
        )

        source_dtype = str(
            archive["X"].dtype
        )

        source_names_raw = [
            str(value)
            for value in archive[
                "kpi_names"
            ].tolist()
        ]

    if (
        len(source_shape) != 3
        or source_shape[1:] != (80, 10)
    ):
        raise RuntimeError(
            f"{checkpoint}: unexpected source shape "
            f"{source_shape}"
        )

    if len(source_names_raw) != 10:
        raise RuntimeError(
            f"{checkpoint}: expected 10 KPI names."
        )

    (
        source_names_normalized,
        source_role_mapping,
    ) = resolve_source_names(
        source_names_raw
    )

    source_records.append(
        {
            "checkpoint":
                checkpoint,

            "relative_path":
                str(
                    path.relative_to(ROOT)
                ),

            "sha256":
                sha256_file(path),

            "shape":
                list(source_shape),

            "dtype":
                source_dtype,

            "kpi_names_raw":
                source_names_raw,

            "kpi_names_normalized":
                source_names_normalized,

            "role_mapping":
                source_role_mapping,
        }
    )


if (
    source_records[0]["shape"]
    != source_records[1]["shape"]
):
    raise RuntimeError(
        "Checkpoint source shapes differ."
    )

if (
    source_records[0][
        "kpi_names_normalized"
    ]
    != source_records[1][
        "kpi_names_normalized"
    ]
):
    raise RuntimeError(
        "Checkpoint source KPI names differ."
    )


# ------------------------------------------------------------
# 2. Formal Sionna-840 sequence metadata contract.
# ------------------------------------------------------------

sequence_paths = sorted(
    SIONNA_DATA_ROOT.glob(
        "FT840_*/sequence.npz"
    )
)

if len(sequence_paths) != 840:
    raise RuntimeError(
        f"Expected 840 Sionna sequence files, "
        f"found {len(sequence_paths)}."
    )


formal_ids = {
    path.parent.name
    for path in sequence_paths
}

if len(formal_ids) != 840:
    raise RuntimeError(
        "Formal Sionna sample IDs are not unique."
    )


sionna_reference_names = None
sionna_reference_mapping = None
sionna_dtype_counts = Counter()
sionna_shape_counts = Counter()

for path in sequence_paths:
    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        required_keys = {
            "kpi_sequence",
            "kpi_names",
            "frame_index",
        }

        if set(archive.files) != required_keys:
            raise RuntimeError(
                f"{path.parent.name}: unexpected keys "
                f"{archive.files}"
            )

        shape = tuple(
            int(value)
            for value in archive[
                "kpi_sequence"
            ].shape
        )

        dtype = str(
            archive[
                "kpi_sequence"
            ].dtype
        )

        names = [
            str(value)
            for value in archive[
                "kpi_names"
            ].tolist()
        ]

        frame_shape = tuple(
            int(value)
            for value in archive[
                "frame_index"
            ].shape
        )

    if shape != (80, 10):
        raise RuntimeError(
            f"{path.parent.name}: shape={shape}"
        )

    if frame_shape != (80,):
        raise RuntimeError(
            f"{path.parent.name}: "
            f"frame shape={frame_shape}"
        )

    normalized_names, role_mapping = (
        resolve_sionna_names(
            names
        )
    )

    if sionna_reference_names is None:
        sionna_reference_names = (
            normalized_names
        )

        sionna_reference_mapping = (
            role_mapping
        )

    if normalized_names != sionna_reference_names:
        raise RuntimeError(
            f"{path.parent.name}: KPI-name mismatch."
        )

    if role_mapping != sionna_reference_mapping:
        raise RuntimeError(
            f"{path.parent.name}: KPI-role mismatch."
        )

    sionna_shape_counts[
        str(shape)
    ] += 1

    sionna_dtype_counts[
        dtype
    ] += 1


# ------------------------------------------------------------
# 3. Recover the already-frozen formal analysis825 membership.
#    No new exclusions or post-hoc sample selection are allowed.
# ------------------------------------------------------------

membership_candidates = []


# 3A. Search individual files containing exactly 825 formal IDs.
for search_root in SEARCH_ROOTS:
    if not search_root.exists():
        continue

    for path in search_root.rglob("*"):
        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in TEXT_MEMBERSHIP_EXTENSIONS
        ):
            continue

        if path.stat().st_size > 100 * 1024 * 1024:
            continue

        path_lower = str(path).lower()

        if not any(
            token in path_lower
            for token in SEARCH_PATH_TOKENS
        ):
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            continue

        ids = set(
            SAMPLE_ID_PATTERN.findall(
                text
            )
        ) & formal_ids

        if len(ids) == 825:
            membership_candidates.append(
                {
                    "method":
                        "single_file_exact_825",

                    "source":
                        str(
                            path.relative_to(ROOT)
                        ),

                    "ids":
                        ids,
                }
            )


# 3B. Aggregate per-sample causal PASS/valid reports.
status_groups = defaultdict(set)
status_group_files = defaultdict(int)

for search_root in SEARCH_ROOTS:
    if not search_root.exists():
        continue

    for path in search_root.rglob("*.json"):
        if not path.is_file():
            continue

        path_lower = str(path).lower()

        if "causal" not in path_lower:
            continue

        if path.stat().st_size > 10 * 1024 * 1024:
            continue

        try:
            payload = load_json(path)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        sample_id = payload.get(
            "sample_id"
        )

        if sample_id not in formal_ids:
            continue

        status = str(
            payload.get(
                "status",
                "",
            )
        ).upper()

        is_valid = (
            status
            in {
                "PASS",
                "VALID",
                "CAUSAL_VALID",
            }
            or payload.get(
                "causal_valid"
            )
            is True
            or payload.get(
                "eligible_for_analysis"
            )
            is True
            or payload.get(
                "analysis_eligible"
            )
            is True
        )

        if not is_valid:
            continue

        group = nearest_causal_ancestor(
            path
        )

        status_groups[group].add(
            sample_id
        )

        status_group_files[group] += 1


for group, ids in status_groups.items():
    if len(ids) == 825:
        membership_candidates.append(
            {
                "method":
                    "aggregated_causal_valid_reports",

                "source":
                    group,

                "source_file_count":
                    status_group_files[group],

                "ids":
                    ids,
            }
        )


if not membership_candidates:
    top_counts = sorted(
        [
            (
                group,
                len(ids),
                status_group_files[group],
            )
            for group, ids
            in status_groups.items()
        ],
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )[:20]

    print(
        "TOP CAUSAL STATUS GROUP COUNTS"
    )

    for row in top_counts:
        print(row)

    raise RuntimeError(
        "Could not recover an exact frozen "
        "825-sample membership set."
    )


reference_analysis_ids = set(
    membership_candidates[0]["ids"]
)

for candidate in membership_candidates[1:]:
    if (
        set(candidate["ids"])
        != reference_analysis_ids
    ):
        raise RuntimeError(
            "Conflicting 825-sample membership sets "
            "were discovered."
        )


analysis_ids = sorted(
    reference_analysis_ids
)

if len(analysis_ids) != 825:
    raise RuntimeError(
        "Analysis membership count is not 825."
    )

if not set(analysis_ids).issubset(
    formal_ids
):
    raise RuntimeError(
        "Analysis membership includes unknown IDs."
    )


# ------------------------------------------------------------
# 4. Verify labels and expected 825 composition.
# ------------------------------------------------------------

label_counts = Counter()
physical_count = 0
control_count = 0

for sample_id in analysis_ids:
    metadata_path = (
        SIONNA_DATA_ROOT
        / sample_id
        / "metadata.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            metadata_path
        )

    metadata = load_json(
        metadata_path
    )

    if metadata.get(
        "sample_id"
    ) != sample_id:
        raise RuntimeError(
            f"{sample_id}: metadata ID mismatch."
        )

    label = normalize(
        metadata.get(
            "label",
            "",
        )
    )

    if not label:
        raise RuntimeError(
            f"{sample_id}: missing label."
        )

    label_counts[label] += 1

    if label in CONTROL_LABELS:
        control_count += 1
    else:
        physical_count += 1


if physical_count != 706:
    raise RuntimeError(
        f"Expected 706 physical samples, "
        f"found {physical_count}."
    )

if control_count != 119:
    raise RuntimeError(
        f"Expected 119 controls, "
        f"found {control_count}."
    )


SIONNA_IDS_PATH.write_text(
    "\n".join(
        analysis_ids
    )
    + "\n",
    encoding="utf-8",
)


membership_rows = []

for candidate in membership_candidates:
    membership_rows.append(
        {
            "method":
                candidate["method"],

            "source":
                candidate["source"],

            "source_file_count":
                candidate.get(
                    "source_file_count",
                    1,
                ),

            "sample_count":
                len(candidate["ids"]),

            "membership_sha256":
                hashlib.sha256(
                    (
                        "\n".join(
                            sorted(
                                candidate["ids"]
                            )
                        )
                        + "\n"
                    ).encode("utf-8")
                ).hexdigest(),
        }
    )


with MEMBERSHIP_SOURCES_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            membership_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        membership_rows
    )


# ------------------------------------------------------------
# 5. Verify canonical Zenodo hashes without opening data rows.
# ------------------------------------------------------------

zenodo_rows = load_csv(
    ZENODO_INVENTORY_PATH
)

if len(zenodo_rows) != 10:
    raise RuntimeError(
        "Zenodo inventory count is not 10."
    )

zenodo_total_size = 0

for row in zenodo_rows:
    path = ROOT / row["relative_path"]

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    if sha256_file(path) != row["sha256"]:
        raise RuntimeError(
            f"Zenodo hash mismatch: {path.name}"
        )

    zenodo_total_size += (
        path.stat().st_size
    )


if zenodo_total_size != 8892495:
    raise RuntimeError(
        "Unexpected Zenodo total byte size."
    )


# ------------------------------------------------------------
# 6. Write frozen cross-system asset contract.
# ------------------------------------------------------------

asset_rows = [
    {
        "system":
            "source_reconstruction",

        "canonical_asset":
            str(
                SOURCE_CP2_PATH.relative_to(
                    ROOT
                )
            ),

        "sample_count":
            source_records[1]["shape"][0],

        "time_length":
            source_records[1]["shape"][1],

        "kpi_count":
            source_records[1]["shape"][2],

        "reliability_target":
            "PER_proxy",

        "scope":
            (
                "Revision-time locked uncoded "
                "48-bit packet-error proxy."
            ),
    },
    {
        "system":
            "sionna_independent_link",

        "canonical_asset":
            str(
                SIONNA_DATA_ROOT.relative_to(
                    ROOT
                )
            ),

        "sample_count":
            825,

        "time_length":
            80,

        "kpi_count":
            10,

        "reliability_target":
            sionna_reference_mapping[
                "bler_post_ldpc"
            ],

        "scope":
            (
                "Frozen causal-valid formal "
                "analysis subset only."
            ),
    },
    {
        "system":
            "zenodo_real_oran",

        "canonical_asset":
            str(
                Path(
                    zenodo_rows[0][
                        "relative_path"
                    ]
                ).parent
            ),

        "sample_count":
            10,

        "time_length":
            "",

        "kpi_count":
            14,

        "reliability_target":
            "dl_bler",

        "scope":
            (
                "Ten public real-measurement "
                "KPI traces; no labels or "
                "thresholded diagnosis."
            ),
    },
]


with ASSET_CONTRACT_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            asset_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        asset_rows
    )


summary = {
    "schema":
        "phyguard.cross_system_"
        "reliability_metadata_contract.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "LOCKED_BEFORE_RELIABILITY_VALUE_ANALYSIS",

    "source": {
        "canonical_checkpoint":
            "checkpoint2",

        "checkpoint_telemetry_files_byte_identical":
            True,

        "telemetry_sha256":
            source_hash_2,

        "shape":
            source_records[1]["shape"],

        "dtype":
            source_records[1]["dtype"],

        "kpi_names":
            source_records[1][
                "kpi_names_normalized"
            ],

        "role_mapping":
            source_records[1][
                "role_mapping"
            ],
    },

    "sionna": {
        "formal_generated_count":
            840,

        "analysis_count":
            825,

        "physical_count":
            physical_count,

        "control_count":
            control_count,

        "excluded_from_analysis_count":
            len(
                formal_ids
                - set(analysis_ids)
            ),

        "shape_counts":
            dict(
                sionna_shape_counts
            ),

        "dtype_counts":
            dict(
                sionna_dtype_counts
            ),

        "kpi_names":
            sionna_reference_names,

        "role_mapping":
            sionna_reference_mapping,

        "label_counts":
            dict(
                sorted(
                    label_counts.items()
                )
            ),

        "analysis_membership_sha256":
            sha256_file(
                SIONNA_IDS_PATH
            ),

        "membership_source_count":
            len(
                membership_candidates
            ),
    },

    "zenodo": {
        "canonical_file_count":
            10,

        "total_size_bytes":
            zenodo_total_size,

        "columns": [
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

        "value_rows_opened":
            False,
    },

    "intended_scientific_role": {
        "source":
            (
                "Characterize the uncoded "
                "48-bit packet-error proxy."
            ),

        "sionna":
            (
                "Test whether post-LDPC BLER "
                "preserves the same reliability "
                "direction in an independent "
                "coded link."
            ),

        "zenodo":
            (
                "Test the SINR-to-DL-BLER "
                "direction in real O-RAN "
                "measurements while controlling "
                "for file and MCS adaptation."
            ),
    },

    "methodological_boundary": {
        "source_kpi_values_analyzed":
            False,

        "sionna_kpi_values_analyzed":
            False,

        "zenodo_data_rows_read":
            False,

        "reliability_correlations_computed":
            False,

        "statistical_tests_computed":
            False,

        "thresholds_selected":
            False,

        "models_trained":
            False,

        "endpoints_changed":
            False,

        "analysis_subset_changed":
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
        "reliability_metadata_contract_manifest.v1",

    "status":
        summary["status"],

    "summary_sha256":
        sha256_file(
            SUMMARY_PATH
        ),

    "sionna_ids_sha256":
        sha256_file(
            SIONNA_IDS_PATH
        ),

    "membership_sources_sha256":
        sha256_file(
            MEMBERSHIP_SOURCES_PATH
        ),

    "asset_contract_sha256":
        sha256_file(
            ASSET_CONTRACT_PATH
        ),

    "source_telemetry_sha256":
        source_hash_2,

    "zenodo_inventory_sha256":
        sha256_file(
            ZENODO_INVENTORY_PATH
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


print("status:", summary["status"])

print("\nSOURCE CONTRACT")
print(
    "checkpoint_files_byte_identical:",
    True,
)
print(
    "shape:",
    source_records[1]["shape"],
)
print(
    "dtype:",
    source_records[1]["dtype"],
)
print(
    "kpi_names:",
    source_records[1][
        "kpi_names_normalized"
    ],
)
print(
    "role_mapping:",
    source_records[1][
        "role_mapping"
    ],
)

print("\nSIONNA CONTRACT")
print("formal_generated_count: 840")
print("analysis_count:", len(analysis_ids))
print("physical_count:", physical_count)
print("control_count:", control_count)
print(
    "excluded_from_analysis_count:",
    len(
        formal_ids
        - set(analysis_ids)
    ),
)
print(
    "kpi_names:",
    sionna_reference_names,
)
print(
    "role_mapping:",
    sionna_reference_mapping,
)
print(
    "label_counts:",
    dict(
        sorted(
            label_counts.items()
        )
    ),
)
print(
    "membership_source_count:",
    len(membership_candidates),
)
print(
    "analysis_membership_sha256:",
    sha256_file(
        SIONNA_IDS_PATH
    ),
)

print("\nMEMBERSHIP SOURCES")
for row in membership_rows:
    print(
        row["method"],
        "|",
        row["source"],
        "| sample_count=",
        row["sample_count"],
        "| sha256=",
        row["membership_sha256"],
    )

print("\nZENODO CONTRACT")
print("canonical_file_count: 10")
print(
    "total_size_bytes:",
    zenodo_total_size,
)
print("value_rows_opened: False")

print("\nMETHOD BOUNDARY")
for key, value in summary[
    "methodological_boundary"
].items():
    print(
        key + ":",
        value,
    )

print("\nsummary:", SUMMARY_PATH)
print("sionna_ids:", SIONNA_IDS_PATH)
print(
    "membership_sources:",
    MEMBERSHIP_SOURCES_PATH,
)
print(
    "asset_contract:",
    ASSET_CONTRACT_PATH,
)
print("manifest:", MANIFEST_PATH)

print(
    "\nCROSS_SYSTEM_RELIABILITY_"
    "METADATA_CONTRACT_V1_PASS"
)
