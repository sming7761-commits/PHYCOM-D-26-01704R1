import copy
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/phyguard_revision")

V1_PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "cross_system_reliability_statistical_protocol_v1.json"
)

V1_SUMMARY_PATH = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_statistical_protocol_v1"
    / "summary.json"
)

V1_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "cross_system_reliability_statistical_protocol_v1.json"
)

SOURCE_TELEMETRY_PATH = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
    / "data"
    / "full"
    / "telemetry.npz"
)

SIONNA_IDS_PATH = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_metadata_contract_v1"
    / "sionna_analysis825_ids.txt"
)

SIONNA_DATA_ROOT = (
    ROOT
    / "data"
    / "sionna_formal_test840"
)

ZENODO_INVENTORY_PATH = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_asset_inventory_v2"
    / "zenodo_canonical_inventory.csv"
)

FAILED_V1_OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_statistics_v1"
)

V2_PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "cross_system_reliability_statistical_protocol_v2.json"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_statistical_protocol_v2"
)

MCS_AUDIT_PATH = (
    OUTPUT_ROOT
    / "mcs_encoding_audit.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "cross_system_reliability_statistical_protocol_v2.json"
)

EXPECTED_V1_STATUS = (
    "LOCKED_BEFORE_RELIABILITY_STATISTICAL_EXECUTION"
)

EXPECTED_SIONNA_MEMBERSHIP_SHA256 = (
    "a8aa74f95df897a57c82f60ea0d2f233"
    "53eaa023ec0446b079dd8a88d7de462a"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


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


def normalize_name(value):
    if isinstance(value, bytes):
        value = value.decode(
            "utf-8",
            errors="replace",
        )

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def summarize_mcs(
    system,
    asset,
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = values[
        np.isfinite(values)
    ]

    if finite.size == 0:
        raise RuntimeError(
            f"{system}: no finite MCS values."
        )

    deviations = np.abs(
        finite
        - np.rint(finite)
    )

    noninteger = (
        deviations > 1e-6
    )

    return {
        "system":
            system,

        "asset":
            asset,

        "finite_count":
            int(finite.size),

        "minimum":
            float(np.min(finite)),

        "maximum":
            float(np.max(finite)),

        "unique_count":
            int(
                np.unique(finite).size
            ),

        "noninteger_count":
            int(
                np.sum(noninteger)
            ),

        "noninteger_fraction":
            float(
                np.mean(noninteger)
            ),

        "maximum_integer_deviation":
            float(
                np.max(deviations)
            ),

        "encoding_classification":
            (
                "DISCRETE_INTEGER_INDEX"
                if not np.any(noninteger)
                else "CONTINUOUS_OR_AGGREGATED_MCS"
            ),
    }


required_paths = [
    V1_PROTOCOL_PATH,
    V1_SUMMARY_PATH,
    V1_MANIFEST_PATH,
    SOURCE_TELEMETRY_PATH,
    SIONNA_IDS_PATH,
    SIONNA_DATA_ROOT,
    ZENODO_INVENTORY_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)


v1_protocol = load_json(
    V1_PROTOCOL_PATH
)

v1_summary = load_json(
    V1_SUMMARY_PATH
)

v1_manifest = load_json(
    V1_MANIFEST_PATH
)


if (
    v1_protocol.get("status")
    != EXPECTED_V1_STATUS
):
    raise RuntimeError(
        "V1 protocol status mismatch."
    )

if (
    v1_summary.get("status")
    != EXPECTED_V1_STATUS
):
    raise RuntimeError(
        "V1 summary status mismatch."
    )

if (
    v1_manifest.get("status")
    != EXPECTED_V1_STATUS
):
    raise RuntimeError(
        "V1 manifest status mismatch."
    )

if (
    sha256_file(V1_PROTOCOL_PATH)
    != v1_manifest["protocol_sha256"]
):
    raise RuntimeError(
        "V1 protocol checksum mismatch."
    )

if (
    sha256_file(SIONNA_IDS_PATH)
    != EXPECTED_SIONNA_MEMBERSHIP_SHA256
):
    raise RuntimeError(
        "Sionna membership checksum mismatch."
    )

if V2_PROTOCOL_PATH.exists():
    raise RuntimeError(
        "V2 protocol already exists."
    )

if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "V2 protocol output already exists."
    )


# ------------------------------------------------------------
# 1. Metadata-only MCS encoding audit.
#    No SINR, BLER, PER or performance relationship is read.
# ------------------------------------------------------------

audit_rows = []


with np.load(
    SOURCE_TELEMETRY_PATH,
    allow_pickle=False,
) as archive:
    source_names = [
        normalize_name(value)
        for value in archive[
            "kpi_names"
        ].tolist()
    ]

    source_x = archive["X"]

    source_mcs_index = (
        source_names.index("mcs")
    )

    source_mcs = np.asarray(
        source_x[
            :,
            :,
            source_mcs_index,
        ],
        dtype=np.float64,
    ).reshape(-1)


source_audit = summarize_mcs(
    system="source_reconstruction",
    asset=str(
        SOURCE_TELEMETRY_PATH.relative_to(
            ROOT
        )
    ),
    values=source_mcs,
)

audit_rows.append(
    source_audit
)


analysis_ids = [
    line.strip()
    for line in SIONNA_IDS_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

if len(analysis_ids) != 825:
    raise RuntimeError(
        "Expected 825 Sionna IDs."
    )


sionna_mcs_parts = []

for sample_id in analysis_ids:
    sequence_path = (
        SIONNA_DATA_ROOT
        / sample_id
        / "sequence.npz"
    )

    if not sequence_path.exists():
        raise FileNotFoundError(
            sequence_path
        )

    with np.load(
        sequence_path,
        allow_pickle=False,
    ) as archive:
        names = [
            normalize_name(value)
            for value in archive[
                "kpi_names"
            ].tolist()
        ]

        sequence = archive[
            "kpi_sequence"
        ]

        mcs_index = names.index(
            "mcs_index"
        )

        sionna_mcs_parts.append(
            np.asarray(
                sequence[
                    :,
                    mcs_index,
                ],
                dtype=np.float64,
            )
        )


sionna_mcs = np.concatenate(
    sionna_mcs_parts
)

sionna_audit = summarize_mcs(
    system="sionna_analysis825",
    asset=str(
        SIONNA_DATA_ROOT.relative_to(
            ROOT
        )
    ),
    values=sionna_mcs,
)

audit_rows.append(
    sionna_audit
)


zenodo_inventory = load_csv(
    ZENODO_INVENTORY_PATH
)

if len(zenodo_inventory) != 10:
    raise RuntimeError(
        "Expected ten Zenodo files."
    )


zenodo_mcs_parts = []

for row in zenodo_inventory:
    path = ROOT / row[
        "relative_path"
    ]

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    if sha256_file(path) != row[
        "sha256"
    ]:
        raise RuntimeError(
            f"Zenodo checksum mismatch: "
            f"{path.name}"
        )

    frame = pd.read_csv(
        path,
        usecols=["phy_mcs"],
    )

    mcs = pd.to_numeric(
        frame["phy_mcs"],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    file_audit = summarize_mcs(
        system="zenodo_file",
        asset=str(
            path.relative_to(ROOT)
        ),
        values=mcs,
    )

    audit_rows.append(
        file_audit
    )

    zenodo_mcs_parts.append(
        mcs
    )


zenodo_all_mcs = np.concatenate(
    zenodo_mcs_parts
)

zenodo_audit = summarize_mcs(
    system="zenodo_all_files",
    asset=str(
        Path(
            zenodo_inventory[0][
                "relative_path"
            ]
        ).parent
    ),
    values=zenodo_all_mcs,
)

audit_rows.append(
    zenodo_audit
)


if (
    source_audit[
        "noninteger_count"
    ]
    != 0
):
    raise RuntimeError(
        "Source MCS is not integer encoded."
    )

if (
    sionna_audit[
        "noninteger_count"
    ]
    != 0
):
    raise RuntimeError(
        "Sionna MCS is not integer encoded."
    )

if (
    zenodo_audit[
        "finite_count"
    ]
    != 59798
):
    raise RuntimeError(
        "Unexpected Zenodo MCS count."
    )

if (
    zenodo_audit[
        "noninteger_count"
    ]
    != 57584
):
    raise RuntimeError(
        "Unexpected Zenodo noninteger MCS count."
    )

if (
    zenodo_audit[
        "noninteger_fraction"
    ]
    < 0.95
):
    raise RuntimeError(
        "Zenodo MCS is not predominantly continuous."
    )


# ------------------------------------------------------------
# 2. Record state of the failed Stage 60E2-v1 execution.
# ------------------------------------------------------------

failed_v1_files = []

if FAILED_V1_OUTPUT_ROOT.exists():
    failed_v1_files = sorted(
        str(
            path.relative_to(ROOT)
        )
        for path in FAILED_V1_OUTPUT_ROOT.rglob(
            "*"
        )
        if path.is_file()
    )


# ------------------------------------------------------------
# 3. Build protocol v2 without reading outcome relationships.
# ------------------------------------------------------------

v2_protocol = copy.deepcopy(
    v1_protocol
)

v2_protocol["schema"] = (
    "phyguard.cross_system_reliability_"
    "statistical_protocol.v2"
)

v2_protocol["created_at_utc"] = (
    datetime.now(
        timezone.utc
    ).isoformat()
)

v2_protocol["status"] = (
    "LOCKED_BEFORE_RELIABILITY_STATISTICAL_EXECUTION"
)

v2_protocol["supersedes"] = {
    "protocol":
        str(
            V1_PROTOCOL_PATH.relative_to(
                ROOT
            )
        ),

    "protocol_sha256":
        sha256_file(
            V1_PROTOCOL_PATH
        ),

    "reason":
        (
            "The V1 protocol incorrectly assumed that "
            "Zenodo phy_mcs was a discrete integer index. "
            "A metadata-only encoding audit showed that "
            "57584 of 59798 finite values were noninteger. "
            "No SINR-BLER association was inspected when "
            "making this amendment."
        ),

    "source_analysis_changed":
        False,

    "sionna_analysis_changed":
        False,

    "zenodo_predictor_changed":
        False,

    "zenodo_outcome_changed":
        False,

    "zenodo_expected_direction_changed":
        False,

    "zenodo_control_variable_changed":
        False,

    "zenodo_control_method_changed":
        True,
}


v2_protocol[
    "common_preprocessing"
][
    "mcs_canonicalization"
] = {
    "source_and_sionna":
        (
            "MCS must be finite and within 1e-6 "
            "of an integer. Canonicalize using "
            "nearest-integer rounding."
        ),

    "zenodo":
        (
            "Treat phy_mcs as a continuous or "
            "aggregated telemetry variable. Do not "
            "round, discretize, bin, or threshold it."
        ),

    "metadata_basis": {
        "source_noninteger_fraction":
            source_audit[
                "noninteger_fraction"
            ],

        "sionna_noninteger_fraction":
            sionna_audit[
                "noninteger_fraction"
            ],

        "zenodo_noninteger_fraction":
            zenodo_audit[
                "noninteger_fraction"
            ],
    },
}


v1_zenodo = copy.deepcopy(
    v1_protocol[
        "zenodo_analysis"
    ]
)

v2_protocol[
    "zenodo_analysis"
] = {
    "file_count":
        10,

    "predictor":
        "dl_sinr",

    "outcome":
        "dl_bler",

    "continuous_control":
        "phy_mcs",

    "mcs_encoding":
        {
            "classification":
                "CONTINUOUS_OR_AGGREGATED_MCS",

            "finite_value_count":
                zenodo_audit[
                    "finite_count"
                ],

            "noninteger_value_count":
                zenodo_audit[
                    "noninteger_count"
                ],

            "noninteger_fraction":
                zenodo_audit[
                    "noninteger_fraction"
                ],

            "rounding_allowed":
                False,

            "binning_allowed":
                False,
        },

    "bler_unit_normalization":
        v1_zenodo[
            "bler_unit_normalization"
        ],

    "minimum_complete_rows_per_file":
        200,

    "primary_file_effect":
        (
            "Partial Spearman correlation between "
            "dl_sinr and dl_bler while controlling "
            "continuous phy_mcs."
        ),

    "partial_spearman_algorithm": [
        (
            "Within each recording file, retain rows "
            "with finite dl_sinr, normalized dl_bler, "
            "and phy_mcs."
        ),
        (
            "Replace dl_sinr, normalized dl_bler, and "
            "phy_mcs by average ranks."
        ),
        (
            "Regress the dl_sinr ranks on an intercept "
            "and the phy_mcs ranks using ordinary least "
            "squares."
        ),
        (
            "Regress the dl_bler ranks on the same "
            "intercept and phy_mcs ranks."
        ),
        (
            "Compute Pearson correlation between the "
            "two residual vectors."
        ),
    ],

    "eligibility_failures": [
        "fewer than 200 complete rows",
        "constant ranked dl_sinr",
        "constant ranked dl_bler",
        "constant ranked phy_mcs",
        "zero residual variance",
        "nonfinite partial correlation",
    ],

    "primary_aggregate":
        v1_zenodo[
            "primary_aggregate"
        ],

    "exact_direction_test":
        v1_zenodo[
            "exact_direction_test"
        ],

    "cluster_bootstrap":
        v1_zenodo[
            "cluster_bootstrap"
        ],

    "support_rule":
        (
            "All ten files must be eligible; the "
            "equal-weight Fisher-z aggregate partial "
            "Spearman correlation must be negative; "
            "the exact one-sided file sign-flip "
            "p-value must be below 0.05; and the upper "
            "bound of the file-cluster bootstrap "
            "interval must be below zero."
        ),

    "mandatory_reporting": [
        "eligible file count",
        "excluded file count and fixed reason",
        "partial Spearman effect for every file",
        "aggregate partial Spearman correlation",
        "95% file-cluster bootstrap interval",
        "exact one-sided p-value",
        "number of files with negative effects",
        "BLER unit decision for every file",
        "complete-row count for every file",
        "MCS encoding decision",
    ],

    "exploratory_secondary_endpoints":
        v1_zenodo[
            "exploratory_secondary_endpoints"
        ],

    "secondary_multiplicity":
        v1_zenodo[
            "secondary_multiplicity"
        ],
}


prohibited = list(
    v2_protocol[
        "prohibited_posthoc_actions"
    ]
)

for item in [
    (
        "Round Zenodo phy_mcs to an integer "
        "after observing its encoding."
    ),
    (
        "Bin or discretize Zenodo phy_mcs "
        "after observing results."
    ),
    (
        "Replace continuous-MCS partial "
        "Spearman with the invalid V1 "
        "integer-stratified estimator."
    ),
]:
    if item not in prohibited:
        prohibited.append(item)

v2_protocol[
    "prohibited_posthoc_actions"
] = prohibited


v2_protocol[
    "methodological_boundary"
] = {
    "source_kpi_values_analyzed":
        False,

    "sionna_kpi_values_analyzed":
        False,

    "zenodo_mcs_encoding_values_inspected":
        True,

    "zenodo_sinr_values_analyzed":
        False,

    "zenodo_bler_values_analyzed":
        False,

    "zenodo_sinr_bler_relationship_analyzed":
        False,

    "correlations_computed_for_protocol_amendment":
        False,

    "statistical_tests_computed_for_protocol_amendment":
        False,

    "models_trained":
        False,

    "thresholds_selected":
        False,

    "predictor_changed":
        False,

    "outcome_changed":
        False,

    "expected_direction_changed":
        False,

    "source_analysis_changed":
        False,

    "sionna_analysis_changed":
        False,

    "zenodo_mcs_rounded":
        False,

    "zenodo_mcs_binned":
        False,
}


# ------------------------------------------------------------
# 4. Write frozen v2 protocol assets.
# ------------------------------------------------------------

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

V2_PROTOCOL_PATH.write_text(
    json.dumps(
        v2_protocol,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


with MCS_AUDIT_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            audit_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        audit_rows
    )


summary = {
    "schema":
        "phyguard.cross_system_reliability_"
        "statistical_protocol_summary.v2",

    "status":
        v2_protocol["status"],

    "supersedes_protocol":
        str(
            V1_PROTOCOL_PATH.relative_to(
                ROOT
            )
        ),

    "supersedes_protocol_sha256":
        sha256_file(
            V1_PROTOCOL_PATH
        ),

    "amendment_trigger":
        (
            "Zenodo phy_mcs encoding is predominantly "
            "noninteger and therefore incompatible with "
            "the V1 integer-MCS stratification."
        ),

    "source_mcs": source_audit,

    "sionna_mcs": sionna_audit,

    "zenodo_mcs": zenodo_audit,

    "source_primary_unchanged":
        True,

    "sionna_primary_unchanged":
        True,

    "zenodo_predictor_unchanged":
        "dl_sinr",

    "zenodo_outcome_unchanged":
        "dl_bler",

    "zenodo_expected_direction_unchanged":
        "negative",

    "zenodo_primary_v2":
        (
            "Within-file partial Spearman correlation "
            "between dl_sinr and dl_bler controlling "
            "continuous phy_mcs."
        ),

    "failed_v1_output_file_count":
        len(
            failed_v1_files
        ),

    "failed_v1_output_files":
        failed_v1_files,

    "protocol_path":
        str(
            V2_PROTOCOL_PATH.relative_to(
                ROOT
            )
        ),

    "protocol_sha256":
        sha256_file(
            V2_PROTOCOL_PATH
        ),

    "mcs_audit_path":
        str(
            MCS_AUDIT_PATH.relative_to(
                ROOT
            )
        ),

    "mcs_audit_sha256":
        sha256_file(
            MCS_AUDIT_PATH
        ),

    "methodological_boundary":
        v2_protocol[
            "methodological_boundary"
        ],
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
        "phyguard.cross_system_reliability_"
        "statistical_protocol_manifest.v2",

    "status":
        v2_protocol["status"],

    "v1_protocol_sha256":
        sha256_file(
            V1_PROTOCOL_PATH
        ),

    "v2_protocol_sha256":
        sha256_file(
            V2_PROTOCOL_PATH
        ),

    "summary_sha256":
        sha256_file(
            SUMMARY_PATH
        ),

    "mcs_audit_sha256":
        sha256_file(
            MCS_AUDIT_PATH
        ),

    "sionna_membership_sha256":
        sha256_file(
            SIONNA_IDS_PATH
        ),

    "zenodo_inventory_sha256":
        sha256_file(
            ZENODO_INVENTORY_PATH
        ),

    "methodological_boundary":
        v2_protocol[
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


print(
    "status:",
    v2_protocol["status"],
)

print("\nMCS ENCODING AUDIT")
print(
    "source_noninteger_count:",
    source_audit[
        "noninteger_count"
    ],
)
print(
    "source_encoding:",
    source_audit[
        "encoding_classification"
    ],
)
print(
    "sionna_noninteger_count:",
    sionna_audit[
        "noninteger_count"
    ],
)
print(
    "sionna_encoding:",
    sionna_audit[
        "encoding_classification"
    ],
)
print(
    "zenodo_finite_count:",
    zenodo_audit[
        "finite_count"
    ],
)
print(
    "zenodo_noninteger_count:",
    zenodo_audit[
        "noninteger_count"
    ],
)
print(
    "zenodo_noninteger_fraction:",
    format(
        zenodo_audit[
            "noninteger_fraction"
        ],
        ".12f",
    ),
)
print(
    "zenodo_encoding:",
    zenodo_audit[
        "encoding_classification"
    ],
)

print("\nPROTOCOL AMENDMENT")
print(
    "source_analysis_changed: False"
)
print(
    "sionna_analysis_changed: False"
)
print(
    "zenodo_predictor_changed: False"
)
print(
    "zenodo_outcome_changed: False"
)
print(
    "zenodo_expected_direction_changed: False"
)
print(
    "zenodo_primary_v2:",
    (
        "partial Spearman controlling "
        "continuous phy_mcs"
    ),
)
print(
    "zenodo_mcs_rounding_allowed: False"
)
print(
    "zenodo_mcs_binning_allowed: False"
)
print(
    "zenodo_sinr_bler_relationship_analyzed: False"
)
print(
    "failed_v1_output_file_count:",
    len(
        failed_v1_files
    ),
)

print("\nv2_protocol:", V2_PROTOCOL_PATH)
print("mcs_audit:", MCS_AUDIT_PATH)
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)

print(
    "\nCROSS_SYSTEM_RELIABILITY_"
    "STATISTICAL_PROTOCOL_V2_PASS"
)
