import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/root/phyguard_revision")

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "kuleuven_signal_analysis_protocol_v1.json"
)

FEATURE_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_unlabeled_features_v1"
)

FEATURE_TENSOR_PATH = (
    FEATURE_ROOT
    / "features_160x80x10.npy"
)

FEATURE_NAMES_PATH = (
    FEATURE_ROOT
    / "feature_names.json"
)

LINEAGE_PATH = (
    FEATURE_ROOT
    / "unlabeled_sequence_lineage.csv"
)

FEATURE_SUMMARY_PATH = (
    FEATURE_ROOT
    / "summary.json"
)

FEATURE_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_unlabeled_features_v1.json"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_unlabeled_endpoints_v1"
)

ENDPOINT_TABLE_PATH = (
    OUTPUT_ROOT
    / "unlabeled_sequence_endpoints.csv"
)

ENDPOINT_NAMES_PATH = (
    OUTPUT_ROOT
    / "endpoint_names.json"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_unlabeled_endpoints_v1.json"
)

EXPECTED_DATASET_SNAPSHOT_SHA256 = (
    "0134ffb74c227ff41e4c10052045308d5"
    "b8cb61414f3ba54324c09815e6f0734"
)

EXPECTED_FEATURE_SNAPSHOT_SHA256 = (
    "7fc2b1b1330c9aa01c94df29afd7d179"
    "b14a6d497d9d1b8a131722461a7dde86"
)

EXPECTED_FEATURE_NAMES = [
    "primary_best_rx_power_db",
    "primary_total_power_db",
    "secondary_total_power_db",
    "tx_layer_power_gap_db",
    "primary_rx_beam_entropy",
    "primary_best_second_beam_margin_db",
    "primary_frequency_selectivity_db",
    "primary_temporal_complex_coherence",
    "primary_rx_beam_switch_rate",
    "primary_symbol_power_volatility_db",
]

ENDPOINT_NAMES = [
    "primary_best_rx_power_excursion_db",
    "deep_fade_occupancy_3db",
    "maximum_contiguous_deep_fade_fraction",
    "rx_beam_entropy_excursion",
    "temporal_coherence_excursion",
    "frequency_selectivity_excursion_db",
    "mean_rx_beam_switch_rate",
    "power_volatility_q95_db",
]

EXPECTED_SHAPE = (160, 80, 10)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def quantile(values, probability):
    return float(
        np.quantile(
            values,
            probability,
            method="linear",
        )
    )


def longest_true_run(values):
    longest = 0
    current = 0

    for value in values:
        if bool(value):
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 0

    return longest


required_paths = [
    PROTOCOL_PATH,
    FEATURE_TENSOR_PATH,
    FEATURE_NAMES_PATH,
    LINEAGE_PATH,
    FEATURE_SUMMARY_PATH,
    FEATURE_MANIFEST_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)


protocol = load_json(
    PROTOCOL_PATH
)

feature_summary = load_json(
    FEATURE_SUMMARY_PATH
)

feature_manifest = load_json(
    FEATURE_MANIFEST_PATH
)

feature_names_record = load_json(
    FEATURE_NAMES_PATH
)


if (
    protocol.get("status")
    != "LOCKED_BEFORE_SIGNAL_VALUE_ANALYSIS"
):
    raise RuntimeError(
        "Unexpected Stage 59D protocol status."
    )


if feature_summary.get("status") != "PASS":
    raise RuntimeError(
        "Stage 59E summary is not PASS."
    )


if feature_manifest.get("status") != "PASS":
    raise RuntimeError(
        "Stage 59E manifest is not PASS."
    )


if (
    feature_summary.get(
        "dataset_snapshot_sha256"
    )
    != EXPECTED_DATASET_SNAPSHOT_SHA256
):
    raise RuntimeError(
        "Dataset snapshot mismatch."
    )


if (
    feature_summary.get(
        "unlabeled_feature_snapshot_sha256"
    )
    != EXPECTED_FEATURE_SNAPSHOT_SHA256
):
    raise RuntimeError(
        "Unlabeled feature snapshot mismatch."
    )


if feature_summary.get(
    "methodological_boundary",
    {},
).get("labels_joined") is not False:
    raise RuntimeError(
        "Stage 59E labels_joined boundary failed."
    )


if feature_summary.get(
    "methodological_boundary",
    {},
).get("label_manifest_opened") is not False:
    raise RuntimeError(
        "Stage 59E label-manifest boundary failed."
    )


if (
    feature_names_record.get(
        "feature_names"
    )
    != EXPECTED_FEATURE_NAMES
):
    raise RuntimeError(
        "Feature-name order mismatch."
    )


if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Stage 59F output already exists; "
        "refusing to overwrite it."
    )


tensor = np.load(
    FEATURE_TENSOR_PATH,
    allow_pickle=False,
)


if tensor.shape != EXPECTED_SHAPE:
    raise RuntimeError(
        f"Unexpected tensor shape: {tensor.shape}"
    )


if tensor.dtype != np.float64:
    raise RuntimeError(
        f"Unexpected tensor dtype: {tensor.dtype}"
    )


if not np.all(np.isfinite(tensor)):
    raise RuntimeError(
        "Feature tensor contains non-finite values."
    )


sequence_uids = []

with LINEAGE_PATH.open(
    "r",
    encoding="utf-8",
    newline="",
) as stream:
    reader = csv.DictReader(stream)

    for expected_index, row in enumerate(reader):
        actual_index = int(
            row["sequence_index"]
        )

        if actual_index != expected_index:
            raise RuntimeError(
                "Lineage sequence-index order mismatch."
            )

        uid = row["sequence_uid"]

        if not uid.startswith("KU26G_"):
            raise RuntimeError(
                f"Unexpected opaque UID: {uid}"
            )

        sequence_uids.append(uid)


if len(sequence_uids) != EXPECTED_SHAPE[0]:
    raise RuntimeError(
        "Lineage sequence count mismatch."
    )


if len(set(sequence_uids)) != len(sequence_uids):
    raise RuntimeError(
        "Duplicate opaque sequence UID."
    )


endpoint_rows = []


for sequence_index, uid in enumerate(
    sequence_uids
):
    values = tensor[
        sequence_index
    ]

    primary_power = values[:, 0]
    entropy = values[:, 4]
    frequency_selectivity = values[:, 6]
    temporal_coherence = values[:, 7]
    beam_switch_rate = values[:, 8]
    power_volatility = values[:, 9]

    primary_q05 = quantile(
        primary_power,
        0.05,
    )

    primary_q95 = quantile(
        primary_power,
        0.95,
    )

    deep_fade_mask = (
        primary_power
        <= primary_q95 - 3.0
    )

    endpoint_values = {
        "primary_best_rx_power_excursion_db":
            primary_q95 - primary_q05,

        "deep_fade_occupancy_3db":
            float(
                np.mean(
                    deep_fade_mask
                )
            ),

        "maximum_contiguous_deep_fade_fraction":
            float(
                longest_true_run(
                    deep_fade_mask
                )
                / values.shape[0]
            ),

        "rx_beam_entropy_excursion":
            (
                quantile(
                    entropy,
                    0.95,
                )
                - quantile(
                    entropy,
                    0.05,
                )
            ),

        "temporal_coherence_excursion":
            (
                quantile(
                    temporal_coherence,
                    0.95,
                )
                - quantile(
                    temporal_coherence,
                    0.05,
                )
            ),

        "frequency_selectivity_excursion_db":
            (
                quantile(
                    frequency_selectivity,
                    0.95,
                )
                - quantile(
                    frequency_selectivity,
                    0.05,
                )
            ),

        "mean_rx_beam_switch_rate":
            float(
                np.mean(
                    beam_switch_rate
                )
            ),

        "power_volatility_q95_db":
            quantile(
                power_volatility,
                0.95,
            ),
    }


    endpoint_array = np.asarray(
        [
            endpoint_values[name]
            for name in ENDPOINT_NAMES
        ],
        dtype=np.float64,
    )


    if not np.all(
        np.isfinite(endpoint_array)
    ):
        raise RuntimeError(
            f"Non-finite endpoint for {uid}."
        )


    for bounded_name in [
        "deep_fade_occupancy_3db",
        "maximum_contiguous_deep_fade_fraction",
        "mean_rx_beam_switch_rate",
    ]:
        bounded_value = endpoint_values[
            bounded_name
        ]

        if not 0.0 <= bounded_value <= 1.0:
            raise RuntimeError(
                f"Bounded endpoint outside range: "
                f"{uid}, {bounded_name}, "
                f"{bounded_value}"
            )


    endpoint_rows.append(
        {
            "sequence_index":
                sequence_index,

            "sequence_uid":
                uid,

            **{
                name:
                    format(
                        endpoint_values[name],
                        ".17g",
                    )
                for name in ENDPOINT_NAMES
            },
        }
    )


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)


with ENDPOINT_TABLE_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "sequence_index",
            "sequence_uid",
            *ENDPOINT_NAMES,
        ],
    )

    writer.writeheader()
    writer.writerows(
        endpoint_rows
    )


ENDPOINT_NAMES_PATH.write_text(
    json.dumps(
        {
            "schema":
                "phyguard.kuleuven."
                "endpoint_names.v1",

            "endpoint_count":
                len(
                    ENDPOINT_NAMES
                ),

            "primary_endpoint":
                ENDPOINT_NAMES[0],

            "secondary_endpoints":
                ENDPOINT_NAMES[1:],

            "endpoint_names":
                ENDPOINT_NAMES,

            "quantile_method":
                "numpy_linear",

            "deep_fade_threshold_db":
                3.0,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


endpoint_table_sha256 = sha256_file(
    ENDPOINT_TABLE_PATH
)

endpoint_names_sha256 = sha256_file(
    ENDPOINT_NAMES_PATH
)

protocol_sha256 = sha256_file(
    PROTOCOL_PATH
)


unlabeled_endpoint_snapshot_sha256 = (
    hashlib.sha256(
        (
            EXPECTED_FEATURE_SNAPSHOT_SHA256
            + "|"
            + protocol_sha256
            + "|"
            + endpoint_table_sha256
            + "|"
            + endpoint_names_sha256
        ).encode("utf-8")
    ).hexdigest()
)


summary = {
    "schema":
        "phyguard.kuleuven."
        "unlabeled_endpoints.v1",

    "status":
        "PASS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "dataset_snapshot_sha256":
        EXPECTED_DATASET_SNAPSHOT_SHA256,

    "unlabeled_feature_snapshot_sha256":
        EXPECTED_FEATURE_SNAPSHOT_SHA256,

    "unlabeled_endpoint_snapshot_sha256":
        unlabeled_endpoint_snapshot_sha256,

    "sequence_count":
        len(
            endpoint_rows
        ),

    "endpoint_count":
        len(
            ENDPOINT_NAMES
        ),

    "primary_endpoint":
        ENDPOINT_NAMES[0],

    "secondary_endpoint_count":
        len(
            ENDPOINT_NAMES
        ) - 1,

    "finite_endpoint_value_count":
        len(endpoint_rows)
        * len(ENDPOINT_NAMES),

    "expected_endpoint_value_count":
        160
        * len(ENDPOINT_NAMES),

    "files": {
        "endpoint_table_path":
            str(
                ENDPOINT_TABLE_PATH
            ),

        "endpoint_table_sha256":
            endpoint_table_sha256,

        "endpoint_names_path":
            str(
                ENDPOINT_NAMES_PATH
            ),

        "endpoint_names_sha256":
            endpoint_names_sha256,
    },

    "methodological_boundary": {
        "signal_values_read":
            True,

        "features_read":
            True,

        "sequence_endpoints_computed":
            True,

        "labels_joined":
            False,

        "label_manifest_opened":
            False,

        "activity_numbers_parsed":
            False,

        "models_trained":
            False,

        "predictions_computed":
            False,

        "classification_performance_computed":
            False,

        "statistical_tests_computed":
            False,

        "thresholds_selected":
            False,

        "endpoint_values_printed":
            False,

        "endpoint_definitions_changed":
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
        "phyguard.kuleuven."
        "unlabeled_endpoints_manifest.v1",

    "status":
        "PASS",

    "dataset_snapshot_sha256":
        EXPECTED_DATASET_SNAPSHOT_SHA256,

    "unlabeled_feature_snapshot_sha256":
        EXPECTED_FEATURE_SNAPSHOT_SHA256,

    "unlabeled_endpoint_snapshot_sha256":
        unlabeled_endpoint_snapshot_sha256,

    "sequence_count":
        len(
            endpoint_rows
        ),

    "endpoint_count":
        len(
            ENDPOINT_NAMES
        ),

    "protocol_sha256":
        protocol_sha256,

    "summary_sha256":
        sha256_file(
            SUMMARY_PATH
        ),

    "files": [
        {
            "relative_path":
                str(
                    path.relative_to(
                        ROOT
                    )
                ),

            "size_bytes":
                path.stat().st_size,

            "sha256":
                sha256_file(path),
        }
        for path in [
            ENDPOINT_TABLE_PATH,
            ENDPOINT_NAMES_PATH,
            SUMMARY_PATH,
        ]
    ],

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
    "dataset_snapshot_sha256:",
    EXPECTED_DATASET_SNAPSHOT_SHA256,
)
print(
    "unlabeled_feature_snapshot_sha256:",
    EXPECTED_FEATURE_SNAPSHOT_SHA256,
)
print(
    "unlabeled_endpoint_snapshot_sha256:",
    unlabeled_endpoint_snapshot_sha256,
)
print("sequence_count:", len(endpoint_rows))
print("endpoint_count:", len(ENDPOINT_NAMES))
print(
    "primary_endpoint:",
    ENDPOINT_NAMES[0],
)
print(
    "finite_endpoint_value_count:",
    len(endpoint_rows)
    * len(ENDPOINT_NAMES),
)
print(
    "expected_endpoint_value_count:",
    160
    * len(ENDPOINT_NAMES),
)
print("labels_joined: False")
print("label_manifest_opened: False")
print("activity_numbers_parsed: False")
print("models_trained: False")
print("performance_computed: False")
print("statistical_tests_computed: False")
print("thresholds_selected: False")
print("endpoint_values_printed: False")
print("endpoint_table:", ENDPOINT_TABLE_PATH)
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)
print(
    "\nKULEUVEN_UNLABELED_"
    "ENDPOINTS_V1_PASS"
)
