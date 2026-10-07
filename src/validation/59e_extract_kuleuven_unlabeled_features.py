import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/root/phyguard_revision")

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "kuleuven_signal_analysis_protocol_v1.json"
)

PROTOCOL_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_signal_analysis_protocol_v1.json"
)

DOWNLOAD_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_scenario1_locked_download_v1"
    / "downloads"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_unlabeled_features_v1"
)

FEATURE_TENSOR_PATH = (
    OUTPUT_ROOT
    / "features_160x80x10.npy"
)

FEATURE_LONG_PATH = (
    OUTPUT_ROOT
    / "features_long.csv"
)

LINEAGE_PATH = (
    OUTPUT_ROOT
    / "unlabeled_sequence_lineage.csv"
)

QC_PATH = (
    OUTPUT_ROOT
    / "feature_qc.csv"
)

FEATURE_NAMES_PATH = (
    OUTPUT_ROOT
    / "feature_names.json"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_unlabeled_features_v1.json"
)

EXPECTED_DATASET_SNAPSHOT_SHA256 = (
    "0134ffb74c227ff41e4c10052045308d5"
    "b8cb61414f3ba54324c09815e6f0734"
)

EXPECTED_FILE_COUNT = 160
EXPECTED_FILE_SIZE = 12_800_000
EXPECTED_COMPLEX_SAMPLES = 3_200_000

SYMBOL_COUNT = 2000
SUBCARRIER_COUNT = 100
TX_LAYER_COUNT = 2
RX_BEAM_COUNT = 8

TIME_STEP_COUNT = 80
SYMBOLS_PER_TIME_STEP = 25
TIME_STEP_DURATION_MS = 125.0

SCALE = 16384.0
EPS = 1e-12

FEATURE_NAMES = [
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
        path.read_text(encoding="utf-8")
    )


def power_to_db(value):
    return 10.0 * np.log10(
        np.maximum(value, EPS)
    )


def sequence_uid(raw_sha256):
    digest = hashlib.sha256(
        (
            EXPECTED_DATASET_SNAPSHOT_SHA256
            + "|"
            + raw_sha256
        ).encode("utf-8")
    ).hexdigest()

    return "KU26G_" + digest[:20]


for required_path in [
    PROTOCOL_PATH,
    PROTOCOL_MANIFEST_PATH,
    DOWNLOAD_ROOT,
]:
    if not required_path.exists():
        raise FileNotFoundError(
            required_path
        )


protocol = load_json(
    PROTOCOL_PATH
)

protocol_manifest = load_json(
    PROTOCOL_MANIFEST_PATH
)


if (
    protocol.get("status")
    != "LOCKED_BEFORE_SIGNAL_VALUE_ANALYSIS"
):
    raise RuntimeError(
        "Stage 59D protocol is not in the expected locked state."
    )


if protocol_manifest.get("status") != "PASS":
    raise RuntimeError(
        "Stage 59D manifest is not PASS."
    )


if (
    protocol.get("dataset_snapshot_sha256")
    != EXPECTED_DATASET_SNAPSHOT_SHA256
):
    raise RuntimeError(
        "Protocol dataset snapshot mismatch."
    )


if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Stage 59E output already exists; refusing to overwrite."
    )


raw_files = sorted(
    DOWNLOAD_ROOT.glob("*.bin"),
    key=lambda path: path.name,
)


if len(raw_files) != EXPECTED_FILE_COUNT:
    raise RuntimeError(
        f"Expected {EXPECTED_FILE_COUNT} BIN files, "
        f"found {len(raw_files)}."
    )


for path in raw_files:
    if path.stat().st_size != EXPECTED_FILE_SIZE:
        raise RuntimeError(
            f"Unexpected file size: {path}"
        )


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)


feature_tensor = np.empty(
    (
        EXPECTED_FILE_COUNT,
        TIME_STEP_COUNT,
        len(FEATURE_NAMES),
    ),
    dtype=np.float64,
)


lineage_rows = []
qc_rows = []
snapshot_lines = []
uids = set()


for sequence_index, path in enumerate(
    raw_files
):
    raw = np.memmap(
        path,
        dtype=">i2",
        mode="r",
    )

    expected_int16_count = (
        EXPECTED_COMPLEX_SAMPLES * 2
    )

    if raw.size != expected_int16_count:
        raise RuntimeError(
            f"Decoded int16 count mismatch for {path.name}: "
            f"{raw.size} != {expected_int16_count}"
        )

    raw_view = raw.reshape(
        SYMBOL_COUNT,
        SUBCARRIER_COUNT,
        2,
        TX_LAYER_COUNT,
        4,
        2,
    )

    file_features = np.empty(
        (
            TIME_STEP_COUNT,
            len(FEATURE_NAMES),
        ),
        dtype=np.float64,
    )

    raw_digest = hashlib.sha256()


    for time_index in range(
        TIME_STEP_COUNT
    ):
        symbol_start = (
            time_index
            * SYMBOLS_PER_TIME_STEP
        )

        symbol_end = (
            symbol_start
            + SYMBOLS_PER_TIME_STEP
        )

        stored_block = raw_view[
            symbol_start:symbol_end
        ]

        raw_digest.update(
            stored_block.tobytes(
                order="C"
            )
        )

        ordered = np.transpose(
            stored_block,
            (
                0,
                1,
                3,
                2,
                4,
                5,
            ),
        ).reshape(
            SYMBOLS_PER_TIME_STEP,
            SUBCARRIER_COUNT,
            TX_LAYER_COUNT,
            RX_BEAM_COUNT,
            2,
        )

        imaginary = (
            ordered[..., 0]
            .astype(
                np.float64,
                copy=False,
            )
            / SCALE
        )

        real = (
            ordered[..., 1]
            .astype(
                np.float64,
                copy=False,
            )
            / SCALE
        )

        power = (
            real * real
            + imaginary * imaginary
        )

        primary_power = power[
            :,
            :,
            0,
            :,
        ]

        secondary_power = power[
            :,
            :,
            1,
            :,
        ]

        primary_rx_power = (
            primary_power.mean(
                axis=(0, 1)
            )
        )

        primary_total_power = float(
            primary_power.mean()
        )

        secondary_total_power = float(
            secondary_power.mean()
        )

        best_rx_power = float(
            np.max(
                primary_rx_power
            )
        )

        sorted_rx_power = np.sort(
            primary_rx_power
        )

        strongest_power = float(
            sorted_rx_power[-1]
        )

        second_power = float(
            sorted_rx_power[-2]
        )

        power_sum = float(
            primary_rx_power.sum()
        )

        if power_sum > EPS:
            shares = (
                primary_rx_power
                / power_sum
            )

            positive_shares = shares[
                shares > 0
            ]

            entropy = float(
                -np.sum(
                    positive_shares
                    * np.log(
                        positive_shares
                    )
                )
                / math.log(
                    RX_BEAM_COUNT
                )
            )
        else:
            entropy = 0.0

        subcarrier_power = (
            primary_power.mean(
                axis=(0, 2)
            )
        )

        frequency_selectivity = float(
            np.std(
                power_to_db(
                    subcarrier_power
                ),
                ddof=0,
            )
        )

        primary_complex = (
            real[
                :,
                :,
                0,
                :,
            ]
            + 1j
            * imaginary[
                :,
                :,
                0,
                :,
            ]
        )

        previous = primary_complex[
            :-1
        ].reshape(
            SYMBOLS_PER_TIME_STEP - 1,
            -1,
        )

        following = primary_complex[
            1:
        ].reshape(
            SYMBOLS_PER_TIME_STEP - 1,
            -1,
        )

        numerator = np.abs(
            np.sum(
                previous
                * np.conjugate(
                    following
                ),
                axis=1,
            )
        )

        denominator = np.sqrt(
            np.sum(
                np.abs(
                    previous
                ) ** 2,
                axis=1,
            )
            * np.sum(
                np.abs(
                    following
                ) ** 2,
                axis=1,
            )
        )

        correlations = (
            numerator
            / np.maximum(
                denominator,
                EPS,
            )
        )

        temporal_coherence = float(
            np.mean(
                correlations
            )
        )

        per_symbol_rx_power = (
            primary_power.mean(
                axis=1
            )
        )

        best_beam_per_symbol = np.argmax(
            per_symbol_rx_power,
            axis=1,
        )

        beam_switch_rate = float(
            np.mean(
                best_beam_per_symbol[1:]
                != best_beam_per_symbol[:-1]
            )
        )

        per_symbol_best_power = np.max(
            per_symbol_rx_power,
            axis=1,
        )

        symbol_power_volatility = float(
            np.std(
                power_to_db(
                    per_symbol_best_power
                ),
                ddof=0,
            )
        )

        values = np.asarray(
            [
                float(
                    power_to_db(
                        best_rx_power
                    )
                ),
                float(
                    power_to_db(
                        primary_total_power
                    )
                ),
                float(
                    power_to_db(
                        secondary_total_power
                    )
                ),
                float(
                    power_to_db(
                        primary_total_power
                    )
                    - power_to_db(
                        secondary_total_power
                    )
                ),
                entropy,
                float(
                    power_to_db(
                        strongest_power
                    )
                    - power_to_db(
                        second_power
                    )
                ),
                frequency_selectivity,
                temporal_coherence,
                beam_switch_rate,
                symbol_power_volatility,
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(values)
        ):
            raise RuntimeError(
                f"Non-finite features in "
                f"{path.name}, time index {time_index}."
            )

        if not (
            -1e-10
            <= values[4]
            <= 1.0 + 1e-10
        ):
            raise RuntimeError(
                f"Entropy outside range in "
                f"{path.name}, time index {time_index}."
            )

        if not (
            -1e-10
            <= values[7]
            <= 1.0 + 1e-10
        ):
            raise RuntimeError(
                f"Coherence outside range in "
                f"{path.name}, time index {time_index}: "
                f"{values[7]}"
            )

        if not (
            0.0
            <= values[8]
            <= 1.0
        ):
            raise RuntimeError(
                f"Beam-switch rate outside range in "
                f"{path.name}, time index {time_index}."
            )

        file_features[
            time_index
        ] = values


    raw_sha256 = (
        raw_digest.hexdigest()
    )

    uid = sequence_uid(
        raw_sha256
    )

    if uid in uids:
        raise RuntimeError(
            f"Duplicate opaque sequence UID: {uid}"
        )

    uids.add(uid)

    feature_tensor[
        sequence_index
    ] = file_features

    lineage_rows.append(
        {
            "sequence_index":
                sequence_index,

            "sequence_uid":
                uid,

            "filename":
                path.name,

            "size_bytes":
                path.stat().st_size,

            "raw_sha256":
                raw_sha256,
        }
    )

    snapshot_lines.append(
        f"{path.name}|"
        f"{path.stat().st_size}|"
        f"{raw_sha256}"
    )

    qc_row = {
        "sequence_index":
            sequence_index,

        "sequence_uid":
            uid,

        "time_step_count":
            TIME_STEP_COUNT,

        "feature_count":
            len(
                FEATURE_NAMES
            ),

        "finite_value_count":
            int(
                np.isfinite(
                    file_features
                ).sum()
            ),

        "expected_value_count":
            int(
                file_features.size
            ),

        "qc_status":
            "PASS",
    }

    for feature_index, feature_name in enumerate(
        FEATURE_NAMES
    ):
        qc_row[
            feature_name + "_min"
        ] = float(
            np.min(
                file_features[
                    :,
                    feature_index,
                ]
            )
        )

        qc_row[
            feature_name + "_max"
        ] = float(
            np.max(
                file_features[
                    :,
                    feature_index,
                ]
            )
        )

    qc_rows.append(
        qc_row
    )

    del file_features
    del raw_view
    del raw

    print(
        f"[{sequence_index + 1:03d}/"
        f"{EXPECTED_FILE_COUNT}]",
        uid,
        "| shape=80x10",
        "| finite=800/800",
        "| raw_sha256=",
        raw_sha256,
        flush=True,
    )


rebuilt_snapshot_payload = (
    "\n".join(
        snapshot_lines
    )
    + "\n"
).encode("utf-8")

rebuilt_dataset_snapshot_sha256 = (
    hashlib.sha256(
        rebuilt_snapshot_payload
    ).hexdigest()
)


if (
    rebuilt_dataset_snapshot_sha256
    != EXPECTED_DATASET_SNAPSHOT_SHA256
):
    raise RuntimeError(
        "Rebuilt dataset snapshot mismatch: "
        f"{rebuilt_dataset_snapshot_sha256}"
    )


np.save(
    FEATURE_TENSOR_PATH,
    feature_tensor,
    allow_pickle=False,
)


FEATURE_NAMES_PATH.write_text(
    json.dumps(
        {
            "schema":
                "phyguard.kuleuven."
                "feature_names.v1",

            "feature_count":
                len(
                    FEATURE_NAMES
                ),

            "feature_names":
                FEATURE_NAMES,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


with LINEAGE_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            lineage_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        lineage_rows
    )


with QC_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            qc_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        qc_rows
    )


with FEATURE_LONG_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    fieldnames = [
        "sequence_uid",
        "time_index",
        "time_start_ms",
        "time_end_ms",
        *FEATURE_NAMES,
    ]

    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for sequence_index, lineage in enumerate(
        lineage_rows
    ):
        for time_index in range(
            TIME_STEP_COUNT
        ):
            row = {
                "sequence_uid":
                    lineage[
                        "sequence_uid"
                    ],

                "time_index":
                    time_index,

                "time_start_ms":
                    time_index
                    * TIME_STEP_DURATION_MS,

                "time_end_ms":
                    (
                        time_index + 1
                    )
                    * TIME_STEP_DURATION_MS,
            }

            for feature_index, feature_name in enumerate(
                FEATURE_NAMES
            ):
                row[
                    feature_name
                ] = format(
                    feature_tensor[
                        sequence_index,
                        time_index,
                        feature_index,
                    ],
                    ".12g",
                )

            writer.writerow(row)


feature_tensor_sha256 = sha256_file(
    FEATURE_TENSOR_PATH
)

feature_long_sha256 = sha256_file(
    FEATURE_LONG_PATH
)

lineage_sha256 = sha256_file(
    LINEAGE_PATH
)

qc_sha256 = sha256_file(
    QC_PATH
)

feature_names_sha256 = sha256_file(
    FEATURE_NAMES_PATH
)


unlabeled_feature_snapshot_sha256 = (
    hashlib.sha256(
        (
            EXPECTED_DATASET_SNAPSHOT_SHA256
            + "|"
            + sha256_file(
                PROTOCOL_PATH
            )
            + "|"
            + feature_tensor_sha256
            + "|"
            + feature_long_sha256
            + "|"
            + lineage_sha256
            + "|"
            + qc_sha256
            + "|"
            + feature_names_sha256
        ).encode("utf-8")
    ).hexdigest()
)


summary = {
    "schema":
        "phyguard.kuleuven."
        "unlabeled_features.v1",

    "status":
        "PASS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "dataset_snapshot_sha256":
        EXPECTED_DATASET_SNAPSHOT_SHA256,

    "rebuilt_dataset_snapshot_sha256":
        rebuilt_dataset_snapshot_sha256,

    "unlabeled_feature_snapshot_sha256":
        unlabeled_feature_snapshot_sha256,

    "sequence_count":
        EXPECTED_FILE_COUNT,

    "time_step_count":
        TIME_STEP_COUNT,

    "feature_count":
        len(
            FEATURE_NAMES
        ),

    "tensor_shape":
        list(
            feature_tensor.shape
        ),

    "tensor_dtype":
        str(
            feature_tensor.dtype
        ),

    "finite_value_count":
        int(
            np.isfinite(
                feature_tensor
            ).sum()
        ),

    "expected_value_count":
        int(
            feature_tensor.size
        ),

    "qc_pass_sequence_count":
        len(
            qc_rows
        ),

    "qc_failure_sequence_count":
        0,

    "files": {
        "feature_tensor_path":
            str(
                FEATURE_TENSOR_PATH
            ),

        "feature_tensor_sha256":
            feature_tensor_sha256,

        "feature_long_path":
            str(
                FEATURE_LONG_PATH
            ),

        "feature_long_sha256":
            feature_long_sha256,

        "lineage_path":
            str(
                LINEAGE_PATH
            ),

        "lineage_sha256":
            lineage_sha256,

        "qc_path":
            str(
                QC_PATH
            ),

        "qc_sha256":
            qc_sha256,

        "feature_names_path":
            str(
                FEATURE_NAMES_PATH
            ),

        "feature_names_sha256":
            feature_names_sha256,
    },

    "methodological_boundary": {
        "signal_values_read":
            True,

        "features_computed":
            True,

        "labels_joined":
            False,

        "label_manifest_opened":
            False,

        "activity_numbers_parsed":
            False,

        "sequence_endpoints_computed":
            False,

        "models_trained":
            False,

        "predictions_computed":
            False,

        "classification_performance_computed":
            False,

        "thresholds_selected":
            False,

        "feature_values_printed":
            False,

        "files_selected_by_signal_values":
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
        "unlabeled_features_manifest.v1",

    "status":
        "PASS",

    "dataset_snapshot_sha256":
        EXPECTED_DATASET_SNAPSHOT_SHA256,

    "unlabeled_feature_snapshot_sha256":
        unlabeled_feature_snapshot_sha256,

    "sequence_count":
        EXPECTED_FILE_COUNT,

    "tensor_shape":
        list(
            feature_tensor.shape
        ),

    "protocol_sha256":
        sha256_file(
            PROTOCOL_PATH
        ),

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
            FEATURE_TENSOR_PATH,
            FEATURE_LONG_PATH,
            LINEAGE_PATH,
            QC_PATH,
            FEATURE_NAMES_PATH,
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


print()
print("status: PASS")
print(
    "dataset_snapshot_sha256:",
    EXPECTED_DATASET_SNAPSHOT_SHA256,
)
print(
    "rebuilt_dataset_snapshot_sha256:",
    rebuilt_dataset_snapshot_sha256,
)
print(
    "unlabeled_feature_snapshot_sha256:",
    unlabeled_feature_snapshot_sha256,
)
print(
    "tensor_shape:",
    list(
        feature_tensor.shape
    ),
)
print(
    "tensor_dtype:",
    feature_tensor.dtype,
)
print(
    "finite_value_count:",
    int(
        np.isfinite(
            feature_tensor
        ).sum()
    ),
)
print(
    "expected_value_count:",
    int(
        feature_tensor.size
    ),
)
print("qc_pass_sequence_count: 160")
print("qc_failure_sequence_count: 0")
print("signal_values_read: True")
print("features_computed: True")
print("labels_joined: False")
print("label_manifest_opened: False")
print("sequence_endpoints_computed: False")
print("models_trained: False")
print("performance_computed: False")
print("thresholds_selected: False")
print("feature_values_printed: False")
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)
print(
    "\nKULEUVEN_UNLABELED_"
    "FEATURE_EXTRACTION_V1_PASS"
)
