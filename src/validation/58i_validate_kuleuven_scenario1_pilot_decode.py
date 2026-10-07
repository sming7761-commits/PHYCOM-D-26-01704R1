import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/root/phyguard_revision")

INVENTORY_PATH = (
    ROOT
    / "artifacts"
    / "public_real_measurement_package_structure_audit_v1"
    / "kuleuven_file_inventory.csv"
)

CONVERTER_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_official_converter_audit_v1"
)

PY_CONVERTER_PATH = (
    CONVERTER_ROOT
    / "binary_to_csv_converter.py"
)

C_CONVERTER_PATH = (
    CONVERTER_ROOT
    / "binary2npy.c"
)

CONVERTER_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_official_converter_audit_v1.json"
)

PLAN_PATH = (
    ROOT
    / "configs"
    / "kuleuven_scenario1_pilot_decode_plan_v1.json"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_scenario1_pilot_decode_v1"
)

DOWNLOAD_ROOT = OUTPUT_ROOT / "downloads"

DECODE_RECORDS_PATH = (
    OUTPUT_ROOT
    / "decode_probe_records.json"
)

SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_scenario1_pilot_decode_v1.json"
)


EXPECTED_PY_SHA256 = (
    "e0a177d6f4ab09155bdf2ea4895856df"
    "d9beed96c96e169829a61405dda7f9a5"
)

EXPECTED_C_SHA256 = (
    "c164a1a6ea6a2a19492225e894c2a259"
    "9485dcf9bd10623c0681dac661b7065f"
)

EXPECTED_FILE_SIZE = 12_800_000

EXPECTED_COMPLEX_SAMPLE_COUNT = 3_200_000

RX_BEAM_COUNT = 8
SUBCARRIER_COUNT = 100
LAYER_COUNT = 2
SYMBOL_COUNT = 2000

SCALE = 16384.0


PILOT_FILES = [
    {
        "filename":
            "comm_sym2000_int10_10s_"
            "scen1_activity01_rep1.bin",

        "scenario":
            1,

        "activity":
            1,

        "repetition":
            1,

        "author_activity_description":
            (
                "Walk straight crossing the link "
                "(from c to f) at slow speed."
            ),

        "pilot_role":
            "contains_author_defined_link_crossing",

        "sequence_level_interpretation":
            (
                "The 10-second sequence contains a "
                "physical link-crossing blockage event; "
                "the entire sequence is not assumed to "
                "be continuously blocked."
            ),
    },
    {
        "filename":
            "comm_sym2000_int10_10s_"
            "scen1_activity03_rep1.bin",

        "scenario":
            1,

        "activity":
            3,

        "repetition":
            1,

        "author_activity_description":
            (
                "Walk from c to m and U-turn "
                "before blocking at slow speed."
            ),

        "pilot_role":
            "author_defined_nonblocking_trajectory",

        "sequence_level_interpretation":
            (
                "The trajectory is explicitly designed "
                "to turn before blocking the link."
            ),
    },
]


PROBE_COORDINATES = [
    {
        "symbol": 0,
        "subcarrier": 0,
        "layer": 0,
        "rx_beam": 0,
    },
    {
        "symbol": 0,
        "subcarrier": 0,
        "layer": 1,
        "rx_beam": 7,
    },
    {
        "symbol": 0,
        "subcarrier": 99,
        "layer": 0,
        "rx_beam": 4,
    },
    {
        "symbol": 1,
        "subcarrier": 0,
        "layer": 0,
        "rx_beam": 0,
    },
    {
        "symbol": 999,
        "subcarrier": 49,
        "layer": 1,
        "rx_beam": 3,
    },
    {
        "symbol": 1999,
        "subcarrier": 99,
        "layer": 1,
        "rx_beam": 7,
    },
]


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
        path.read_text(
            encoding="utf-8"
        )
    )


def flat_index(
    symbol,
    subcarrier,
    layer,
    rx_beam,
):
    if not 0 <= symbol < SYMBOL_COUNT:
        raise ValueError("Invalid symbol index.")

    if not 0 <= subcarrier < SUBCARRIER_COUNT:
        raise ValueError("Invalid subcarrier index.")

    if not 0 <= layer < LAYER_COUNT:
        raise ValueError("Invalid layer index.")

    if not 0 <= rx_beam < RX_BEAM_COUNT:
        raise ValueError("Invalid Rx-beam index.")

    antenna_group = rx_beam // 4
    antenna_offset = rx_beam % 4

    per_symbol = (
        LAYER_COUNT
        * RX_BEAM_COUNT
        * SUBCARRIER_COUNT
    )

    return (
        symbol * per_symbol
        + subcarrier
        * LAYER_COUNT
        * RX_BEAM_COUNT
        + antenna_group
        * 4
        * LAYER_COUNT
        + layer * 4
        + antenna_offset
    )


def download_file(file_id, output_path):
    url = (
        "https://rdr.kuleuven.be/api/access/"
        f"datafile/{file_id}?format=original"
    )

    command = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "4",
        "--retry-delay",
        "3",
        "--retry-all-errors",
        "--connect-timeout",
        "20",
        "--max-time",
        "600",
        "--max-filesize",
        str(20 * 1024 * 1024),
        "--output",
        str(output_path),
        url,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    return {
        "url":
            url,

        "returncode":
            int(result.returncode),

        "stderr":
            result.stderr[-4000:],
    }


required_paths = [
    INVENTORY_PATH,
    PY_CONVERTER_PATH,
    C_CONVERTER_PATH,
    CONVERTER_MANIFEST_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)


converter_manifest = load_json(
    CONVERTER_MANIFEST_PATH
)

if converter_manifest.get("status") != "PASS":
    raise RuntimeError(
        "Official converter audit is not PASS."
    )


if sha256_file(
    PY_CONVERTER_PATH
) != EXPECTED_PY_SHA256:
    raise RuntimeError(
        "Python converter SHA256 mismatch."
    )


if sha256_file(
    C_CONVERTER_PATH
) != EXPECTED_C_SHA256:
    raise RuntimeError(
        "C converter SHA256 mismatch."
    )


if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Pilot decode output already exists; "
        "refusing to overwrite it."
    )


if PLAN_PATH.exists():
    raise RuntimeError(
        "Pilot decode plan already exists; "
        "refusing to overwrite it."
    )


with INVENTORY_PATH.open(
    "r",
    encoding="utf-8",
    newline="",
) as stream:
    inventory_rows = list(
        csv.DictReader(stream)
    )


inventory_by_filename = {}

for row in inventory_rows:
    filename = str(
        row.get("filename")
        or ""
    )

    if filename:
        inventory_by_filename.setdefault(
            filename,
            [],
        ).append(row)


resolved_files = []


for specification in PILOT_FILES:
    matches = inventory_by_filename.get(
        specification["filename"],
        [],
    )

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one inventory match for "
            f"{specification['filename']}; "
            f"found {len(matches)}."
        )

    row = matches[0]

    if str(
        row.get("restricted")
    ).lower() not in {
        "false",
        "0",
        "",
        "none",
    }:
        raise RuntimeError(
            "Pilot file is marked restricted: "
            f"{specification['filename']}"
        )

    size_bytes = int(
        row.get("size_bytes")
        or 0
    )

    if size_bytes != EXPECTED_FILE_SIZE:
        raise RuntimeError(
            "Unexpected repository size for "
            f"{specification['filename']}: "
            f"{size_bytes}"
        )

    resolved_files.append(
        {
            **specification,

            "file_id":
                int(
                    row["file_id"]
                ),

            "repository_size_bytes":
                size_bytes,

            "directory_label":
                row.get(
                    "directory_label"
                ),
        }
    )


plan = {
    "schema":
        "phyguard.kuleuven."
        "scenario1_pilot_decode_plan.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "LOCKED_BEFORE_MEASUREMENT_DOWNLOAD",

    "scope":
        (
            "Binary-access and deterministic-decoder "
            "validation only."
        ),

    "official_converter_contract": {
        "python_converter_path":
            str(
                PY_CONVERTER_PATH
            ),

        "python_converter_sha256":
            EXPECTED_PY_SHA256,

        "c_converter_path":
            str(
                C_CONVERTER_PATH
            ),

        "c_converter_sha256":
            EXPECTED_C_SHA256,

        "sample_storage":
            (
                "4 bytes per complex sample: "
                "big-endian int16 imaginary followed "
                "by big-endian int16 real."
            ),

        "scaling":
            "real_and_imaginary_divided_by_16384",

        "amplitude":
            "hypot(real, imaginary)",

        "phase":
            "atan2(imaginary, real)",
    },

    "expected_dimensions": {
        "symbols":
            SYMBOL_COUNT,

        "subcarriers":
            SUBCARRIER_COUNT,

        "layers":
            LAYER_COUNT,

        "rx_beams":
            RX_BEAM_COUNT,

        "complex_sample_count":
            EXPECTED_COMPLEX_SAMPLE_COUNT,

        "file_size_bytes":
            EXPECTED_FILE_SIZE,
    },

    "pilot_files":
        resolved_files,

    "probe_coordinates":
        PROBE_COORDINATES,

    "methodological_boundary": {
        "model_training":
            False,

        "model_prediction":
            False,

        "classification_performance":
            False,

        "threshold_selection":
            False,

        "event_interval_inference":
            False,

        "cross_file_value_comparison":
            False,

        "full_scenario1_download":
            False,

        "example_25_6_mb_file_used":
            False,

        "existing_sionna_assets_modified":
            False,

        "existing_locked_models_modified":
            False,
    },
}


PLAN_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

PLAN_PATH.write_text(
    json.dumps(
        plan,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

plan_sha256 = sha256_file(
    PLAN_PATH
)


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

DOWNLOAD_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)


decode_records = []

download_records = []


for record in resolved_files:
    output_path = (
        DOWNLOAD_ROOT
        / record["filename"]
    )

    download_result = download_file(
        record["file_id"],
        output_path,
    )

    if download_result["returncode"] != 0:
        raise RuntimeError(
            "Download failed for "
            f"{record['filename']}: "
            f"{download_result['stderr']}"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"Downloaded file is missing: {output_path}"
        )

    actual_size = output_path.stat().st_size

    if actual_size != EXPECTED_FILE_SIZE:
        raise RuntimeError(
            "Downloaded file size mismatch for "
            f"{record['filename']}: "
            f"{actual_size} != {EXPECTED_FILE_SIZE}"
        )

    raw = np.memmap(
        output_path,
        dtype=">i2",
        mode="r",
    )

    expected_int16_count = (
        EXPECTED_COMPLEX_SAMPLE_COUNT
        * 2
    )

    if raw.size != expected_int16_count:
        raise RuntimeError(
            "Raw int16 count mismatch for "
            f"{record['filename']}: "
            f"{raw.size} != {expected_int16_count}"
        )

    pairs = raw.reshape(
        EXPECTED_COMPLEX_SAMPLE_COUNT,
        2,
    )

    file_probe_records = []

    for coordinate in PROBE_COORDINATES:
        index = flat_index(
            symbol=coordinate["symbol"],
            subcarrier=coordinate[
                "subcarrier"
            ],
            layer=coordinate["layer"],
            rx_beam=coordinate[
                "rx_beam"
            ],
        )

        imaginary_int16 = int(
            pairs[index, 0]
        )

        real_int16 = int(
            pairs[index, 1]
        )

        real_value = (
            real_int16
            / SCALE
        )

        imaginary_value = (
            imaginary_int16
            / SCALE
        )

        amplitude = math.hypot(
            real_value,
            imaginary_value,
        )

        phase = math.atan2(
            imaginary_value,
            real_value,
        )

        if not all(
            math.isfinite(value)
            for value in [
                real_value,
                imaginary_value,
                amplitude,
                phase,
            ]
        ):
            raise RuntimeError(
                "Non-finite decoded value."
            )

        file_probe_records.append(
            {
                **coordinate,

                "flat_index":
                    index,

                "stored_imaginary_int16":
                    imaginary_int16,

                "stored_real_int16":
                    real_int16,

                "imaginary":
                    imaginary_value,

                "real":
                    real_value,

                "amplitude":
                    amplitude,

                "phase_radians":
                    phase,
            }
        )

    decode_records.append(
        {
            "filename":
                record["filename"],

            "file_id":
                record["file_id"],

            "pilot_role":
                record["pilot_role"],

            "repository_size_bytes":
                record[
                    "repository_size_bytes"
                ],

            "downloaded_size_bytes":
                actual_size,

            "sha256":
                sha256_file(
                    output_path
                ),

            "raw_int16_count":
                int(
                    raw.size
                ),

            "complex_sample_count":
                EXPECTED_COMPLEX_SAMPLE_COUNT,

            "probe_count":
                len(
                    file_probe_records
                ),

            "probes":
                file_probe_records,
        }
    )

    download_records.append(
        {
            "filename":
                record["filename"],

            "file_id":
                record["file_id"],

            "url":
                download_result["url"],

            "size_bytes":
                actual_size,

            "sha256":
                sha256_file(
                    output_path
                ),
        }
    )

    del pairs
    del raw


DECODE_RECORDS_PATH.write_text(
    json.dumps(
        decode_records,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


summary = {
    "schema":
        "phyguard.kuleuven."
        "scenario1_pilot_decode.v1",

    "status":
        "PASS",

    "plan_path":
        str(PLAN_PATH),

    "plan_sha256":
        plan_sha256,

    "downloaded_file_count":
        len(
            download_records
        ),

    "downloaded_total_size_bytes":
        int(
            sum(
                record["size_bytes"]
                for record in download_records
            )
        ),

    "files":
        download_records,

    "decoder_contract": {
        "storage_dtype":
            "big_endian_int16_pairs",

        "pair_order":
            "imaginary_then_real",

        "scale":
            SCALE,

        "complex_sample_count_per_file":
            EXPECTED_COMPLEX_SAMPLE_COUNT,

        "probe_replay_count":
            int(
                sum(
                    record["probe_count"]
                    for record in decode_records
                )
            ),

        "probe_replay_failure_count":
            0,
    },

    "methodological_boundary":
        plan[
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


manifest_files = [
    PLAN_PATH,
    DECODE_RECORDS_PATH,
    SUMMARY_PATH,
]

manifest_files.extend(
    sorted(
        DOWNLOAD_ROOT.glob("*.bin")
    )
)


manifest = {
    "schema":
        "phyguard.kuleuven."
        "scenario1_pilot_decode_manifest.v1",

    "status":
        "PASS",

    "plan_sha256":
        plan_sha256,

    "downloaded_file_count":
        len(
            download_records
        ),

    "downloaded_total_size_bytes":
        summary[
            "downloaded_total_size_bytes"
        ],

    "decoder_probe_replay_count":
        summary[
            "decoder_contract"
        ][
            "probe_replay_count"
        ],

    "decoder_probe_failure_count":
        0,

    "model_training":
        False,

    "model_prediction":
        False,

    "classification_performance":
        False,

    "threshold_selection":
        False,

    "cross_file_value_comparison":
        False,

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
        for path in manifest_files
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
    "plan_sha256:",
    plan_sha256,
)
print(
    "downloaded_file_count:",
    len(
        download_records
    ),
)
print(
    "downloaded_total_size_bytes:",
    summary[
        "downloaded_total_size_bytes"
    ],
)
print(
    "complex_sample_count_per_file:",
    EXPECTED_COMPLEX_SAMPLE_COUNT,
)
print(
    "decoder_probe_replay_count:",
    manifest[
        "decoder_probe_replay_count"
    ],
)
print(
    "decoder_probe_failure_count:",
    0,
)
print("model_training: False")
print("model_prediction: False")
print("classification_performance: False")
print("threshold_selection: False")
print("cross_file_value_comparison: False")
print("full_scenario1_download: False")


print("\nPILOT FILES")

for record in decode_records:
    print(
        record["filename"],
        "| role=",
        record["pilot_role"],
        "| bytes=",
        record["downloaded_size_bytes"],
        "| samples=",
        record["complex_sample_count"],
        "| sha256=",
        record["sha256"],
    )


print("\noutput_root:", OUTPUT_ROOT)
print("plan:", PLAN_PATH)
print("decode_records:", DECODE_RECORDS_PATH)
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)

print(
    "\nKULEUVEN_SCENARIO1_"
    "PILOT_DECODE_PASS"
)
