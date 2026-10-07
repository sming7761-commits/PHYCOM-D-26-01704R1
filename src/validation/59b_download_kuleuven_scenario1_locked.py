import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "kuleuven_scenario1_external_validation_protocol_v1.json"
)

FILE_MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "kuleuven_scenario1_external_validation_protocol_v1"
    / "scenario1_file_manifest.csv"
)

PILOT_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_scenario1_pilot_decode_v1"
    / "downloads"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_scenario1_locked_download_v1"
)

DOWNLOAD_ROOT = OUTPUT_ROOT / "downloads"
QUARANTINE_ROOT = OUTPUT_ROOT / "quarantine"

PROGRESS_PATH = OUTPUT_ROOT / "progress.json"
FILE_RECORDS_PATH = OUTPUT_ROOT / "download_file_records.csv"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_scenario1_locked_download_v1.json"
)

EXPECTED_FILE_COUNT = 160
EXPECTED_FILE_SIZE = 12_800_000
EXPECTED_TOTAL_SIZE = 2_048_000_000

MINIMUM_FREE_BYTES = 3_000_000_000


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def download_one(file_id, output_path):
    url = (
        "https://rdr.kuleuven.be/api/access/"
        f"datafile/{file_id}?format=original"
    )

    temporary = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    if temporary.exists():
        temporary.unlink()

    command = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "6",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--max-time",
        "1800",
        "--max-filesize",
        str(20 * 1024 * 1024),
        "--output",
        str(temporary),
        url,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Download failed for file_id={file_id}: "
            f"{result.stderr[-2000:]}"
        )

    if not temporary.exists():
        raise RuntimeError(
            f"Temporary download missing: {temporary}"
        )

    size_bytes = temporary.stat().st_size

    if size_bytes != EXPECTED_FILE_SIZE:
        raise RuntimeError(
            f"Downloaded size mismatch for {output_path.name}: "
            f"{size_bytes} != {EXPECTED_FILE_SIZE}"
        )

    os.replace(
        temporary,
        output_path,
    )

    return url


for path in [
    PROTOCOL_PATH,
    FILE_MANIFEST_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(path)


protocol = json.loads(
    PROTOCOL_PATH.read_text(
        encoding="utf-8"
    )
)

if (
    protocol.get("status")
    != "LOCKED_BEFORE_FULL_SCENARIO1_DOWNLOAD"
):
    raise RuntimeError(
        "Scenario 1 protocol is not in the expected locked state."
    )


with FILE_MANIFEST_PATH.open(
    "r",
    encoding="utf-8",
    newline="",
) as stream:
    rows = list(
        csv.DictReader(stream)
    )


if len(rows) != EXPECTED_FILE_COUNT:
    raise RuntimeError(
        f"Expected {EXPECTED_FILE_COUNT} locked files, "
        f"found {len(rows)}."
    )


if sum(
    int(row["size_bytes"])
    for row in rows
) != EXPECTED_TOTAL_SIZE:
    raise RuntimeError(
        "Locked manifest total size mismatch."
    )


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

DOWNLOAD_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

QUARANTINE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


free_bytes = shutil.disk_usage(
    ROOT
).free

already_present_bytes = sum(
    path.stat().st_size
    for path in DOWNLOAD_ROOT.glob("*.bin")
    if path.stat().st_size == EXPECTED_FILE_SIZE
)

remaining_bytes = (
    EXPECTED_TOTAL_SIZE
    - already_present_bytes
)

required_free_bytes = min(
    MINIMUM_FREE_BYTES,
    remaining_bytes + 500_000_000,
)

if free_bytes < required_free_bytes:
    raise RuntimeError(
        f"Insufficient disk space: free={free_bytes}, "
        f"required={required_free_bytes}"
    )


records = []


for index, row in enumerate(
    rows,
    start=1,
):
    filename = row["filename"]
    file_id = int(row["file_id"])

    target = DOWNLOAD_ROOT / filename
    source = None
    url = None

    if target.exists():
        if target.stat().st_size == EXPECTED_FILE_SIZE:
            source = "REUSED_COMPLETE_DOWNLOAD"
        else:
            quarantine = (
                QUARANTINE_ROOT
                / (
                    filename
                    + f".invalid_size_{target.stat().st_size}"
                )
            )

            os.replace(
                target,
                quarantine,
            )

    if source is None:
        pilot = PILOT_ROOT / filename

        if (
            pilot.exists()
            and pilot.stat().st_size
            == EXPECTED_FILE_SIZE
        ):
            shutil.copy2(
                pilot,
                target,
            )

            source = "COPIED_FROM_STAGE58I_PILOT"

        else:
            url = download_one(
                file_id,
                target,
            )

            source = "DOWNLOADED_FROM_REPOSITORY"

    if target.stat().st_size != EXPECTED_FILE_SIZE:
        raise RuntimeError(
            f"Final size mismatch: {filename}"
        )

    record = {
        "sequence_id":
            row["sequence_id"],

        "filename":
            filename,

        "file_id":
            file_id,

        "activity":
            int(row["activity"]),

        "repetition":
            int(row["repetition"]),

        "sequence_label":
            row["sequence_label"],

        "sequence_label_id":
            int(row["sequence_label_id"]),

        "size_bytes":
            target.stat().st_size,

        "sha256":
            sha256_file(target),

        "acquisition_source":
            source,

        "download_url":
            url,
    }

    records.append(
        record
    )

    completed_bytes = sum(
        item["size_bytes"]
        for item in records
    )

    atomic_json(
        PROGRESS_PATH,
        {
            "status":
                "RUNNING",

            "updated_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "completed_file_count":
                len(records),

            "expected_file_count":
                EXPECTED_FILE_COUNT,

            "completed_size_bytes":
                completed_bytes,

            "expected_total_size_bytes":
                EXPECTED_TOTAL_SIZE,

            "last_completed_filename":
                filename,
        },
    )

    print(
        f"[{index:03d}/{EXPECTED_FILE_COUNT}]",
        filename,
        "|",
        source,
        "| sha256=",
        record["sha256"],
        flush=True,
    )


if len(records) != EXPECTED_FILE_COUNT:
    raise RuntimeError(
        "Final downloaded file count mismatch."
    )


downloaded_total_size = sum(
    record["size_bytes"]
    for record in records
)

if downloaded_total_size != EXPECTED_TOTAL_SIZE:
    raise RuntimeError(
        "Final downloaded byte count mismatch."
    )


with FILE_RECORDS_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            records[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(records)


source_counts = {}

for record in records:
    key = record["acquisition_source"]

    source_counts[key] = (
        source_counts.get(key, 0)
        + 1
    )


summary = {
    "schema":
        "phyguard.kuleuven."
        "scenario1_locked_download.v1",

    "status":
        "PASS",

    "protocol_path":
        str(PROTOCOL_PATH),

    "protocol_sha256":
        sha256_file(
            PROTOCOL_PATH
        ),

    "locked_file_manifest_path":
        str(FILE_MANIFEST_PATH),

    "locked_file_manifest_sha256":
        sha256_file(
            FILE_MANIFEST_PATH
        ),

    "downloaded_file_count":
        len(records),

    "downloaded_total_size_bytes":
        downloaded_total_size,

    "positive_sequence_count":
        sum(
            record["sequence_label_id"] == 1
            for record in records
        ),

    "negative_sequence_count":
        sum(
            record["sequence_label_id"] == 0
            for record in records
        ),

    "acquisition_source_counts":
        source_counts,

    "methodological_boundary": {
        "models_trained":
            False,

        "model_predictions_computed":
            False,

        "classification_performance_computed":
            False,

        "thresholds_selected":
            False,

        "event_intervals_inferred":
            False,

        "files_selected_by_performance":
            False,

        "existing_sionna_assets_modified":
            False,

        "existing_locked_models_modified":
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
        "scenario1_locked_download_manifest.v1",

    "status":
        "PASS",

    "file_count":
        len(records),

    "total_size_bytes":
        downloaded_total_size,

    "protocol_sha256":
        summary["protocol_sha256"],

    "file_manifest_sha256":
        summary[
            "locked_file_manifest_sha256"
        ],

    "download_records_sha256":
        sha256_file(
            FILE_RECORDS_PATH
        ),

    "summary_sha256":
        sha256_file(
            SUMMARY_PATH
        ),

    "files":
        records,

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


atomic_json(
    PROGRESS_PATH,
    {
        "status":
            "PASS",

        "updated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "completed_file_count":
            len(records),

        "expected_file_count":
            EXPECTED_FILE_COUNT,

        "completed_size_bytes":
            downloaded_total_size,

        "expected_total_size_bytes":
            EXPECTED_TOTAL_SIZE,
    },
)


print()
print("status: PASS")
print(
    "downloaded_file_count:",
    len(records),
)
print(
    "downloaded_total_size_bytes:",
    downloaded_total_size,
)
print(
    "positive_sequence_count:",
    summary["positive_sequence_count"],
)
print(
    "negative_sequence_count:",
    summary["negative_sequence_count"],
)
print(
    "acquisition_source_counts:",
    source_counts,
)
print("models_trained: False")
print("performance_computed: False")
print("thresholds_selected: False")
print("event_intervals_inferred: False")
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)
print(
    "\nKULEUVEN_SCENARIO1_"
    "LOCKED_DOWNLOAD_V1_PASS"
)
