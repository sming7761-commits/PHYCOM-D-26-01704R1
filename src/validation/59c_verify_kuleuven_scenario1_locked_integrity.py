import csv
import hashlib
import json
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

PROTOCOL_PATH = (
    ROOT / "configs"
    / "kuleuven_scenario1_external_validation_protocol_v1.json"
)

LOCKED_LIST_PATH = (
    ROOT / "artifacts"
    / "kuleuven_scenario1_external_validation_protocol_v1"
    / "scenario1_file_manifest.csv"
)

DOWNLOAD_ROOT = (
    ROOT / "artifacts"
    / "kuleuven_scenario1_locked_download_v1"
)

DOWNLOAD_DIR = DOWNLOAD_ROOT / "downloads"
DOWNLOAD_SUMMARY_PATH = DOWNLOAD_ROOT / "summary.json"
DOWNLOAD_RECORDS_PATH = DOWNLOAD_ROOT / "download_file_records.csv"

DOWNLOAD_MANIFEST_PATH = (
    ROOT / "manifests"
    / "kuleuven_scenario1_locked_download_v1.json"
)

OUTPUT_ROOT = (
    ROOT / "artifacts"
    / "kuleuven_scenario1_locked_integrity_v1"
)

INTEGRITY_CSV_PATH = OUTPUT_ROOT / "file_integrity.csv"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

MANIFEST_PATH = (
    ROOT / "manifests"
    / "kuleuven_scenario1_locked_integrity_v1.json"
)

EXPECTED_COUNT = 160
EXPECTED_SIZE = 12_800_000
EXPECTED_TOTAL = 2_048_000_000


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


def load_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


required = [
    PROTOCOL_PATH,
    LOCKED_LIST_PATH,
    DOWNLOAD_SUMMARY_PATH,
    DOWNLOAD_RECORDS_PATH,
    DOWNLOAD_MANIFEST_PATH,
    DOWNLOAD_DIR,
]

for path in required:
    if not path.exists():
        raise FileNotFoundError(path)


protocol = load_json(PROTOCOL_PATH)
download_summary = load_json(DOWNLOAD_SUMMARY_PATH)
download_manifest = load_json(DOWNLOAD_MANIFEST_PATH)

if protocol.get("status") != "LOCKED_BEFORE_FULL_SCENARIO1_DOWNLOAD":
    raise RuntimeError("Unexpected protocol status.")

if download_summary.get("status") != "PASS":
    raise RuntimeError("Stage 59B summary is not PASS.")

if download_manifest.get("status") != "PASS":
    raise RuntimeError("Stage 59B manifest is not PASS.")


locked_rows = load_csv(LOCKED_LIST_PATH)
download_rows = load_csv(DOWNLOAD_RECORDS_PATH)

if len(locked_rows) != EXPECTED_COUNT:
    raise RuntimeError(
        f"Locked list count mismatch: {len(locked_rows)}"
    )

if len(download_rows) != EXPECTED_COUNT:
    raise RuntimeError(
        f"Download-record count mismatch: {len(download_rows)}"
    )


locked_by_name = {
    row["filename"]: row
    for row in locked_rows
}

download_by_name = {
    row["filename"]: row
    for row in download_rows
}

manifest_by_name = {
    row["filename"]: row
    for row in download_manifest.get("files", [])
}


if set(locked_by_name) != set(download_by_name):
    raise RuntimeError(
        "Locked-list and download-record filename sets differ."
    )

if set(locked_by_name) != set(manifest_by_name):
    raise RuntimeError(
        "Locked-list and download-manifest filename sets differ."
    )


actual_files = sorted(
    path.name
    for path in DOWNLOAD_DIR.glob("*.bin")
)

if set(actual_files) != set(locked_by_name):
    raise RuntimeError(
        "Download directory contains missing or extra BIN files."
    )


if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Stage 59C output already exists; refusing to overwrite."
    )

OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)


integrity_rows = []
snapshot_lines = []


for index, filename in enumerate(
    sorted(locked_by_name),
    start=1,
):
    locked = locked_by_name[filename]
    downloaded = download_by_name[filename]
    manifested = manifest_by_name[filename]

    path = DOWNLOAD_DIR / filename

    size_bytes = path.stat().st_size

    if size_bytes != EXPECTED_SIZE:
        raise RuntimeError(
            f"Size mismatch for {filename}: {size_bytes}"
        )

    actual_sha256 = sha256_file(path)

    recorded_sha256 = downloaded["sha256"]
    manifested_sha256 = manifested["sha256"]

    if actual_sha256 != recorded_sha256:
        raise RuntimeError(
            f"Download-record SHA256 mismatch: {filename}"
        )

    if actual_sha256 != manifested_sha256:
        raise RuntimeError(
            f"Download-manifest SHA256 mismatch: {filename}"
        )

    if int(locked["file_id"]) != int(downloaded["file_id"]):
        raise RuntimeError(
            f"File-ID mismatch: {filename}"
        )

    if (
        int(locked["sequence_label_id"])
        != int(downloaded["sequence_label_id"])
    ):
        raise RuntimeError(
            f"Label mismatch: {filename}"
        )

    integrity_rows.append(
        {
            "sequence_id": locked["sequence_id"],
            "filename": filename,
            "file_id": int(locked["file_id"]),
            "activity": int(locked["activity"]),
            "repetition": int(locked["repetition"]),
            "sequence_label": locked["sequence_label"],
            "sequence_label_id": int(
                locked["sequence_label_id"]
            ),
            "size_bytes": size_bytes,
            "sha256": actual_sha256,
            "integrity_status": "PASS",
        }
    )

    snapshot_lines.append(
        f"{filename}|{size_bytes}|{actual_sha256}"
    )

    print(
        f"[{index:03d}/{EXPECTED_COUNT}]",
        filename,
        "| PASS",
        flush=True,
    )


total_size = sum(
    row["size_bytes"]
    for row in integrity_rows
)

positive_count = sum(
    row["sequence_label_id"] == 1
    for row in integrity_rows
)

negative_count = sum(
    row["sequence_label_id"] == 0
    for row in integrity_rows
)


if total_size != EXPECTED_TOTAL:
    raise RuntimeError(
        f"Total-size mismatch: {total_size}"
    )

if positive_count != 100 or negative_count != 60:
    raise RuntimeError(
        "Final label-count mismatch."
    )


snapshot_payload = (
    "\n".join(snapshot_lines) + "\n"
).encode("utf-8")

dataset_snapshot_sha256 = hashlib.sha256(
    snapshot_payload
).hexdigest()


with INTEGRITY_CSV_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(integrity_rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(integrity_rows)


summary = {
    "schema":
        "phyguard.kuleuven."
        "scenario1_locked_integrity.v1",

    "status": "PASS",

    "dataset_snapshot_sha256":
        dataset_snapshot_sha256,

    "verified_file_count":
        len(integrity_rows),

    "verified_total_size_bytes":
        total_size,

    "positive_sequence_count":
        positive_count,

    "negative_sequence_count":
        negative_count,

    "sha256_failure_count":
        0,

    "missing_file_count":
        0,

    "extra_file_count":
        0,

    "source_protocol_sha256":
        sha256_file(PROTOCOL_PATH),

    "source_locked_list_sha256":
        sha256_file(LOCKED_LIST_PATH),

    "source_download_summary_sha256":
        sha256_file(DOWNLOAD_SUMMARY_PATH),

    "source_download_manifest_sha256":
        sha256_file(DOWNLOAD_MANIFEST_PATH),

    "methodological_boundary": {
        "signal_values_analyzed": False,
        "features_computed": False,
        "models_trained": False,
        "predictions_computed": False,
        "performance_computed": False,
        "thresholds_selected": False,
        "event_intervals_inferred": False,
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
        "scenario1_locked_integrity_manifest.v1",

    "status": "PASS",

    "dataset_snapshot_sha256":
        dataset_snapshot_sha256,

    "verified_file_count":
        len(integrity_rows),

    "verified_total_size_bytes":
        total_size,

    "integrity_csv_sha256":
        sha256_file(INTEGRITY_CSV_PATH),

    "summary_sha256":
        sha256_file(SUMMARY_PATH),

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


print()
print("status: PASS")
print("verified_file_count:", len(integrity_rows))
print("verified_total_size_bytes:", total_size)
print("positive_sequence_count:", positive_count)
print("negative_sequence_count:", negative_count)
print("sha256_failure_count: 0")
print(
    "dataset_snapshot_sha256:",
    dataset_snapshot_sha256,
)
print("signal_values_analyzed: False")
print("features_computed: False")
print("performance_computed: False")
print("thresholds_selected: False")
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)
print(
    "\nKULEUVEN_SCENARIO1_"
    "LOCKED_INTEGRITY_V1_PASS"
)
