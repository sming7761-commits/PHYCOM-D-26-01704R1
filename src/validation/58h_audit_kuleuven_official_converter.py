import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_official_converter_audit_v1"
)

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_official_converter_audit_v1.json"
)

FILES = [
    {
        "file_id": 28010,
        "filename": "binary_to_csv_converter.py",
        "expected_size": 1318,
    },
    {
        "file_id": 27546,
        "filename": "binary2npy.c",
        "expected_size": 8284,
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


def download_file(file_id, output_path, limit_bytes):
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
        "180",
        "--max-filesize",
        str(limit_bytes),
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
        "url": url,
        "returncode": int(result.returncode),
        "stderr": result.stderr[-4000:],
    }


if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Converter audit output already exists; "
        "refusing to overwrite it."
    )

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)


records = []


for record in FILES:
    output_path = (
        OUTPUT_ROOT
        / record["filename"]
    )

    result = download_file(
        record["file_id"],
        output_path,
        limit_bytes=1024 * 1024,
    )

    if result["returncode"] != 0:
        raise RuntimeError(
            f"Download failed for {record['filename']}: "
            f"{result['stderr']}"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"Downloaded file is missing: {output_path}"
        )

    raw = output_path.read_bytes()

    beginning = raw[:256].lower()

    if (
        b"<html" in beginning
        or b"<!doctype html" in beginning
    ):
        raise RuntimeError(
            f"HTML returned instead of source: "
            f"{record['filename']}"
        )

    actual_size = output_path.stat().st_size

    if actual_size != record["expected_size"]:
        raise RuntimeError(
            f"Unexpected size for {record['filename']}: "
            f"{actual_size} != {record['expected_size']}"
        )

    records.append(
        {
            "file_id":
                record["file_id"],

            "filename":
                record["filename"],

            "expected_size":
                record["expected_size"],

            "actual_size":
                actual_size,

            "sha256":
                sha256_file(
                    output_path
                ),

            "relative_path":
                str(
                    output_path.relative_to(
                        ROOT
                    )
                ),

            "download_url":
                result["url"],
        }
    )


manifest = {
    "schema":
        "phyguard.kuleuven."
        "official_converter_audit.v1",

    "status":
        "PASS",

    "source_file_count":
        len(records),

    "measurement_binary_opened":
        False,

    "example_measurement_binary_opened":
        False,

    "models_trained":
        False,

    "labels_constructed":
        False,

    "performance_computed":
        False,

    "files":
        records,
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
    "source_file_count:",
    len(records),
)
print("measurement_binary_opened: False")
print("example_measurement_binary_opened: False")
print("models_trained: False")
print("labels_constructed: False")
print("performance_computed: False")


for record in records:
    path = (
        OUTPUT_ROOT
        / record["filename"]
    )

    print()
    print("=" * 88)
    print(
        record["filename"],
        "| bytes=",
        record["actual_size"],
        "| sha256=",
        record["sha256"],
    )
    print("=" * 88)

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        print(
            f"{line_number:04d}: {line}"
        )


print()
print("output_root:", OUTPUT_ROOT)
print("manifest:", MANIFEST_PATH)
print(
    "KULEUVEN_OFFICIAL_CONVERTER_AUDIT_PASS"
)
