import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

FORMAL_PLAN = (
    ROOT
    / "configs"
    / "sionna_formal_test840_plan_v1.json"
)

FORMAL_CSV = (
    ROOT
    / "configs"
    / "sionna_formal_test840_plan_v1.csv"
)

FORMAL_LOCK = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_plan_lock.json"
)

CREATE_SCRIPT = (
    ROOT
    / "scripts"
    / "39_create_sionna_formal_test840_plan.py"
)

CREATE_LOG = (
    ROOT
    / "logs"
    / "39_create_sionna_formal_test840_plan.log"
)

POLICY = (
    ROOT
    / "configs"
    / "sionna_causal_validation_policy_v2.json"
)

PREVALIDATION_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_prevalidation210_causal_v2_final.zip"
)

PREVALIDATION_SHA_FILE = Path(
    str(PREVALIDATION_ARCHIVE) + ".sha256"
)

PREVALIDATION_FREEZE_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_prevalidation210_causal_v2_freeze_manifest.json"
)

SOURCE_PREVALIDATION_PLAN = (
    ROOT
    / "configs"
    / "sionna_prevalidation210_plan_v1.json"
)

SMOKE_PLAN = (
    ROOT
    / "configs"
    / "smoke24_plan_v1.json"
)

FREEZE_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_protocol_freeze_manifest.json"
)

ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal_test840_protocol_locked.zip"
)

ARCHIVE_SHA_FILE = Path(
    str(ARCHIVE) + ".sha256"
)

EXPECTED_FORMAL_PLAN_SHA256 = (
    "b767fe07cd17eef0c42ec58cfe801732"
    "d38f7124ed5859915195e21223083437"
)

EXPECTED_POLICY_SHA256 = (
    "510b0ba97393a63a199d65414c2b44660"
    "e7140c1696582202b1603c7197f879d"
)

EXPECTED_PREVALIDATION_ARCHIVE_SHA256 = (
    "da8d6dfe0e5fa443bae14cf6501012b8"
    "5982e8bb02221c9e1b5caa76d128461f"
)

EXPECTED_LABEL_COUNTS = {
    "normal": 60,
    "interference": 180,
    "blockage": 180,
    "mobility": 180,
    "adaptation_mismatch": 180,
    "nonphysical_goodput": 60,
}

EXPECTED_CHANNEL_COUNTS = {
    "CDL-A": 280,
    "CDL-B": 280,
    "CDL-C": 280,
}

EXPECTED_SEVERITY_COUNTS = {
    "control": 120,
    "mild": 240,
    "moderate": 240,
    "severe": 240,
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


def sha256_json(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


required_paths = [
    FORMAL_PLAN,
    FORMAL_CSV,
    FORMAL_LOCK,
    CREATE_SCRIPT,
    CREATE_LOG,
    POLICY,
    PREVALIDATION_ARCHIVE,
    PREVALIDATION_SHA_FILE,
    PREVALIDATION_FREEZE_MANIFEST,
    SOURCE_PREVALIDATION_PLAN,
    SMOKE_PLAN,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)


formal_plan = json.loads(
    FORMAL_PLAN.read_text(
        encoding="utf-8"
    )
)

formal_lock = json.loads(
    FORMAL_LOCK.read_text(
        encoding="utf-8"
    )
)

policy = json.loads(
    POLICY.read_text(
        encoding="utf-8"
    )
)

prevalidation_freeze = json.loads(
    PREVALIDATION_FREEZE_MANIFEST.read_text(
        encoding="utf-8"
    )
)


errors = []

stored_plan_hash = formal_plan.get(
    "plan_sha256"
)

plan_without_hash = dict(
    formal_plan
)

plan_without_hash.pop(
    "plan_sha256",
    None,
)

recomputed_plan_hash = sha256_json(
    plan_without_hash
)

if (
    stored_plan_hash
    != EXPECTED_FORMAL_PLAN_SHA256
):
    errors.append(
        "Stored formal-plan SHA256 mismatch."
    )

if (
    recomputed_plan_hash
    != EXPECTED_FORMAL_PLAN_SHA256
):
    errors.append(
        "Recomputed formal-plan SHA256 mismatch."
    )

if formal_plan.get("sample_count") != 840:
    errors.append(
        "Formal plan sample count is not 840."
    )

if formal_plan.get("stratum_count") != 42:
    errors.append(
        "Formal plan stratum count is not 42."
    )

if (
    formal_plan.get(
        "replicates_per_stratum"
    )
    != 20
):
    errors.append(
        "Formal plan replicates per stratum "
        "is not 20."
    )

samples = formal_plan.get(
    "samples",
    []
)

if len(samples) != 840:
    errors.append(
        f"Formal sample-list count={len(samples)}, "
        "expected 840."
    )


sample_ids = [
    sample.get("sample_id")
    for sample in samples
]

seeds = [
    int(sample.get("seed"))
    for sample in samples
]

if len(set(sample_ids)) != 840:
    errors.append(
        "Formal sample IDs are not unique."
    )

if len(set(seeds)) != 840:
    errors.append(
        "Formal seeds are not unique."
    )


label_counts = Counter(
    sample.get("label")
    for sample in samples
)

channel_counts = Counter(
    sample.get("channel_model")
    for sample in samples
)

severity_counts = Counter(
    sample.get("severity")
    for sample in samples
)

stratum_counts = Counter(
    sample.get("formal_test_stratum")
    for sample in samples
)


if dict(label_counts) != EXPECTED_LABEL_COUNTS:
    errors.append(
        f"Unexpected label counts: "
        f"{dict(label_counts)}"
    )

if dict(channel_counts) != EXPECTED_CHANNEL_COUNTS:
    errors.append(
        f"Unexpected channel counts: "
        f"{dict(channel_counts)}"
    )

if dict(severity_counts) != EXPECTED_SEVERITY_COUNTS:
    errors.append(
        f"Unexpected severity counts: "
        f"{dict(severity_counts)}"
    )

if len(stratum_counts) != 42:
    errors.append(
        "Unique formal stratum count is not 42."
    )

if set(stratum_counts.values()) != {20}:
    errors.append(
        "Not every formal stratum contains 20 samples."
    )


for sample in samples:
    sample_id = sample.get("sample_id")

    required_false_flags = [
        "eligible_for_training",
        "eligible_for_threshold_tuning",
        "eligible_for_model_selection",
        "eligible_for_feature_selection",
        "eligible_for_calibration",
        "policy_modification_allowed",
        "prevalidation_overlap_allowed",
    ]

    for field in required_false_flags:
        if sample.get(field) is not False:
            errors.append(
                f"{sample_id}: {field} is not false."
            )

    if sample.get("formal_test") is not True:
        errors.append(
            f"{sample_id}: formal_test is not true."
        )

    if (
        sample.get(
            "causal_validation_policy_sha256"
        )
        != EXPECTED_POLICY_SHA256
    ):
        errors.append(
            f"{sample_id}: policy SHA256 mismatch."
        )


if formal_lock.get("status") != "PASS":
    errors.append(
        "Formal plan-lock manifest is not PASS."
    )

if formal_lock.get("sample_count") != 840:
    errors.append(
        "Plan-lock sample count is not 840."
    )

if (
    formal_lock.get("plan_sha256")
    != EXPECTED_FORMAL_PLAN_SHA256
):
    errors.append(
        "Plan-lock SHA256 mismatch."
    )

for overlap_field in [
    "prevalidation_sample_id_overlap_count",
    "prevalidation_seed_overlap_count",
    "smoke24_sample_id_overlap_count",
    "smoke24_seed_overlap_count",
]:
    if formal_lock.get(overlap_field) != 0:
        errors.append(
            f"{overlap_field} is not zero."
        )


if (
    policy.get("policy_sha256")
    != EXPECTED_POLICY_SHA256
):
    errors.append(
        "Causal-validation policy SHA256 mismatch."
    )

if policy.get("status") != "LOCKED":
    errors.append(
        "Causal-validation policy is not LOCKED."
    )


prevalidation_archive_hash = sha256_file(
    PREVALIDATION_ARCHIVE
)

if (
    prevalidation_archive_hash
    != EXPECTED_PREVALIDATION_ARCHIVE_SHA256
):
    errors.append(
        "Frozen Prevalidation-210 archive hash mismatch."
    )


sidecar_text = (
    PREVALIDATION_SHA_FILE
    .read_text(
        encoding="utf-8"
    )
    .strip()
)

if (
    EXPECTED_PREVALIDATION_ARCHIVE_SHA256
    not in sidecar_text
):
    errors.append(
        "Prevalidation SHA256 sidecar mismatch."
    )


if (
    prevalidation_freeze.get("status")
    != "PASS"
):
    errors.append(
        "Prevalidation freeze manifest is not PASS."
    )

if (
    prevalidation_freeze.get(
        "sample_count"
    )
    != 210
):
    errors.append(
        "Prevalidation freeze sample count "
        "is not 210."
    )

if (
    prevalidation_freeze
    .get("causal_validation_v2", {})
    .get("policy_sha256")
    != EXPECTED_POLICY_SHA256
):
    errors.append(
        "Prevalidation freeze policy hash mismatch."
    )


if errors:
    print("status: FAIL")

    for error in errors:
        print("-", error)

    raise RuntimeError(
        "Formal Test-840 protocol validation failed."
    )


files = [
    FORMAL_PLAN,
    FORMAL_CSV,
    FORMAL_LOCK,
    CREATE_SCRIPT,
    CREATE_LOG,
    POLICY,
    PREVALIDATION_FREEZE_MANIFEST,
    PREVALIDATION_SHA_FILE,
    SOURCE_PREVALIDATION_PLAN,
    SMOKE_PLAN,
]


file_records = []

for path in sorted(
    files,
    key=lambda value: str(
        value.relative_to(ROOT)
    ),
):
    file_records.append(
        {
            "relative_path":
                str(path.relative_to(ROOT)),
            "sha256":
                sha256_file(path),
            "size_bytes":
                path.stat().st_size,
        }
    )


freeze_manifest = {
    "schema":
        "phyguard.sionna."
        "formal_test840.protocol_freeze.v1",

    "status":
        "PASS",

    "purpose":
        "Freeze the disjoint Formal Test-840 "
        "plan and causal-validation policy before "
        "formal sample generation.",

    "sample_count":
        840,

    "stratum_count":
        42,

    "samples_per_stratum":
        20,

    "plan_sha256":
        EXPECTED_FORMAL_PLAN_SHA256,

    "causal_validation_policy_sha256":
        EXPECTED_POLICY_SHA256,

    "source_prevalidation_archive_sha256":
        EXPECTED_PREVALIDATION_ARCHIVE_SHA256,

    "seed_range": [
        min(seeds),
        max(seeds),
    ],

    "unique_sample_id_count":
        len(set(sample_ids)),

    "unique_seed_count":
        len(set(seeds)),

    "overlap_counts": {
        "prevalidation_sample_ids": 0,
        "prevalidation_seeds": 0,
        "smoke24_sample_ids": 0,
        "smoke24_seeds": 0,
    },

    "label_counts":
        dict(label_counts),

    "channel_counts":
        dict(channel_counts),

    "severity_counts":
        dict(severity_counts),

    "methodological_boundary": {
        "training_allowed":
            False,
        "threshold_tuning_allowed":
            False,
        "model_selection_allowed":
            False,
        "feature_selection_allowed":
            False,
        "calibration_allowed":
            False,
        "policy_modification_allowed":
            False,
        "policy_locked_before_generation":
            True,
    },

    "files":
        file_records,
}

FREEZE_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True,
)

FREEZE_MANIFEST.write_text(
    json.dumps(
        freeze_manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

files.append(
    FREEZE_MANIFEST
)


archive_manifest = {
    "schema":
        "phyguard.archive_manifest.v1",

    "status":
        "PASS",

    "archive_role":
        "Formal Test-840 protocol lock",

    "formal_plan_sha256":
        EXPECTED_FORMAL_PLAN_SHA256,

    "causal_validation_policy_sha256":
        EXPECTED_POLICY_SHA256,

    "source_prevalidation_archive_sha256":
        EXPECTED_PREVALIDATION_ARCHIVE_SHA256,

    "file_count":
        len(files),

    "files": [
        {
            "relative_path":
                str(path.relative_to(ROOT)),
            "sha256":
                sha256_file(path),
            "size_bytes":
                path.stat().st_size,
        }
        for path in sorted(
            files,
            key=lambda value: str(
                value.relative_to(ROOT)
            ),
        )
    ],
}

archive_manifest_bytes = json.dumps(
    archive_manifest,
    ensure_ascii=False,
    indent=2,
).encode("utf-8")


ARCHIVE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if ARCHIVE.exists():
    ARCHIVE.unlink()

fixed_time = (
    1980,
    1,
    1,
    0,
    0,
    0,
)

with zipfile.ZipFile(
    ARCHIVE,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=9,
) as archive:

    for path in sorted(
        files,
        key=lambda value: str(
            value.relative_to(ROOT)
        ),
    ):
        relative = str(
            path.relative_to(ROOT)
        )

        info = zipfile.ZipInfo(
            relative,
            date_time=fixed_time,
        )

        info.compress_type = (
            zipfile.ZIP_DEFLATED
        )

        info.external_attr = (
            0o644 << 16
        )

        archive.writestr(
            info,
            path.read_bytes(),
        )

    manifest_info = zipfile.ZipInfo(
        "archive_manifest.json",
        date_time=fixed_time,
    )

    manifest_info.compress_type = (
        zipfile.ZIP_DEFLATED
    )

    manifest_info.external_attr = (
        0o644 << 16
    )

    archive.writestr(
        manifest_info,
        archive_manifest_bytes,
    )


archive_sha256 = sha256_file(
    ARCHIVE
)

ARCHIVE_SHA_FILE.write_text(
    f"{archive_sha256}  "
    f"{ARCHIVE.name}\n",
    encoding="utf-8",
)


with zipfile.ZipFile(
    ARCHIVE,
    mode="r",
) as archive:
    bad_file = archive.testzip()
    archive_names = archive.namelist()

if bad_file is not None:
    raise RuntimeError(
        f"ZIP integrity failure: {bad_file}"
    )


expected_archive_entries = (
    len(files) + 1
)

if (
    len(archive_names)
    != expected_archive_entries
):
    raise RuntimeError(
        f"Archive entry count="
        f"{len(archive_names)}, "
        f"expected "
        f"{expected_archive_entries}."
    )


print("status: PASS")
print("sample_count: 840")
print("stratum_count: 42")
print("samples_per_stratum: 20")
print(
    "formal_plan_sha256:",
    EXPECTED_FORMAL_PLAN_SHA256,
)
print(
    "policy_sha256:",
    EXPECTED_POLICY_SHA256,
)
print(
    "source_prevalidation_archive_sha256:",
    EXPECTED_PREVALIDATION_ARCHIVE_SHA256,
)
print("unique_sample_ids: 840")
print("unique_seeds: 840")
print(
    "seed_range:",
    [min(seeds), max(seeds)],
)
print(
    "all_overlap_counts_zero:",
    True,
)
print(
    "policy_modification_allowed:",
    False,
)
print("archive:", ARCHIVE)
print(
    "archive_sha256:",
    archive_sha256,
)
print(
    "archive_entry_count:",
    len(archive_names),
)
print("zip_test: PASS")
print(
    "freeze_manifest:",
    FREEZE_MANIFEST,
)
print(
    "sha256_file:",
    ARCHIVE_SHA_FILE,
)

print(
    "\nSIONNA_FORMAL_TEST840_"
    "PROTOCOL_FREEZE_PASS"
)
