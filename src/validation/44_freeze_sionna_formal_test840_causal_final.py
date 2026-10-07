import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

PLAN_PATH = (
    ROOT
    / "configs"
    / "sionna_formal_test840_plan_v1.json"
)

PLAN_CSV = (
    ROOT
    / "configs"
    / "sionna_formal_test840_plan_v1.csv"
)

POLICY_PATH = (
    ROOT
    / "configs"
    / "sionna_causal_validation_policy_v2.json"
)

PLAN_LOCK = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_plan_lock.json"
)

PROTOCOL_FREEZE = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_protocol_freeze_manifest.json"
)

GENERATION_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_generation_manifest.json"
)

RUNNER_AMENDMENT = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_runner_qc_amendment_v1.json"
)

QC_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_artifact_qc_reconciled_v2.json"
)

CAUSAL_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_causal_validation_v2.json"
)

CAUSAL_CSV = (
    ROOT
    / "results"
    / "sionna_formal_test840_causal_validation_v2.csv"
)

DATA_ROOT = (
    ROOT
    / "data"
    / "sionna_formal_test840"
)

RESULT_ROOT = (
    ROOT
    / "results"
    / "sionna_formal_test840"
)

PROTOCOL_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal_test840_protocol_locked.zip"
)

PROTOCOL_ARCHIVE_SHA = Path(
    str(PROTOCOL_ARCHIVE) + ".sha256"
)

FREEZE_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_causal_final_freeze_manifest.json"
)

ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal_test840_causal_final.zip"
)

ARCHIVE_SHA = Path(
    str(ARCHIVE) + ".sha256"
)

EXPECTED_PLAN_SHA256 = (
    "b767fe07cd17eef0c42ec58cfe801732"
    "d38f7124ed5859915195e21223083437"
)

EXPECTED_POLICY_SHA256 = (
    "510b0ba97393a63a199d65414c2b44660"
    "e7140c1696582202b1603c7197f879d"
)

EXPECTED_PROTOCOL_ARCHIVE_SHA256 = (
    "ae721c16860d75cfce303c877f20125a"
    "9ee7583f7f6e8da659711c46339a3f5e"
)

EXPECTED_EVIDENCE_SHA256 = (
    "2a003bc8ecce4243309b297be36740669"
    "c04a8e7dfa27384a5f2de33ec3cc297"
)

EXPECTED_QC_EXCLUDED_ID = (
    "FT840_CDLB_NORMAL_R10"
)


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


def aggregate_original_evidence(samples):
    digest = hashlib.sha256()
    records = []

    for sample in sorted(
        samples,
        key=lambda item: item["sample_id"],
    ):
        sample_id = sample["sample_id"]

        paths = [
            DATA_ROOT / sample_id / "sequence.npz",
            DATA_ROOT / sample_id / "metadata.json",
            RESULT_ROOT / sample_id / "report.json",
        ]

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)

            relative = str(path.relative_to(ROOT))
            file_hash = sha256_file(path)

            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_hash.encode("ascii"))
            digest.update(b"\n")

            records.append(
                {
                    "relative_path": relative,
                    "sha256": file_hash,
                    "size_bytes": path.stat().st_size,
                }
            )

    return digest.hexdigest(), records


SCRIPT_PATHS = [
    ROOT / "scripts/41_generate_sionna_formal_test840.py",
    ROOT / "scripts/42_reconcile_formal_test840_artifact_qc.py",
    ROOT / "scripts/43_apply_locked_causal_v2_to_formal_test840.py",
]

LOG_PATHS = [
    ROOT / "logs/41_generate_sionna_formal_test840_driver.log",
    ROOT / "logs/42_reconcile_formal_test840_artifact_qc.log",
    ROOT / "logs/43_apply_locked_causal_v2_to_formal_test840.log",
]

OPTIONAL_LOG_PATHS = [
    ROOT / "logs/41_generate_sionna_formal_test840_audit.log",
    ROOT / "logs/41_generate_sionna_formal_test840.pid",
]

required = [
    PLAN_PATH,
    PLAN_CSV,
    POLICY_PATH,
    PLAN_LOCK,
    PROTOCOL_FREEZE,
    GENERATION_MANIFEST,
    RUNNER_AMENDMENT,
    QC_MANIFEST,
    CAUSAL_MANIFEST,
    CAUSAL_CSV,
    PROTOCOL_ARCHIVE,
    PROTOCOL_ARCHIVE_SHA,
] + SCRIPT_PATHS + LOG_PATHS

for path in required:
    if not path.exists():
        raise FileNotFoundError(path)


plan = load_json(PLAN_PATH)
policy = load_json(POLICY_PATH)
qc = load_json(QC_MANIFEST)
causal = load_json(CAUSAL_MANIFEST)


errors = []

if plan.get("plan_sha256") != EXPECTED_PLAN_SHA256:
    errors.append("Formal-plan SHA256 mismatch.")

if policy.get("policy_sha256") != EXPECTED_POLICY_SHA256:
    errors.append("Causal-policy SHA256 mismatch.")

if policy.get("status") != "LOCKED":
    errors.append("Causal policy is not LOCKED.")

samples = plan.get("samples", [])

if len(samples) != 840:
    errors.append(
        f"Formal sample count={len(samples)}, expected 840."
    )

if sha256_file(PROTOCOL_ARCHIVE) != EXPECTED_PROTOCOL_ARCHIVE_SHA256:
    errors.append("Formal protocol archive SHA256 mismatch.")

if qc.get("artifact_qc_passed_count") != 839:
    errors.append("Expected 839 artifact-QC passes.")

if qc.get("artifact_qc_failed_count") != 1:
    errors.append("Expected one artifact-QC exclusion.")

if qc.get("artifact_qc_failed_sample_ids") != [
    EXPECTED_QC_EXCLUDED_ID
]:
    errors.append("Unexpected artifact-QC exclusion.")

if causal.get("status") != "EVALUATION_COMPLETE":
    errors.append("Causal evaluation is not complete.")

if causal.get("planned_sample_count") != 840:
    errors.append("Causal planned count is not 840.")

if causal.get("artifact_qc_eligible_count") != 839:
    errors.append("Causal eligible count is not 839.")

if causal.get("artifact_qc_excluded_count") != 1:
    errors.append("Causal QC exclusion count is not one.")

if causal.get("causal_v2_evaluated_count") != 839:
    errors.append("Causal evaluated count is not 839.")

if causal.get("causal_v2_passed_count") != 825:
    errors.append("Expected 825 causal-v2 passes.")

if causal.get("causal_v2_failed_count") != 14:
    errors.append("Expected 14 causal-v2 failures.")


causal_records = causal.get("records", [])

if len(causal_records) != 840:
    errors.append(
        f"Causal record count={len(causal_records)}, expected 840."
    )

causal_failed_records = [
    record
    for record in causal_records
    if record.get("causal_v2_status") == "FAIL"
]

if len(causal_failed_records) != 14:
    errors.append("Causal failed-record count is not 14.")

non_mild_failures = [
    record["sample_id"]
    for record in causal_failed_records
    if record.get("severity") != "mild"
]

if non_mild_failures:
    errors.append(
        "Formal causal failures outside mild severity: "
        + str(non_mild_failures)
    )


expected_label_summary = {
    "adaptation_mismatch": {
        "planned_count": 180,
        "qc_excluded_count": 0,
        "evaluated_count": 180,
        "causal_v2_pass_count": 173,
        "causal_v2_fail_count": 7,
    },
    "blockage": {
        "planned_count": 180,
        "qc_excluded_count": 0,
        "evaluated_count": 180,
        "causal_v2_pass_count": 178,
        "causal_v2_fail_count": 2,
    },
    "interference": {
        "planned_count": 180,
        "qc_excluded_count": 0,
        "evaluated_count": 180,
        "causal_v2_pass_count": 175,
        "causal_v2_fail_count": 5,
    },
    "mobility": {
        "planned_count": 180,
        "qc_excluded_count": 0,
        "evaluated_count": 180,
        "causal_v2_pass_count": 180,
        "causal_v2_fail_count": 0,
    },
    "nonphysical_goodput": {
        "planned_count": 60,
        "qc_excluded_count": 0,
        "evaluated_count": 60,
        "causal_v2_pass_count": 60,
        "causal_v2_fail_count": 0,
    },
    "normal": {
        "planned_count": 60,
        "qc_excluded_count": 1,
        "evaluated_count": 59,
        "causal_v2_pass_count": 59,
        "causal_v2_fail_count": 0,
    },
}

if causal.get("label_summary") != expected_label_summary:
    errors.append("Formal causal label summary mismatch.")


evidence_hash, evidence_records = (
    aggregate_original_evidence(samples)
)

if evidence_hash != EXPECTED_EVIDENCE_SHA256:
    errors.append("Original evidence aggregate SHA256 mismatch.")

if len(evidence_records) != 2520:
    errors.append(
        f"Original evidence file count={len(evidence_records)}, "
        "expected 2520."
    )


sequence_files = list(
    DATA_ROOT.rglob("sequence.npz")
)

metadata_files = list(
    DATA_ROOT.rglob("metadata.json")
)

report_files = list(
    RESULT_ROOT.rglob("report.json")
)

reconciled_qc_files = list(
    RESULT_ROOT.rglob(
        "generation_qc_reconciled_v2.json"
    )
)

causal_record_files = list(
    RESULT_ROOT.rglob(
        "causal_validation_v2.json"
    )
)

if len(sequence_files) != 840:
    errors.append(
        f"Sequence file count={len(sequence_files)}, expected 840."
    )

if len(metadata_files) != 840:
    errors.append(
        f"Metadata file count={len(metadata_files)}, expected 840."
    )

if len(report_files) != 840:
    errors.append(
        f"Legacy report count={len(report_files)}, expected 840."
    )

if len(reconciled_qc_files) != 840:
    errors.append(
        "Reconciled per-sample QC file count is not 840."
    )

if len(causal_record_files) != 839:
    errors.append(
        "Per-sample causal-v2 file count is not 839."
    )


if errors:
    print("status: FAIL")

    for error in errors:
        print("-", error)

    raise RuntimeError(
        "Formal Test-840 final freeze validation failed."
    )


files = set(
    [
        PLAN_PATH,
        PLAN_CSV,
        POLICY_PATH,
        PLAN_LOCK,
        PROTOCOL_FREEZE,
        GENERATION_MANIFEST,
        RUNNER_AMENDMENT,
        QC_MANIFEST,
        CAUSAL_MANIFEST,
        CAUSAL_CSV,
        PROTOCOL_ARCHIVE,
        PROTOCOL_ARCHIVE_SHA,
    ]
    + SCRIPT_PATHS
    + LOG_PATHS
)

for path in OPTIONAL_LOG_PATHS:
    if path.exists():
        files.add(path)

for directory in (
    DATA_ROOT,
    RESULT_ROOT,
):
    for path in directory.rglob("*"):
        if path.is_file():
            files.add(path)


file_records = []

for path in sorted(
    files,
    key=lambda item: str(item.relative_to(ROOT)),
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


failure_label_counts = Counter(
    record["label"]
    for record in causal_failed_records
)

failure_check_counts = Counter(
    check
    for record in causal_failed_records
    for check in record.get(
        "failed_hard_checks",
        [],
    )
)


freeze_manifest = {
    "schema":
        "phyguard.sionna."
        "formal_test840.causal_final_freeze.v1",

    "status": "PASS",

    "purpose":
        "Freeze all preregistered Formal Test-840 "
        "simulation artifacts, artifact-QC outcomes, "
        "and results from the causal-validation policy "
        "locked before formal testing.",

    "formal_plan_sha256":
        EXPECTED_PLAN_SHA256,

    "causal_validation_policy_sha256":
        EXPECTED_POLICY_SHA256,

    "formal_protocol_archive_sha256":
        EXPECTED_PROTOCOL_ARCHIVE_SHA256,

    "planned_sample_count": 840,

    "artifact_qc": {
        "passed_count": 839,
        "excluded_count": 1,
        "excluded_sample_ids": [
            EXPECTED_QC_EXCLUDED_ID
        ],
    },

    "causal_validation_v2": {
        "evaluated_count": 839,
        "passed_count": 825,
        "failed_count": 14,
        "pass_rate_evaluated":
            825 / 839,
        "pass_rate_planned":
            825 / 840,
        "all_failures_mild_severity":
            True,
        "failure_label_counts":
            dict(failure_label_counts),
        "failed_hard_check_counts":
            dict(failure_check_counts),
    },

    "label_summary":
        expected_label_summary,

    "original_evidence": {
        "file_count": 2520,
        "aggregate_sha256":
            EXPECTED_EVIDENCE_SHA256,
        "unchanged": True,
    },

    "methodological_boundary": {
        "policy_locked_before_formal_test":
            True,
        "policy_modified_after_results":
            False,
        "failed_samples_replaced":
            False,
        "simulation_rerun_after_results":
            False,
        "original_sequences_modified":
            False,
        "original_metadata_modified":
            False,
        "original_reports_modified":
            False,
        "training_performed":
            False,
        "threshold_tuning_performed":
            False,
        "model_selection_performed":
            False,
    },

    "archive_entry_count":
        len(file_records) + 1,

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

files.add(FREEZE_MANIFEST)


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
        key=lambda item: str(item.relative_to(ROOT)),
    ):
        relative = str(path.relative_to(ROOT))

        info = zipfile.ZipInfo(
            relative,
            date_time=fixed_time,
        )

        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16

        archive.writestr(
            info,
            path.read_bytes(),
        )


archive_sha256 = sha256_file(ARCHIVE)

ARCHIVE_SHA.write_text(
    f"{archive_sha256}  {ARCHIVE.name}\n",
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

if len(archive_names) != len(files):
    raise RuntimeError(
        f"Archive entry count={len(archive_names)}, "
        f"expected {len(files)}."
    )


print("status: PASS")
print("planned_sample_count: 840")
print("artifact_qc_passed_count: 839")
print("artifact_qc_excluded_count: 1")
print(
    "artifact_qc_excluded_sample_ids:",
    [EXPECTED_QC_EXCLUDED_ID],
)
print("causal_v2_evaluated_count: 839")
print("causal_v2_passed_count: 825")
print("causal_v2_failed_count: 14")
print(
    "causal_v2_pass_rate_evaluated:",
    825 / 839,
)
print(
    "all_causal_failures_mild:",
    True,
)
print(
    "original_evidence_aggregate_sha256:",
    EXPECTED_EVIDENCE_SHA256,
)
print("original_evidence_file_count: 2520")
print("archive:", ARCHIVE)
print("archive_sha256:", archive_sha256)
print("archive_entry_count:", len(archive_names))
print("zip_test: PASS")
print("freeze_manifest:", FREEZE_MANIFEST)
print("sha256_file:", ARCHIVE_SHA)

print(
    "\nSIONNA_FORMAL_TEST840_"
    "CAUSAL_FINAL_FREEZE_PASS"
)
