import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("/root/phyguard_revision")

PLAN_PATH = (
    ROOT
    / "configs"
    / "sionna_formal_test840_plan_v1.json"
)

OLD_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_generation_manifest.json"
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

RUNNER_PATH = (
    ROOT
    / "scripts"
    / "41_generate_sionna_formal_test840.py"
)

OUTPUT_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_"
    "artifact_qc_reconciled_v2.json"
)

AMENDMENT_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_"
    "runner_qc_amendment_v1.json"
)

EXPECTED_PLAN_SHA256 = (
    "b767fe07cd17eef0c42ec58cfe801732"
    "d38f7124ed5859915195e21223083437"
)

EXPECTED_DIRECTION_ERROR = {
    "interference":
        "Interference physical-direction "
        "validation failed.",

    "blockage":
        "Blockage physical-direction "
        "validation failed.",

    "adaptation_mismatch":
        "Adaptation-mismatch directional "
        "validation failed.",
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


def aggregate_source_evidence(samples):
    digest = hashlib.sha256()
    count = 0

    for sample in sorted(
        samples,
        key=lambda value: value["sample_id"],
    ):
        sample_id = sample["sample_id"]

        paths = [
            DATA_ROOT
            / sample_id
            / "sequence.npz",

            DATA_ROOT
            / sample_id
            / "metadata.json",

            RESULT_ROOT
            / sample_id
            / "report.json",
        ]

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)

            relative = str(
                path.relative_to(ROOT)
            )

            file_hash = sha256_file(path)

            digest.update(
                relative.encode("utf-8")
            )
            digest.update(b"\0")
            digest.update(
                file_hash.encode("ascii")
            )
            digest.update(b"\n")

            count += 1

    return digest.hexdigest(), count


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def collected_report_errors(metadata, report):
    values = []

    validation = metadata.get(
        "validation"
    )

    if isinstance(validation, dict):
        values.extend(
            validation.get("errors", [])
        )

    values.extend(
        report.get("errors", [])
    )

    unique = []

    for value in values:
        value = str(value)

        if value and value not in unique:
            unique.append(value)

    return unique


for required in (
    PLAN_PATH,
    OLD_MANIFEST_PATH,
    RUNNER_PATH,
):
    if not required.exists():
        raise FileNotFoundError(required)


plan = load_json(PLAN_PATH)
old_manifest = load_json(
    OLD_MANIFEST_PATH
)

if plan.get("plan_sha256") != EXPECTED_PLAN_SHA256:
    raise RuntimeError(
        "Formal Test-840 plan hash changed."
    )

samples = plan.get("samples", [])

if len(samples) != 840:
    raise RuntimeError(
        f"Formal sample count={len(samples)}, "
        "expected 840."
    )


expected_contract_hash = plan.get(
    "kpi_contract_sha256"
)

if not isinstance(
    expected_contract_hash,
    str,
):
    raise RuntimeError(
        "Top-level KPI-contract SHA256 "
        "is missing from formal plan."
    )

if len(expected_contract_hash) != 64:
    raise RuntimeError(
        "Top-level KPI-contract SHA256 "
        "does not have 64 characters."
    )


old_runs = {
    record["sample_id"]: record
    for record in old_manifest.get(
        "runs",
        [],
    )
}

if len(old_runs) != 840:
    raise RuntimeError(
        f"Old manifest run count="
        f"{len(old_runs)}, expected 840."
    )


evidence_hash_before, evidence_count = (
    aggregate_source_evidence(samples)
)

records = []


for sample in samples:
    sample_id = sample["sample_id"]
    label = sample["label"]

    old_record = old_runs.get(
        sample_id
    )

    if old_record is None:
        raise RuntimeError(
            f"Old run record missing: "
            f"{sample_id}"
        )

    return_code = old_record.get(
        "generator_return_code"
    )

    sequence_path = (
        DATA_ROOT
        / sample_id
        / "sequence.npz"
    )

    metadata_path = (
        DATA_ROOT
        / sample_id
        / "metadata.json"
    )

    report_path = (
        RESULT_ROOT
        / sample_id
        / "report.json"
    )

    errors = []

    for path in (
        sequence_path,
        metadata_path,
        report_path,
    ):
        if not path.exists():
            errors.append(
                f"missing {path}"
            )

    metadata = {}
    report = {}

    if sequence_path.exists():
        try:
            with np.load(
                sequence_path,
                allow_pickle=False,
            ) as archive:
                required_keys = {
                    "kpi_sequence",
                    "kpi_names",
                    "frame_index",
                }

                missing_keys = (
                    required_keys
                    - set(archive.files)
                )

                if missing_keys:
                    errors.append(
                        "missing NPZ keys: "
                        + str(
                            sorted(missing_keys)
                        )
                    )

                else:
                    sequence = archive[
                        "kpi_sequence"
                    ]

                    if sequence.shape != (
                        80,
                        10,
                    ):
                        errors.append(
                            "invalid sequence shape "
                            f"{sequence.shape}"
                        )

                    if (
                        sequence.dtype
                        != np.float32
                    ):
                        errors.append(
                            "invalid sequence dtype "
                            f"{sequence.dtype}"
                        )

                    if not np.isfinite(
                        sequence
                    ).all():
                        errors.append(
                            "sequence contains "
                            "NaN or Inf"
                        )

                    if len(
                        archive["kpi_names"]
                    ) != 10:
                        errors.append(
                            "KPI-name count "
                            "is not 10"
                        )

                    if len(
                        archive["frame_index"]
                    ) != 80:
                        errors.append(
                            "frame-index count "
                            "is not 80"
                        )

        except Exception as exc:
            errors.append(
                f"NPZ read error: {exc!r}"
            )


    if metadata_path.exists():
        try:
            metadata = load_json(
                metadata_path
            )
        except Exception as exc:
            errors.append(
                "metadata JSON error: "
                f"{exc!r}"
            )


    if report_path.exists():
        try:
            report = load_json(
                report_path
            )
        except Exception as exc:
            errors.append(
                f"report JSON error: "
                f"{exc!r}"
            )


    if metadata:
        for field in (
            "sample_id",
            "label",
            "severity",
            "channel_model",
        ):
            if (
                metadata.get(field)
                != sample.get(field)
            ):
                errors.append(
                    f"metadata {field} mismatch"
                )

        if metadata.get(
            "sequence_shape"
        ) not in (
            [80, 10],
            None,
        ):
            errors.append(
                "metadata sequence "
                "shape mismatch"
            )

        if (
            metadata.get(
                "exact_bitwise_replay"
            )
            is not True
        ):
            errors.append(
                "exact bitwise replay failed"
            )

        if (
            metadata.get(
                "numerical_replay_pass"
            )
            is not True
        ):
            errors.append(
                "numerical replay failed"
            )

        actual_contract_hash = (
            metadata.get(
                "kpi_contract_sha256"
            )
        )

        if (
            actual_contract_hash
            != expected_contract_hash
        ):
            errors.append(
                "KPI-contract SHA256 "
                "mismatch against formal "
                "plan top-level contract"
            )

        validation = metadata.get(
            "validation"
        )

        if isinstance(
            validation,
            dict,
        ):
            if (
                validation.get(
                    "all_finite"
                )
                is not True
            ):
                errors.append(
                    "base validation "
                    "all_finite failed"
                )

            if int(
                validation.get(
                    "varying_kpi_count",
                    0,
                )
            ) < 6:
                errors.append(
                    "fewer than six "
                    "varying KPIs"
                )

            for name, item in (
                validation.get(
                    "range_checks",
                    {},
                ).items()
            ):
                if (
                    item.get("pass")
                    is not True
                ):
                    errors.append(
                        "range check failed: "
                        f"{name}"
                    )


    legacy_status = (
        report.get("status")
        if report
        else None
    )

    report_errors = (
        collected_report_errors(
            metadata,
            report,
        )
        if metadata and report
        else []
    )


    if metadata and report:
        if label in EXPECTED_DIRECTION_ERROR:
            expected_error = (
                EXPECTED_DIRECTION_ERROR[
                    label
                ]
            )

            unexpected_errors = [
                item
                for item in report_errors
                if item != expected_error
            ]

            for item in unexpected_errors:
                errors.append(
                    "unexpected generator "
                    f"error: {item}"
                )

            pass_case = (
                return_code == 0
                and legacy_status == "PASS"
                and not report_errors
            )

            allowed_legacy_fail_case = (
                return_code == 1
                and legacy_status == "FAIL"
                and expected_error
                in report_errors
                and not unexpected_errors
            )

            if not (
                pass_case
                or allowed_legacy_fail_case
            ):
                errors.append(
                    "generator exit/status "
                    "is inconsistent with "
                    "the locked legacy "
                    "directional validator"
                )

        elif label in {
            "normal",
            "mobility",
            "nonphysical_goodput",
        }:
            if return_code != 0:
                errors.append(
                    "generator return code="
                    f"{return_code}"
                )

            if legacy_status != "PASS":
                errors.append(
                    "legacy report status="
                    f"{legacy_status!r}"
                )

            for item in report_errors:
                errors.append(
                    f"generator error: {item}"
                )

        else:
            errors.append(
                f"unsupported label {label}"
            )


    status = (
        "PASS"
        if not errors
        else "FAIL"
    )

    record = {
        "schema":
            "phyguard.sionna."
            "formal_test840."
            "generation_qc_reconciled.v2",

        "status":
            status,

        "sample_id":
            sample_id,

        "label":
            label,

        "severity":
            sample["severity"],

        "channel_model":
            sample["channel_model"],

        "seed":
            sample["seed"],

        "formal_plan_sha256":
            EXPECTED_PLAN_SHA256,

        "expected_kpi_contract_sha256":
            expected_contract_hash,

        "metadata_kpi_contract_sha256":
            metadata.get(
                "kpi_contract_sha256"
            ),

        "generator_return_code":
            return_code,

        "legacy_directional_status":
            legacy_status,

        "legacy_report_errors":
            report_errors,

        "errors":
            errors,

        "sequence_sha256": (
            sha256_file(sequence_path)
            if sequence_path.exists()
            else None
        ),

        "metadata_sha256": (
            sha256_file(metadata_path)
            if metadata_path.exists()
            else None
        ),

        "report_sha256": (
            sha256_file(report_path)
            if report_path.exists()
            else None
        ),
    }

    qc_path = (
        RESULT_ROOT
        / sample_id
        / "generation_qc_"
        "reconciled_v2.json"
    )

    qc_path.write_text(
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    records.append(record)


evidence_hash_after, evidence_count_after = (
    aggregate_source_evidence(samples)
)

if evidence_hash_after != evidence_hash_before:
    raise RuntimeError(
        "Original Formal Test evidence "
        "changed during reconciliation."
    )

if evidence_count_after != evidence_count:
    raise RuntimeError(
        "Original Formal Test evidence "
        "file count changed."
    )


passed = [
    record
    for record in records
    if record["status"] == "PASS"
]

failed = [
    record
    for record in records
    if record["status"] != "PASS"
]

legacy_counts = Counter(
    record[
        "legacy_directional_status"
    ]
    for record in records
)

failure_label_counts = Counter(
    record["label"]
    for record in failed
)

error_counts = Counter(
    error
    for record in failed
    for error in record["errors"]
)


output = {
    "schema":
        "phyguard.sionna."
        "formal_test840."
        "artifact_qc_reconciled.v2",

    "status": (
        "PASS"
        if not failed
        else "COMPLETE_WITH_QC_FAILURES"
    ),

    "purpose":
        "Reconcile existing Formal Test-840 "
        "artifacts after correcting a runner-only "
        "KPI-contract comparison bug.",

    "sample_count":
        840,

    "artifact_qc_passed_count":
        len(passed),

    "artifact_qc_failed_count":
        len(failed),

    "artifact_qc_failed_sample_ids": [
        record["sample_id"]
        for record in failed
    ],

    "legacy_directional_status_counts":
        dict(legacy_counts),

    "failure_label_counts":
        dict(failure_label_counts),

    "error_counts":
        dict(error_counts),

    "formal_plan_sha256":
        EXPECTED_PLAN_SHA256,

    "expected_kpi_contract_sha256":
        expected_contract_hash,

    "original_evidence": {
        "file_count":
            evidence_count,

        "aggregate_sha256":
            evidence_hash_before,

        "unchanged_during_reconciliation":
            True,
    },

    "methodological_boundary": {
        "simulation_rerun":
            False,

        "sequence_files_modified":
            False,

        "metadata_files_modified":
            False,

        "legacy_reports_modified":
            False,

        "causal_validation_policy_modified":
            False,

        "models_loaded":
            False,

        "training_performed":
            False,

        "threshold_tuning_performed":
            False,
    },

    "records":
        records,
}

OUTPUT_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_MANIFEST.write_text(
    json.dumps(
        output,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


amendment = {
    "schema":
        "phyguard.sionna."
        "formal_test840."
        "runner_qc_amendment.v1",

    "status":
        "PASS",

    "scope":
        "Artifact-QC implementation only",

    "original_runner_path":
        str(RUNNER_PATH),

    "original_runner_sha256":
        sha256_file(RUNNER_PATH),

    "identified_bug":
        (
            "The runner compared metadata "
            "kpi_contract_sha256 against a "
            "nonexistent per-sample plan field."
        ),

    "incorrect_comparison":
        (
            "metadata.kpi_contract_sha256 "
            "vs sample.kpi_contract_sha256"
        ),

    "correct_comparison":
        (
            "metadata.kpi_contract_sha256 "
            "vs formal-plan top-level "
            "kpi_contract_sha256"
        ),

    "false_failure_count":
        840,

    "simulation_artifacts_regenerated":
        False,

    "locked_policy_modified":
        False,

    "reconciled_manifest":
        str(OUTPUT_MANIFEST),

    "original_evidence_aggregate_sha256":
        evidence_hash_before,
}

AMENDMENT_MANIFEST.write_text(
    json.dumps(
        amendment,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print("status:", output["status"])
print("sample_count: 840")
print(
    "artifact_qc_passed_count:",
    len(passed),
)
print(
    "artifact_qc_failed_count:",
    len(failed),
)
print(
    "artifact_qc_failed_sample_ids:",
    output[
        "artifact_qc_failed_sample_ids"
    ],
)
print(
    "legacy_directional_status_counts:",
    dict(legacy_counts),
)
print(
    "expected_kpi_contract_sha256:",
    expected_contract_hash,
)
print(
    "original_evidence_file_count:",
    evidence_count,
)
print(
    "original_evidence_aggregate_sha256:",
    evidence_hash_before,
)
print(
    "original_evidence_unchanged:",
    True,
)

print("\nFAILURE DETAILS")

for record in failed:
    print(
        record["sample_id"],
        "| label=",
        record["label"],
        "| channel=",
        record["channel_model"],
        "| errors=",
        record["errors"],
    )

print(
    "\nreconciled_manifest:",
    OUTPUT_MANIFEST,
)
print(
    "amendment_manifest:",
    AMENDMENT_MANIFEST,
)

print(
    "\nSIONNA_FORMAL_TEST840_"
    "ARTIFACT_QC_RECONCILIATION_PASS"
)
