import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

PLAN_PATH = (
    ROOT
    / "configs"
    / "sionna_formal_test840_plan_v1.json"
)

POLICY_PATH = (
    ROOT
    / "configs"
    / "sionna_causal_validation_policy_v2.json"
)

QC_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_artifact_qc_reconciled_v2.json"
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

OUTPUT_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_causal_validation_v2.json"
)

OUTPUT_CSV = (
    ROOT
    / "results"
    / "sionna_formal_test840_causal_validation_v2.csv"
)

EXPECTED_PLAN_SHA256 = (
    "b767fe07cd17eef0c42ec58cfe801732"
    "d38f7124ed5859915195e21223083437"
)

EXPECTED_POLICY_SHA256 = (
    "510b0ba97393a63a199d65414c2b44660"
    "e7140c1696582202b1603c7197f879d"
)

EXPECTED_SOURCE_EVIDENCE_SHA256 = (
    "2a003bc8ecce4243309b297be36740669"
    "c04a8e7dfa27384a5f2de33ec3cc297"
)

EXPECTED_QC_EXCLUDED_ID = (
    "FT840_CDLB_NORMAL_R10"
)

SEVERITY_VALUES = {
    "interference": {
        "mild": 6.0,
        "moderate": 3.0,
        "severe": 0.0,
    },
    "blockage": {
        "mild": -6.0,
        "moderate": -9.0,
        "severe": -12.0,
    },
    "adaptation_mismatch": {
        "mild": 30.0,
        "moderate": 45.0,
        "severe": 60.0,
    },
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


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def first_number(*values):
    for value in values:
        if finite_number(value):
            return float(value)

    return None


def close_enough(left, right, tolerance=1e-6):
    return (
        finite_number(left)
        and finite_number(right)
        and abs(float(left) - float(right))
        <= tolerance
    )


def aggregate_original_evidence(samples):
    digest = hashlib.sha256()
    count = 0

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

            count += 1

    return digest.hexdigest(), count


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


for required in (
    PLAN_PATH,
    POLICY_PATH,
    QC_MANIFEST_PATH,
):
    if not required.exists():
        raise FileNotFoundError(required)


plan = load_json(PLAN_PATH)
policy = load_json(POLICY_PATH)
qc_manifest = load_json(QC_MANIFEST_PATH)


if plan.get("plan_sha256") != EXPECTED_PLAN_SHA256:
    raise RuntimeError(
        "Formal Test-840 plan hash changed."
    )

if policy.get("policy_sha256") != EXPECTED_POLICY_SHA256:
    raise RuntimeError(
        "Causal-validation policy hash changed."
    )

if policy.get("status") != "LOCKED":
    raise RuntimeError(
        "Causal-validation policy is not LOCKED."
    )

samples = plan.get("samples", [])

if len(samples) != 840:
    raise RuntimeError(
        f"Formal sample count={len(samples)}, expected 840."
    )

if qc_manifest.get("sample_count") != 840:
    raise RuntimeError(
        "Reconciled QC manifest sample count is not 840."
    )

if qc_manifest.get("artifact_qc_passed_count") != 839:
    raise RuntimeError(
        "Expected 839 artifact-QC passes."
    )

if qc_manifest.get("artifact_qc_failed_count") != 1:
    raise RuntimeError(
        "Expected one artifact-QC failure."
    )

qc_failed_ids = qc_manifest.get(
    "artifact_qc_failed_sample_ids",
    [],
)

if qc_failed_ids != [EXPECTED_QC_EXCLUDED_ID]:
    raise RuntimeError(
        f"Unexpected QC exclusion list: {qc_failed_ids}"
    )


qc_by_id = {
    record["sample_id"]: record
    for record in qc_manifest.get("records", [])
}

if len(qc_by_id) != 840:
    raise RuntimeError(
        "Reconciled QC record count is not 840."
    )


evidence_hash_before, evidence_count = (
    aggregate_original_evidence(samples)
)

if evidence_hash_before != EXPECTED_SOURCE_EVIDENCE_SHA256:
    raise RuntimeError(
        "Original Formal Test evidence hash changed."
    )

if evidence_count != 2520:
    raise RuntimeError(
        f"Original evidence file count={evidence_count}, expected 2520."
    )


records = []


for sample in samples:
    sample_id = sample["sample_id"]
    label = sample["label"]
    severity = sample["severity"]

    qc_record = qc_by_id.get(sample_id)

    if qc_record is None:
        raise RuntimeError(
            f"QC record missing: {sample_id}"
        )

    if qc_record.get("status") != "PASS":
        records.append(
            {
                "sample_id": sample_id,
                "label": label,
                "severity": severity,
                "channel_model":
                    sample["channel_model"],
                "seed": sample["seed"],
                "artifact_qc_status":
                    qc_record.get("status"),
                "causal_v2_status":
                    "NOT_EVALUATED_QC_EXCLUDED",
                "failed_hard_checks": [],
                "hard_checks": {},
                "audit_values": {},
                "exclusion_reason":
                    qc_record.get("errors", []),
            }
        )

        continue


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

    metadata = load_json(metadata_path)
    report = load_json(report_path)

    intervention = (
        metadata
        .get("generator_config", {})
        .get("intervention", {})
    )

    runtime = metadata.get(
        "runtime_first",
        {}
    )

    direction = report.get(
        "directional_validation",
        {}
    )

    hard_checks = {
        "artifact_qc_passed": True,
        "formal_plan_binding":
            sample.get(
                "causal_validation_policy_sha256"
            )
            == EXPECTED_POLICY_SHA256,
    }

    audit_values = {}


    if label == "interference":
        expected_target = (
            SEVERITY_VALUES[
                "interference"
            ][severity]
        )

        target_sir = first_number(
            report.get(
                "target_interference_sir_db"
            ),
            intervention.get(
                "target_sir_db"
            ),
        )

        achieved_sir = first_number(
            report.get(
                "mean_achieved_event_sir_db"
            ),
            runtime.get(
                "mean_achieved_event_sir_db"
            ),
        )

        sinr_change = first_number(
            direction.get("sinr_change_db")
        )

        evm_ratio = first_number(
            direction.get("evm_ratio")
        )

        reference_ber = first_number(
            direction.get(
                "reference_ber_pre_ldpc"
            )
        )

        event_ber = first_number(
            direction.get(
                "event_ber_pre_ldpc"
            )
        )

        hard_checks.update(
            {
                "intervention_type":
                    intervention.get("type")
                    == (
                        "independent_cochannel_"
                        "wideband_interference"
                    ),

                "independent_rng":
                    intervention.get(
                        "independent_rng"
                    )
                    is True,

                "target_sir_matches_severity":
                    close_enough(
                        target_sir,
                        expected_target,
                    ),

                "achieved_sir_within_2db":
                    (
                        target_sir is not None
                        and achieved_sir is not None
                        and abs(
                            achieved_sir
                            - target_sir
                        ) <= 2.0
                    ),

                "sinr_decreases_at_least_2db":
                    (
                        sinr_change is not None
                        and sinr_change <= -2.0
                    ),

                "evm_ratio_at_least_1p10":
                    (
                        evm_ratio is not None
                        and evm_ratio >= 1.10
                    ),

                "pre_ldpc_ber_increases":
                    (
                        reference_ber is not None
                        and event_ber is not None
                        and event_ber
                        > reference_ber + 1e-4
                    ),
            }
        )

        audit_values = {
            "target_sir_db": target_sir,
            "achieved_sir_db": achieved_sir,
            "achieved_minus_target_db": (
                achieved_sir - target_sir
                if (
                    achieved_sir is not None
                    and target_sir is not None
                )
                else None
            ),
            "rsrp_shift_db":
                direction.get("rsrp_shift_db"),
            "rssi_increase_db":
                direction.get("rssi_increase_db"),
            "rssi_excess_increase_db":
                direction.get(
                    "rssi_excess_increase_db"
                ),
            "sinr_change_db": sinr_change,
            "evm_ratio": evm_ratio,
            "reference_ber": reference_ber,
            "event_ber": event_ber,
        }


    elif label == "adaptation_mismatch":
        expected_peak = (
            SEVERITY_VALUES[
                "adaptation_mismatch"
            ][severity]
        )

        peak_phase = first_number(
            report.get(
                "peak_phase_error_deg"
            ),
            intervention.get(
                "peak_phase_error_deg"
            ),
        )

        mean_phase = first_number(
            report.get(
                "mean_event_phase_error_deg"
            ),
            runtime.get(
                "mean_event_phase_error_deg"
            ),
        )

        phase_ratio = (
            mean_phase / peak_phase
            if (
                mean_phase is not None
                and peak_phase not in (
                    None,
                    0.0,
                )
            )
            else None
        )

        evm_ratio = first_number(
            direction.get("evm_ratio")
        )

        reference_ber = first_number(
            direction.get(
                "reference_ber_pre_ldpc"
            )
        )

        event_ber = first_number(
            direction.get(
                "event_ber_pre_ldpc"
            )
        )

        reference_bler = first_number(
            direction.get(
                "reference_bler_post_ldpc"
            )
        )

        event_bler = first_number(
            direction.get(
                "event_bler_post_ldpc"
            )
        )

        reference_goodput = first_number(
            direction.get(
                "reference_goodput"
            )
        )

        event_goodput = first_number(
            direction.get(
                "event_goodput"
            )
        )

        receiver_degradation = bool(
            (
                evm_ratio is not None
                and evm_ratio >= 1.20
            )
            or (
                reference_ber is not None
                and event_ber is not None
                and event_ber
                > reference_ber + 1e-4
            )
            or (
                reference_bler is not None
                and event_bler is not None
                and event_bler > reference_bler
            )
        )

        hard_checks.update(
            {
                "intervention_type":
                    intervention.get("type")
                    == (
                        "receiver_adaptation_"
                        "state_phase_mismatch"
                    ),

                "peak_phase_matches_severity":
                    close_enough(
                        peak_phase,
                        expected_peak,
                    ),

                "mean_to_peak_phase_ratio":
                    (
                        phase_ratio is not None
                        and 0.85
                        <= phase_ratio
                        <= 0.90
                    ),

                "propagation_channel_unchanged":
                    intervention.get(
                        "propagation_channel_unchanged"
                    )
                    is True,

                "transmit_configuration_unchanged":
                    intervention.get(
                        "transmit_configuration_unchanged"
                    )
                    is True,

                "receiver_degradation_present":
                    receiver_degradation,

                "goodput_nonimproving":
                    (
                        reference_goodput is not None
                        and event_goodput is not None
                        and event_goodput
                        <= reference_goodput + 1e-7
                    ),
            }
        )

        audit_values = {
            "peak_phase_error_deg":
                peak_phase,
            "mean_event_phase_error_deg":
                mean_phase,
            "mean_to_peak_phase_ratio":
                phase_ratio,
            "rsrp_change_db":
                direction.get("rsrp_change_db"),
            "rssi_change_db":
                direction.get("rssi_change_db"),
            "sinr_change_db":
                direction.get("sinr_change_db"),
            "evm_ratio": evm_ratio,
            "reference_ber": reference_ber,
            "event_ber": event_ber,
            "reference_bler": reference_bler,
            "event_bler": event_bler,
            "reference_goodput":
                reference_goodput,
            "event_goodput":
                event_goodput,
        }


    elif label == "blockage":
        expected_peak = (
            SEVERITY_VALUES[
                "blockage"
            ][severity]
        )

        peak_attenuation = first_number(
            report.get(
                "blockage_peak_attenuation_db"
            ),
            intervention.get(
                "peak_attenuation_db"
            ),
        )

        mean_attenuation = first_number(
            report.get(
                "mean_applied_event_attenuation_db"
            ),
            runtime.get(
                "mean_applied_event_attenuation_db"
            ),
        )

        attenuation_ratio = (
            abs(mean_attenuation)
            / abs(peak_attenuation)
            if (
                mean_attenuation is not None
                and peak_attenuation not in (
                    None,
                    0.0,
                )
            )
            else None
        )

        rsrp_change = first_number(
            direction.get("rsrp_change_db")
        )

        rssi_change = first_number(
            direction.get("rssi_change_db")
        )

        sinr_change = first_number(
            direction.get("sinr_change_db")
        )

        evm_ratio = first_number(
            direction.get("evm_ratio")
        )

        reference_ber = first_number(
            direction.get(
                "reference_ber_pre_ldpc"
            )
        )

        event_ber = first_number(
            direction.get(
                "event_ber_pre_ldpc"
            )
        )

        reference_bler = first_number(
            direction.get(
                "reference_bler_post_ldpc"
            )
        )

        event_bler = first_number(
            direction.get(
                "event_bler_post_ldpc"
            )
        )

        reference_goodput = first_number(
            direction.get(
                "reference_goodput"
            )
        )

        event_goodput = first_number(
            direction.get(
                "event_goodput"
            )
        )

        power_checks = {
            "rsrp_decrease_ge_2db":
                (
                    rsrp_change is not None
                    and rsrp_change <= -2.0
                ),

            "rssi_decrease_ge_0p5db":
                (
                    rssi_change is not None
                    and rssi_change <= -0.5
                ),
        }

        impairment_checks = {
            "sinr_decrease_ge_2db":
                (
                    sinr_change is not None
                    and sinr_change <= -2.0
                ),

            "evm_ratio_ge_1p10":
                (
                    evm_ratio is not None
                    and evm_ratio >= 1.10
                ),

            "pre_ldpc_ber_increase":
                (
                    reference_ber is not None
                    and event_ber is not None
                    and event_ber
                    > reference_ber + 1e-4
                ),

            "post_ldpc_bler_increase":
                (
                    reference_bler is not None
                    and event_bler is not None
                    and event_bler > reference_bler
                ),

            "goodput_decrease":
                (
                    reference_goodput is not None
                    and event_goodput is not None
                    and event_goodput
                    < reference_goodput - 1e-7
                ),
        }

        hard_checks.update(
            {
                "intervention_type":
                    intervention.get("type")
                    == (
                        "time_localized_excess_"
                        "path_loss"
                    ),

                "peak_attenuation_matches_severity":
                    close_enough(
                        peak_attenuation,
                        expected_peak,
                    ),

                "mean_to_peak_attenuation_ratio":
                    (
                        attenuation_ratio is not None
                        and 0.80
                        <= attenuation_ratio
                        <= 1.00
                    ),

                "noise_power_unchanged":
                    intervention.get(
                        "noise_power_unchanged"
                    )
                    is True,

                "power_domain_response":
                    any(power_checks.values()),

                "at_least_two_impairment_responses":
                    sum(
                        impairment_checks.values()
                    ) >= 2,
            }
        )

        audit_values = {
            "peak_attenuation_db":
                peak_attenuation,
            "mean_applied_event_attenuation_db":
                mean_attenuation,
            "mean_to_peak_attenuation_ratio":
                attenuation_ratio,
            "power_checks": power_checks,
            "impairment_checks":
                impairment_checks,
            "rsrp_change_db": rsrp_change,
            "rssi_change_db": rssi_change,
            "sinr_change_db": sinr_change,
            "evm_ratio": evm_ratio,
        }


    elif label in {
        "normal",
        "mobility",
    }:
        hard_checks[
            "locked_v1_validator_pass"
        ] = report.get("status") == "PASS"


    elif label == "nonphysical_goodput":
        consistency_error = first_number(
            report.get(
                "goodput_consistency_error"
            ),
            metadata.get(
                "goodput_consistency_error"
            ),
        )

        hard_checks.update(
            {
                "locked_control_validator_pass":
                    report.get("status")
                    == "PASS",

                "other_nine_kpis_unchanged":
                    report.get(
                        "other_nine_kpis_bitwise_unchanged",
                        metadata.get(
                            "other_nine_kpis_bitwise_unchanged"
                        ),
                    )
                    is True,

                "goodput_consistency_error_ge_0p20":
                    (
                        consistency_error is not None
                        and consistency_error >= 0.20
                    ),
            }
        )

        audit_values = {
            "goodput_consistency_error":
                consistency_error,
            "source_normal_id":
                metadata.get(
                    "source_normal_id"
                ),
        }


    else:
        raise RuntimeError(
            f"Unsupported label: {label}"
        )


    failed_hard_checks = [
        name
        for name, passed
        in hard_checks.items()
        if passed is not True
    ]

    causal_status = (
        "PASS"
        if not failed_hard_checks
        else "FAIL"
    )

    record = {
        "sample_id": sample_id,
        "label": label,
        "severity": severity,
        "channel_model":
            sample["channel_model"],
        "seed": sample["seed"],
        "artifact_qc_status": "PASS",
        "causal_v2_status":
            causal_status,
        "hard_checks":
            hard_checks,
        "failed_hard_checks":
            failed_hard_checks,
        "audit_values":
            audit_values,
        "exclusion_reason": [],
    }

    record_path = (
        RESULT_ROOT
        / sample_id
        / "causal_validation_v2.json"
    )

    record_path.write_text(
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    records.append(record)


evidence_hash_after, evidence_count_after = (
    aggregate_original_evidence(samples)
)

if evidence_hash_after != evidence_hash_before:
    raise RuntimeError(
        "Original Formal Test evidence changed "
        "during causal evaluation."
    )

if evidence_count_after != evidence_count:
    raise RuntimeError(
        "Original evidence file count changed."
    )


evaluated = [
    record
    for record in records
    if record["causal_v2_status"]
    in {"PASS", "FAIL"}
]

causal_passed = [
    record
    for record in evaluated
    if record["causal_v2_status"]
    == "PASS"
]

causal_failed = [
    record
    for record in evaluated
    if record["causal_v2_status"]
    == "FAIL"
]

qc_excluded = [
    record
    for record in records
    if record["causal_v2_status"]
    == "NOT_EVALUATED_QC_EXCLUDED"
]


status_counts = Counter(
    record["causal_v2_status"]
    for record in records
)

failure_check_counts = Counter(
    check
    for record in causal_failed
    for check in record[
        "failed_hard_checks"
    ]
)

label_summary = {}

for label in sorted(
    set(
        record["label"]
        for record in records
    )
):
    label_records = [
        record
        for record in records
        if record["label"] == label
    ]

    label_summary[label] = {
        "planned_count":
            len(label_records),

        "qc_excluded_count":
            sum(
                record["causal_v2_status"]
                == "NOT_EVALUATED_QC_EXCLUDED"
                for record in label_records
            ),

        "evaluated_count":
            sum(
                record["causal_v2_status"]
                in {"PASS", "FAIL"}
                for record in label_records
            ),

        "causal_v2_pass_count":
            sum(
                record["causal_v2_status"]
                == "PASS"
                for record in label_records
            ),

        "causal_v2_fail_count":
            sum(
                record["causal_v2_status"]
                == "FAIL"
                for record in label_records
            ),
    }


output = {
    "schema":
        "phyguard.sionna."
        "formal_test840."
        "causal_validation_v2",

    "status":
        "EVALUATION_COMPLETE",

    "formal_plan_sha256":
        EXPECTED_PLAN_SHA256,

    "causal_validation_policy_sha256":
        EXPECTED_POLICY_SHA256,

    "planned_sample_count":
        840,

    "artifact_qc_eligible_count":
        len(evaluated),

    "artifact_qc_excluded_count":
        len(qc_excluded),

    "artifact_qc_excluded_sample_ids": [
        record["sample_id"]
        for record in qc_excluded
    ],

    "causal_v2_evaluated_count":
        len(evaluated),

    "causal_v2_passed_count":
        len(causal_passed),

    "causal_v2_failed_count":
        len(causal_failed),

    "causal_v2_failed_sample_ids": [
        record["sample_id"]
        for record in causal_failed
    ],

    "status_counts":
        dict(status_counts),

    "failed_hard_check_counts":
        dict(
            failure_check_counts.most_common()
        ),

    "label_summary":
        label_summary,

    "original_evidence": {
        "file_count":
            evidence_count,

        "aggregate_sha256":
            evidence_hash_before,

        "unchanged_during_evaluation":
            True,
    },

    "methodological_boundary": {
        "policy_locked_before_formal_test":
            True,

        "policy_modified_after_results":
            False,

        "simulation_rerun":
            False,

        "failed_samples_replaced":
            False,

        "original_sequences_modified":
            False,

        "original_metadata_modified":
            False,

        "original_legacy_reports_modified":
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


with OUTPUT_CSV.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "sample_id",
            "label",
            "severity",
            "channel_model",
            "seed",
            "artifact_qc_status",
            "causal_v2_status",
            "failed_hard_checks",
        ],
    )

    writer.writeheader()

    for record in records:
        writer.writerow(
            {
                "sample_id":
                    record["sample_id"],
                "label":
                    record["label"],
                "severity":
                    record["severity"],
                "channel_model":
                    record["channel_model"],
                "seed":
                    record["seed"],
                "artifact_qc_status":
                    record[
                        "artifact_qc_status"
                    ],
                "causal_v2_status":
                    record[
                        "causal_v2_status"
                    ],
                "failed_hard_checks":
                    " | ".join(
                        record[
                            "failed_hard_checks"
                        ]
                    ),
            }
        )


print("status: EVALUATION_COMPLETE")
print("planned_sample_count: 840")
print(
    "artifact_qc_eligible_count:",
    len(evaluated),
)
print(
    "artifact_qc_excluded_count:",
    len(qc_excluded),
)
print(
    "artifact_qc_excluded_sample_ids:",
    [
        record["sample_id"]
        for record in qc_excluded
    ],
)
print(
    "causal_v2_evaluated_count:",
    len(evaluated),
)
print(
    "causal_v2_passed_count:",
    len(causal_passed),
)
print(
    "causal_v2_failed_count:",
    len(causal_failed),
)
print(
    "causal_v2_failed_sample_ids:",
    [
        record["sample_id"]
        for record in causal_failed
    ],
)
print(
    "failed_hard_check_counts:",
    dict(
        failure_check_counts.most_common()
    ),
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

print("\nLABEL SUMMARY")

for label, value in label_summary.items():
    print(label, value)


if causal_failed:
    print("\nCAUSAL V2 FAILED RECORDS")

    for record in causal_failed:
        print(
            record["sample_id"],
            "| label=",
            record["label"],
            "| severity=",
            record["severity"],
            "| channel=",
            record["channel_model"],
            "| failed_checks=",
            record["failed_hard_checks"],
        )


print("\nmanifest:", OUTPUT_MANIFEST)
print("csv:", OUTPUT_CSV)

print(
    "\nSIONNA_FORMAL_TEST840_"
    "CAUSAL_V2_EVALUATION_PASS"
)
