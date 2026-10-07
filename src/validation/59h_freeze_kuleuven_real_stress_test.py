import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

STAT_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_preregistered_statistics_v1"
)

STAT_SUMMARY = STAT_ROOT / "summary.json"
ENDPOINT_RESULTS = STAT_ROOT / "endpoint_results.csv"
JOINED_TABLE = STAT_ROOT / "joined_sequence_endpoints.csv"
ACTIVITY_TABLE = STAT_ROOT / "activity_median_endpoints.csv"
BOOTSTRAP_FILE = STAT_ROOT / "primary_cluster_bootstrap.npz"
PERMUTATION_FILE = STAT_ROOT / "activity_permutation_distributions.npz"

STAT_MANIFEST = (
    ROOT
    / "manifests"
    / "kuleuven_preregistered_statistics_v1.json"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_real_stress_test_decision_v1"
)

DECISION_PATH = OUTPUT_ROOT / "decision_record.json"
ARCHIVE_CONTENTS_PATH = OUTPUT_ROOT / "archive_contents.json"

RESULTS_ROOT = ROOT / "results"

ARCHIVE_PATH = (
    RESULTS_ROOT
    / "phyguard_kuleuven_real_stress_test_final.zip"
)

CHECKSUM_PATH = (
    RESULTS_ROOT
    / "phyguard_kuleuven_real_stress_test_final.zip.sha256"
)

EXPECTED_DATASET_SNAPSHOT = (
    "0134ffb74c227ff41e4c10052045308d5"
    "b8cb61414f3ba54324c09815e6f0734"
)

EXPECTED_FEATURE_SNAPSHOT = (
    "7fc2b1b1330c9aa01c94df29afd7d179"
    "b14a6d497d9d1b8a131722461a7dde86"
)

EXPECTED_ENDPOINT_SNAPSHOT = (
    "6bf09fd1d793fc12ffe420870f7443571"
    "6b0e07e9658c66a8feb4f248040a8be"
)


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


required_paths = [
    STAT_SUMMARY,
    ENDPOINT_RESULTS,
    JOINED_TABLE,
    ACTIVITY_TABLE,
    BOOTSTRAP_FILE,
    PERMUTATION_FILE,
    STAT_MANIFEST,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)


summary = load_json(STAT_SUMMARY)
manifest = load_json(STAT_MANIFEST)


if summary.get("status") != "PASS":
    raise RuntimeError(
        "Stage 59G summary is not PASS."
    )

if manifest.get("status") != "PASS":
    raise RuntimeError(
        "Stage 59G manifest is not PASS."
    )

if (
    summary.get("dataset_snapshot_sha256")
    != EXPECTED_DATASET_SNAPSHOT
):
    raise RuntimeError(
        "Dataset snapshot mismatch."
    )

if (
    summary.get("unlabeled_feature_snapshot_sha256")
    != EXPECTED_FEATURE_SNAPSHOT
):
    raise RuntimeError(
        "Feature snapshot mismatch."
    )

if (
    summary.get("unlabeled_endpoint_snapshot_sha256")
    != EXPECTED_ENDPOINT_SNAPSHOT
):
    raise RuntimeError(
        "Endpoint snapshot mismatch."
    )


population = summary["population"]
boundary = summary["methodological_boundary"]
primary = summary["primary_result"]
secondary = summary["secondary_results"]


if population["sequence_count"] != 160:
    raise RuntimeError(
        "Unexpected sequence count."
    )

if population["activity_count"] != 32:
    raise RuntimeError(
        "Unexpected activity count."
    )

if population["sequence_exclusion_count"] != 0:
    raise RuntimeError(
        "Unexpected sequence exclusion."
    )

if boundary.get("all_160_sequences_used") is not True:
    raise RuntimeError(
        "Not all sequences were used."
    )

if boundary.get("samples_excluded") is not False:
    raise RuntimeError(
        "Sample-exclusion boundary failed."
    )

if boundary.get("thresholds_selected") is not False:
    raise RuntimeError(
        "Threshold boundary failed."
    )

if boundary.get("endpoint_signs_reversed") is not False:
    raise RuntimeError(
        "Endpoint-sign boundary failed."
    )

if boundary.get("endpoint_definitions_changed") is not False:
    raise RuntimeError(
        "Endpoint-definition boundary failed."
    )


primary_supported = bool(
    primary["activity_auroc"] > 0.5
    and primary["permutation_p_raw"] < 0.05
)

holm_significant_secondary = [
    row["endpoint"]
    for row in secondary
    if row["permutation_p_holm"] < 0.05
]

exploratory_secondary = [
    {
        "endpoint":
            row["endpoint"],

        "sequence_auroc":
            row["sequence_auroc"],

        "activity_auroc":
            row["activity_auroc"],

        "cliffs_delta":
            row["activity_cliffs_delta"],

        "p_raw":
            row["permutation_p_raw"],

        "p_holm":
            row["permutation_p_holm"],
    }
    for row in secondary
    if row["permutation_p_raw"] < 0.05
]


if primary_supported:
    raise RuntimeError(
        "Unexpected primary-support classification."
    )

if holm_significant_secondary:
    raise RuntimeError(
        "Unexpected Holm-significant secondary endpoint."
    )


decision = {
    "schema":
        "phyguard.kuleuven."
        "real_stress_test_decision.v1",

    "created_at_utc":
        datetime.now(timezone.utc).isoformat(),

    "status":
        "FROZEN",

    "dataset_role":
        (
            "Preregistered mechanism-specific external "
            "real-measurement stress test and boundary case."
        ),

    "dataset_snapshot_sha256":
        EXPECTED_DATASET_SNAPSHOT,

    "unlabeled_feature_snapshot_sha256":
        EXPECTED_FEATURE_SNAPSHOT,

    "unlabeled_endpoint_snapshot_sha256":
        EXPECTED_ENDPOINT_SNAPSHOT,

    "population": population,

    "confirmatory_conclusion": {
        "primary_endpoint":
            primary["endpoint"],

        "primary_supported":
            False,

        "sequence_auroc":
            primary["sequence_auroc"],

        "sequence_auroc_ci95": [
            primary[
                "bootstrap_sequence_auroc_ci_low"
            ],
            primary[
                "bootstrap_sequence_auroc_ci_high"
            ],
        ],

        "sequence_average_precision":
            primary["sequence_average_precision"],

        "activity_auroc":
            primary["activity_auroc"],

        "activity_auroc_ci95": [
            primary[
                "bootstrap_activity_auroc_ci_low"
            ],
            primary[
                "bootstrap_activity_auroc_ci_high"
            ],
        ],

        "activity_cliffs_delta":
            primary["activity_cliffs_delta"],

        "permutation_p_one_sided":
            primary["permutation_p_raw"],

        "positive_activity_median":
            primary["activity_positive_median"],

        "negative_activity_median":
            primary["activity_negative_median"],
    },

    "secondary_conclusion": {
        "holm_significant_endpoint_count":
            0,

        "holm_significant_endpoints":
            [],

        "exploratory_unadjusted_signals":
            exploratory_secondary,

        "interpretation":
            (
                "No secondary endpoint remained significant "
                "after Holm family-wise correction. Any "
                "endpoint with an unadjusted p-value below "
                "0.05 is exploratory only."
            ),
    },

    "scientific_interpretation": {
        "supported":
            (
                "The independently collected 26-GHz "
                "measurements expose a real-world boundary "
                "case in which the preregistered received-"
                "power excursion signature does not "
                "separate link-crossing and pre-blocking "
                "U-turn trajectories."
            ),

        "not_supported":
            (
                "The KU Leuven experiment does not provide "
                "positive confirmation of the preregistered "
                "blockage-evidence endpoint."
            ),

        "reviewer_relevance": {
            "reviewer_3_comment_1":
                (
                    "Strong evidence against a "
                    "self-fulfilling evaluation: an "
                    "independent real-measurement test was "
                    "preregistered, retained, and reported "
                    "despite a null primary result."
                ),

            "reviewer_1_comment_1":
                (
                    "Provides real public measurement "
                    "evaluation but not positive external "
                    "validation; further real-KPI evidence "
                    "is still required."
                ),

            "reviewer_1_comment_6":
                (
                    "Strengthens statistical rigor and "
                    "limitations through preregistration, "
                    "cluster bootstrap, permutation testing, "
                    "and complete sample retention."
                ),
        },
    },

    "manuscript_safe_wording_en":
        (
            "On the preregistered KU Leuven 26-GHz "
            "real-measurement stress test, the primary "
            "received-power excursion endpoint did not "
            "discriminate author-defined link-crossing "
            "trajectories from trajectories that turned "
            "before blocking (sequence-level AUROC 0.473; "
            "activity-level AUROC 0.463; one-sided "
            "permutation p=0.644). No secondary endpoint "
            "remained significant after Holm correction. "
            "We retain this null result as an independent "
            "boundary case, demonstrating that the proposed "
            "physics evidence is not guaranteed by the "
            "label-construction mechanism and does not "
            "transfer uniformly across measurement systems."
        ),

    "prohibited_posthoc_actions": [
        "Reverse the primary endpoint direction",
        "Replace the preregistered primary endpoint",
        "Promote an unadjusted secondary endpoint to confirmatory",
        "Select a threshold using the KU Leuven labels",
        "Exclude activities or repetitions based on results",
        "Relabel trajectories using measured signal values",
        "Report thresholded accuracy from this dataset",
        "Claim positive external validation from this experiment",
    ],

    "next_action":
        (
            "Freeze this experiment as a real-world boundary "
            "case and continue with the independent Zenodo "
            "O-RAN KPI and measured DL-BLER validation."
        ),

    "source_summary_sha256":
        sha256_file(STAT_SUMMARY),

    "source_manifest_sha256":
        sha256_file(STAT_MANIFEST),
}


if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Stage 59H output already exists; "
        "refusing to overwrite."
    )

if ARCHIVE_PATH.exists():
    raise RuntimeError(
        "Final KU Leuven archive already exists."
    )


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


DECISION_PATH.write_text(
    json.dumps(
        decision,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


archive_sources = [
    ROOT
    / "configs"
    / "kuleuven_scenario1_external_validation_protocol_v1.json",

    ROOT
    / "configs"
    / "kuleuven_signal_analysis_protocol_v1.json",

    ROOT
    / "configs"
    / "kuleuven_statistical_execution_contract_v1.json",

    ROOT
    / "artifacts"
    / "kuleuven_scenario1_external_validation_protocol_v1"
    / "scenario1_file_manifest.csv",

    ROOT
    / "artifacts"
    / "kuleuven_scenario1_locked_integrity_v1"
    / "summary.json",

    ROOT
    / "artifacts"
    / "kuleuven_scenario1_locked_integrity_v1"
    / "file_integrity.csv",

    ROOT
    / "artifacts"
    / "kuleuven_unlabeled_features_v1"
    / "summary.json",

    ROOT
    / "artifacts"
    / "kuleuven_unlabeled_features_v1"
    / "feature_names.json",

    ROOT
    / "artifacts"
    / "kuleuven_unlabeled_features_v1"
    / "unlabeled_sequence_lineage.csv",

    ROOT
    / "artifacts"
    / "kuleuven_unlabeled_endpoints_v1"
    / "summary.json",

    ROOT
    / "artifacts"
    / "kuleuven_unlabeled_endpoints_v1"
    / "endpoint_names.json",

    ROOT
    / "artifacts"
    / "kuleuven_unlabeled_endpoints_v1"
    / "unlabeled_sequence_endpoints.csv",

    JOINED_TABLE,
    ACTIVITY_TABLE,
    ENDPOINT_RESULTS,
    BOOTSTRAP_FILE,
    PERMUTATION_FILE,
    STAT_SUMMARY,
    STAT_MANIFEST,
    DECISION_PATH,

    ROOT
    / "scripts"
    / "59d_freeze_kuleuven_signal_analysis_protocol.py",

    ROOT
    / "scripts"
    / "59e_extract_kuleuven_unlabeled_features.py",

    ROOT
    / "scripts"
    / "59f_compute_kuleuven_unlabeled_endpoints.py",

    ROOT
    / "scripts"
    / "59g0_freeze_kuleuven_statistical_execution_contract.py",

    ROOT
    / "scripts"
    / "59g_run_kuleuven_preregistered_statistics.py",
]


for path in archive_sources:
    if not path.exists():
        raise FileNotFoundError(path)


archive_records = [
    {
        "relative_path":
            str(path.relative_to(ROOT)),

        "size_bytes":
            path.stat().st_size,

        "sha256":
            sha256_file(path),
    }
    for path in archive_sources
]


ARCHIVE_CONTENTS_PATH.write_text(
    json.dumps(
        {
            "schema":
                "phyguard.kuleuven."
                "real_stress_test_archive_contents.v1",

            "raw_measurement_binaries_included":
                False,

            "raw_measurement_note":
                (
                    "The 2.048-GB raw binaries are not "
                    "duplicated in this archive. Their "
                    "complete per-file hashes and dataset "
                    "snapshot are preserved in the included "
                    "integrity assets."
                ),

            "file_count":
                len(archive_records),

            "files":
                archive_records,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


archive_sources.append(
    ARCHIVE_CONTENTS_PATH
)


with zipfile.ZipFile(
    ARCHIVE_PATH,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=9,
) as archive:
    for path in sorted(
        archive_sources,
        key=lambda item: str(
            item.relative_to(ROOT)
        ),
    ):
        relative_path = str(
            path.relative_to(ROOT)
        )

        info = zipfile.ZipInfo(
            filename=relative_path,
            date_time=(
                2026,
                7,
                31,
                0,
                0,
                0,
            ),
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


archive_sha256 = sha256_file(
    ARCHIVE_PATH
)


CHECKSUM_PATH.write_text(
    (
        archive_sha256
        + "  "
        + ARCHIVE_PATH.name
        + "\n"
    ),
    encoding="utf-8",
)


print("status: PASS")
print("confirmatory_primary_supported: False")
print(
    "primary_sequence_auroc:",
    format(
        primary["sequence_auroc"],
        ".6f",
    ),
)
print(
    "primary_activity_auroc:",
    format(
        primary["activity_auroc"],
        ".6f",
    ),
)
print(
    "primary_permutation_p:",
    format(
        primary["permutation_p_raw"],
        ".8f",
    ),
)
print("holm_significant_secondary_count: 0")
print(
    "exploratory_unadjusted_signal_count:",
    len(exploratory_secondary),
)
print("all_160_sequences_retained: True")
print("posthoc_endpoint_switching_allowed: False")
print("raw_measurement_binaries_in_archive: False")
print("archive:", ARCHIVE_PATH)
print("archive_sha256:", archive_sha256)
print("checksum_file:", CHECKSUM_PATH)
print(
    "\nKULEUVEN_REAL_STRESS_TEST_FINAL_FREEZE_PASS"
)
