import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

ANALYSIS_PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "kuleuven_signal_analysis_protocol_v1.json"
)

ENDPOINT_SUMMARY_PATH = (
    ROOT
    / "artifacts"
    / "kuleuven_unlabeled_endpoints_v1"
    / "summary.json"
)

ENDPOINT_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_unlabeled_endpoints_v1.json"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_statistical_execution_contract_v1"
)

CONTRACT_PATH = (
    ROOT
    / "configs"
    / "kuleuven_statistical_execution_contract_v1.json"
)

SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_statistical_execution_contract_v1.json"
)

EXPECTED_DATASET_SNAPSHOT_SHA256 = (
    "0134ffb74c227ff41e4c10052045308d5"
    "b8cb61414f3ba54324c09815e6f0734"
)

EXPECTED_FEATURE_SNAPSHOT_SHA256 = (
    "7fc2b1b1330c9aa01c94df29afd7d179"
    "b14a6d497d9d1b8a131722461a7dde86"
)

EXPECTED_ENDPOINT_SNAPSHOT_SHA256 = (
    "6bf09fd1d793fc12ffe420870f7443571"
    "6b0e07e9658c66a8feb4f248040a8be"
)

PRIMARY_ENDPOINT = (
    "primary_best_rx_power_excursion_db"
)

SECONDARY_ENDPOINTS = [
    "deep_fade_occupancy_3db",
    "maximum_contiguous_deep_fade_fraction",
    "rx_beam_entropy_excursion",
    "temporal_coherence_excursion",
    "frequency_selectivity_excursion_db",
    "mean_rx_beam_switch_rate",
    "power_volatility_q95_db",
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
        path.read_text(encoding="utf-8")
    )


for path in [
    ANALYSIS_PROTOCOL_PATH,
    ENDPOINT_SUMMARY_PATH,
    ENDPOINT_MANIFEST_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(path)


analysis_protocol = load_json(
    ANALYSIS_PROTOCOL_PATH
)

endpoint_summary = load_json(
    ENDPOINT_SUMMARY_PATH
)

endpoint_manifest = load_json(
    ENDPOINT_MANIFEST_PATH
)


if (
    analysis_protocol.get("status")
    != "LOCKED_BEFORE_SIGNAL_VALUE_ANALYSIS"
):
    raise RuntimeError(
        "Unexpected Stage 59D protocol status."
    )


if endpoint_summary.get("status") != "PASS":
    raise RuntimeError(
        "Stage 59F summary is not PASS."
    )


if endpoint_manifest.get("status") != "PASS":
    raise RuntimeError(
        "Stage 59F manifest is not PASS."
    )


if (
    endpoint_summary.get(
        "dataset_snapshot_sha256"
    )
    != EXPECTED_DATASET_SNAPSHOT_SHA256
):
    raise RuntimeError(
        "Dataset snapshot mismatch."
    )


if (
    endpoint_summary.get(
        "unlabeled_feature_snapshot_sha256"
    )
    != EXPECTED_FEATURE_SNAPSHOT_SHA256
):
    raise RuntimeError(
        "Feature snapshot mismatch."
    )


if (
    endpoint_summary.get(
        "unlabeled_endpoint_snapshot_sha256"
    )
    != EXPECTED_ENDPOINT_SNAPSHOT_SHA256
):
    raise RuntimeError(
        "Endpoint snapshot mismatch."
    )


boundary = endpoint_summary.get(
    "methodological_boundary",
    {},
)


for field in [
    "labels_joined",
    "label_manifest_opened",
    "activity_numbers_parsed",
    "classification_performance_computed",
    "statistical_tests_computed",
    "thresholds_selected",
]:
    if boundary.get(field) is not False:
        raise RuntimeError(
            f"Stage 59F boundary failed: {field}"
        )


if OUTPUT_ROOT.exists() or CONTRACT_PATH.exists():
    raise RuntimeError(
        "Stage 59G0 output already exists; "
        "refusing to overwrite it."
    )


contract = {
    "schema":
        "phyguard.kuleuven."
        "statistical_execution_contract.v1",

    "created_at_utc":
        datetime.now(timezone.utc).isoformat(),

    "status":
        "LOCKED_BEFORE_LABEL_JOIN_AND_STATISTICAL_ANALYSIS",

    "dataset_snapshot_sha256":
        EXPECTED_DATASET_SNAPSHOT_SHA256,

    "unlabeled_feature_snapshot_sha256":
        EXPECTED_FEATURE_SNAPSHOT_SHA256,

    "unlabeled_endpoint_snapshot_sha256":
        EXPECTED_ENDPOINT_SNAPSHOT_SHA256,

    "population": {
        "sequence_count":
            160,

        "activity_cluster_count":
            32,

        "positive_activity_count":
            20,

        "negative_activity_count":
            12,

        "repetitions_per_activity":
            5,

        "positive_sequence_count":
            100,

        "negative_sequence_count":
            60,

        "positive_prevalence":
            0.625,
    },

    "primary_endpoint": {
        "name":
            PRIMARY_ENDPOINT,

        "direction":
            "higher_values_indicate_blockage_event",

        "confirmatory":
            True,
    },

    "secondary_endpoints": {
        "names":
            SECONDARY_ENDPOINTS,

        "confirmatory":
            False,

        "direction":
            (
                "No directional alternative was fixed in "
                "Stage 59D; therefore all secondary "
                "permutation tests are two-sided."
            ),

        "multiplicity_adjustment":
            "Holm across the seven secondary endpoints",
    },

    "activity_aggregation": {
        "method":
            "median across the five repetitions",

        "grouping_key":
            "activity",

        "aggregation_performed_after_label_join":
            True,

        "missing_repetition_policy":
            "analysis failure; no imputation",
    },

    "reported_discrimination_metrics": {
        "sequence_level": [
            "AUROC",
            "average_precision",
        ],

        "activity_level": [
            (
                "AUROC computed from the 32 "
                "activity-level median endpoint values"
            ),
        ],

        "score_orientation":
            (
                "Raw endpoint orientation is preserved; "
                "scores are never multiplied by minus one "
                "after labels are observed."
            ),

        "average_precision_baseline":
            0.625,

        "thresholded_metrics":
            False,
    },

    "bootstrap_contract": {
        "method":
            "stratified activity-cluster bootstrap",

        "resamples":
            10_000,

        "confidence_level":
            0.95,

        "seed":
            59031,

        "interval":
            "percentile interval at 2.5% and 97.5%",

        "sampling": {
            "positive_clusters":
                (
                    "Sample 20 positive activities with "
                    "replacement from the 20 positive "
                    "activities."
                ),

            "negative_clusters":
                (
                    "Sample 12 negative activities with "
                    "replacement from the 12 negative "
                    "activities."
                ),

            "sequence_metrics":
                (
                    "Retain all five repetitions for every "
                    "sampled activity-cluster occurrence."
                ),

            "activity_metric":
                (
                    "Use the median endpoint value of each "
                    "sampled activity-cluster occurrence."
                ),

            "duplicate_cluster_occurrences":
                (
                    "Duplicated bootstrap clusters are "
                    "treated as separate resampled cluster "
                    "occurrences."
                ),
        },

        "metrics": [
            "sequence_level_AUROC",
            "sequence_level_average_precision",
            "activity_level_AUROC",
        ],

        "invalid_resample_policy":
            (
                "Not expected under stratified sampling; "
                "any invalid resample causes analysis failure."
            ),
    },

    "primary_permutation_test": {
        "endpoint":
            PRIMARY_ENDPOINT,

        "analysis_unit":
            "activity-level median",

        "test_statistic":
            "activity-level AUROC",

        "permutations":
            100_000,

        "seed":
            59032,

        "alternative":
            "positive_greater",

        "label_permutation":
            (
                "Randomly assign exactly 20 of the 32 "
                "activity clusters to the positive class "
                "and the remaining 12 to the negative class."
            ),

        "p_value_formula":
            (
                "(1 + count(permuted_AUROC >= "
                "observed_AUROC)) / "
                "(number_of_permutations + 1)"
            ),
    },

    "secondary_permutation_tests": {
        "analysis_unit":
            "activity-level median",

        "test_statistic":
            "absolute(activity-level AUROC - 0.5)",

        "permutations_per_endpoint":
            100_000,

        "seed_rule":
            (
                "59040 plus the zero-based secondary "
                "endpoint index"
            ),

        "alternative":
            "two_sided",

        "label_permutation":
            (
                "Randomly assign exactly 20 positive and "
                "12 negative activity labels."
            ),

        "raw_p_value_formula":
            (
                "(1 + count(abs(permuted_AUROC - 0.5) "
                ">= abs(observed_AUROC - 0.5))) / "
                "(number_of_permutations + 1)"
            ),

        "adjustment":
            "Holm family-wise correction over seven tests",
    },

    "effect_size": {
        "name":
            "Cliff's delta",

        "analysis_unit":
            "activity-level median",

        "orientation":
            "positive activities minus negative activities",

        "formula":
            (
                "(number of positive-negative pairs with "
                "positive value greater than negative value "
                "minus number with lower value) divided by "
                "20 times 12"
            ),

        "ties":
            "zero contribution",
    },

    "descriptive_statistics": {
        "sequence_level_by_class": [
            "count",
            "median",
            "interquartile range",
        ],

        "activity_level_by_class": [
            "count",
            "median",
            "interquartile range",
        ],

        "printed_before_analysis_completion":
            False,
    },

    "execution_order": [
        (
            "Verify frozen dataset, feature, endpoint, "
            "protocol, and contract hashes."
        ),
        (
            "Read the frozen endpoint table and frozen "
            "opaque lineage table."
        ),
        (
            "Read the Stage 59A author-defined label "
            "manifest for the first time."
        ),
        (
            "Join labels by filename through the frozen "
            "lineage table."
        ),
        (
            "Verify 160 one-to-one joins, 32 activities, "
            "five repetitions per activity, 100 positive "
            "and 60 negative sequences."
        ),
        (
            "Compute all fixed descriptive and "
            "discrimination metrics."
        ),
        (
            "Run fixed bootstrap and permutation analyses."
        ),
        (
            "Write all results without endpoint selection, "
            "threshold fitting, or sample exclusion."
        ),
    ],

    "failure_conditions": [
        "Any source hash mismatch",
        "Any duplicate or missing join",
        "Any non-finite endpoint",
        "Any activity without five repetitions",
        "Any deviation from 20 positive and 12 negative activities",
        "Any sequence exclusion",
        "Any threshold fitting",
        "Any endpoint sign reversal",
    ],

    "methodological_boundary": {
        "label_manifest_opened":
            False,

        "labels_joined":
            False,

        "endpoint_values_printed":
            False,

        "statistical_results_computed":
            False,

        "models_trained":
            False,

        "thresholds_selected":
            False,

        "samples_excluded":
            False,
    },

    "source_assets": {
        "analysis_protocol_path":
            str(ANALYSIS_PROTOCOL_PATH),

        "analysis_protocol_sha256":
            sha256_file(
                ANALYSIS_PROTOCOL_PATH
            ),

        "endpoint_summary_path":
            str(ENDPOINT_SUMMARY_PATH),

        "endpoint_summary_sha256":
            sha256_file(
                ENDPOINT_SUMMARY_PATH
            ),

        "endpoint_manifest_path":
            str(ENDPOINT_MANIFEST_PATH),

        "endpoint_manifest_sha256":
            sha256_file(
                ENDPOINT_MANIFEST_PATH
            ),
    },
}


CONTRACT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

CONTRACT_PATH.write_text(
    json.dumps(
        contract,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)


summary = {
    "schema":
        "phyguard.kuleuven."
        "statistical_execution_contract_summary.v1",

    "status":
        "PASS",

    "contract_status":
        contract["status"],

    "dataset_snapshot_sha256":
        EXPECTED_DATASET_SNAPSHOT_SHA256,

    "unlabeled_feature_snapshot_sha256":
        EXPECTED_FEATURE_SNAPSHOT_SHA256,

    "unlabeled_endpoint_snapshot_sha256":
        EXPECTED_ENDPOINT_SNAPSHOT_SHA256,

    "primary_endpoint":
        PRIMARY_ENDPOINT,

    "primary_permutation_statistic":
        "activity-level AUROC",

    "secondary_endpoint_count":
        len(SECONDARY_ENDPOINTS),

    "bootstrap_resamples":
        10_000,

    "primary_permutations":
        100_000,

    "secondary_permutations_per_endpoint":
        100_000,

    "label_manifest_opened":
        False,

    "labels_joined":
        False,

    "statistical_results_computed":
        False,

    "models_trained":
        False,

    "thresholds_selected":
        False,

    "contract_path":
        str(CONTRACT_PATH),

    "contract_sha256":
        sha256_file(CONTRACT_PATH),
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
        "statistical_execution_contract_manifest.v1",

    "status":
        "PASS",

    "contract_sha256":
        sha256_file(CONTRACT_PATH),

    "summary_sha256":
        sha256_file(SUMMARY_PATH),

    "dataset_snapshot_sha256":
        EXPECTED_DATASET_SNAPSHOT_SHA256,

    "unlabeled_endpoint_snapshot_sha256":
        EXPECTED_ENDPOINT_SNAPSHOT_SHA256,

    "methodological_boundary":
        contract["methodological_boundary"],
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
    "contract_status:",
    contract["status"],
)
print(
    "dataset_snapshot_sha256:",
    EXPECTED_DATASET_SNAPSHOT_SHA256,
)
print(
    "unlabeled_feature_snapshot_sha256:",
    EXPECTED_FEATURE_SNAPSHOT_SHA256,
)
print(
    "unlabeled_endpoint_snapshot_sha256:",
    EXPECTED_ENDPOINT_SNAPSHOT_SHA256,
)
print("primary_endpoint:", PRIMARY_ENDPOINT)
print(
    "primary_permutation_statistic:",
    "activity-level AUROC",
)
print(
    "secondary_endpoint_count:",
    len(SECONDARY_ENDPOINTS),
)
print("bootstrap_resamples: 10000")
print("primary_permutations: 100000")
print(
    "secondary_permutations_per_endpoint:",
    100000,
)
print("label_manifest_opened: False")
print("labels_joined: False")
print("statistical_results_computed: False")
print("models_trained: False")
print("thresholds_selected: False")
print("contract:", CONTRACT_PATH)
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)
print(
    "\nKULEUVEN_STATISTICAL_"
    "EXECUTION_CONTRACT_V1_PASS"
)
