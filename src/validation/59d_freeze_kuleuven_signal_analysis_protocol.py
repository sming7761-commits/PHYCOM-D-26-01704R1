import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

INTEGRITY_SUMMARY_PATH = (
    ROOT
    / "artifacts"
    / "kuleuven_scenario1_locked_integrity_v1"
    / "summary.json"
)

INTEGRITY_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_scenario1_locked_integrity_v1.json"
)

SEQUENCE_PROTOCOL_PATH = (
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

CONVERTER_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_official_converter_audit_v1.json"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_signal_analysis_protocol_v1"
)

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "kuleuven_signal_analysis_protocol_v1.json"
)

SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_signal_analysis_protocol_v1.json"
)

EXPECTED_DATASET_SNAPSHOT_SHA256 = (
    "0134ffb74c227ff41e4c10052045308d5"
    "b8cb61414f3ba54324c09815e6f0734"
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


required_paths = [
    INTEGRITY_SUMMARY_PATH,
    INTEGRITY_MANIFEST_PATH,
    SEQUENCE_PROTOCOL_PATH,
    FILE_MANIFEST_PATH,
    CONVERTER_MANIFEST_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)


integrity_summary = load_json(
    INTEGRITY_SUMMARY_PATH
)

integrity_manifest = load_json(
    INTEGRITY_MANIFEST_PATH
)

sequence_protocol = load_json(
    SEQUENCE_PROTOCOL_PATH
)

converter_manifest = load_json(
    CONVERTER_MANIFEST_PATH
)


if integrity_summary.get("status") != "PASS":
    raise RuntimeError(
        "Stage 59C integrity summary is not PASS."
    )

if integrity_manifest.get("status") != "PASS":
    raise RuntimeError(
        "Stage 59C integrity manifest is not PASS."
    )

if converter_manifest.get("status") != "PASS":
    raise RuntimeError(
        "Official converter manifest is not PASS."
    )

if (
    integrity_summary.get(
        "dataset_snapshot_sha256"
    )
    != EXPECTED_DATASET_SNAPSHOT_SHA256
):
    raise RuntimeError(
        "Dataset snapshot SHA256 does not match "
        "the locked Stage 59C snapshot."
    )

if integrity_summary.get("verified_file_count") != 160:
    raise RuntimeError(
        "Expected 160 verified files."
    )

if (
    sequence_protocol.get("status")
    != "LOCKED_BEFORE_FULL_SCENARIO1_DOWNLOAD"
):
    raise RuntimeError(
        "Unexpected sequence-protocol status."
    )


if OUTPUT_ROOT.exists() or PROTOCOL_PATH.exists():
    raise RuntimeError(
        "Stage 59D output already exists; "
        "refusing to overwrite it."
    )


protocol = {
    "schema":
        "phyguard.kuleuven."
        "signal_analysis_protocol.v1",

    "created_at_utc":
        datetime.now(timezone.utc).isoformat(),

    "status":
        "LOCKED_BEFORE_SIGNAL_VALUE_ANALYSIS",

    "dataset_snapshot_sha256":
        EXPECTED_DATASET_SNAPSHOT_SHA256,

    "scientific_role": {
        "role":
            (
                "Mechanism-specific external validation of "
                "physics-grounded blockage evidence on real "
                "26-GHz hardware measurements."
            ),

        "supported_claim":
            (
                "Author-defined link-crossing trajectories "
                "produce stronger blockage-related channel "
                "evidence than trajectories that explicitly "
                "turn before blocking."
            ),

        "unsupported_claims": [
            "Full four-class external validation",
            "Per-symbol blockage localization",
            "Continuous blockage throughout each sequence",
            "End-to-end anomaly detection",
            "Universal superiority over all temporal models",
        ],
    },

    "raw_decoder": {
        "file_count":
            160,

        "bytes_per_file":
            12_800_000,

        "complex_samples_per_file":
            3_200_000,

        "storage_dtype":
            "big-endian signed int16 pairs",

        "pair_order":
            [
                "imaginary",
                "real",
            ],

        "scale_divisor":
            16384.0,

        "canonical_shape":
            [
                2000,
                100,
                2,
                8,
            ],

        "canonical_axes":
            [
                "symbol",
                "subcarrier",
                "tx_layer",
                "rx_beam",
            ],

        "official_flat_index_formula":
            (
                "symbol*1600 + subcarrier*16 + "
                "(rx_beam//4)*8 + tx_layer*4 + "
                "(rx_beam%4)"
            ),

        "main_communication_tx_layer":
            0,

        "secondary_tx_layer":
            1,
    },

    "temporal_adapter": {
        "input_symbol_count":
            2000,

        "output_time_step_count":
            80,

        "symbols_per_time_step":
            25,

        "capture_interval_per_symbol_ms":
            5.0,

        "time_step_duration_ms":
            125.0,

        "sequence_duration_seconds":
            10.0,

        "overlap":
            0,

        "padding":
            "none",

        "resampling":
            "none",
    },

    "numerical_constants": {
        "power_epsilon":
            1e-12,

        "normalization_epsilon":
            1e-12,

        "entropy_log_base":
            "natural",

        "deep_fade_threshold_db":
            3.0,

        "quantile_method":
            "numpy_linear",
    },

    "multivariate_feature_contract": [
        {
            "index": 0,
            "name":
                "primary_best_rx_power_db",

            "formula":
                (
                    "10*log10(max over rx_beam of mean "
                    "squared magnitude over the 25 symbols "
                    "and 100 subcarriers for tx_layer 0)"
                ),
        },
        {
            "index": 1,
            "name":
                "primary_total_power_db",

            "formula":
                (
                    "10*log10(mean squared magnitude over "
                    "symbols, subcarriers, and rx beams "
                    "for tx_layer 0)"
                ),
        },
        {
            "index": 2,
            "name":
                "secondary_total_power_db",

            "formula":
                (
                    "10*log10(mean squared magnitude over "
                    "symbols, subcarriers, and rx beams "
                    "for tx_layer 1)"
                ),
        },
        {
            "index": 3,
            "name":
                "tx_layer_power_gap_db",

            "formula":
                (
                    "primary_total_power_db minus "
                    "secondary_total_power_db"
                ),
        },
        {
            "index": 4,
            "name":
                "primary_rx_beam_entropy",

            "formula":
                (
                    "Normalized entropy of the eight "
                    "tx_layer-0 receive-beam power shares"
                ),

            "range":
                [0.0, 1.0],
        },
        {
            "index": 5,
            "name":
                "primary_best_second_beam_margin_db",

            "formula":
                (
                    "Power in dB of the strongest receive "
                    "beam minus the second-strongest beam "
                    "for tx_layer 0"
                ),
        },
        {
            "index": 6,
            "name":
                "primary_frequency_selectivity_db",

            "formula":
                (
                    "Standard deviation across 100 "
                    "subcarriers of subcarrier power in dB, "
                    "averaged over symbols and receive beams"
                ),
        },
        {
            "index": 7,
            "name":
                "primary_temporal_complex_coherence",

            "formula":
                (
                    "Mean absolute normalized complex "
                    "correlation between adjacent symbols, "
                    "computed across subcarriers and receive "
                    "beams for tx_layer 0"
                ),

            "range":
                [0.0, 1.0],
        },
        {
            "index": 8,
            "name":
                "primary_rx_beam_switch_rate",

            "formula":
                (
                    "Fraction of the 24 adjacent-symbol "
                    "transitions in which the strongest "
                    "receive beam changes for tx_layer 0"
                ),

            "range":
                [0.0, 1.0],
        },
        {
            "index": 9,
            "name":
                "primary_symbol_power_volatility_db",

            "formula":
                (
                    "Standard deviation within the time "
                    "step of symbol-level strongest-beam "
                    "power in dB for tx_layer 0"
                ),
        },
    ],

    "sequence_level_endpoints": {
        "primary_endpoint": {
            "name":
                "primary_best_rx_power_excursion_db",

            "formula":
                (
                    "95th percentile minus 5th percentile "
                    "of primary_best_rx_power_db across "
                    "the 80 time steps"
                ),

            "directional_hypothesis":
                (
                    "BLOCKAGE_EVENT_PRESENT is greater than "
                    "NO_LINK_BLOCKAGE"
                ),

            "selection_basis":
                (
                    "Pre-specified physical expectation "
                    "that link crossing produces a deeper "
                    "within-sequence received-power change."
                ),
        },

        "secondary_endpoints": [
            {
                "name":
                    "deep_fade_occupancy_3db",

                "formula":
                    (
                        "Fraction of time steps whose "
                        "primary_best_rx_power_db is at "
                        "least 3 dB below its sequence-level "
                        "95th percentile"
                    ),
            },
            {
                "name":
                    "maximum_contiguous_deep_fade_fraction",

                "formula":
                    (
                        "Longest contiguous run satisfying "
                        "the fixed 3-dB deep-fade condition "
                        "divided by 80"
                    ),
            },
            {
                "name":
                    "rx_beam_entropy_excursion",

                "formula":
                    (
                        "95th minus 5th percentile of "
                        "primary_rx_beam_entropy"
                    ),
            },
            {
                "name":
                    "temporal_coherence_excursion",

                "formula":
                    (
                        "95th minus 5th percentile of "
                        "primary_temporal_complex_coherence"
                    ),
            },
            {
                "name":
                    "frequency_selectivity_excursion_db",

                "formula":
                    (
                        "95th minus 5th percentile of "
                        "primary_frequency_selectivity_db"
                    ),
            },
            {
                "name":
                    "mean_rx_beam_switch_rate",

                "formula":
                    (
                        "Mean primary_rx_beam_switch_rate "
                        "across 80 time steps"
                    ),
            },
            {
                "name":
                    "power_volatility_q95_db",

                "formula":
                    (
                        "95th percentile of "
                        "primary_symbol_power_volatility_db"
                    ),
            },
        ],
    },

    "anti_leakage_execution_order": [
        (
            "Decode and extract all 160 unlabeled "
            "80-by-10 sequences."
        ),
        (
            "Perform only mechanical QC: file size, "
            "shape, finite values, and feature ranges."
        ),
        (
            "Freeze the complete unlabeled feature table "
            "and its SHA256 manifest."
        ),
        (
            "Compute sequence-level endpoints without "
            "accessing labels."
        ),
        (
            "Freeze the unlabeled endpoint table and "
            "its SHA256 manifest."
        ),
        (
            "Only then join the author-defined labels "
            "from the locked Stage 59A manifest."
        ),
        (
            "Run the pre-specified statistical analysis "
            "without threshold fitting or endpoint changes."
        ),
    ],

    "quality_control": {
        "allowed_exclusions": [
            "Missing file",
            "Incorrect byte count",
            "Incorrect decoded sample count",
            "Non-finite decoded values",
            "Non-finite derived feature values",
        ],

        "disallowed_exclusions": [
            "Unexpected signal appearance",
            "Weak blockage effect",
            "Model disagreement",
            "Poor classification result",
            "Activity-specific performance",
        ],

        "label_blind_qc":
            True,
    },

    "statistical_analysis": {
        "unit_of_observation":
            "ten-second measurement sequence",

        "cluster_variable":
            "activity",

        "cluster_count":
            32,

        "repetitions_per_cluster":
            5,

        "primary_discrimination_metrics": [
            "sequence-level AUROC",
            "sequence-level average precision",
            (
                "activity-level AUROC using the median "
                "of five repetitions"
            ),
        ],

        "confidence_intervals": {
            "method":
                "activity-cluster bootstrap",

            "resamples":
                10_000,

            "confidence_level":
                0.95,

            "seed":
                59031,
        },

        "primary_hypothesis_test": {
            "method":
                (
                    "Monte Carlo permutation of the "
                    "20 positive and 12 negative activity "
                    "labels across 32 activity clusters"
                ),

            "permutations":
                100_000,

            "alternative":
                "positive_greater",

            "seed":
                59032,
        },

        "effect_size":
            (
                "Cliff's delta computed from the "
                "32 activity-level median endpoint values"
            ),

        "secondary_endpoint_adjustment":
            "Holm family-wise correction",

        "thresholded_accuracy_reported":
            False,

        "threshold_selected":
            False,

        "model_fitting":
            False,
    },

    "methodological_boundary": {
        "signal_values_read":
            False,

        "features_computed":
            False,

        "labels_joined":
            False,

        "models_trained":
            False,

        "predictions_computed":
            False,

        "classification_performance_computed":
            False,

        "thresholds_selected":
            False,

        "endpoint_definitions_changed_after_signal_read":
            False,

        "existing_sionna_assets_modified":
            False,

        "existing_locked_models_modified":
            False,
    },

    "source_assets": {
        "integrity_summary_path":
            str(INTEGRITY_SUMMARY_PATH),

        "integrity_summary_sha256":
            sha256_file(
                INTEGRITY_SUMMARY_PATH
            ),

        "integrity_manifest_path":
            str(INTEGRITY_MANIFEST_PATH),

        "integrity_manifest_sha256":
            sha256_file(
                INTEGRITY_MANIFEST_PATH
            ),

        "sequence_protocol_path":
            str(SEQUENCE_PROTOCOL_PATH),

        "sequence_protocol_sha256":
            sha256_file(
                SEQUENCE_PROTOCOL_PATH
            ),

        "file_manifest_path":
            str(FILE_MANIFEST_PATH),

        "file_manifest_sha256":
            sha256_file(
                FILE_MANIFEST_PATH
            ),

        "converter_manifest_path":
            str(CONVERTER_MANIFEST_PATH),

        "converter_manifest_sha256":
            sha256_file(
                CONVERTER_MANIFEST_PATH
            ),
    },
}


PROTOCOL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

PROTOCOL_PATH.write_text(
    json.dumps(
        protocol,
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
        "signal_analysis_protocol_summary.v1",

    "status":
        "PASS",

    "protocol_status":
        protocol["status"],

    "dataset_snapshot_sha256":
        EXPECTED_DATASET_SNAPSHOT_SHA256,

    "sequence_count":
        160,

    "time_steps_per_sequence":
        80,

    "feature_count":
        10,

    "primary_endpoint":
        (
            "primary_best_rx_power_excursion_db"
        ),

    "activity_cluster_count":
        32,

    "bootstrap_resamples":
        10_000,

    "permutation_count":
        100_000,

    "signal_values_read":
        False,

    "features_computed":
        False,

    "labels_joined":
        False,

    "models_trained":
        False,

    "performance_computed":
        False,

    "thresholds_selected":
        False,

    "protocol_path":
        str(PROTOCOL_PATH),

    "protocol_sha256":
        sha256_file(PROTOCOL_PATH),
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
        "signal_analysis_protocol_manifest.v1",

    "status":
        "PASS",

    "protocol_sha256":
        sha256_file(PROTOCOL_PATH),

    "summary_sha256":
        sha256_file(SUMMARY_PATH),

    "dataset_snapshot_sha256":
        EXPECTED_DATASET_SNAPSHOT_SHA256,

    "methodological_boundary":
        protocol["methodological_boundary"],
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
    "protocol_status:",
    protocol["status"],
)
print(
    "dataset_snapshot_sha256:",
    EXPECTED_DATASET_SNAPSHOT_SHA256,
)
print("sequence_count: 160")
print("time_steps_per_sequence: 80")
print("feature_count: 10")
print(
    "primary_endpoint:",
    "primary_best_rx_power_excursion_db",
)
print("activity_cluster_count: 32")
print("bootstrap_resamples: 10000")
print("permutation_count: 100000")
print("signal_values_read: False")
print("features_computed: False")
print("labels_joined: False")
print("models_trained: False")
print("performance_computed: False")
print("thresholds_selected: False")
print("protocol:", PROTOCOL_PATH)
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)
print(
    "\nKULEUVEN_SIGNAL_ANALYSIS_"
    "PROTOCOL_V1_PASS"
)
