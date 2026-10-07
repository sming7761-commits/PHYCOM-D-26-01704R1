import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

METADATA_SUMMARY_PATH = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_metadata_contract_v1"
    / "summary.json"
)

METADATA_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "cross_system_reliability_metadata_contract_v1.json"
)

SIONNA_IDS_PATH = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_metadata_contract_v1"
    / "sionna_analysis825_ids.txt"
)

ZENODO_INVENTORY_PATH = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_asset_inventory_v2"
    / "zenodo_canonical_inventory.csv"
)

FORMULA_CONTRACT_PATH = (
    ROOT
    / "artifacts"
    / "reconstructed_48bit_per_formula_contract_v1"
    / "summary.json"
)

SOURCE_TELEMETRY_PATH = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
    / "data"
    / "full"
    / "telemetry.npz"
)

SOURCE_METADATA_PATH = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
    / "data"
    / "full"
    / "metadata.csv"
)

SIONNA_DATA_ROOT = (
    ROOT
    / "data"
    / "sionna_formal_test840"
)

OUTPUT_CONFIG_PATH = (
    ROOT
    / "configs"
    / "cross_system_reliability_statistical_protocol_v1.json"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_statistical_protocol_v1"
)

SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "cross_system_reliability_statistical_protocol_v1.json"
)


EXPECTED_MEMBERSHIP_SHA256 = (
    "a8aa74f95df897a57c82f60ea0d2f233"
    "53eaa023ec0446b079dd8a88d7de462a"
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
        path.read_text(
            encoding="utf-8"
        )
    )


def load_csv(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        return list(
            csv.DictReader(stream)
        )


required_paths = [
    METADATA_SUMMARY_PATH,
    METADATA_MANIFEST_PATH,
    SIONNA_IDS_PATH,
    ZENODO_INVENTORY_PATH,
    FORMULA_CONTRACT_PATH,
    SOURCE_TELEMETRY_PATH,
    SOURCE_METADATA_PATH,
    SIONNA_DATA_ROOT,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)


metadata_summary = load_json(
    METADATA_SUMMARY_PATH
)

metadata_manifest = load_json(
    METADATA_MANIFEST_PATH
)

formula_contract = load_json(
    FORMULA_CONTRACT_PATH
)

zenodo_rows = load_csv(
    ZENODO_INVENTORY_PATH
)


if (
    metadata_summary.get("status")
    != "LOCKED_BEFORE_RELIABILITY_VALUE_ANALYSIS"
):
    raise RuntimeError(
        "Stage 60E0 metadata contract has unexpected status."
    )

if (
    metadata_manifest.get("status")
    != "LOCKED_BEFORE_RELIABILITY_VALUE_ANALYSIS"
):
    raise RuntimeError(
        "Stage 60E0 manifest has unexpected status."
    )

if formula_contract.get("status") != "PASS":
    raise RuntimeError(
        "Stage 60C formula contract is not PASS."
    )

if (
    formula_contract.get("packet_surrogate_bits")
    != 48
):
    raise RuntimeError(
        "Unexpected packet-surrogate length."
    )

if (
    sha256_file(SIONNA_IDS_PATH)
    != EXPECTED_MEMBERSHIP_SHA256
):
    raise RuntimeError(
        "Frozen Sionna-825 membership hash mismatch."
    )

if len(zenodo_rows) != 10:
    raise RuntimeError(
        "Expected exactly 10 Zenodo files."
    )

if metadata_summary["source"]["shape"] != [
    1080,
    80,
    10,
]:
    raise RuntimeError(
        "Unexpected source telemetry shape."
    )

if metadata_summary["sionna"]["analysis_count"] != 825:
    raise RuntimeError(
        "Unexpected Sionna analysis count."
    )

if metadata_summary["sionna"]["physical_count"] != 706:
    raise RuntimeError(
        "Unexpected Sionna physical count."
    )

if metadata_summary["sionna"]["control_count"] != 119:
    raise RuntimeError(
        "Unexpected Sionna control count."
    )

if metadata_summary["zenodo"]["value_rows_opened"] is not False:
    raise RuntimeError(
        "Zenodo rows were already marked as opened."
    )


if OUTPUT_CONFIG_PATH.exists():
    raise RuntimeError(
        "Stage 60E1 protocol config already exists."
    )

if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Stage 60E1 output already exists."
    )


protocol = {
    "schema":
        "phyguard.cross_system_reliability_"
        "statistical_protocol.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "LOCKED_BEFORE_RELIABILITY_STATISTICAL_EXECUTION",

    "scientific_question":
        (
            "Does increasing SINR consistently correspond "
            "to decreasing link-error probability across "
            "the revision-time source reconstruction, the "
            "independent coded Sionna link, and public "
            "real O-RAN measurements?"
        ),

    "claim_scope":
        {
            "supported_claim":
                (
                    "The 48-bit uncoded PER quantity is a "
                    "trend-level reliability proxy whose "
                    "expected SINR-to-error direction can be "
                    "compared with coded BLER and measured "
                    "DL-BLER."
                ),

            "prohibited_claims": [
                (
                    "The 48-bit PER proxy is numerically "
                    "calibrated to post-LDPC BLER."
                ),
                (
                    "The 48-bit proxy reproduces HARQ "
                    "failure probability."
                ),
                (
                    "Error-rate magnitudes are directly "
                    "comparable across the three systems."
                ),
                (
                    "A significant association establishes "
                    "causal equivalence between systems."
                ),
            ],
        },

    "canonical_assets":
        {
            "source_telemetry":
                str(
                    SOURCE_TELEMETRY_PATH.relative_to(
                        ROOT
                    )
                ),

            "source_metadata":
                str(
                    SOURCE_METADATA_PATH.relative_to(
                        ROOT
                    )
                ),

            "source_telemetry_sha256":
                sha256_file(
                    SOURCE_TELEMETRY_PATH
                ),

            "sionna_root":
                str(
                    SIONNA_DATA_ROOT.relative_to(
                        ROOT
                    )
                ),

            "sionna_analysis_ids":
                str(
                    SIONNA_IDS_PATH.relative_to(
                        ROOT
                    )
                ),

            "sionna_analysis_ids_sha256":
                sha256_file(
                    SIONNA_IDS_PATH
                ),

            "zenodo_inventory":
                str(
                    ZENODO_INVENTORY_PATH.relative_to(
                        ROOT
                    )
                ),

            "zenodo_inventory_sha256":
                sha256_file(
                    ZENODO_INVENTORY_PATH
                ),

            "zenodo_file_count":
                10,
        },

    "common_preprocessing":
        {
            "missing_values":
                (
                    "Use listwise deletion only for the "
                    "variables required by the current "
                    "endpoint. No imputation."
                ),

            "mcs_canonicalization":
                {
                    "rule":
                        (
                            "MCS must be finite and within "
                            "1e-6 of an integer. Canonicalize "
                            "using nearest-integer rounding."
                        ),

                    "otherwise":
                        "Fail the analysis.",
                },

            "ties":
                (
                    "Use average ranks for tied values."
                ),

            "smoothing":
                False,

            "temporal_resampling":
                False,

            "lag_selection":
                False,

            "sign_reversal":
                False,

            "outlier_trimming":
                False,

            "threshold_selection":
                False,
        },

    "mcs_stratified_rank_correlation":
        {
            "name":
                "MCS-stratified Spearman correlation",

            "algorithm": [
                (
                    "Within each independent sequence or "
                    "recording file, divide observations by "
                    "the canonical integer MCS value."
                ),
                (
                    "Discard an MCS stratum only when it has "
                    "fewer than the preregistered minimum "
                    "number of finite observations or when "
                    "SINR or the error variable is constant."
                ),
                (
                    "Within each retained MCS stratum, "
                    "replace SINR and the error variable by "
                    "average ranks."
                ),
                (
                    "Center and standardize both rank "
                    "variables within that MCS stratum."
                ),
                (
                    "Concatenate the standardized within-"
                    "stratum ranks and compute their Pearson "
                    "correlation."
                ),
            ],

            "interpretation":
                (
                    "This estimates a monotonic SINR-error "
                    "association after removing between-MCS "
                    "level differences."
                ),

            "expected_direction":
                "negative",
        },

    "source_analysis":
        {
            "sample_count":
                1080,

            "sequence_shape":
                [80, 10],

            "predictor":
                "SINR_dB",

            "outcome":
                "PER_proxy",

            "stratification":
                "MCS",

            "minimum_rows_per_mcs_stratum":
                8,

            "minimum_retained_rows_per_sequence":
                24,

            "primary_sequence_effect":
                "MCS-stratified Spearman correlation",

            "primary_aggregate":
                (
                    "Median of eligible sequence-level "
                    "correlations."
                ),

            "bootstrap":
                {
                    "iterations":
                        10000,

                    "seed":
                        60011,

                    "resampling_unit":
                        "sequence",

                    "strata":
                        "regime x label",

                    "interval":
                        "percentile 95%",
                },

            "direction_test":
                {
                    "method":
                        "Monte Carlo sign-flip test",

                    "iterations":
                        100000,

                    "seed":
                        60012,

                    "alternative":
                        "aggregate correlation < 0",

                    "plus_one_correction":
                        True,
                },

            "support_rule":
                (
                    "The median correlation is negative and "
                    "the upper bound of its 95% bootstrap "
                    "interval is below zero."
                ),

            "secondary_descriptive_endpoints": [
                (
                    "Sequence-level Spearman correlation "
                    "between BER and PER_proxy; expected "
                    "positive."
                ),
                (
                    "Absolute deviation between observed "
                    "PER_proxy and 1-(1-BER)^48; descriptive "
                    "only because bit errors need not be "
                    "independent."
                ),
            ],
        },

    "sionna_analysis":
        {
            "generated_sample_count":
                840,

            "locked_analysis_sample_count":
                825,

            "physical_count":
                706,

            "control_count":
                119,

            "analysis_membership_sha256":
                EXPECTED_MEMBERSHIP_SHA256,

            "sequence_shape":
                [80, 10],

            "predictor":
                "SINR_dB",

            "outcome":
                "BLER_post_LDPC",

            "stratification":
                "MCS_index",

            "minimum_rows_per_mcs_stratum":
                8,

            "minimum_retained_rows_per_sequence":
                24,

            "primary_sequence_effect":
                "MCS-stratified Spearman correlation",

            "primary_aggregate":
                (
                    "Median of eligible sequence-level "
                    "correlations."
                ),

            "bootstrap":
                {
                    "iterations":
                        10000,

                    "seed":
                        60021,

                    "resampling_unit":
                        "sample",

                    "strata":
                        "label",

                    "interval":
                        "percentile 95%",
                },

            "direction_test":
                {
                    "method":
                        "Monte Carlo sign-flip test",

                    "iterations":
                        100000,

                    "seed":
                        60022,

                    "alternative":
                        "aggregate correlation < 0",

                    "plus_one_correction":
                        True,
                },

            "support_rule":
                (
                    "The median correlation is negative and "
                    "the upper bound of its 95% bootstrap "
                    "interval is below zero."
                ),

            "mandatory_subgroup_reporting": [
                "all 825 samples",
                "physical samples",
                "control samples",
                "interference",
                "blockage",
                "mobility",
                "adaptation_mismatch",
                "normal",
                "nonphysical_goodput",
            ],

            "secondary_descriptive_endpoint":
                (
                    "Sequence-level Spearman correlation "
                    "between pre-LDPC BER and post-LDPC "
                    "BLER; expected positive."
                ),
        },

    "zenodo_analysis":
        {
            "file_count":
                10,

            "predictor":
                "dl_sinr",

            "outcome":
                "dl_bler",

            "stratification":
                "phy_mcs",

            "bler_unit_normalization":
                {
                    "ratio_rule":
                        (
                            "When every finite BLER value is "
                            "within [0,1+1e-8], retain it as "
                            "a ratio."
                        ),

                    "percentage_rule":
                        (
                            "Otherwise, when every finite "
                            "value is within [0,100+1e-8] "
                            "and at least one value exceeds "
                            "1+1e-8, divide all BLER values "
                            "by 100."
                        ),

                    "invalid_rule":
                        (
                            "Any other range fails the "
                            "analysis."
                        ),

                    "post_normalization":
                        (
                            "Clip only numerical tolerance "
                            "violations within 1e-8 to [0,1]."
                        ),
                },

            "minimum_rows_per_mcs_stratum":
                20,

            "minimum_retained_rows_per_file":
                200,

            "primary_file_effect":
                "MCS-stratified Spearman correlation",

            "primary_aggregate":
                {
                    "transform":
                        (
                            "Apply Fisher arctanh to each "
                            "eligible file-level correlation."
                        ),

                    "weighting":
                        "Equal weight per recording file",

                    "aggregate":
                        (
                            "Arithmetic mean in Fisher-z "
                            "space, transformed back using "
                            "tanh."
                        ),
                },

            "exact_direction_test":
                {
                    "method":
                        (
                            "Enumerate all 2^10 sign-flip "
                            "patterns over the ten independent "
                            "file-level Fisher-z effects."
                        ),

                    "alternative":
                        "aggregate correlation < 0",

                    "significance_level":
                        0.05,
                },

            "cluster_bootstrap":
                {
                    "iterations":
                        10000,

                    "seed":
                        60031,

                    "resampling_unit":
                        "recording file",

                    "within_file_rows_resampled":
                        False,

                    "interval":
                        "percentile 95%",
                },

            "support_rule":
                (
                    "The aggregate correlation is negative, "
                    "the exact one-sided sign-flip p-value "
                    "is below 0.05, and the upper bound of "
                    "the file-cluster bootstrap interval is "
                    "below zero."
                ),

            "mandatory_reporting": [
                "eligible file count",
                "excluded file count and fixed reason",
                "effect for every file",
                "aggregate correlation",
                "95% file-cluster bootstrap interval",
                "exact one-sided p-value",
                "number of files with negative effects",
                "BLER unit decision for every file",
            ],

            "exploratory_secondary_endpoints": [
                {
                    "name":
                        "unadjusted_sinr_bler",

                    "variables":
                        "dl_sinr versus dl_bler",

                    "expected_direction":
                        "negative",
                },
                {
                    "name":
                        "cqi_bler",

                    "variables":
                        "mac_dl_cqi versus dl_bler",

                    "expected_direction":
                        "negative",
                },
                {
                    "name":
                        "spectral_efficiency_bler",

                    "variables":
                        "se versus dl_bler",

                    "expected_direction":
                        "negative",
                },
                {
                    "name":
                        "sinr_mcs",

                    "variables":
                        "dl_sinr versus phy_mcs",

                    "expected_direction":
                        "positive",
                },
            ],

            "secondary_multiplicity":
                {
                    "family_size":
                        4,

                    "correction":
                        "Holm family-wise correction",

                    "role":
                        (
                            "Exploratory only; no secondary "
                            "endpoint may replace the primary "
                            "endpoint."
                        ),
                },
        },

    "cross_system_interpretation":
        {
            "full_directional_support":
                (
                    "Source, Sionna, and Zenodo all satisfy "
                    "their preregistered support rules."
                ),

            "coded_and_real_support":
                (
                    "Sionna and Zenodo satisfy their support "
                    "rules, supporting transfer of the "
                    "reliability direction beyond the "
                    "uncoded source proxy."
                ),

            "partial_support":
                (
                    "At least one but not all systems "
                    "satisfy their support rules."
                ),

            "no_support":
                (
                    "None of the three systems satisfies its "
                    "support rule."
                ),

            "magnitude_comparison_allowed":
                False,

            "calibration_equivalence_allowed":
                False,
        },

    "prohibited_posthoc_actions": [
        "Reverse the expected SINR-error direction",
        "Replace dl_bler as the Zenodo primary outcome",
        "Replace the MCS-stratified primary endpoint",
        "Change the frozen Sionna-825 membership",
        "Choose a temporal lag after observing results",
        "Smooth BLER or SINR after observing results",
        "Remove recording files because their effects are unfavorable",
        "Promote a secondary endpoint to primary",
        "Select a diagnostic threshold using these reliability analyses",
        "Claim equality between PER_proxy and coded BLER magnitudes",
    ],

    "methodological_boundary":
        {
            "source_kpi_values_analyzed":
                False,

            "sionna_kpi_values_analyzed":
                False,

            "zenodo_data_rows_read":
                False,

            "correlations_computed":
                False,

            "statistical_tests_computed":
                False,

            "models_trained":
                False,

            "thresholds_selected":
                False,

            "endpoint_directions_changed":
                False,

            "analysis_subset_changed":
                False,
        },
}


OUTPUT_CONFIG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CONFIG_PATH.write_text(
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
        "phyguard.cross_system_reliability_"
        "statistical_protocol_summary.v1",

    "status":
        protocol["status"],

    "scientific_question":
        protocol["scientific_question"],

    "source_primary":
        {
            "predictor":
                "SINR_dB",

            "outcome":
                "PER_proxy",

            "stratification":
                "MCS",

            "expected_direction":
                "negative",

            "sample_count":
                1080,
        },

    "sionna_primary":
        {
            "predictor":
                "SINR_dB",

            "outcome":
                "BLER_post_LDPC",

            "stratification":
                "MCS_index",

            "expected_direction":
                "negative",

            "analysis_count":
                825,

            "membership_sha256":
                EXPECTED_MEMBERSHIP_SHA256,
        },

    "zenodo_primary":
        {
            "predictor":
                "dl_sinr",

            "outcome":
                "dl_bler",

            "stratification":
                "phy_mcs",

            "expected_direction":
                "negative",

            "file_count":
                10,

            "inference":
                (
                    "Exact file-level sign-flip test "
                    "and file-cluster bootstrap."
                ),
        },

    "protocol_path":
        str(
            OUTPUT_CONFIG_PATH.relative_to(
                ROOT
            )
        ),

    "protocol_sha256":
        sha256_file(
            OUTPUT_CONFIG_PATH
        ),

    "methodological_boundary":
        protocol[
            "methodological_boundary"
        ],
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
        "phyguard.cross_system_reliability_"
        "statistical_protocol_manifest.v1",

    "status":
        protocol["status"],

    "protocol_sha256":
        sha256_file(
            OUTPUT_CONFIG_PATH
        ),

    "summary_sha256":
        sha256_file(
            SUMMARY_PATH
        ),

    "metadata_summary_sha256":
        sha256_file(
            METADATA_SUMMARY_PATH
        ),

    "metadata_manifest_sha256":
        sha256_file(
            METADATA_MANIFEST_PATH
        ),

    "formula_contract_sha256":
        sha256_file(
            FORMULA_CONTRACT_PATH
        ),

    "sionna_membership_sha256":
        sha256_file(
            SIONNA_IDS_PATH
        ),

    "zenodo_inventory_sha256":
        sha256_file(
            ZENODO_INVENTORY_PATH
        ),

    "methodological_boundary":
        protocol[
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


print(
    "status:",
    protocol["status"],
)

print("\nPRIMARY ENDPOINTS")
print(
    "source:",
    "SINR_dB -> PER_proxy",
    "| stratified by MCS",
    "| expected negative",
)
print(
    "sionna:",
    "SINR_dB -> BLER_post_LDPC",
    "| stratified by MCS_index",
    "| expected negative",
)
print(
    "zenodo:",
    "dl_sinr -> dl_bler",
    "| stratified by phy_mcs",
    "| expected negative",
)

print("\nFROZEN POPULATIONS")
print("source_sequence_count: 1080")
print("sionna_analysis_count: 825")
print(
    "sionna_membership_sha256:",
    EXPECTED_MEMBERSHIP_SHA256,
)
print("zenodo_file_count: 10")

print("\nZENODO INFERENCE")
print(
    "file_effect:",
    "MCS-stratified Spearman",
)
print(
    "aggregate:",
    "equal-weight Fisher-z mean",
)
print(
    "test:",
    "exact 2^10 file sign-flip",
)
print(
    "confidence_interval:",
    "10000 file-cluster bootstrap",
)
print(
    "secondary_correction:",
    "Holm across 4 exploratory endpoints",
)

print("\nMETHOD BOUNDARY")
for key, value in protocol[
    "methodological_boundary"
].items():
    print(
        key + ":",
        value,
    )

print("\nprotocol:", OUTPUT_CONFIG_PATH)
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)

print(
    "\nCROSS_SYSTEM_RELIABILITY_"
    "STATISTICAL_PROTOCOL_V1_PASS"
)
