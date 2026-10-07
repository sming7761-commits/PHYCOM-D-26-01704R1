import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path("/root/phyguard_revision")

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "cross_system_reliability_statistical_protocol_v2.json"
)

PROTOCOL_SUMMARY_PATH = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_statistical_protocol_v2"
    / "summary.json"
)

PROTOCOL_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "cross_system_reliability_statistical_protocol_v2.json"
)

METADATA_CONTRACT_PATH = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_metadata_contract_v1"
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

SIONNA_IDS_PATH = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_metadata_contract_v1"
    / "sionna_analysis825_ids.txt"
)

SIONNA_DATA_ROOT = (
    ROOT
    / "data"
    / "sionna_formal_test840"
)

ZENODO_INVENTORY_PATH = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_asset_inventory_v2"
    / "zenodo_canonical_inventory.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_statistics_v2"
)

SOURCE_EFFECTS_PATH = (
    OUTPUT_ROOT
    / "source_sequence_effects.csv"
)

SIONNA_EFFECTS_PATH = (
    OUTPUT_ROOT
    / "sionna_sequence_effects.csv"
)

SIONNA_SUBGROUP_PATH = (
    OUTPUT_ROOT
    / "sionna_subgroup_results.csv"
)

ZENODO_FILE_EFFECTS_PATH = (
    OUTPUT_ROOT
    / "zenodo_file_effects.csv"
)

ZENODO_SECONDARY_PATH = (
    OUTPUT_ROOT
    / "zenodo_secondary_results.csv"
)

DISTRIBUTIONS_PATH = (
    OUTPUT_ROOT
    / "statistical_distributions.npz"
)

SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "cross_system_reliability_statistics_v2.json"
)

EXPECTED_MEMBERSHIP_SHA256 = (
    "a8aa74f95df897a57c82f60ea0d2f233"
    "53eaa023ec0446b079dd8a88d7de462a"
)

EXPECTED_STATUS = (
    "LOCKED_BEFORE_RELIABILITY_STATISTICAL_EXECUTION"
)

CONTROL_LABELS = {
    "normal",
    "nonphysical_goodput",
}

MANDATORY_SIONNA_GROUPS = [
    "all_825_samples",
    "physical_samples",
    "control_samples",
    "interference",
    "blockage",
    "mobility",
    "adaptation_mismatch",
    "normal",
    "nonphysical_goodput",
]


# STAGE60E2_V2_FINAL_JSON_SERIALIZATION_FIX
def json_safe(value):
    """Convert non-finite summary values to JSON null."""
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, np.generic):
        return json_safe(value.item())

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None

    return value


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


def normalize(value):
    import re

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def decode_name(value):
    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    return str(value)


def safe_spearman(x, y):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[finite]
    y = y[finite]

    if len(x) < 3:
        return math.nan, len(x), "too_few_rows"

    if np.ptp(x) <= 0:
        return math.nan, len(x), "constant_predictor"

    if np.ptp(y) <= 0:
        return math.nan, len(x), "constant_outcome"

    rank_x = rankdata(
        x,
        method="average",
    )

    rank_y = rankdata(
        y,
        method="average",
    )

    effect = float(
        np.corrcoef(
            rank_x,
            rank_y,
        )[0, 1]
    )

    if not np.isfinite(effect):
        return math.nan, len(x), "nonfinite_effect"

    return effect, len(x), "eligible"


def mcs_stratified_spearman(
    predictor,
    outcome,
    mcs,
    minimum_rows_per_stratum,
    minimum_retained_rows,
):
    predictor = np.asarray(
        predictor,
        dtype=np.float64,
    )

    outcome = np.asarray(
        outcome,
        dtype=np.float64,
    )

    mcs = np.asarray(
        mcs,
        dtype=np.float64,
    )

    finite = (
        np.isfinite(predictor)
        & np.isfinite(outcome)
        & np.isfinite(mcs)
    )

    predictor = predictor[finite]
    outcome = outcome[finite]
    mcs = mcs[finite]

    finite_row_count = len(
        predictor
    )

    if finite_row_count < minimum_retained_rows:
        return {
            "eligible": False,
            "effect": math.nan,
            "finite_row_count": finite_row_count,
            "retained_row_count": 0,
            "candidate_stratum_count": 0,
            "retained_stratum_count": 0,
            "reason": "too_few_finite_rows",
        }

    rounded_mcs = np.rint(
        mcs
    )

    if np.any(
        np.abs(
            mcs - rounded_mcs
        )
        > 1e-6
    ):
        raise RuntimeError(
            "MCS contains a finite non-integer value."
        )

    canonical_mcs = rounded_mcs.astype(
        np.int64
    )

    standardized_predictor = []
    standardized_outcome = []

    unique_mcs = sorted(
        np.unique(
            canonical_mcs
        ).tolist()
    )

    retained_strata = 0
    retained_rows = 0

    for mcs_value in unique_mcs:
        mask = (
            canonical_mcs
            == mcs_value
        )

        x = predictor[mask]
        y = outcome[mask]

        if len(x) < minimum_rows_per_stratum:
            continue

        if np.ptp(x) <= 0:
            continue

        if np.ptp(y) <= 0:
            continue

        rank_x = rankdata(
            x,
            method="average",
        ).astype(np.float64)

        rank_y = rankdata(
            y,
            method="average",
        ).astype(np.float64)

        std_x = float(
            np.std(
                rank_x,
                ddof=0,
            )
        )

        std_y = float(
            np.std(
                rank_y,
                ddof=0,
            )
        )

        if std_x <= 0 or std_y <= 0:
            continue

        standardized_predictor.append(
            (
                rank_x
                - np.mean(rank_x)
            )
            / std_x
        )

        standardized_outcome.append(
            (
                rank_y
                - np.mean(rank_y)
            )
            / std_y
        )

        retained_strata += 1
        retained_rows += len(x)

    if retained_rows < minimum_retained_rows:
        return {
            "eligible": False,
            "effect": math.nan,
            "finite_row_count": finite_row_count,
            "retained_row_count": retained_rows,
            "candidate_stratum_count": len(unique_mcs),
            "retained_stratum_count": retained_strata,
            "reason": "too_few_retained_rows",
        }

    concatenated_x = np.concatenate(
        standardized_predictor
    )

    concatenated_y = np.concatenate(
        standardized_outcome
    )

    effect = float(
        np.corrcoef(
            concatenated_x,
            concatenated_y,
        )[0, 1]
    )

    if not np.isfinite(effect):
        return {
            "eligible": False,
            "effect": math.nan,
            "finite_row_count": finite_row_count,
            "retained_row_count": retained_rows,
            "candidate_stratum_count": len(unique_mcs),
            "retained_stratum_count": retained_strata,
            "reason": "nonfinite_effect",
        }

    return {
        "eligible": True,
        "effect": effect,
        "finite_row_count": finite_row_count,
        "retained_row_count": retained_rows,
        "candidate_stratum_count": len(unique_mcs),
        "retained_stratum_count": retained_strata,
        "reason": "eligible",
    }



def partial_spearman_continuous_control(
    predictor,
    outcome,
    control,
    minimum_complete_rows,
):
    predictor = np.asarray(predictor, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)

    finite = (
        np.isfinite(predictor)
        & np.isfinite(outcome)
        & np.isfinite(control)
    )
    predictor = predictor[finite]
    outcome = outcome[finite]
    control = control[finite]

    n = len(predictor)
    base = {
        "complete_row_count": n,
        "control_unique_count": int(np.unique(control).size),
        "predictor_residual_sd": math.nan,
        "outcome_residual_sd": math.nan,
    }

    def fail(reason):
        return {
            **base,
            "eligible": False,
            "effect": math.nan,
            "reason": reason,
        }

    if n < minimum_complete_rows:
        return fail("too_few_complete_rows")
    if np.ptp(predictor) <= 0:
        return fail("constant_predictor")
    if np.ptp(outcome) <= 0:
        return fail("constant_outcome")
    if np.ptp(control) <= 0:
        return fail("constant_control")

    x_rank = rankdata(predictor, method="average").astype(np.float64)
    y_rank = rankdata(outcome, method="average").astype(np.float64)
    c_rank = rankdata(control, method="average").astype(np.float64)

    design = np.column_stack(
        [np.ones(n, dtype=np.float64), c_rank]
    )
    x_beta = np.linalg.lstsq(design, x_rank, rcond=None)[0]
    y_beta = np.linalg.lstsq(design, y_rank, rcond=None)[0]
    x_resid = x_rank - design @ x_beta
    y_resid = y_rank - design @ y_beta

    base["predictor_residual_sd"] = float(np.std(x_resid, ddof=0))
    base["outcome_residual_sd"] = float(np.std(y_resid, ddof=0))

    if base["predictor_residual_sd"] <= 0:
        return fail("zero_predictor_residual_variance")
    if base["outcome_residual_sd"] <= 0:
        return fail("zero_outcome_residual_variance")

    effect = float(np.corrcoef(x_resid, y_resid)[0, 1])
    if not np.isfinite(effect):
        return fail("nonfinite_partial_effect")

    return {
        **base,
        "eligible": True,
        "effect": effect,
        "reason": "eligible",
    }


def stratified_bootstrap_median(
    effects,
    strata,
    iterations,
    seed,
):
    effects = np.asarray(
        effects,
        dtype=np.float64,
    )

    strata = np.asarray(
        strata,
        dtype=object,
    )

    unique_strata = sorted(
        set(
            strata.tolist()
        )
    )

    group_indices = [
        np.flatnonzero(
            strata == stratum
        )
        for stratum in unique_strata
    ]

    rng = np.random.default_rng(
        seed
    )

    distribution = np.empty(
        iterations,
        dtype=np.float64,
    )

    for iteration in range(
        iterations
    ):
        sampled_parts = []

        for indices in group_indices:
            sampled_indices = rng.choice(
                indices,
                size=len(indices),
                replace=True,
            )

            sampled_parts.append(
                effects[sampled_indices]
            )

        distribution[iteration] = float(
            np.median(
                np.concatenate(
                    sampled_parts
                )
            )
        )

    return distribution


def sign_flip_median_test(
    effects,
    iterations,
    seed,
):
    effects = np.asarray(
        effects,
        dtype=np.float64,
    )

    observed = float(
        np.median(
            effects
        )
    )

    rng = np.random.default_rng(
        seed
    )

    distribution = np.empty(
        iterations,
        dtype=np.float64,
    )

    chunk_size = 1000
    offset = 0

    while offset < iterations:
        current = min(
            chunk_size,
            iterations - offset,
        )

        signs = (
            rng.integers(
                0,
                2,
                size=(
                    current,
                    len(effects),
                ),
                dtype=np.int8,
            )
            * 2
            - 1
        )

        distribution[
            offset:
            offset + current
        ] = np.median(
            signs
            * effects[None, :],
            axis=1,
        )

        offset += current

    p_value = (
        1
        + int(
            np.sum(
                distribution
                <= observed
            )
        )
    ) / (
        iterations
        + 1
    )

    return observed, distribution, float(
        p_value
    )


def fisher_z(effect):
    effect = float(
        np.clip(
            effect,
            -1.0 + 1e-12,
            1.0 - 1e-12,
        )
    )

    return float(
        np.arctanh(
            effect
        )
    )


def fisher_aggregate(effects):
    effects = np.asarray(
        effects,
        dtype=np.float64,
    )

    z_values = np.asarray(
        [
            fisher_z(effect)
            for effect in effects
        ],
        dtype=np.float64,
    )

    aggregate_z = float(
        np.mean(
            z_values
        )
    )

    return (
        float(
            np.tanh(
                aggregate_z
            )
        ),
        aggregate_z,
        z_values,
    )


def exact_file_sign_flip(
    effects,
    alternative,
):
    aggregate_effect, observed_z, z_values = (
        fisher_aggregate(
            effects
        )
    )

    statistics = []

    for signs in itertools.product(
        [-1.0, 1.0],
        repeat=len(z_values),
    ):
        statistic = float(
            np.mean(
                z_values
                * np.asarray(
                    signs,
                    dtype=np.float64,
                )
            )
        )

        statistics.append(
            statistic
        )

    statistics = np.asarray(
        statistics,
        dtype=np.float64,
    )

    if alternative == "negative":
        p_value = float(
            np.mean(
                statistics
                <= observed_z
            )
        )
    elif alternative == "positive":
        p_value = float(
            np.mean(
                statistics
                >= observed_z
            )
        )
    else:
        raise ValueError(
            alternative
        )

    return {
        "aggregate_effect":
            aggregate_effect,

        "aggregate_fisher_z":
            observed_z,

        "p_value":
            p_value,

        "distribution":
            statistics,
    }


def file_cluster_bootstrap(
    effects,
    iterations,
    seed,
):
    effects = np.asarray(
        effects,
        dtype=np.float64,
    )

    z_values = np.asarray(
        [
            fisher_z(effect)
            for effect in effects
        ],
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        seed
    )

    distribution = np.empty(
        iterations,
        dtype=np.float64,
    )

    for iteration in range(
        iterations
    ):
        sample = rng.choice(
            z_values,
            size=len(z_values),
            replace=True,
        )

        distribution[iteration] = float(
            np.tanh(
                np.mean(sample)
            )
        )

    return distribution


def percentile_interval(
    distribution,
):
    low, high = np.quantile(
        np.asarray(
            distribution,
            dtype=np.float64,
        ),
        [
            0.025,
            0.975,
        ],
    )

    return float(low), float(high)


def holm_adjust(p_values):
    p_values = np.asarray(
        p_values,
        dtype=np.float64,
    )

    count = len(
        p_values
    )

    order = np.argsort(
        p_values
    )

    adjusted = np.empty(
        count,
        dtype=np.float64,
    )

    running_maximum = 0.0

    for rank, index in enumerate(
        order
    ):
        multiplier = (
            count
            - rank
        )

        candidate = min(
            1.0,
            multiplier
            * p_values[index],
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted[index] = min(
            1.0,
            running_maximum,
        )

    return adjusted


required_paths = [
    PROTOCOL_PATH,
    PROTOCOL_SUMMARY_PATH,
    PROTOCOL_MANIFEST_PATH,
    METADATA_CONTRACT_PATH,
    SOURCE_TELEMETRY_PATH,
    SOURCE_METADATA_PATH,
    SIONNA_IDS_PATH,
    SIONNA_DATA_ROOT,
    ZENODO_INVENTORY_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            path
        )


protocol = load_json(
    PROTOCOL_PATH
)

protocol_summary = load_json(
    PROTOCOL_SUMMARY_PATH
)

protocol_manifest = load_json(
    PROTOCOL_MANIFEST_PATH
)

metadata_contract = load_json(
    METADATA_CONTRACT_PATH
)


if protocol.get("status") != EXPECTED_STATUS:
    raise RuntimeError(
        "Protocol status mismatch."
    )

if protocol_summary.get("status") != EXPECTED_STATUS:
    raise RuntimeError(
        "Protocol summary status mismatch."
    )

if protocol_manifest.get("status") != EXPECTED_STATUS:
    raise RuntimeError(
        "Protocol manifest status mismatch."
    )

if (
    sha256_file(
        PROTOCOL_PATH
    )
    != protocol_manifest["v2_protocol_sha256"]
):
    raise RuntimeError(
        "Protocol checksum mismatch."
    )

if (
    sha256_file(
        SIONNA_IDS_PATH
    )
    != EXPECTED_MEMBERSHIP_SHA256
):
    raise RuntimeError(
        "Sionna membership checksum mismatch."
    )

if (
    metadata_contract.get("status")
    != "LOCKED_BEFORE_RELIABILITY_VALUE_ANALYSIS"
):
    raise RuntimeError(
        "Metadata contract status mismatch."
    )

if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Stage 60E2 output already exists; "
        "refusing to overwrite."
    )

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)


# ============================================================
# 1. SOURCE RECONSTRUCTION
# ============================================================

with np.load(
    SOURCE_TELEMETRY_PATH,
    allow_pickle=False,
) as archive:
    source_x = np.asarray(
        archive["X"],
        dtype=np.float64,
    )

    source_kpi_names = [
        normalize(
            decode_name(value)
        )
        for value in archive[
            "kpi_names"
        ].tolist()
    ]


if source_x.shape != (
    1080,
    80,
    10,
):
    raise RuntimeError(
        f"Unexpected source shape: {source_x.shape}"
    )


source_index = {
    name: index
    for index, name
    in enumerate(
        source_kpi_names
    )
}

for required_name in [
    "sinr_db",
    "ber",
    "per_proxy",
    "mcs",
]:
    if required_name not in source_index:
        raise RuntimeError(
            f"Missing source KPI: {required_name}"
        )


source_metadata = load_csv(
    SOURCE_METADATA_PATH
)

if len(source_metadata) != 1080:
    raise RuntimeError(
        "Source metadata row count is not 1080."
    )


source_metadata_by_index = {}

for row in source_metadata:
    global_index = int(
        row["global_index"]
    )

    if global_index in source_metadata_by_index:
        raise RuntimeError(
            "Duplicate source global_index."
        )

    source_metadata_by_index[
        global_index
    ] = row


if set(
    source_metadata_by_index
) != set(
    range(1080)
):
    raise RuntimeError(
        "Source global_index does not cover 0..1079."
    )


source_effect_rows = []

for index in range(
    1080
):
    sequence = source_x[
        index
    ]

    metadata = source_metadata_by_index[
        index
    ]

    per_values = sequence[
        :,
        source_index["per_proxy"],
    ]

    ber_values = sequence[
        :,
        source_index["ber"],
    ]

    if np.nanmin(per_values) < -1e-8:
        raise RuntimeError(
            "Source PER proxy below zero."
        )

    if np.nanmax(per_values) > 1.0 + 1e-8:
        raise RuntimeError(
            "Source PER proxy above one."
        )

    if np.nanmin(ber_values) < -1e-8:
        raise RuntimeError(
            "Source BER below zero."
        )

    if np.nanmax(ber_values) > 1.0 + 1e-8:
        raise RuntimeError(
            "Source BER above one."
        )

    primary = mcs_stratified_spearman(
        predictor=sequence[
            :,
            source_index["sinr_db"],
        ],
        outcome=per_values,
        mcs=sequence[
            :,
            source_index["mcs"],
        ],
        minimum_rows_per_stratum=8,
        minimum_retained_rows=24,
    )

    ber_per_effect, ber_per_rows, ber_per_reason = (
        safe_spearman(
            ber_values,
            per_values,
        )
    )

    independent_prediction = (
        1.0
        - np.power(
            1.0
            - np.clip(
                ber_values,
                0.0,
                1.0,
            ),
            48,
        )
    )

    analytical_absolute_deviation = float(
        np.nanmean(
            np.abs(
                per_values
                - independent_prediction
            )
        )
    )

    source_effect_rows.append(
        {
            "global_index":
                index,

            "regime":
                normalize(
                    metadata.get(
                        "regime",
                        "",
                    )
                ),

            "label":
                normalize(
                    metadata.get(
                        "label",
                        "",
                    )
                ),

            "severity":
                normalize(
                    metadata.get(
                        "severity",
                        "",
                    )
                ),

            "eligible":
                primary["eligible"],

            "primary_effect":
                primary["effect"],

            "finite_row_count":
                primary[
                    "finite_row_count"
                ],

            "retained_row_count":
                primary[
                    "retained_row_count"
                ],

            "candidate_stratum_count":
                primary[
                    "candidate_stratum_count"
                ],

            "retained_stratum_count":
                primary[
                    "retained_stratum_count"
                ],

            "eligibility_reason":
                primary["reason"],

            "ber_per_spearman":
                ber_per_effect,

            "ber_per_row_count":
                ber_per_rows,

            "ber_per_reason":
                ber_per_reason,

            "mean_absolute_deviation_from_independent_ber_formula":
                analytical_absolute_deviation,
        }
    )


source_eligible_rows = [
    row
    for row in source_effect_rows
    if row["eligible"]
]

source_effects = np.asarray(
    [
        row["primary_effect"]
        for row in source_eligible_rows
    ],
    dtype=np.float64,
)

source_strata = np.asarray(
    [
        (
            row["regime"]
            + "|"
            + row["label"]
        )
        for row in source_eligible_rows
    ],
    dtype=object,
)


if len(source_effects) == 0:
    raise RuntimeError(
        "No eligible source effects."
    )


source_bootstrap = (
    stratified_bootstrap_median(
        effects=source_effects,
        strata=source_strata,
        iterations=10000,
        seed=60011,
    )
)

source_observed, source_signflip, source_p = (
    sign_flip_median_test(
        effects=source_effects,
        iterations=100000,
        seed=60012,
    )
)

source_ci_low, source_ci_high = (
    percentile_interval(
        source_bootstrap
    )
)

source_supported = bool(
    source_observed < 0
    and source_ci_high < 0
)


# ============================================================
# 2. SIONNA CODED LINK
# ============================================================

analysis_ids = [
    line.strip()
    for line in SIONNA_IDS_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

if len(analysis_ids) != 825:
    raise RuntimeError(
        "Sionna analysis ID count is not 825."
    )


sionna_effect_rows = []

for sample_id in analysis_ids:
    sample_root = (
        SIONNA_DATA_ROOT
        / sample_id
    )

    sequence_path = (
        sample_root
        / "sequence.npz"
    )

    metadata_path = (
        sample_root
        / "metadata.json"
    )

    if not sequence_path.exists():
        raise FileNotFoundError(
            sequence_path
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            metadata_path
        )

    metadata = load_json(
        metadata_path
    )

    label = normalize(
        metadata.get(
            "label",
            "",
        )
    )

    with np.load(
        sequence_path,
        allow_pickle=False,
    ) as archive:
        sequence = np.asarray(
            archive[
                "kpi_sequence"
            ],
            dtype=np.float64,
        )

        names = [
            normalize(
                decode_name(value)
            )
            for value in archive[
                "kpi_names"
            ].tolist()
        ]

    if sequence.shape != (
        80,
        10,
    ):
        raise RuntimeError(
            f"{sample_id}: unexpected sequence shape."
        )

    name_index = {
        name: index
        for index, name
        in enumerate(names)
    }

    for required_name in [
        "sinr_db",
        "ber_pre_ldpc",
        "bler_post_ldpc",
        "mcs_index",
    ]:
        if required_name not in name_index:
            raise RuntimeError(
                f"{sample_id}: missing {required_name}"
            )

    bler_values = sequence[
        :,
        name_index[
            "bler_post_ldpc"
        ],
    ]

    ber_values = sequence[
        :,
        name_index[
            "ber_pre_ldpc"
        ],
    ]

    if np.nanmin(bler_values) < -1e-8:
        raise RuntimeError(
            f"{sample_id}: BLER below zero."
        )

    if np.nanmax(bler_values) > 1.0 + 1e-8:
        raise RuntimeError(
            f"{sample_id}: BLER above one."
        )

    primary = mcs_stratified_spearman(
        predictor=sequence[
            :,
            name_index["sinr_db"],
        ],
        outcome=bler_values,
        mcs=sequence[
            :,
            name_index["mcs_index"],
        ],
        minimum_rows_per_stratum=8,
        minimum_retained_rows=24,
    )

    ber_bler_effect, ber_bler_rows, ber_bler_reason = (
        safe_spearman(
            ber_values,
            bler_values,
        )
    )

    sionna_effect_rows.append(
        {
            "sample_id":
                sample_id,

            "label":
                label,

            "sample_type":
                (
                    "control"
                    if label in CONTROL_LABELS
                    else "physical"
                ),

            "eligible":
                primary["eligible"],

            "primary_effect":
                primary["effect"],

            "finite_row_count":
                primary[
                    "finite_row_count"
                ],

            "retained_row_count":
                primary[
                    "retained_row_count"
                ],

            "candidate_stratum_count":
                primary[
                    "candidate_stratum_count"
                ],

            "retained_stratum_count":
                primary[
                    "retained_stratum_count"
                ],

            "eligibility_reason":
                primary["reason"],

            "ber_bler_spearman":
                ber_bler_effect,

            "ber_bler_row_count":
                ber_bler_rows,

            "ber_bler_reason":
                ber_bler_reason,
        }
    )


sionna_eligible_rows = [
    row
    for row in sionna_effect_rows
    if row["eligible"]
]

sionna_effects = np.asarray(
    [
        row["primary_effect"]
        for row in sionna_eligible_rows
    ],
    dtype=np.float64,
)

sionna_strata = np.asarray(
    [
        row["label"]
        for row in sionna_eligible_rows
    ],
    dtype=object,
)


if len(sionna_effects) == 0:
    raise RuntimeError(
        "No eligible Sionna effects."
    )


sionna_bootstrap = (
    stratified_bootstrap_median(
        effects=sionna_effects,
        strata=sionna_strata,
        iterations=10000,
        seed=60021,
    )
)

sionna_observed, sionna_signflip, sionna_p = (
    sign_flip_median_test(
        effects=sionna_effects,
        iterations=100000,
        seed=60022,
    )
)

sionna_ci_low, sionna_ci_high = (
    percentile_interval(
        sionna_bootstrap
    )
)

sionna_supported = bool(
    sionna_observed < 0
    and sionna_ci_high < 0
)


sionna_subgroup_rows = []

for group_name in MANDATORY_SIONNA_GROUPS:
    if group_name == "all_825_samples":
        rows = sionna_effect_rows

    elif group_name == "physical_samples":
        rows = [
            row
            for row in sionna_effect_rows
            if row["sample_type"]
            == "physical"
        ]

    elif group_name == "control_samples":
        rows = [
            row
            for row in sionna_effect_rows
            if row["sample_type"]
            == "control"
        ]

    else:
        rows = [
            row
            for row in sionna_effect_rows
            if row["label"]
            == group_name
        ]

    eligible = [
        row
        for row in rows
        if row["eligible"]
    ]

    effects = np.asarray(
        [
            row["primary_effect"]
            for row in eligible
        ],
        dtype=np.float64,
    )

    sionna_subgroup_rows.append(
        {
            "group":
                group_name,

            "total_count":
                len(rows),

            "eligible_count":
                len(eligible),

            "ineligible_count":
                len(rows)
                - len(eligible),

            "median_effect":
                (
                    float(
                        np.median(
                            effects
                        )
                    )
                    if len(effects)
                    else math.nan
                ),

            "mean_effect":
                (
                    float(
                        np.mean(
                            effects
                        )
                    )
                    if len(effects)
                    else math.nan
                ),

            "negative_effect_count":
                int(
                    np.sum(
                        effects < 0
                    )
                ),

            "negative_effect_fraction":
                (
                    float(
                        np.mean(
                            effects < 0
                        )
                    )
                    if len(effects)
                    else math.nan
                ),
        }
    )


# ============================================================
# 3. ZENODO REAL O-RAN MEASUREMENTS
# ============================================================

zenodo_inventory = load_csv(
    ZENODO_INVENTORY_PATH
)

if len(zenodo_inventory) != 10:
    raise RuntimeError(
        "Zenodo file count is not 10."
    )


zenodo_file_rows = []

secondary_file_effects = {
    "unadjusted_sinr_bler": [],
    "cqi_bler": [],
    "spectral_efficiency_bler": [],
    "sinr_mcs": [],
}

secondary_expected_directions = {
    "unadjusted_sinr_bler": "negative",
    "cqi_bler": "negative",
    "spectral_efficiency_bler": "negative",
    "sinr_mcs": "positive",
}


for inventory_row in zenodo_inventory:
    path = (
        ROOT
        / inventory_row[
            "relative_path"
        ]
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    if (
        sha256_file(path)
        != inventory_row["sha256"]
    ):
        raise RuntimeError(
            f"Zenodo checksum mismatch: {path.name}"
        )

    frame = pd.read_csv(
        path
    )

    required_columns = [
        "phy_mcs",
        "mac_dl_cqi",
        "dl_sinr",
        "se",
        "dl_bler",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"{path.name}: missing columns "
            f"{missing_columns}"
        )

    numeric = pd.DataFrame(
        {
            column:
                pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )
            for column
            in required_columns
        }
    )

    raw_bler = numeric[
        "dl_bler"
    ].to_numpy(
        dtype=np.float64
    )

    finite_bler = raw_bler[
        np.isfinite(
            raw_bler
        )
    ]

    if len(finite_bler) == 0:
        raise RuntimeError(
            f"{path.name}: no finite BLER values."
        )

    raw_bler_min = float(
        np.min(
            finite_bler
        )
    )

    raw_bler_max = float(
        np.max(
            finite_bler
        )
    )

    if (
        raw_bler_min >= -1e-8
        and raw_bler_max <= 1.0 + 1e-8
    ):
        unit_decision = "ratio"
        normalized_bler = raw_bler.copy()

    elif (
        raw_bler_min >= -1e-8
        and raw_bler_max <= 100.0 + 1e-8
        and raw_bler_max > 1.0 + 1e-8
    ):
        unit_decision = "percentage_divided_by_100"
        normalized_bler = (
            raw_bler
            / 100.0
        )

    else:
        raise RuntimeError(
            f"{path.name}: invalid BLER range "
            f"[{raw_bler_min}, {raw_bler_max}]"
        )

    tolerance_mask_low = (
        np.isfinite(
            normalized_bler
        )
        & (
            normalized_bler < 0
        )
        & (
            normalized_bler >= -1e-8
        )
    )

    tolerance_mask_high = (
        np.isfinite(
            normalized_bler
        )
        & (
            normalized_bler > 1
        )
        & (
            normalized_bler <= 1.0 + 1e-8
        )
    )

    normalized_bler[
        tolerance_mask_low
        | tolerance_mask_high
    ] = np.clip(
        normalized_bler[
            tolerance_mask_low
            | tolerance_mask_high
        ],
        0.0,
        1.0,
    )

    primary = partial_spearman_continuous_control(
        predictor=numeric[
            "dl_sinr"
        ].to_numpy(
            dtype=np.float64
        ),
        outcome=normalized_bler,
        control=numeric[
            "phy_mcs"
        ].to_numpy(
            dtype=np.float64
        ),
        minimum_complete_rows=200,
    )

    unadjusted_effect, unadjusted_rows, unadjusted_reason = (
        safe_spearman(
            numeric[
                "dl_sinr"
            ].to_numpy(
                dtype=np.float64
            ),
            normalized_bler,
        )
    )

    cqi_effect, cqi_rows, cqi_reason = (
        safe_spearman(
            numeric[
                "mac_dl_cqi"
            ].to_numpy(
                dtype=np.float64
            ),
            normalized_bler,
        )
    )

    se_effect, se_rows, se_reason = (
        safe_spearman(
            numeric[
                "se"
            ].to_numpy(
                dtype=np.float64
            ),
            normalized_bler,
        )
    )

    sinr_mcs_effect, sinr_mcs_rows, sinr_mcs_reason = (
        safe_spearman(
            numeric[
                "dl_sinr"
            ].to_numpy(
                dtype=np.float64
            ),
            numeric[
                "phy_mcs"
            ].to_numpy(
                dtype=np.float64
            ),
        )
    )

    secondary_values = {
        "unadjusted_sinr_bler":
            (
                unadjusted_effect,
                unadjusted_rows,
                unadjusted_reason,
            ),

        "cqi_bler":
            (
                cqi_effect,
                cqi_rows,
                cqi_reason,
            ),

        "spectral_efficiency_bler":
            (
                se_effect,
                se_rows,
                se_reason,
            ),

        "sinr_mcs":
            (
                sinr_mcs_effect,
                sinr_mcs_rows,
                sinr_mcs_reason,
            ),
    }

    for endpoint, values in secondary_values.items():
        if (
            values[2] == "eligible"
            and np.isfinite(
                values[0]
            )
        ):
            secondary_file_effects[
                endpoint
            ].append(
                {
                    "filename":
                        path.name,

                    "effect":
                        float(
                            values[0]
                        ),
                }
            )

    zenodo_file_rows.append(
        {
            "filename":
                path.name,

            "relative_path":
                inventory_row[
                    "relative_path"
                ],

            "sha256":
                inventory_row[
                    "sha256"
                ],

            "total_row_count":
                len(frame),

            "bler_unit_decision":
                unit_decision,

            "raw_bler_min":
                raw_bler_min,

            "raw_bler_max":
                raw_bler_max,

            "normalized_bler_min":
                float(
                    np.nanmin(
                        normalized_bler
                    )
                ),

            "normalized_bler_max":
                float(
                    np.nanmax(
                        normalized_bler
                    )
                ),

            "primary_eligible":
                primary["eligible"],

            "primary_effect":
                primary["effect"],

            "primary_complete_row_count":
                primary[
                    "complete_row_count"
                ],

            "primary_control_unique_count":
                primary[
                    "control_unique_count"
                ],

            "primary_predictor_residual_sd":
                primary[
                    "predictor_residual_sd"
                ],

            "primary_outcome_residual_sd":
                primary[
                    "outcome_residual_sd"
                ],

            "primary_eligibility_reason":
                primary["reason"],

            "unadjusted_sinr_bler":
                unadjusted_effect,

            "unadjusted_sinr_bler_rows":
                unadjusted_rows,

            "unadjusted_sinr_bler_reason":
                unadjusted_reason,

            "cqi_bler":
                cqi_effect,

            "cqi_bler_rows":
                cqi_rows,

            "cqi_bler_reason":
                cqi_reason,

            "spectral_efficiency_bler":
                se_effect,

            "spectral_efficiency_bler_rows":
                se_rows,

            "spectral_efficiency_bler_reason":
                se_reason,

            "sinr_mcs":
                sinr_mcs_effect,

            "sinr_mcs_rows":
                sinr_mcs_rows,

            "sinr_mcs_reason":
                sinr_mcs_reason,
        }
    )


zenodo_primary_eligible = [
    row
    for row in zenodo_file_rows
    if row["primary_eligible"]
]

zenodo_primary_effects = [
    row["primary_effect"]
    for row in zenodo_primary_eligible
]

zenodo_protocol_complete = bool(
    len(zenodo_primary_effects)
    == 10
)


if len(zenodo_primary_effects) == 0:
    raise RuntimeError(
        "No eligible Zenodo primary file effects."
    )


zenodo_exact = exact_file_sign_flip(
    effects=zenodo_primary_effects,
    alternative="negative",
)

zenodo_bootstrap = (
    file_cluster_bootstrap(
        effects=zenodo_primary_effects,
        iterations=10000,
        seed=60031,
    )
)

zenodo_ci_low, zenodo_ci_high = (
    percentile_interval(
        zenodo_bootstrap
    )
)

zenodo_negative_file_count = int(
    np.sum(
        np.asarray(
            zenodo_primary_effects
        )
        < 0
    )
)

zenodo_supported = bool(
    zenodo_protocol_complete
    and zenodo_exact[
        "aggregate_effect"
    ] < 0
    and zenodo_exact[
        "p_value"
    ] < 0.05
    and zenodo_ci_high < 0
)


zenodo_secondary_rows = []
secondary_raw_p_values = []
secondary_distributions = {}


for endpoint in [
    "unadjusted_sinr_bler",
    "cqi_bler",
    "spectral_efficiency_bler",
    "sinr_mcs",
]:
    records = secondary_file_effects[
        endpoint
    ]

    effects = [
        row["effect"]
        for row in records
    ]

    direction = (
        secondary_expected_directions[
            endpoint
        ]
    )

    if len(effects) == 0:
        aggregate_effect = math.nan
        p_value = 1.0
        distribution = np.asarray(
            [],
            dtype=np.float64,
        )
    else:
        result = exact_file_sign_flip(
            effects=effects,
            alternative=direction,
        )

        aggregate_effect = result[
            "aggregate_effect"
        ]

        p_value = result[
            "p_value"
        ]

        distribution = result[
            "distribution"
        ]

    secondary_raw_p_values.append(
        p_value
    )

    secondary_distributions[
        endpoint
    ] = distribution

    zenodo_secondary_rows.append(
        {
            "endpoint":
                endpoint,

            "expected_direction":
                direction,

            "eligible_file_count":
                len(effects),

            "aggregate_effect":
                aggregate_effect,

            "p_raw":
                p_value,

            "p_holm":
                math.nan,

            "significant_after_holm":
                False,
        }
    )


secondary_adjusted = holm_adjust(
    secondary_raw_p_values
)


for index, adjusted in enumerate(
    secondary_adjusted
):
    zenodo_secondary_rows[
        index
    ]["p_holm"] = float(
        adjusted
    )

    zenodo_secondary_rows[
        index
    ][
        "significant_after_holm"
    ] = bool(
        adjusted < 0.05
    )


# ============================================================
# 4. CROSS-SYSTEM CLASSIFICATION
# ============================================================

support_flags = {
    "source":
        source_supported,

    "sionna":
        sionna_supported,

    "zenodo":
        zenodo_supported,
}


if all(
    support_flags.values()
):
    cross_system_classification = (
        "FULL_DIRECTIONAL_SUPPORT"
    )

elif (
    sionna_supported
    and zenodo_supported
):
    cross_system_classification = (
        "CODED_AND_REAL_SUPPORT"
    )

elif any(
    support_flags.values()
):
    cross_system_classification = (
        "PARTIAL_SUPPORT"
    )

else:
    cross_system_classification = (
        "NO_SUPPORT"
    )


# ============================================================
# 5. WRITE OUTPUTS
# ============================================================

with SOURCE_EFFECTS_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            source_effect_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        source_effect_rows
    )


with SIONNA_EFFECTS_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            sionna_effect_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        sionna_effect_rows
    )


with SIONNA_SUBGROUP_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            sionna_subgroup_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        sionna_subgroup_rows
    )


with ZENODO_FILE_EFFECTS_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            zenodo_file_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        zenodo_file_rows
    )


with ZENODO_SECONDARY_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            zenodo_secondary_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        zenodo_secondary_rows
    )


np.savez_compressed(
    DISTRIBUTIONS_PATH,

    source_bootstrap=
        source_bootstrap,

    source_signflip=
        source_signflip,

    sionna_bootstrap=
        sionna_bootstrap,

    sionna_signflip=
        sionna_signflip,

    zenodo_primary_bootstrap=
        zenodo_bootstrap,

    zenodo_primary_exact_signflip=
        zenodo_exact[
            "distribution"
        ],

    zenodo_secondary_unadjusted_sinr_bler=
        secondary_distributions[
            "unadjusted_sinr_bler"
        ],

    zenodo_secondary_cqi_bler=
        secondary_distributions[
            "cqi_bler"
        ],

    zenodo_secondary_spectral_efficiency_bler=
        secondary_distributions[
            "spectral_efficiency_bler"
        ],

    zenodo_secondary_sinr_mcs=
        secondary_distributions[
            "sinr_mcs"
        ],
)


summary = {
    "schema":
        "phyguard.cross_system_reliability_"
        "statistics.v2",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "PASS",

    "protocol_sha256":
        sha256_file(
            PROTOCOL_PATH
        ),

    "source": {
        "total_sequence_count":
            1080,

        "eligible_sequence_count":
            len(
                source_eligible_rows
            ),

        "ineligible_sequence_count":
            1080
            - len(
                source_eligible_rows
            ),

        "median_mcs_stratified_spearman":
            source_observed,

        "bootstrap_ci95": [
            source_ci_low,
            source_ci_high,
        ],

        "sign_flip_p_one_sided":
            source_p,

        "negative_effect_count":
            int(
                np.sum(
                    source_effects < 0
                )
            ),

        "negative_effect_fraction":
            float(
                np.mean(
                    source_effects < 0
                )
            ),

        "support_rule_satisfied":
            source_supported,

        "secondary_ber_per": {
            "eligible_sequence_count":
                int(
                    np.sum(
                        [
                            np.isfinite(
                                row[
                                    "ber_per_spearman"
                                ]
                            )
                            for row
                            in source_effect_rows
                        ]
                    )
                ),

            "median_spearman":
                float(
                    np.nanmedian(
                        [
                            row[
                                "ber_per_spearman"
                            ]
                            for row
                            in source_effect_rows
                        ]
                    )
                ),
        },

        "secondary_independent_ber_formula": {
            "median_sequence_mean_absolute_deviation":
                float(
                    np.median(
                        [
                            row[
                                "mean_absolute_deviation_from_independent_ber_formula"
                            ]
                            for row
                            in source_effect_rows
                        ]
                    )
                ),

            "interpretation":
                (
                    "Descriptive only; bit errors "
                    "need not be independent."
                ),
        },
    },

    "sionna": {
        "locked_analysis_count":
            825,

        "eligible_sequence_count":
            len(
                sionna_eligible_rows
            ),

        "ineligible_sequence_count":
            825
            - len(
                sionna_eligible_rows
            ),

        "median_mcs_stratified_spearman":
            sionna_observed,

        "bootstrap_ci95": [
            sionna_ci_low,
            sionna_ci_high,
        ],

        "sign_flip_p_one_sided":
            sionna_p,

        "negative_effect_count":
            int(
                np.sum(
                    sionna_effects < 0
                )
            ),

        "negative_effect_fraction":
            float(
                np.mean(
                    sionna_effects < 0
                )
            ),

        "support_rule_satisfied":
            sionna_supported,

        "subgroups":
            sionna_subgroup_rows,

        "secondary_pre_ldpc_ber_to_post_ldpc_bler": {
            "eligible_sequence_count":
                int(
                    np.sum(
                        [
                            np.isfinite(
                                row[
                                    "ber_bler_spearman"
                                ]
                            )
                            for row
                            in sionna_effect_rows
                        ]
                    )
                ),

            "median_spearman":
                float(
                    np.nanmedian(
                        [
                            row[
                                "ber_bler_spearman"
                            ]
                            for row
                            in sionna_effect_rows
                        ]
                    )
                ),
        },
    },

    "zenodo": {
        "file_count":
            10,

        "primary_effect_definition":
            (
                "Within-file partial Spearman correlation "
                "between dl_sinr and dl_bler controlling "
                "continuous phy_mcs."
            ),

        "mcs_encoding":
            "CONTINUOUS_OR_AGGREGATED_MCS",

        "mcs_rounded":
            False,

        "mcs_binned":
            False,

        "eligible_file_count":
            len(
                zenodo_primary_eligible
            ),

        "excluded_file_count":
            10
            - len(
                zenodo_primary_eligible
            ),

        "protocol_complete_all_10_files":
            zenodo_protocol_complete,

        "aggregate_partial_spearman":
            zenodo_exact[
                "aggregate_effect"
            ],

        "aggregate_fisher_z":
            zenodo_exact[
                "aggregate_fisher_z"
            ],

        "bootstrap_ci95": [
            zenodo_ci_low,
            zenodo_ci_high,
        ],

        "exact_sign_flip_p_one_sided":
            zenodo_exact[
                "p_value"
            ],

        "negative_file_count":
            zenodo_negative_file_count,

        "negative_file_fraction":
            (
                zenodo_negative_file_count
                / len(
                    zenodo_primary_effects
                )
            ),

        "support_rule_satisfied":
            zenodo_supported,

        "bler_unit_decisions":
            {
                row["filename"]:
                    row[
                        "bler_unit_decision"
                    ]
                for row
                in zenodo_file_rows
            },

        "secondary_results":
            zenodo_secondary_rows,
    },

    "cross_system": {
        "source_supported":
            source_supported,

        "sionna_supported":
            sionna_supported,

        "zenodo_supported":
            zenodo_supported,

        "classification":
            cross_system_classification,

        "magnitude_comparison_permitted":
            False,

        "calibration_equivalence_permitted":
            False,
    },

    "methodological_boundary": {
        "source_kpi_values_analyzed":
            True,

        "sionna_kpi_values_analyzed":
            True,

        "zenodo_data_rows_read":
            True,

        "endpoint_directions_changed":
            False,

        "zenodo_mcs_control_method_amended_before_outcome_analysis":
            True,

        "zenodo_mcs_rounded":
            False,

        "zenodo_mcs_binned":
            False,

        "analysis_subset_changed":
            False,

        "recording_files_removed_posthoc":
            False,

        "temporal_lag_selected":
            False,

        "smoothing_applied":
            False,

        "thresholds_selected":
            False,

        "models_trained":
            False,

        "secondary_promoted_to_primary":
            False,
    },

    "files": {
        "source_effects":
            str(
                SOURCE_EFFECTS_PATH
            ),

        "sionna_effects":
            str(
                SIONNA_EFFECTS_PATH
            ),

        "sionna_subgroups":
            str(
                SIONNA_SUBGROUP_PATH
            ),

        "zenodo_file_effects":
            str(
                ZENODO_FILE_EFFECTS_PATH
            ),

        "zenodo_secondary":
            str(
                ZENODO_SECONDARY_PATH
            ),

        "distributions":
            str(
                DISTRIBUTIONS_PATH
            ),
    },
}


SUMMARY_PATH.write_text(
    json.dumps(
        json_safe(summary),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ),
    encoding="utf-8",
)


manifest = {
    "schema":
        "phyguard.cross_system_reliability_"
        "statistics_manifest.v2",

    "status":
        "PASS",

    "protocol_sha256":
        sha256_file(
            PROTOCOL_PATH
        ),

    "source_effects_sha256":
        sha256_file(
            SOURCE_EFFECTS_PATH
        ),

    "sionna_effects_sha256":
        sha256_file(
            SIONNA_EFFECTS_PATH
        ),

    "sionna_subgroups_sha256":
        sha256_file(
            SIONNA_SUBGROUP_PATH
        ),

    "zenodo_file_effects_sha256":
        sha256_file(
            ZENODO_FILE_EFFECTS_PATH
        ),

    "zenodo_secondary_sha256":
        sha256_file(
            ZENODO_SECONDARY_PATH
        ),

    "distributions_sha256":
        sha256_file(
            DISTRIBUTIONS_PATH
        ),

    "summary_sha256":
        sha256_file(
            SUMMARY_PATH
        ),

    "cross_system_classification":
        cross_system_classification,

    "methodological_boundary":
        summary[
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


print("status: PASS")

print("\nSOURCE RESULT")
print(
    "eligible_sequence_count:",
    len(source_eligible_rows),
)
print(
    "median_effect:",
    format(
        source_observed,
        ".6f",
    ),
)
print(
    "bootstrap_ci95:",
    (
        format(
            source_ci_low,
            ".6f",
        ),
        format(
            source_ci_high,
            ".6f",
        ),
    ),
)
print(
    "sign_flip_p:",
    format(
        source_p,
        ".8f",
    ),
)
print(
    "support_rule_satisfied:",
    source_supported,
)

print("\nSIONNA RESULT")
print(
    "eligible_sequence_count:",
    len(sionna_eligible_rows),
)
print(
    "median_effect:",
    format(
        sionna_observed,
        ".6f",
    ),
)
print(
    "bootstrap_ci95:",
    (
        format(
            sionna_ci_low,
            ".6f",
        ),
        format(
            sionna_ci_high,
            ".6f",
        ),
    ),
)
print(
    "sign_flip_p:",
    format(
        sionna_p,
        ".8f",
    ),
)
print(
    "support_rule_satisfied:",
    sionna_supported,
)

print("\nSIONNA SUBGROUPS")
for row in sionna_subgroup_rows:
    print(
        row["group"],
        "| eligible=",
        row["eligible_count"],
        "| median=",
        (
            format(
                row["median_effect"],
                ".6f",
            )
            if np.isfinite(
                row["median_effect"]
            )
            else "nan"
        ),
        "| negative_fraction=",
        (
            format(
                row[
                    "negative_effect_fraction"
                ],
                ".6f",
            )
            if np.isfinite(
                row[
                    "negative_effect_fraction"
                ]
            )
            else "nan"
        ),
    )

print("\nZENODO RESULT")
print(
    "eligible_file_count:",
    len(
        zenodo_primary_eligible
    ),
)
print(
    "protocol_complete_all_10_files:",
    zenodo_protocol_complete,
)
print(
    "aggregate_effect:",
    format(
        zenodo_exact[
            "aggregate_effect"
        ],
        ".6f",
    ),
)
print(
    "bootstrap_ci95:",
    (
        format(
            zenodo_ci_low,
            ".6f",
        ),
        format(
            zenodo_ci_high,
            ".6f",
        ),
    ),
)
print(
    "exact_sign_flip_p:",
    format(
        zenodo_exact[
            "p_value"
        ],
        ".8f",
    ),
)
print(
    "negative_file_count:",
    zenodo_negative_file_count,
)
print(
    "support_rule_satisfied:",
    zenodo_supported,
)

print("\nZENODO FILE EFFECTS")
for row in zenodo_file_rows:
    print(
        row["filename"],
        "| rows=",
        row[
            "total_row_count"
        ],
        "| unit=",
        row[
            "bler_unit_decision"
        ],
        "| eligible=",
        row[
            "primary_eligible"
        ],
        "| effect=",
        (
            format(
                row[
                    "primary_effect"
                ],
                ".6f",
            )
            if np.isfinite(
                row[
                    "primary_effect"
                ]
            )
            else "nan"
        ),
        "| complete_rows=",
        row[
            "primary_complete_row_count"
        ],
        "| control_unique=",
        row[
            "primary_control_unique_count"
        ],
    )

print("\nZENODO SECONDARY")
for row in zenodo_secondary_rows:
    print(
        row["endpoint"],
        "| files=",
        row[
            "eligible_file_count"
        ],
        "| effect=",
        format(
            row[
                "aggregate_effect"
            ],
            ".6f",
        ),
        "| p_raw=",
        format(
            row["p_raw"],
            ".8f",
        ),
        "| p_holm=",
        format(
            row["p_holm"],
            ".8f",
        ),
        "| holm_significant=",
        row[
            "significant_after_holm"
        ],
    )

print("\nCROSS-SYSTEM RESULT")
print(
    "source_supported:",
    source_supported,
)
print(
    "sionna_supported:",
    sionna_supported,
)
print(
    "zenodo_supported:",
    zenodo_supported,
)
print(
    "classification:",
    cross_system_classification,
)

print("\nsummary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)

print(
    "\nCROSS_SYSTEM_RELIABILITY_"
    "STATISTICS_V2_PASS"
)
