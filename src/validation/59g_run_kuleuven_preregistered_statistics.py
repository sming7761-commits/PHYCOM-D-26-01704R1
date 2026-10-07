import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("/root/phyguard_revision")

CONTRACT_PATH = (
    ROOT
    / "configs"
    / "kuleuven_statistical_execution_contract_v1.json"
)

CONTRACT_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_statistical_execution_contract_v1.json"
)

ANALYSIS_PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "kuleuven_signal_analysis_protocol_v1.json"
)

ENDPOINT_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_unlabeled_endpoints_v1"
)

ENDPOINT_TABLE_PATH = (
    ENDPOINT_ROOT
    / "unlabeled_sequence_endpoints.csv"
)

ENDPOINT_NAMES_PATH = (
    ENDPOINT_ROOT
    / "endpoint_names.json"
)

ENDPOINT_SUMMARY_PATH = (
    ENDPOINT_ROOT
    / "summary.json"
)

LINEAGE_PATH = (
    ROOT
    / "artifacts"
    / "kuleuven_unlabeled_features_v1"
    / "unlabeled_sequence_lineage.csv"
)

LABEL_MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "kuleuven_scenario1_external_validation_protocol_v1"
    / "scenario1_file_manifest.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_preregistered_statistics_v1"
)

JOINED_TABLE_PATH = (
    OUTPUT_ROOT
    / "joined_sequence_endpoints.csv"
)

ACTIVITY_TABLE_PATH = (
    OUTPUT_ROOT
    / "activity_median_endpoints.csv"
)

ENDPOINT_RESULTS_PATH = (
    OUTPUT_ROOT
    / "endpoint_results.csv"
)

BOOTSTRAP_PATH = (
    OUTPUT_ROOT
    / "primary_cluster_bootstrap.npz"
)

PERMUTATION_PATH = (
    OUTPUT_ROOT
    / "activity_permutation_distributions.npz"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_preregistered_statistics_v1.json"
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

ENDPOINT_NAMES = [
    PRIMARY_ENDPOINT,
    *SECONDARY_ENDPOINTS,
]

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 59031

PRIMARY_PERMUTATIONS = 100_000
PRIMARY_PERMUTATION_SEED = 59032

SECONDARY_PERMUTATIONS = 100_000
SECONDARY_SEED_BASE = 59040

EXPECTED_SEQUENCE_COUNT = 160
EXPECTED_ACTIVITY_COUNT = 32
EXPECTED_POSITIVE_ACTIVITIES = 20
EXPECTED_NEGATIVE_ACTIVITIES = 12
EXPECTED_POSITIVE_SEQUENCES = 100
EXPECTED_NEGATIVE_SEQUENCES = 60
EXPECTED_REPETITIONS = {1, 2, 3, 4, 5}


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


def load_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def binary_auc(labels, scores):
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if labels.shape != scores.shape:
        raise ValueError(
            "Label and score shapes differ."
        )

    positive_count = int(
        np.sum(labels == 1)
    )

    negative_count = int(
        np.sum(labels == 0)
    )

    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "Both classes are required for AUROC."
        )

    ranks = rankdata(
        scores,
        method="average",
    )

    positive_rank_sum = float(
        np.sum(
            ranks[labels == 1]
        )
    )

    auc = (
        positive_rank_sum
        - positive_count
        * (positive_count + 1)
        / 2.0
    ) / (
        positive_count
        * negative_count
    )

    return float(auc)


def cliff_delta(
    positive_values,
    negative_values,
):
    positive_values = np.asarray(
        positive_values,
        dtype=np.float64,
    )

    negative_values = np.asarray(
        negative_values,
        dtype=np.float64,
    )

    differences = (
        positive_values[:, None]
        - negative_values[None, :]
    )

    greater = int(
        np.sum(differences > 0)
    )

    lower = int(
        np.sum(differences < 0)
    )

    denominator = (
        positive_values.size
        * negative_values.size
    )

    return float(
        (greater - lower)
        / denominator
    )


def descriptive(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "count":
            int(values.size),

        "median":
            float(
                np.median(values)
            ),

        "q25":
            float(
                np.quantile(
                    values,
                    0.25,
                    method="linear",
                )
            ),

        "q75":
            float(
                np.quantile(
                    values,
                    0.75,
                    method="linear",
                )
            ),

        "iqr":
            float(
                np.quantile(
                    values,
                    0.75,
                    method="linear",
                )
                - np.quantile(
                    values,
                    0.25,
                    method="linear",
                )
            ),
    }


def percentile_interval(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return (
        float(
            np.quantile(
                values,
                0.025,
                method="linear",
            )
        ),
        float(
            np.quantile(
                values,
                0.975,
                method="linear",
            )
        ),
    )


def holm_adjust(p_values):
    p_values = np.asarray(
        p_values,
        dtype=np.float64,
    )

    count = p_values.size

    order = np.argsort(
        p_values,
        kind="mergesort",
    )

    adjusted_sorted = np.empty(
        count,
        dtype=np.float64,
    )

    running_maximum = 0.0

    for rank_index, original_index in enumerate(
        order
    ):
        multiplier = (
            count - rank_index
        )

        candidate = min(
            1.0,
            multiplier
            * p_values[original_index],
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted_sorted[
            rank_index
        ] = running_maximum

    adjusted = np.empty(
        count,
        dtype=np.float64,
    )

    for rank_index, original_index in enumerate(
        order
    ):
        adjusted[
            original_index
        ] = adjusted_sorted[
            rank_index
        ]

    return adjusted


def permutation_auc_distribution(
    activity_scores,
    positive_count,
    permutations,
    seed,
    batch_size=5000,
):
    activity_scores = np.asarray(
        activity_scores,
        dtype=np.float64,
    )

    activity_count = int(
        activity_scores.size
    )

    negative_count = (
        activity_count
        - positive_count
    )

    ranks = rankdata(
        activity_scores,
        method="average",
    )

    result = np.empty(
        permutations,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        seed
    )

    offset = 0

    while offset < permutations:
        batch = min(
            batch_size,
            permutations - offset,
        )

        random_values = rng.random(
            (
                batch,
                activity_count,
            )
        )

        positive_indices = np.argpartition(
            random_values,
            positive_count - 1,
            axis=1,
        )[:, :positive_count]

        rank_sums = np.sum(
            ranks[
                positive_indices
            ],
            axis=1,
        )

        auc_values = (
            rank_sums
            - positive_count
            * (positive_count + 1)
            / 2.0
        ) / (
            positive_count
            * negative_count
        )

        result[
            offset:offset + batch
        ] = auc_values

        offset += batch

    return result


required_paths = [
    CONTRACT_PATH,
    CONTRACT_MANIFEST_PATH,
    ANALYSIS_PROTOCOL_PATH,
    ENDPOINT_TABLE_PATH,
    ENDPOINT_NAMES_PATH,
    ENDPOINT_SUMMARY_PATH,
    LINEAGE_PATH,
    LABEL_MANIFEST_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)


contract = load_json(
    CONTRACT_PATH
)

contract_manifest = load_json(
    CONTRACT_MANIFEST_PATH
)

endpoint_summary = load_json(
    ENDPOINT_SUMMARY_PATH
)

endpoint_names_record = load_json(
    ENDPOINT_NAMES_PATH
)


if (
    contract.get("status")
    != "LOCKED_BEFORE_LABEL_JOIN_AND_STATISTICAL_ANALYSIS"
):
    raise RuntimeError(
        "Statistical contract is not in the locked state."
    )


if contract_manifest.get("status") != "PASS":
    raise RuntimeError(
        "Statistical-contract manifest is not PASS."
    )


if endpoint_summary.get("status") != "PASS":
    raise RuntimeError(
        "Unlabeled-endpoint summary is not PASS."
    )


if (
    contract.get(
        "dataset_snapshot_sha256"
    )
    != EXPECTED_DATASET_SNAPSHOT_SHA256
):
    raise RuntimeError(
        "Contract dataset snapshot mismatch."
    )


if (
    contract.get(
        "unlabeled_feature_snapshot_sha256"
    )
    != EXPECTED_FEATURE_SNAPSHOT_SHA256
):
    raise RuntimeError(
        "Contract feature snapshot mismatch."
    )


if (
    contract.get(
        "unlabeled_endpoint_snapshot_sha256"
    )
    != EXPECTED_ENDPOINT_SNAPSHOT_SHA256
):
    raise RuntimeError(
        "Contract endpoint snapshot mismatch."
    )


if (
    endpoint_summary.get(
        "unlabeled_endpoint_snapshot_sha256"
    )
    != EXPECTED_ENDPOINT_SNAPSHOT_SHA256
):
    raise RuntimeError(
        "Endpoint summary snapshot mismatch."
    )


if (
    endpoint_names_record.get(
        "endpoint_names"
    )
    != ENDPOINT_NAMES
):
    raise RuntimeError(
        "Endpoint-name order mismatch."
    )


if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Stage 59G output already exists; "
        "refusing to overwrite it."
    )


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)


endpoint_rows = load_csv(
    ENDPOINT_TABLE_PATH
)

lineage_rows = load_csv(
    LINEAGE_PATH
)

label_rows = load_csv(
    LABEL_MANIFEST_PATH
)


if len(endpoint_rows) != EXPECTED_SEQUENCE_COUNT:
    raise RuntimeError(
        "Endpoint-row count mismatch."
    )


if len(lineage_rows) != EXPECTED_SEQUENCE_COUNT:
    raise RuntimeError(
        "Lineage-row count mismatch."
    )


if len(label_rows) != EXPECTED_SEQUENCE_COUNT:
    raise RuntimeError(
        "Label-manifest row count mismatch."
    )


lineage_by_uid = {}

for row in lineage_rows:
    uid = row["sequence_uid"]

    if uid in lineage_by_uid:
        raise RuntimeError(
            f"Duplicate lineage UID: {uid}"
        )

    lineage_by_uid[uid] = row


label_by_filename = {}

for row in label_rows:
    filename = row["filename"]

    if filename in label_by_filename:
        raise RuntimeError(
            f"Duplicate label filename: {filename}"
        )

    label_by_filename[
        filename
    ] = row


joined_rows = []

seen_filenames = set()


for expected_index, endpoint_row in enumerate(
    endpoint_rows
):
    sequence_index = int(
        endpoint_row["sequence_index"]
    )

    if sequence_index != expected_index:
        raise RuntimeError(
            "Endpoint sequence-index order mismatch."
        )

    uid = endpoint_row[
        "sequence_uid"
    ]

    lineage = lineage_by_uid.get(uid)

    if lineage is None:
        raise RuntimeError(
            f"Missing lineage for UID: {uid}"
        )

    if int(
        lineage["sequence_index"]
    ) != sequence_index:
        raise RuntimeError(
            f"Lineage-index mismatch: {uid}"
        )

    filename = lineage[
        "filename"
    ]

    label = label_by_filename.get(
        filename
    )

    if label is None:
        raise RuntimeError(
            f"Missing label for filename: {filename}"
        )

    if filename in seen_filenames:
        raise RuntimeError(
            f"Duplicate joined filename: {filename}"
        )

    seen_filenames.add(
        filename
    )

    joined = {
        "sequence_index":
            sequence_index,

        "sequence_uid":
            uid,

        "filename":
            filename,

        "sequence_id":
            label["sequence_id"],

        "activity":
            int(
                label["activity"]
            ),

        "repetition":
            int(
                label["repetition"]
            ),

        "sequence_label":
            label[
                "sequence_label"
            ],

        "sequence_label_id":
            int(
                label[
                    "sequence_label_id"
                ]
            ),
    }

    for endpoint_name in ENDPOINT_NAMES:
        value = float(
            endpoint_row[
                endpoint_name
            ]
        )

        if not math.isfinite(value):
            raise RuntimeError(
                f"Non-finite endpoint: "
                f"{uid}, {endpoint_name}"
            )

        joined[
            endpoint_name
        ] = value

    joined_rows.append(
        joined
    )


if set(label_by_filename) != seen_filenames:
    raise RuntimeError(
        "Label join contains missing or unused filenames."
    )


positive_sequence_count = sum(
    row["sequence_label_id"] == 1
    for row in joined_rows
)

negative_sequence_count = sum(
    row["sequence_label_id"] == 0
    for row in joined_rows
)


if (
    positive_sequence_count
    != EXPECTED_POSITIVE_SEQUENCES
):
    raise RuntimeError(
        "Positive sequence count mismatch."
    )


if (
    negative_sequence_count
    != EXPECTED_NEGATIVE_SEQUENCES
):
    raise RuntimeError(
        "Negative sequence count mismatch."
    )


activity_groups = {}

for row in joined_rows:
    activity_groups.setdefault(
        row["activity"],
        [],
    ).append(row)


if len(activity_groups) != EXPECTED_ACTIVITY_COUNT:
    raise RuntimeError(
        "Activity-cluster count mismatch."
    )


activity_rows = []


for activity in sorted(
    activity_groups
):
    rows = activity_groups[
        activity
    ]

    repetitions = {
        row["repetition"]
        for row in rows
    }

    if repetitions != EXPECTED_REPETITIONS:
        raise RuntimeError(
            f"Incomplete repetitions for activity {activity}."
        )

    labels = {
        row["sequence_label_id"]
        for row in rows
    }

    if len(labels) != 1:
        raise RuntimeError(
            f"Inconsistent activity label: {activity}"
        )

    activity_row = {
        "activity":
            activity,

        "sequence_label_id":
            int(next(iter(labels))),

        "sequence_label":
            rows[0]["sequence_label"],

        "repetition_count":
            len(rows),
    }

    for endpoint_name in ENDPOINT_NAMES:
        activity_row[
            endpoint_name
        ] = float(
            np.median(
                [
                    row[
                        endpoint_name
                    ]
                    for row in rows
                ]
            )
        )

    activity_rows.append(
        activity_row
    )


positive_activity_count = sum(
    row["sequence_label_id"] == 1
    for row in activity_rows
)

negative_activity_count = sum(
    row["sequence_label_id"] == 0
    for row in activity_rows
)


if (
    positive_activity_count
    != EXPECTED_POSITIVE_ACTIVITIES
):
    raise RuntimeError(
        "Positive activity count mismatch."
    )


if (
    negative_activity_count
    != EXPECTED_NEGATIVE_ACTIVITIES
):
    raise RuntimeError(
        "Negative activity count mismatch."
    )


sequence_labels = np.asarray(
    [
        row["sequence_label_id"]
        for row in joined_rows
    ],
    dtype=np.int64,
)

activity_labels = np.asarray(
    [
        row["sequence_label_id"]
        for row in activity_rows
    ],
    dtype=np.int64,
)


endpoint_results = []


for endpoint_name in ENDPOINT_NAMES:
    sequence_scores = np.asarray(
        [
            row[endpoint_name]
            for row in joined_rows
        ],
        dtype=np.float64,
    )

    activity_scores = np.asarray(
        [
            row[endpoint_name]
            for row in activity_rows
        ],
        dtype=np.float64,
    )

    sequence_auc = binary_auc(
        sequence_labels,
        sequence_scores,
    )

    sequence_ap = float(
        average_precision_score(
            sequence_labels,
            sequence_scores,
        )
    )

    activity_auc = binary_auc(
        activity_labels,
        activity_scores,
    )

    positive_sequence_values = (
        sequence_scores[
            sequence_labels == 1
        ]
    )

    negative_sequence_values = (
        sequence_scores[
            sequence_labels == 0
        ]
    )

    positive_activity_values = (
        activity_scores[
            activity_labels == 1
        ]
    )

    negative_activity_values = (
        activity_scores[
            activity_labels == 0
        ]
    )

    endpoint_results.append(
        {
            "endpoint":
                endpoint_name,

            "endpoint_role":
                (
                    "primary"
                    if endpoint_name
                    == PRIMARY_ENDPOINT
                    else "secondary"
                ),

            "sequence_auroc":
                sequence_auc,

            "sequence_average_precision":
                sequence_ap,

            "activity_auroc":
                activity_auc,

            "activity_cliffs_delta":
                cliff_delta(
                    positive_activity_values,
                    negative_activity_values,
                ),

            "sequence_positive_count":
                int(
                    positive_sequence_values.size
                ),

            "sequence_positive_median":
                descriptive(
                    positive_sequence_values
                )["median"],

            "sequence_positive_q25":
                descriptive(
                    positive_sequence_values
                )["q25"],

            "sequence_positive_q75":
                descriptive(
                    positive_sequence_values
                )["q75"],

            "sequence_negative_count":
                int(
                    negative_sequence_values.size
                ),

            "sequence_negative_median":
                descriptive(
                    negative_sequence_values
                )["median"],

            "sequence_negative_q25":
                descriptive(
                    negative_sequence_values
                )["q25"],

            "sequence_negative_q75":
                descriptive(
                    negative_sequence_values
                )["q75"],

            "activity_positive_count":
                int(
                    positive_activity_values.size
                ),

            "activity_positive_median":
                descriptive(
                    positive_activity_values
                )["median"],

            "activity_positive_q25":
                descriptive(
                    positive_activity_values
                )["q25"],

            "activity_positive_q75":
                descriptive(
                    positive_activity_values
                )["q75"],

            "activity_negative_count":
                int(
                    negative_activity_values.size
                ),

            "activity_negative_median":
                descriptive(
                    negative_activity_values
                )["median"],

            "activity_negative_q25":
                descriptive(
                    negative_activity_values
                )["q25"],

            "activity_negative_q75":
                descriptive(
                    negative_activity_values
                )["q75"],

            "permutation_alternative":
                (
                    "positive_greater"
                    if endpoint_name
                    == PRIMARY_ENDPOINT
                    else "two_sided"
                ),

            "permutation_p_raw":
                None,

            "permutation_p_holm":
                None,

            "bootstrap_sequence_auroc_ci_low":
                None,

            "bootstrap_sequence_auroc_ci_high":
                None,

            "bootstrap_sequence_ap_ci_low":
                None,

            "bootstrap_sequence_ap_ci_high":
                None,

            "bootstrap_activity_auroc_ci_low":
                None,

            "bootstrap_activity_auroc_ci_high":
                None,
        }
    )


result_by_endpoint = {
    row["endpoint"]: row
    for row in endpoint_results
}


# ------------------------------------------------------------
# Primary stratified activity-cluster bootstrap.
# ------------------------------------------------------------

positive_activities = [
    row
    for row in activity_rows
    if row["sequence_label_id"] == 1
]

negative_activities = [
    row
    for row in activity_rows
    if row["sequence_label_id"] == 0
]


def sequence_values_for_activity(
    activity,
    endpoint_name,
):
    rows = activity_groups[
        activity
    ]

    rows = sorted(
        rows,
        key=lambda row: row[
            "repetition"
        ],
    )

    return np.asarray(
        [
            row[endpoint_name]
            for row in rows
        ],
        dtype=np.float64,
    )


positive_sequence_matrix = np.stack(
    [
        sequence_values_for_activity(
            row["activity"],
            PRIMARY_ENDPOINT,
        )
        for row in positive_activities
    ],
    axis=0,
)

negative_sequence_matrix = np.stack(
    [
        sequence_values_for_activity(
            row["activity"],
            PRIMARY_ENDPOINT,
        )
        for row in negative_activities
    ],
    axis=0,
)

positive_activity_scores = np.asarray(
    [
        row[PRIMARY_ENDPOINT]
        for row in positive_activities
    ],
    dtype=np.float64,
)

negative_activity_scores = np.asarray(
    [
        row[PRIMARY_ENDPOINT]
        for row in negative_activities
    ],
    dtype=np.float64,
)


bootstrap_sequence_auc = np.empty(
    BOOTSTRAP_RESAMPLES,
    dtype=np.float64,
)

bootstrap_sequence_ap = np.empty(
    BOOTSTRAP_RESAMPLES,
    dtype=np.float64,
)

bootstrap_activity_auc = np.empty(
    BOOTSTRAP_RESAMPLES,
    dtype=np.float64,
)


bootstrap_rng = np.random.default_rng(
    BOOTSTRAP_SEED
)


for bootstrap_index in range(
    BOOTSTRAP_RESAMPLES
):
    sampled_positive_indices = (
        bootstrap_rng.integers(
            0,
            EXPECTED_POSITIVE_ACTIVITIES,
            size=EXPECTED_POSITIVE_ACTIVITIES,
        )
    )

    sampled_negative_indices = (
        bootstrap_rng.integers(
            0,
            EXPECTED_NEGATIVE_ACTIVITIES,
            size=EXPECTED_NEGATIVE_ACTIVITIES,
        )
    )

    positive_sequence_sample = (
        positive_sequence_matrix[
            sampled_positive_indices
        ].reshape(-1)
    )

    negative_sequence_sample = (
        negative_sequence_matrix[
            sampled_negative_indices
        ].reshape(-1)
    )

    sequence_scores = np.concatenate(
        [
            positive_sequence_sample,
            negative_sequence_sample,
        ]
    )

    sequence_bootstrap_labels = np.concatenate(
        [
            np.ones(
                positive_sequence_sample.size,
                dtype=np.int64,
            ),
            np.zeros(
                negative_sequence_sample.size,
                dtype=np.int64,
            ),
        ]
    )

    bootstrap_sequence_auc[
        bootstrap_index
    ] = binary_auc(
        sequence_bootstrap_labels,
        sequence_scores,
    )

    bootstrap_sequence_ap[
        bootstrap_index
    ] = average_precision_score(
        sequence_bootstrap_labels,
        sequence_scores,
    )

    activity_scores = np.concatenate(
        [
            positive_activity_scores[
                sampled_positive_indices
            ],
            negative_activity_scores[
                sampled_negative_indices
            ],
        ]
    )

    activity_bootstrap_labels = np.concatenate(
        [
            np.ones(
                EXPECTED_POSITIVE_ACTIVITIES,
                dtype=np.int64,
            ),
            np.zeros(
                EXPECTED_NEGATIVE_ACTIVITIES,
                dtype=np.int64,
            ),
        ]
    )

    bootstrap_activity_auc[
        bootstrap_index
    ] = binary_auc(
        activity_bootstrap_labels,
        activity_scores,
    )


sequence_auc_ci = percentile_interval(
    bootstrap_sequence_auc
)

sequence_ap_ci = percentile_interval(
    bootstrap_sequence_ap
)

activity_auc_ci = percentile_interval(
    bootstrap_activity_auc
)


primary_result = result_by_endpoint[
    PRIMARY_ENDPOINT
]

primary_result[
    "bootstrap_sequence_auroc_ci_low"
] = sequence_auc_ci[0]

primary_result[
    "bootstrap_sequence_auroc_ci_high"
] = sequence_auc_ci[1]

primary_result[
    "bootstrap_sequence_ap_ci_low"
] = sequence_ap_ci[0]

primary_result[
    "bootstrap_sequence_ap_ci_high"
] = sequence_ap_ci[1]

primary_result[
    "bootstrap_activity_auroc_ci_low"
] = activity_auc_ci[0]

primary_result[
    "bootstrap_activity_auroc_ci_high"
] = activity_auc_ci[1]


np.savez_compressed(
    BOOTSTRAP_PATH,
    sequence_auroc=
        bootstrap_sequence_auc,

    sequence_average_precision=
        bootstrap_sequence_ap,

    activity_auroc=
        bootstrap_activity_auc,

    bootstrap_seed=np.asarray(
        [BOOTSTRAP_SEED],
        dtype=np.int64,
    ),
)


# ------------------------------------------------------------
# Primary and secondary activity-level permutation tests.
# ------------------------------------------------------------

permutation_arrays = {}


primary_activity_scores_all = np.asarray(
    [
        row[PRIMARY_ENDPOINT]
        for row in activity_rows
    ],
    dtype=np.float64,
)

primary_permutation_auc = (
    permutation_auc_distribution(
        activity_scores=
            primary_activity_scores_all,

        positive_count=
            EXPECTED_POSITIVE_ACTIVITIES,

        permutations=
            PRIMARY_PERMUTATIONS,

        seed=
            PRIMARY_PERMUTATION_SEED,
    )
)


primary_observed_auc = primary_result[
    "activity_auroc"
]

primary_extreme_count = int(
    np.sum(
        primary_permutation_auc
        >= primary_observed_auc
    )
)

primary_p_value = float(
    (
        1
        + primary_extreme_count
    )
    / (
        PRIMARY_PERMUTATIONS
        + 1
    )
)

primary_result[
    "permutation_p_raw"
] = primary_p_value

primary_result[
    "permutation_p_holm"
] = primary_p_value

permutation_arrays[
    "primary_activity_auroc"
] = primary_permutation_auc


secondary_raw_p_values = []


for secondary_index, endpoint_name in enumerate(
    SECONDARY_ENDPOINTS
):
    activity_scores = np.asarray(
        [
            row[endpoint_name]
            for row in activity_rows
        ],
        dtype=np.float64,
    )

    seed = (
        SECONDARY_SEED_BASE
        + secondary_index
    )

    permutation_auc = (
        permutation_auc_distribution(
            activity_scores=
                activity_scores,

            positive_count=
                EXPECTED_POSITIVE_ACTIVITIES,

            permutations=
                SECONDARY_PERMUTATIONS,

            seed=
                seed,
        )
    )

    observed_auc = result_by_endpoint[
        endpoint_name
    ][
        "activity_auroc"
    ]

    observed_distance = abs(
        observed_auc - 0.5
    )

    extreme_count = int(
        np.sum(
            np.abs(
                permutation_auc
                - 0.5
            )
            >= observed_distance
        )
    )

    raw_p_value = float(
        (
            1
            + extreme_count
        )
        / (
            SECONDARY_PERMUTATIONS
            + 1
        )
    )

    result_by_endpoint[
        endpoint_name
    ][
        "permutation_p_raw"
    ] = raw_p_value

    secondary_raw_p_values.append(
        raw_p_value
    )

    permutation_arrays[
        (
            f"secondary_{secondary_index}_"
            f"{endpoint_name}"
        )
    ] = permutation_auc


secondary_adjusted_p_values = holm_adjust(
    secondary_raw_p_values
)


for endpoint_name, adjusted_p_value in zip(
    SECONDARY_ENDPOINTS,
    secondary_adjusted_p_values,
):
    result_by_endpoint[
        endpoint_name
    ][
        "permutation_p_holm"
    ] = float(
        adjusted_p_value
    )


permutation_arrays[
    "primary_seed"
] = np.asarray(
    [PRIMARY_PERMUTATION_SEED],
    dtype=np.int64,
)

permutation_arrays[
    "secondary_seeds"
] = np.asarray(
    [
        SECONDARY_SEED_BASE
        + index
        for index in range(
            len(
                SECONDARY_ENDPOINTS
            )
        )
    ],
    dtype=np.int64,
)


np.savez_compressed(
    PERMUTATION_PATH,
    **permutation_arrays,
)


# ------------------------------------------------------------
# Write complete results after all calculations have finished.
# ------------------------------------------------------------

if not OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Stage 59G output directory disappeared unexpectedly."
    )


with JOINED_TABLE_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            joined_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        joined_rows
    )


with ACTIVITY_TABLE_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            activity_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        activity_rows
    )


with ENDPOINT_RESULTS_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            endpoint_results[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        endpoint_results
    )


primary_summary = {
    key: value
    for key, value
    in primary_result.items()
}


summary = {
    "schema":
        "phyguard.kuleuven."
        "preregistered_statistics.v1",

    "status":
        "PASS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "dataset_snapshot_sha256":
        EXPECTED_DATASET_SNAPSHOT_SHA256,

    "unlabeled_feature_snapshot_sha256":
        EXPECTED_FEATURE_SNAPSHOT_SHA256,

    "unlabeled_endpoint_snapshot_sha256":
        EXPECTED_ENDPOINT_SNAPSHOT_SHA256,

    "statistical_contract_path":
        str(CONTRACT_PATH),

    "statistical_contract_sha256":
        sha256_file(
            CONTRACT_PATH
        ),

    "population": {
        "sequence_count":
            len(joined_rows),

        "positive_sequence_count":
            positive_sequence_count,

        "negative_sequence_count":
            negative_sequence_count,

        "activity_count":
            len(activity_rows),

        "positive_activity_count":
            positive_activity_count,

        "negative_activity_count":
            negative_activity_count,

        "sequence_exclusion_count":
            0,
    },

    "primary_result":
        primary_summary,

    "secondary_results": [
        row
        for row in endpoint_results
        if row["endpoint_role"]
        == "secondary"
    ],

    "analysis_contract": {
        "bootstrap_resamples":
            BOOTSTRAP_RESAMPLES,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "primary_permutations":
            PRIMARY_PERMUTATIONS,

        "primary_permutation_seed":
            PRIMARY_PERMUTATION_SEED,

        "secondary_permutations_per_endpoint":
            SECONDARY_PERMUTATIONS,

        "secondary_seed_base":
            SECONDARY_SEED_BASE,

        "secondary_adjustment":
            "Holm",
    },

    "files": {
        "joined_sequence_table":
            str(
                JOINED_TABLE_PATH
            ),

        "activity_median_table":
            str(
                ACTIVITY_TABLE_PATH
            ),

        "endpoint_results":
            str(
                ENDPOINT_RESULTS_PATH
            ),

        "bootstrap_distributions":
            str(
                BOOTSTRAP_PATH
            ),

        "permutation_distributions":
            str(
                PERMUTATION_PATH
            ),
    },

    "methodological_boundary": {
        "label_manifest_opened":
            True,

        "labels_joined":
            True,

        "all_160_sequences_used":
            True,

        "samples_excluded":
            False,

        "models_trained":
            False,

        "thresholds_selected":
            False,

        "thresholded_metrics_computed":
            False,

        "endpoint_signs_reversed":
            False,

        "endpoint_definitions_changed":
            False,

        "performance_based_endpoint_selection":
            False,

        "performance_based_sample_selection":
            False,
    },
}


SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


manifest_files = [
    JOINED_TABLE_PATH,
    ACTIVITY_TABLE_PATH,
    ENDPOINT_RESULTS_PATH,
    BOOTSTRAP_PATH,
    PERMUTATION_PATH,
    SUMMARY_PATH,
]


manifest = {
    "schema":
        "phyguard.kuleuven."
        "preregistered_statistics_manifest.v1",

    "status":
        "PASS",

    "dataset_snapshot_sha256":
        EXPECTED_DATASET_SNAPSHOT_SHA256,

    "unlabeled_feature_snapshot_sha256":
        EXPECTED_FEATURE_SNAPSHOT_SHA256,

    "unlabeled_endpoint_snapshot_sha256":
        EXPECTED_ENDPOINT_SNAPSHOT_SHA256,

    "statistical_contract_sha256":
        sha256_file(
            CONTRACT_PATH
        ),

    "sequence_count":
        len(joined_rows),

    "activity_count":
        len(activity_rows),

    "sequence_exclusion_count":
        0,

    "summary_sha256":
        sha256_file(
            SUMMARY_PATH
        ),

    "files": [
        {
            "relative_path":
                str(
                    path.relative_to(
                        ROOT
                    )
                ),

            "size_bytes":
                path.stat().st_size,

            "sha256":
                sha256_file(path),
        }
        for path in manifest_files
    ],

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
print("sequence_count:", len(joined_rows))
print(
    "positive_sequence_count:",
    positive_sequence_count,
)
print(
    "negative_sequence_count:",
    negative_sequence_count,
)
print("activity_count:", len(activity_rows))
print(
    "positive_activity_count:",
    positive_activity_count,
)
print(
    "negative_activity_count:",
    negative_activity_count,
)
print("sequence_exclusion_count: 0")

print("\nPRIMARY ENDPOINT")
print(
    "name:",
    PRIMARY_ENDPOINT,
)
print(
    "sequence_auroc:",
    format(
        primary_result[
            "sequence_auroc"
        ],
        ".6f",
    ),
)
print(
    "sequence_auroc_ci95:",
    (
        format(
            primary_result[
                "bootstrap_sequence_auroc_ci_low"
            ],
            ".6f",
        ),
        format(
            primary_result[
                "bootstrap_sequence_auroc_ci_high"
            ],
            ".6f",
        ),
    ),
)
print(
    "sequence_average_precision:",
    format(
        primary_result[
            "sequence_average_precision"
        ],
        ".6f",
    ),
)
print(
    "sequence_average_precision_ci95:",
    (
        format(
            primary_result[
                "bootstrap_sequence_ap_ci_low"
            ],
            ".6f",
        ),
        format(
            primary_result[
                "bootstrap_sequence_ap_ci_high"
            ],
            ".6f",
        ),
    ),
)
print(
    "activity_auroc:",
    format(
        primary_result[
            "activity_auroc"
        ],
        ".6f",
    ),
)
print(
    "activity_auroc_ci95:",
    (
        format(
            primary_result[
                "bootstrap_activity_auroc_ci_low"
            ],
            ".6f",
        ),
        format(
            primary_result[
                "bootstrap_activity_auroc_ci_high"
            ],
            ".6f",
        ),
    ),
)
print(
    "activity_cliffs_delta:",
    format(
        primary_result[
            "activity_cliffs_delta"
        ],
        ".6f",
    ),
)
print(
    "primary_permutation_p_one_sided:",
    format(
        primary_result[
            "permutation_p_raw"
        ],
        ".8f",
    ),
)
print(
    "positive_activity_median:",
    format(
        primary_result[
            "activity_positive_median"
        ],
        ".6f",
    ),
)
print(
    "negative_activity_median:",
    format(
        primary_result[
            "activity_negative_median"
        ],
        ".6f",
    ),
)

print("\nSECONDARY ENDPOINTS")

for row in endpoint_results:
    if row["endpoint_role"] != "secondary":
        continue

    print(
        row["endpoint"],
        "| sequence_auc=",
        format(
            row["sequence_auroc"],
            ".6f",
        ),
        "| activity_auc=",
        format(
            row["activity_auroc"],
            ".6f",
        ),
        "| delta=",
        format(
            row["activity_cliffs_delta"],
            ".6f",
        ),
        "| p_raw=",
        format(
            row["permutation_p_raw"],
            ".8f",
        ),
        "| p_holm=",
        format(
            row["permutation_p_holm"],
            ".8f",
        ),
    )

print()
print("all_160_sequences_used: True")
print("samples_excluded: False")
print("models_trained: False")
print("thresholds_selected: False")
print("thresholded_metrics_computed: False")
print("endpoint_signs_reversed: False")
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)
print(
    "\nKULEUVEN_PREREGISTERED_"
    "STATISTICS_V1_PASS"
)
