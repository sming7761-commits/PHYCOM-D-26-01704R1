import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/phyguard_revision")

RESULT_ROOT = (
    ROOT
    / "results"
    / "sionna_formal825_detector_conditioned_diagnosis_v1"
)

PREDICTIONS_PATH = (
    RESULT_ROOT
    / "predictions.csv"
)

PER_REPEAT_METRICS_PATH = (
    RESULT_ROOT
    / "per_repeat_metrics.csv"
)

PER_REPEAT_DELTAS_PATH = (
    RESULT_ROOT
    / "per_repeat_deltas_vs_locked_interval.csv"
)

DIAGNOSIS_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal825_detector_conditioned_diagnosis_v1.json"
)

PROPOSAL_EVALUATION_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal825_upstream_detector_evaluation_v1.json"
)

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "sionna_formal825_upstream_detector_protocol_v1.json"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "sionna_formal825_detector_conditioned_statistics_v1"
)

TEMP_ROOT = Path(
    str(OUTPUT_ROOT) + "_tmp"
)

OUTPUT_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal825_detector_conditioned_statistics_v1.json"
)

EXPECTED_PROTOCOL_SHA256 = (
    "4a9b09b82e82efa4940602a73a25cdd"
    "aa05361af4c5edee2ae73f46860996aa2"
)

DETECTORS = [
    "robust_energy",
    "pca_reconstruction",
    "isolation_forest",
]

PHYSICAL_CLASSES = [
    "adaptation_mismatch",
    "blockage",
    "interference",
    "mobility",
]

CONTROL_CLASSES = [
    "normal",
    "nonphysical_goodput",
]

BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20260801
BOOTSTRAP_CHUNK_SIZE = 200

T_CRITICAL_95_DF4 = 2.7764451051977987

PAIRED_METRICS = [
    "selected_accuracy",
    "physical_selection_rate",
    "physical_exact_diagnosis_rate",
    "control_abstention_rate",
    "false_specific_rate",
    "overall_output_rate",
    "overall_task_success_rate",
]

CHANGE_METRICS = [
    "prediction_change_rate",
    "selection_change_rate",
    "task_success_change_rate",
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


def bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    if mapped.isna().any():
        raise RuntimeError(
            "Cannot convert one or more values to Boolean."
        )

    return mapped.astype(bool)


def percentile_interval(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    finite = values[
        np.isfinite(values)
    ]

    if len(finite) == 0:
        return {
            "lower_95": None,
            "upper_95": None,
        }

    return {
        "lower_95":
            float(
                np.quantile(
                    finite,
                    0.025,
                )
            ),

        "upper_95":
            float(
                np.quantile(
                    finite,
                    0.975,
                )
            ),
    }


def rank_correlation(x, y):
    x = pd.Series(
        np.asarray(
            x,
            dtype=float,
        )
    )

    y = pd.Series(
        np.asarray(
            y,
            dtype=float,
        )
    )

    valid = (
        np.isfinite(x.to_numpy())
        & np.isfinite(y.to_numpy())
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return None

    ranked_x = x.rank(
        method="average"
    ).to_numpy(dtype=float)

    ranked_y = y.rank(
        method="average"
    ).to_numpy(dtype=float)

    if (
        np.std(ranked_x) == 0
        or np.std(ranked_y) == 0
    ):
        return None

    return float(
        np.corrcoef(
            ranked_x,
            ranked_y,
        )[0, 1]
    )


def build_condition_metrics(
    selected,
    exact_correct,
    task_success,
    physical_mask,
    control_mask,
):
    selected = np.asarray(
        selected,
        dtype=bool,
    )

    exact_correct = np.asarray(
        exact_correct,
        dtype=bool,
    )

    task_success = np.asarray(
        task_success,
        dtype=bool,
    )

    physical_mask = np.asarray(
        physical_mask,
        dtype=bool,
    )

    control_mask = np.asarray(
        control_mask,
        dtype=bool,
    )

    selected_physical = (
        selected
        & physical_mask[:, None]
    )

    selected_count = (
        selected_physical.sum(axis=0)
    )

    correct_selected_count = (
        (
            exact_correct
            & selected_physical
        )
        .sum(axis=0)
    )

    selected_accuracy = np.divide(
        correct_selected_count,
        selected_count,
        out=np.full(
            selected_count.shape,
            np.nan,
            dtype=float,
        ),
        where=selected_count > 0,
    )

    physical_count = int(
        physical_mask.sum()
    )

    control_count = int(
        control_mask.sum()
    )

    physical_selection = (
        selected_physical.sum(axis=0)
        / physical_count
    )

    physical_exact = (
        (
            exact_correct
            & physical_mask[:, None]
        )
        .sum(axis=0)
        / physical_count
    )

    control_abstention = (
        (
            (~selected)
            & control_mask[:, None]
        )
        .sum(axis=0)
        / control_count
    )

    return {
        "selected_accuracy":
            float(
                np.nanmean(
                    selected_accuracy
                )
            ),

        "physical_selection_rate":
            float(
                np.mean(
                    physical_selection
                )
            ),

        "physical_exact_diagnosis_rate":
            float(
                np.mean(
                    physical_exact
                )
            ),

        "control_abstention_rate":
            float(
                np.mean(
                    control_abstention
                )
            ),

        "false_specific_rate":
            float(
                1.0
                - np.mean(
                    control_abstention
                )
            ),

        "overall_output_rate":
            float(
                np.mean(selected)
            ),

        "overall_task_success_rate":
            float(
                np.mean(
                    task_success
                )
            ),
    }


def bootstrap_paired_detector(
    detector_frame,
    detector_index,
):
    repeats = sorted(
        detector_frame[
            "repeat"
        ].unique().tolist()
    )

    if repeats != [0, 1, 2, 3, 4]:
        raise RuntimeError(
            "Unexpected repeat set."
        )

    sample_ids = sorted(
        detector_frame[
            "sample_id"
        ].unique().tolist()
    )

    if len(sample_ids) != 825:
        raise RuntimeError(
            "Expected 825 sample IDs."
        )

    sample_index = {
        sample_id: index
        for index, sample_id
        in enumerate(sample_ids)
    }

    repeat_index = {
        repeat: index
        for index, repeat
        in enumerate(repeats)
    }

    shape = (
        len(sample_ids),
        len(repeats),
    )

    detector_selected = np.zeros(
        shape,
        dtype=bool,
    )

    locked_selected = np.zeros(
        shape,
        dtype=bool,
    )

    detector_exact = np.zeros(
        shape,
        dtype=bool,
    )

    locked_exact = np.zeros(
        shape,
        dtype=bool,
    )

    detector_task = np.zeros(
        shape,
        dtype=bool,
    )

    locked_task = np.zeros(
        shape,
        dtype=bool,
    )

    prediction_changed = np.zeros(
        shape,
        dtype=bool,
    )

    selection_changed = np.zeros(
        shape,
        dtype=bool,
    )

    task_changed = np.zeros(
        shape,
        dtype=bool,
    )

    truth_by_sample = {}

    for row in detector_frame.itertuples(
        index=False
    ):
        i = sample_index[
            str(row.sample_id)
        ]

        j = repeat_index[
            int(row.repeat)
        ]

        detector_selected[
            i,
            j,
        ] = bool(
            row.detector_selected
        )

        locked_selected[
            i,
            j,
        ] = bool(
            row.locked_interval_selected
        )

        detector_exact[
            i,
            j,
        ] = bool(
            row.detector_exact_correct
        )

        locked_exact[
            i,
            j,
        ] = bool(
            row.locked_interval_exact_correct
        )

        detector_task[
            i,
            j,
        ] = bool(
            row.detector_task_success
        )

        locked_task[
            i,
            j,
        ] = bool(
            row.locked_interval_task_success
        )

        prediction_changed[
            i,
            j,
        ] = bool(
            row.prediction_changed
        )

        selection_changed[
            i,
            j,
        ] = bool(
            row.selection_changed
        )

        task_changed[
            i,
            j,
        ] = bool(
            row.task_success_changed
        )

        truth_by_sample[
            str(row.sample_id)
        ] = str(row.truth)

    truth = np.asarray(
        [
            truth_by_sample[
                sample_id
            ]
            for sample_id
            in sample_ids
        ],
        dtype=object,
    )

    physical_mask = np.isin(
        truth,
        PHYSICAL_CLASSES,
    )

    control_mask = np.isin(
        truth,
        CONTROL_CLASSES,
    )

    detector_point = build_condition_metrics(
        detector_selected,
        detector_exact,
        detector_task,
        physical_mask,
        control_mask,
    )

    locked_point = build_condition_metrics(
        locked_selected,
        locked_exact,
        locked_task,
        physical_mask,
        control_mask,
    )

    point_changes = {
        "prediction_change_rate":
            float(
                prediction_changed.mean()
            ),

        "selection_change_rate":
            float(
                selection_changed.mean()
            ),

        "task_success_change_rate":
            float(
                task_changed.mean()
            ),
    }

    detector_draws = {
        metric: np.empty(
            BOOTSTRAP_REPETITIONS,
            dtype=np.float64,
        )
        for metric in PAIRED_METRICS
    }

    locked_draws = {
        metric: np.empty(
            BOOTSTRAP_REPETITIONS,
            dtype=np.float64,
        )
        for metric in PAIRED_METRICS
    }

    change_draws = {
        metric: np.empty(
            BOOTSTRAP_REPETITIONS,
            dtype=np.float64,
        )
        for metric in CHANGE_METRICS
    }

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
        + detector_index
    )

    completed = 0

    while completed < BOOTSTRAP_REPETITIONS:
        chunk_size = min(
            BOOTSTRAP_CHUNK_SIZE,
            BOOTSTRAP_REPETITIONS
            - completed,
        )

        indices = rng.integers(
            0,
            len(sample_ids),
            size=(
                chunk_size,
                len(sample_ids),
            ),
        )

        sampled_physical = (
            physical_mask[
                indices
            ]
        )

        sampled_control = (
            control_mask[
                indices
            ]
        )

        detector_selected_chunk = (
            detector_selected[
                indices
            ]
        )

        locked_selected_chunk = (
            locked_selected[
                indices
            ]
        )

        detector_exact_chunk = (
            detector_exact[
                indices
            ]
        )

        locked_exact_chunk = (
            locked_exact[
                indices
            ]
        )

        detector_task_chunk = (
            detector_task[
                indices
            ]
        )

        locked_task_chunk = (
            locked_task[
                indices
            ]
        )

        physical_denominator = (
            sampled_physical.sum(
                axis=1
            )[:, None]
        )

        control_denominator = (
            sampled_control.sum(
                axis=1
            )[:, None]
        )

        def calculate_chunk(
            selected_chunk,
            exact_chunk,
            task_chunk,
        ):
            selected_physical = (
                selected_chunk
                & sampled_physical[
                    :,
                    :,
                    None,
                ]
            )

            selected_count = (
                selected_physical.sum(
                    axis=1
                )
            )

            selected_correct = (
                (
                    exact_chunk
                    & selected_physical
                )
                .sum(axis=1)
            )

            selected_accuracy = np.divide(
                selected_correct,
                selected_count,
                out=np.full(
                    selected_count.shape,
                    np.nan,
                    dtype=float,
                ),
                where=selected_count > 0,
            )

            physical_selection = (
                selected_physical.sum(
                    axis=1
                )
                / physical_denominator
            )

            physical_exact = (
                (
                    exact_chunk
                    & sampled_physical[
                        :,
                        :,
                        None,
                    ]
                )
                .sum(axis=1)
                / physical_denominator
            )

            control_abstention = (
                (
                    (~selected_chunk)
                    & sampled_control[
                        :,
                        :,
                        None,
                    ]
                )
                .sum(axis=1)
                / control_denominator
            )

            return {
                "selected_accuracy":
                    np.nanmean(
                        selected_accuracy,
                        axis=1,
                    ),

                "physical_selection_rate":
                    np.mean(
                        physical_selection,
                        axis=1,
                    ),

                "physical_exact_diagnosis_rate":
                    np.mean(
                        physical_exact,
                        axis=1,
                    ),

                "control_abstention_rate":
                    np.mean(
                        control_abstention,
                        axis=1,
                    ),

                "false_specific_rate":
                    1.0
                    - np.mean(
                        control_abstention,
                        axis=1,
                    ),

                "overall_output_rate":
                    selected_chunk.mean(
                        axis=(1, 2)
                    ),

                "overall_task_success_rate":
                    task_chunk.mean(
                        axis=(1, 2)
                    ),
            }

        detector_values = calculate_chunk(
            detector_selected_chunk,
            detector_exact_chunk,
            detector_task_chunk,
        )

        locked_values = calculate_chunk(
            locked_selected_chunk,
            locked_exact_chunk,
            locked_task_chunk,
        )

        section = slice(
            completed,
            completed + chunk_size,
        )

        for metric in PAIRED_METRICS:
            detector_draws[
                metric
            ][section] = detector_values[
                metric
            ]

            locked_draws[
                metric
            ][section] = locked_values[
                metric
            ]

        change_draws[
            "prediction_change_rate"
        ][section] = (
            prediction_changed[
                indices
            ].mean(
                axis=(1, 2)
            )
        )

        change_draws[
            "selection_change_rate"
        ][section] = (
            selection_changed[
                indices
            ].mean(
                axis=(1, 2)
            )
        )

        change_draws[
            "task_success_change_rate"
        ][section] = (
            task_changed[
                indices
            ].mean(
                axis=(1, 2)
            )
        )

        completed += chunk_size

    rows = []

    draw_arrays = {}

    for metric in PAIRED_METRICS:
        detector_interval = percentile_interval(
            detector_draws[
                metric
            ]
        )

        locked_interval = percentile_interval(
            locked_draws[
                metric
            ]
        )

        delta_draw = (
            detector_draws[
                metric
            ]
            - locked_draws[
                metric
            ]
        )

        delta_interval = percentile_interval(
            delta_draw
        )

        rows.append(
            {
                "detector":
                    detector_frame[
                        "detector"
                    ].iloc[0],

                "metric":
                    metric,

                "detector_estimate":
                    detector_point[
                        metric
                    ],

                "detector_lower_95":
                    detector_interval[
                        "lower_95"
                    ],

                "detector_upper_95":
                    detector_interval[
                        "upper_95"
                    ],

                "locked_interval_estimate":
                    locked_point[
                        metric
                    ],

                "locked_interval_lower_95":
                    locked_interval[
                        "lower_95"
                    ],

                "locked_interval_upper_95":
                    locked_interval[
                        "upper_95"
                    ],

                "delta_estimate":
                    detector_point[
                        metric
                    ]
                    - locked_point[
                        metric
                    ],

                "delta_lower_95":
                    delta_interval[
                        "lower_95"
                    ],

                "delta_upper_95":
                    delta_interval[
                        "upper_95"
                    ],

                "bootstrap_repetitions":
                    BOOTSTRAP_REPETITIONS,

                "cluster_unit":
                    "sample_id",

                "five_repeats_kept_together":
                    True,
            }
        )

        draw_arrays[
            f"detector__{metric}"
        ] = detector_draws[
            metric
        ]

        draw_arrays[
            f"locked__{metric}"
        ] = locked_draws[
            metric
        ]

        draw_arrays[
            f"delta__{metric}"
        ] = delta_draw

    for metric in CHANGE_METRICS:
        interval = percentile_interval(
            change_draws[
                metric
            ]
        )

        rows.append(
            {
                "detector":
                    detector_frame[
                        "detector"
                    ].iloc[0],

                "metric":
                    metric,

                "detector_estimate":
                    point_changes[
                        metric
                    ],

                "detector_lower_95":
                    interval[
                        "lower_95"
                    ],

                "detector_upper_95":
                    interval[
                        "upper_95"
                    ],

                "locked_interval_estimate":
                    None,

                "locked_interval_lower_95":
                    None,

                "locked_interval_upper_95":
                    None,

                "delta_estimate":
                    None,

                "delta_lower_95":
                    None,

                "delta_upper_95":
                    None,

                "bootstrap_repetitions":
                    BOOTSTRAP_REPETITIONS,

                "cluster_unit":
                    "sample_id",

                "five_repeats_kept_together":
                    True,
            }
        )

        draw_arrays[
            metric
        ] = change_draws[
            metric
        ]

    return rows, draw_arrays


def add_iou_bin(frame):
    frame = frame.copy()

    frame[
        "iou_bin"
    ] = "not_applicable"

    physical = frame[
        "localization_eligible"
    ]

    iou = frame.loc[
        physical,
        "interval_iou",
    ].astype(float)

    labels = np.select(
        [
            iou == 0.0,
            (iou > 0.0)
            & (iou < 0.3),
            (iou >= 0.3)
            & (iou < 0.5),
            iou >= 0.5,
        ],
        [
            "IoU = 0",
            "0 < IoU < 0.3",
            "0.3 <= IoU < 0.5",
            "IoU >= 0.5",
        ],
        default="invalid",
    )

    frame.loc[
        physical,
        "iou_bin",
    ] = labels

    if (
        frame[
            "iou_bin"
        ] == "invalid"
    ).any():
        raise RuntimeError(
            "Invalid IoU bin assignment."
        )

    return frame


def descriptive_group_metrics(frame):
    physical = frame[
        "truth"
    ].isin(
        PHYSICAL_CLASSES
    )

    detector_selected_physical = (
        frame[
            "detector_selected"
        ]
        & physical
    )

    locked_selected_physical = (
        frame[
            "locked_interval_selected"
        ]
        & physical
    )

    detector_selected_accuracy = None

    if detector_selected_physical.sum() > 0:
        detector_selected_accuracy = float(
            frame.loc[
                detector_selected_physical,
                "detector_exact_correct",
            ].mean()
        )

    locked_selected_accuracy = None

    if locked_selected_physical.sum() > 0:
        locked_selected_accuracy = float(
            frame.loc[
                locked_selected_physical,
                "locked_interval_exact_correct",
            ].mean()
        )

    return {
        "sample_count":
            int(
                frame[
                    "sample_id"
                ].nunique()
            ),

        "prediction_count":
            int(len(frame)),

        "detector_selection_rate":
            float(
                frame[
                    "detector_selected"
                ].mean()
            ),

        "locked_selection_rate":
            float(
                frame[
                    "locked_interval_selected"
                ].mean()
            ),

        "delta_selection_rate":
            float(
                frame[
                    "detector_selected"
                ].mean()
                - frame[
                    "locked_interval_selected"
                ].mean()
            ),

        "detector_task_success_rate":
            float(
                frame[
                    "detector_task_success"
                ].mean()
            ),

        "locked_task_success_rate":
            float(
                frame[
                    "locked_interval_task_success"
                ].mean()
            ),

        "delta_task_success_rate":
            float(
                frame[
                    "detector_task_success"
                ].mean()
                - frame[
                    "locked_interval_task_success"
                ].mean()
            ),

        "detector_selected_accuracy":
            detector_selected_accuracy,

        "locked_selected_accuracy":
            locked_selected_accuracy,

        "prediction_change_rate":
            float(
                frame[
                    "prediction_changed"
                ].mean()
            ),

        "selection_change_rate":
            float(
                frame[
                    "selection_changed"
                ].mean()
            ),

        "task_success_change_rate":
            float(
                frame[
                    "task_success_changed"
                ].mean()
            ),
    }


def aggregate_sample_level(frame):
    return (
        frame.groupby(
            [
                "detector",
                "sample_id",
            ],
            sort=True,
            as_index=False,
        )
        .agg(
            truth=(
                "truth",
                "first",
            ),

            severity=(
                "severity",
                "first",
            ),

            channel_model=(
                "channel_model",
                "first",
            ),

            localization_eligible=(
                "localization_eligible",
                "first",
            ),

            interval_iou=(
                "interval_iou",
                "first",
            ),

            iou_bin=(
                "iou_bin",
                "first",
            ),

            detector_selection_rate=(
                "detector_selected",
                "mean",
            ),

            locked_selection_rate=(
                "locked_interval_selected",
                "mean",
            ),

            detector_task_success_rate=(
                "detector_task_success",
                "mean",
            ),

            locked_task_success_rate=(
                "locked_interval_task_success",
                "mean",
            ),

            prediction_change_rate=(
                "prediction_changed",
                "mean",
            ),

            selection_change_rate=(
                "selection_changed",
                "mean",
            ),

            task_success_change_rate=(
                "task_success_changed",
                "mean",
            ),
        )
    )


def bootstrap_sample_group(
    frame,
    metric,
    seed,
):
    values = frame[
        metric
    ].to_numpy(dtype=float)

    if len(values) == 0:
        raise RuntimeError(
            "Cannot bootstrap empty group."
        )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.integers(
        0,
        len(values),
        size=(
            BOOTSTRAP_REPETITIONS,
            len(values),
        ),
    )

    draws = values[
        indices
    ].mean(axis=1)

    interval = percentile_interval(
        draws
    )

    return {
        "estimate":
            float(
                values.mean()
            ),

        "lower_95":
            interval[
                "lower_95"
            ],

        "upper_95":
            interval[
                "upper_95"
            ],
    }


for required in (
    PREDICTIONS_PATH,
    PER_REPEAT_METRICS_PATH,
    PER_REPEAT_DELTAS_PATH,
    DIAGNOSIS_MANIFEST_PATH,
    PROPOSAL_EVALUATION_MANIFEST_PATH,
    PROTOCOL_PATH,
):
    if not required.exists():
        raise FileNotFoundError(
            required
        )


if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Final statistics output already exists; "
        "refusing to overwrite it."
    )

if TEMP_ROOT.exists():
    shutil.rmtree(
        TEMP_ROOT
    )

TEMP_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


if (
    sha256_file(PROTOCOL_PATH)
    != EXPECTED_PROTOCOL_SHA256
):
    raise RuntimeError(
        "Locked protocol SHA256 mismatch."
    )


diagnosis_manifest = load_json(
    DIAGNOSIS_MANIFEST_PATH
)

proposal_evaluation_manifest = load_json(
    PROPOSAL_EVALUATION_MANIFEST_PATH
)


if diagnosis_manifest.get("status") != "PASS":
    raise RuntimeError(
        "Detector-conditioned diagnosis "
        "manifest is not PASS."
    )

if (
    diagnosis_manifest.get(
        "prediction_count"
    )
    != 12375
):
    raise RuntimeError(
        "Diagnosis prediction count "
        "is not 12375."
    )

if (
    diagnosis_manifest.get(
        "deterministic_prediction_replay"
    )
    != "12375/12375"
):
    raise RuntimeError(
        "Prediction replay contract mismatch."
    )

if (
    proposal_evaluation_manifest.get("status")
    != "PASS"
):
    raise RuntimeError(
        "Proposal evaluation manifest is not PASS."
    )


source_records = {
    record[
        "relative_path"
    ]:
        record
    for record in diagnosis_manifest.get(
        "files",
        [],
    )
}

for relative_path, record in (
    source_records.items()
):
    path = ROOT / relative_path

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    if (
        sha256_file(path)
        != record[
            "sha256"
        ]
    ):
        raise RuntimeError(
            "Detector-conditioned result changed: "
            f"{relative_path}"
        )


predictions = pd.read_csv(
    PREDICTIONS_PATH
)

per_repeat_metrics = pd.read_csv(
    PER_REPEAT_METRICS_PATH
)

per_repeat_deltas = pd.read_csv(
    PER_REPEAT_DELTAS_PATH
)


if len(predictions) != 12375:
    raise RuntimeError(
        "Prediction CSV row count "
        "is not 12375."
    )

if len(per_repeat_metrics) != 15:
    raise RuntimeError(
        "Per-repeat metric count "
        "is not 15."
    )

if len(per_repeat_deltas) != 15:
    raise RuntimeError(
        "Per-repeat delta count "
        "is not 15."
    )


boolean_columns = [
    "localization_eligible",
    "detector_selected",
    "detector_exact_correct",
    "detector_task_success",
    "locked_interval_selected",
    "locked_interval_exact_correct",
    "locked_interval_task_success",
    "selection_changed",
    "prediction_changed",
    "task_success_changed",
]

for column in boolean_columns:
    predictions[
        column
    ] = bool_series(
        predictions[
            column
        ]
    )


predictions = add_iou_bin(
    predictions
)


for detector in DETECTORS:
    frame = predictions[
        predictions[
            "detector"
        ] == detector
    ]

    if len(frame) != 4125:
        raise RuntimeError(
            f"{detector}: expected 4125 rows."
        )

    if (
        frame[
            "sample_id"
        ].nunique()
        != 825
    ):
        raise RuntimeError(
            f"{detector}: expected 825 samples."
        )

    if (
        frame[
            "repeat"
        ].nunique()
        != 5
    ):
        raise RuntimeError(
            f"{detector}: expected five repeats."
        )

    counts = frame.groupby(
        "sample_id"
    ).size()

    if not (
        counts == 5
    ).all():
        raise RuntimeError(
            f"{detector}: each sample must have "
            "five repeat predictions."
        )


paired_bootstrap_rows = []
bootstrap_arrays = {}


for detector_index, detector in enumerate(
    DETECTORS
):
    print(
        "paired_bootstrap_detector:",
        detector,
    )

    frame = predictions[
        predictions[
            "detector"
        ] == detector
    ].copy()

    rows, arrays = bootstrap_paired_detector(
        frame,
        detector_index,
    )

    paired_bootstrap_rows.extend(
        rows
    )

    for key, value in arrays.items():
        bootstrap_arrays[
            f"{detector}__{key}"
        ] = value


paired_bootstrap = pd.DataFrame(
    paired_bootstrap_rows
)


repeat_ci_rows = []

delta_columns = [
    column
    for column in per_repeat_deltas.columns
    if column.startswith(
        "delta_"
    )
]


for detector in DETECTORS:
    frame = per_repeat_deltas[
        per_repeat_deltas[
            "detector"
        ] == detector
    ]

    for column in delta_columns:
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        finite = values[
            np.isfinite(values)
        ]

        if len(finite) == 0:
            continue

        mean = float(
            finite.mean()
        )

        std = float(
            finite.std(
                ddof=1
            )
        )

        standard_error = (
            std
            / np.sqrt(
                len(finite)
            )
        )

        margin = (
            T_CRITICAL_95_DF4
            * standard_error
        )

        repeat_ci_rows.append(
            {
                "detector":
                    detector,

                "metric":
                    column,

                "repeat_count":
                    int(
                        len(finite)
                    ),

                "mean":
                    mean,

                "standard_deviation":
                    std,

                "lower_95":
                    mean - margin,

                "upper_95":
                    mean + margin,

                "interval_type":
                    "t interval across five locked reconstruction repeats",
            }
        )


repeat_ci = pd.DataFrame(
    repeat_ci_rows
)


stratified_rows = []


group_specs = [
    (
        "severity",
        ["detector", "severity"],
    ),
    (
        "label",
        ["detector", "truth"],
    ),
    (
        "channel",
        [
            "detector",
            "channel_model",
        ],
    ),
]


for dimension, columns in group_specs:
    grouped = predictions.groupby(
        columns,
        sort=True,
        dropna=False,
    )

    for keys, frame in grouped:
        if not isinstance(
            keys,
            tuple,
        ):
            keys = (keys,)

        key_map = {
            column: value
            for column, value
            in zip(
                columns,
                keys,
            )
        }

        stratified_rows.append(
            {
                "dimension":
                    dimension,

                "group":
                    "|".join(
                        str(
                            key_map[column]
                        )
                        for column
                        in columns
                    ),

                **key_map,

                **descriptive_group_metrics(
                    frame
                ),
            }
        )


physical_predictions = predictions[
    predictions[
        "localization_eligible"
    ]
].copy()


for (
    detector,
    iou_bin,
), frame in physical_predictions.groupby(
    [
        "detector",
        "iou_bin",
    ],
    sort=True,
):
    stratified_rows.append(
        {
            "dimension":
                "iou_bin",

            "group":
                f"{detector}|{iou_bin}",

            "detector":
                detector,

            "iou_bin":
                iou_bin,

            **descriptive_group_metrics(
                frame
            ),
        }
    )


stratified = pd.DataFrame(
    stratified_rows
)


sample_level = aggregate_sample_level(
    predictions
)

sample_level[
    "delta_selection_rate"
] = (
    sample_level[
        "detector_selection_rate"
    ]
    - sample_level[
        "locked_selection_rate"
    ]
)

sample_level[
    "delta_task_success_rate"
] = (
    sample_level[
        "detector_task_success_rate"
    ]
    - sample_level[
        "locked_task_success_rate"
    ]
)


stratified_bootstrap_rows = []

sample_group_specs = [
    (
        "severity",
        ["detector", "severity"],
    ),
    (
        "label",
        ["detector", "truth"],
    ),
    (
        "channel",
        [
            "detector",
            "channel_model",
        ],
    ),
]


physical_sample_level = sample_level[
    sample_level[
        "localization_eligible"
    ]
].copy()


bootstrap_groups = []


for dimension, columns in sample_group_specs:
    grouped = sample_level.groupby(
        columns,
        sort=True,
        dropna=False,
    )

    for keys, frame in grouped:
        if not isinstance(
            keys,
            tuple,
        ):
            keys = (keys,)

        key_map = {
            column: value
            for column, value
            in zip(
                columns,
                keys,
            )
        }

        bootstrap_groups.append(
            (
                dimension,
                key_map,
                frame,
            )
        )


for (
    detector,
    iou_bin,
), frame in physical_sample_level.groupby(
    [
        "detector",
        "iou_bin",
    ],
    sort=True,
):
    bootstrap_groups.append(
        (
            "iou_bin",
            {
                "detector":
                    detector,

                "iou_bin":
                    iou_bin,
            },
            frame,
        )
    )


sample_metrics = [
    "detector_selection_rate",
    "locked_selection_rate",
    "delta_selection_rate",
    "detector_task_success_rate",
    "locked_task_success_rate",
    "delta_task_success_rate",
    "prediction_change_rate",
    "selection_change_rate",
    "task_success_change_rate",
]


for group_index, (
    dimension,
    key_map,
    frame,
) in enumerate(
    bootstrap_groups
):
    for metric_index, metric in enumerate(
        sample_metrics
    ):
        result = bootstrap_sample_group(
            frame,
            metric,
            BOOTSTRAP_SEED
            + 100
            + group_index * 20
            + metric_index,
        )

        stratified_bootstrap_rows.append(
            {
                "dimension":
                    dimension,

                "group":
                    "|".join(
                        str(value)
                        for value
                        in key_map.values()
                    ),

                **key_map,

                "sample_count":
                    int(
                        len(frame)
                    ),

                "metric":
                    metric,

                "estimate":
                    result[
                        "estimate"
                    ],

                "lower_95":
                    result[
                        "lower_95"
                    ],

                "upper_95":
                    result[
                        "upper_95"
                    ],

                "bootstrap_repetitions":
                    BOOTSTRAP_REPETITIONS,

                "resampling_unit":
                    "sample_id",

                "five_repeats_kept_together":
                    True,
            }
        )


stratified_bootstrap = pd.DataFrame(
    stratified_bootstrap_rows
)


association_rows = []


association_outcomes = [
    "detector_selection_rate",
    "detector_task_success_rate",
    "delta_selection_rate",
    "delta_task_success_rate",
    "prediction_change_rate",
    "task_success_change_rate",
]


for detector in DETECTORS:
    frame = physical_sample_level[
        physical_sample_level[
            "detector"
        ] == detector
    ]

    for outcome in association_outcomes:
        rho = rank_correlation(
            frame[
                "interval_iou"
            ],
            frame[
                outcome
            ],
        )

        association_rows.append(
            {
                "detector":
                    detector,

                "sample_count":
                    int(
                        len(frame)
                    ),

                "predictor":
                    "interval_iou",

                "outcome":
                    outcome,

                "spearman_rho":
                    rho,

                "interpretation":
                    "Descriptive rank association; "
                    "not a causal estimate.",
            }
        )


association = pd.DataFrame(
    association_rows
)


transition_rows = []


def append_transition_group(
    dimension,
    key_map,
    frame,
):
    detector_success = frame[
        "detector_task_success"
    ].to_numpy(dtype=bool)

    locked_success = frame[
        "locked_interval_task_success"
    ].to_numpy(dtype=bool)

    detector_selected = frame[
        "detector_selected"
    ].to_numpy(dtype=bool)

    locked_selected = frame[
        "locked_interval_selected"
    ].to_numpy(dtype=bool)

    transition_rows.append(
        {
            "dimension":
                dimension,

            "group":
                "|".join(
                    str(value)
                    for value
                    in key_map.values()
                ),

            **key_map,

            "sample_count":
                int(
                    frame[
                        "sample_id"
                    ].nunique()
                ),

            "prediction_count":
                int(len(frame)),

            "both_task_success":
                int(
                    (
                        detector_success
                        & locked_success
                    ).sum()
                ),

            "detector_gain":
                int(
                    (
                        detector_success
                        & (~locked_success)
                    ).sum()
                ),

            "detector_loss":
                int(
                    (
                        (~detector_success)
                        & locked_success
                    ).sum()
                ),

            "both_task_failure":
                int(
                    (
                        (~detector_success)
                        & (~locked_success)
                    ).sum()
                ),

            "net_task_success_change_rate":
                float(
                    detector_success.mean()
                    - locked_success.mean()
                ),

            "both_selected":
                int(
                    (
                        detector_selected
                        & locked_selected
                    ).sum()
                ),

            "detector_only_selected":
                int(
                    (
                        detector_selected
                        & (~locked_selected)
                    ).sum()
                ),

            "locked_only_selected":
                int(
                    (
                        (~detector_selected)
                        & locked_selected
                    ).sum()
                ),

            "both_abstained":
                int(
                    (
                        (~detector_selected)
                        & (~locked_selected)
                    ).sum()
                ),

            "net_selection_change_rate":
                float(
                    detector_selected.mean()
                    - locked_selected.mean()
                ),
        }
    )


for detector in DETECTORS:
    detector_frame = predictions[
        predictions[
            "detector"
        ] == detector
    ]

    append_transition_group(
        "overall",
        {
            "detector":
                detector,
        },
        detector_frame,
    )

    for severity, frame in (
        detector_frame.groupby(
            "severity",
            sort=True,
        )
    ):
        append_transition_group(
            "severity",
            {
                "detector":
                    detector,

                "severity":
                    severity,
            },
            frame,
        )

    for truth, frame in (
        detector_frame.groupby(
            "truth",
            sort=True,
        )
    ):
        append_transition_group(
            "label",
            {
                "detector":
                    detector,

                "truth":
                    truth,
            },
            frame,
        )

    physical_frame = detector_frame[
        detector_frame[
            "localization_eligible"
        ]
    ]

    for iou_bin, frame in (
        physical_frame.groupby(
            "iou_bin",
            sort=True,
        )
    ):
        append_transition_group(
            "iou_bin",
            {
                "detector":
                    detector,

                "iou_bin":
                    iou_bin,
            },
            frame,
        )


transitions = pd.DataFrame(
    transition_rows
)


paired_bootstrap_path = (
    TEMP_ROOT
    / "paired_sample_cluster_bootstrap.csv"
)

repeat_ci_path = (
    TEMP_ROOT
    / "repeat_level_delta_confidence_intervals.csv"
)

stratified_path = (
    TEMP_ROOT
    / "stratified_diagnostic_metrics.csv"
)

stratified_bootstrap_path = (
    TEMP_ROOT
    / "stratified_sample_cluster_bootstrap.csv"
)

association_path = (
    TEMP_ROOT
    / "iou_diagnostic_associations.csv"
)

transitions_path = (
    TEMP_ROOT
    / "paired_outcome_transitions.csv"
)

bootstrap_draws_path = (
    TEMP_ROOT
    / "paired_bootstrap_draws.npz"
)

summary_path = (
    TEMP_ROOT
    / "summary.json"
)


paired_bootstrap.to_csv(
    paired_bootstrap_path,
    index=False,
)

repeat_ci.to_csv(
    repeat_ci_path,
    index=False,
)

stratified.to_csv(
    stratified_path,
    index=False,
)

stratified_bootstrap.to_csv(
    stratified_bootstrap_path,
    index=False,
)

association.to_csv(
    association_path,
    index=False,
)

transitions.to_csv(
    transitions_path,
    index=False,
)

np.savez_compressed(
    bootstrap_draws_path,
    **bootstrap_arrays,
)


summary = {
    "schema":
        "phyguard.sionna.formal825."
        "detector_conditioned_statistics.v1",

    "status":
        "PASS",

    "analysis_scope": {
        "sample_count":
            825,

        "physical_sample_count":
            706,

        "control_sample_count":
            119,

        "detector_count":
            3,

        "model_repeat_count":
            5,

        "prediction_count":
            12375,

        "paired_locked_interval_prediction_count":
            4125,
    },

    "paired_bootstrap": {
        "repetitions":
            BOOTSTRAP_REPETITIONS,

        "base_seed":
            BOOTSTRAP_SEED,

        "cluster_unit":
            "sample_id",

        "five_repeats_kept_together":
            True,

        "detector_and_locked_predictions_paired":
            True,
    },

    "iou_bins": [
        "IoU = 0",
        "0 < IoU < 0.3",
        "0.3 <= IoU < 0.5",
        "IoU >= 0.5",
    ],

    "paired_bootstrap_results":
        paired_bootstrap.to_dict(
            orient="records"
        ),

    "iou_associations":
        association.to_dict(
            orient="records"
        ),

    "methodological_boundary": {
        "new_predictions_generated":
            False,

        "models_modified":
            False,

        "detector_parameters_modified":
            False,

        "candidate_window_modified":
            False,

        "thresholds_modified":
            False,

        "post_result_detector_selection":
            False,

        "all_three_detectors_reported":
            True,

        "ground_truth_used_for_statistics_only":
            True,

        "association_claimed_as_causal":
            False,
    },
}


summary_path.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


TEMP_ROOT.rename(
    OUTPUT_ROOT
)


output_files = sorted(
    path
    for path in OUTPUT_ROOT.rglob("*")
    if path.is_file()
)


manifest = {
    "schema":
        "phyguard.sionna.formal825."
        "detector_conditioned_statistics_manifest.v1",

    "status":
        "PASS",

    "protocol_sha256":
        EXPECTED_PROTOCOL_SHA256,

    "diagnosis_manifest_path":
        str(
            DIAGNOSIS_MANIFEST_PATH
        ),

    "diagnosis_manifest_sha256":
        sha256_file(
            DIAGNOSIS_MANIFEST_PATH
        ),

    "source_integrity": {
        "diagnosis_files_unchanged":
            True,

        "new_predictions_generated":
            False,

        "prediction_count":
            12375,

        "sample_count":
            825,

        "detector_count":
            3,

        "repeat_count":
            5,
    },

    "bootstrap": {
        "repetitions":
            BOOTSTRAP_REPETITIONS,

        "base_seed":
            BOOTSTRAP_SEED,

        "cluster_unit":
            "sample_id",

        "five_repeats_kept_together":
            True,

        "conditions_paired":
            True,
    },

    "methodological_boundary": {
        "model_training":
            False,

        "threshold_tuning":
            False,

        "threshold_calibration":
            False,

        "detector_parameter_tuning":
            False,

        "candidate_window_tuning":
            False,

        "detector_selection":
            False,

        "post_result_pipeline_modification":
            False,
    },

    "files": [
        {
            "relative_path":
                str(
                    path.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                sha256_file(path),

            "size_bytes":
                path.stat().st_size,
        }
        for path in output_files
    ],
}


OUTPUT_MANIFEST_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_MANIFEST_PATH.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print("status: PASS")
print("sample_count: 825")
print("physical_sample_count: 706")
print("control_sample_count: 119")
print("detector_count: 3")
print("model_repeat_count: 5")
print("prediction_count: 12375")
print(
    "bootstrap_repetitions:",
    BOOTSTRAP_REPETITIONS,
)
print(
    "paired_cluster_unit: sample_id"
)
print(
    "five_repeats_kept_together:",
    True,
)
print(
    "new_predictions_generated:",
    False,
)
print(
    "detector_selection:",
    False,
)


print("\nMAIN PAIRED BOOTSTRAP 95% INTERVALS")

main_metrics = paired_bootstrap[
    paired_bootstrap[
        "metric"
    ].isin(
        [
            "selected_accuracy",
            "physical_selection_rate",
            "physical_exact_diagnosis_rate",
            "control_abstention_rate",
            "false_specific_rate",
        ]
    )
]

print(
    main_metrics[
        [
            "detector",
            "metric",
            "detector_estimate",
            "locked_interval_estimate",
            "delta_estimate",
            "delta_lower_95",
            "delta_upper_95",
        ]
    ].to_string(
        index=False
    )
)


print("\nDETECTOR × SEVERITY")

severity_view = stratified[
    stratified[
        "dimension"
    ] == "severity"
][
    [
        "detector",
        "severity",
        "sample_count",
        "detector_selection_rate",
        "locked_selection_rate",
        "delta_selection_rate",
        "detector_task_success_rate",
        "locked_task_success_rate",
        "delta_task_success_rate",
        "prediction_change_rate",
    ]
]

print(
    severity_view.to_string(
        index=False
    )
)


print("\nDETECTOR × IoU BIN")

iou_view = stratified[
    stratified[
        "dimension"
    ] == "iou_bin"
][
    [
        "detector",
        "iou_bin",
        "sample_count",
        "detector_selection_rate",
        "locked_selection_rate",
        "delta_selection_rate",
        "detector_task_success_rate",
        "locked_task_success_rate",
        "delta_task_success_rate",
        "prediction_change_rate",
    ]
]

print(
    iou_view.to_string(
        index=False
    )
)


print("\nIoU ASSOCIATIONS")

print(
    association[
        [
            "detector",
            "predictor",
            "outcome",
            "spearman_rho",
        ]
    ].to_string(
        index=False
    )
)


print("\nOVERALL PAIRED TRANSITIONS")

overall_transitions = transitions[
    transitions[
        "dimension"
    ] == "overall"
]

print(
    overall_transitions[
        [
            "detector",
            "prediction_count",
            "both_task_success",
            "detector_gain",
            "detector_loss",
            "both_task_failure",
            "net_task_success_change_rate",
            "detector_only_selected",
            "locked_only_selected",
            "net_selection_change_rate",
        ]
    ].to_string(
        index=False
    )
)


print("\noutput_root:", OUTPUT_ROOT)
print(
    "paired_bootstrap:",
    OUTPUT_ROOT
    / "paired_sample_cluster_bootstrap.csv",
)
print(
    "stratified:",
    OUTPUT_ROOT
    / "stratified_diagnostic_metrics.csv",
)
print(
    "association:",
    OUTPUT_ROOT
    / "iou_diagnostic_associations.csv",
)
print(
    "transitions:",
    OUTPUT_ROOT
    / "paired_outcome_transitions.csv",
)
print(
    "manifest:",
    OUTPUT_MANIFEST_PATH,
)

print(
    "\nFORMAL825_DETECTOR_CONDITIONED_"
    "STATISTICS_PASS"
)
