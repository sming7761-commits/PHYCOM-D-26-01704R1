import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/phyguard_revision")

PLAN_PATH = (
    ROOT
    / "configs"
    / "sionna_formal_test840_plan_v1.json"
)

CAUSAL_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_causal_validation_v2.json"
)

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "sionna_formal825_upstream_detector_protocol_v1.json"
)

PROTOCOL_LOCK_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal825_upstream_detector_protocol_lock_v1.json"
)

PROPOSAL_ROOT = (
    ROOT
    / "results"
    / "sionna_formal825_upstream_detector_proposals_v1"
)

PROPOSAL_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal825_upstream_detector_proposals_v1.json"
)

IFOREST_MODEL_PATH = (
    ROOT
    / "artifacts"
    / "frozen_upstream_detectors"
    / "formal825_source_fitted_isolation_forest_v1.joblib"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "sionna_formal825_upstream_detector_evaluation_v1"
)

TEMP_ROOT = Path(
    str(OUTPUT_ROOT) + "_tmp"
)

OUTPUT_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "sionna_formal825_upstream_detector_evaluation_v1.json"
)

EXPECTED_PLAN_SHA256 = (
    "b767fe07cd17eef0c42ec58cfe801732"
    "d38f7124ed5859915195e21223083437"
)

EXPECTED_PROTOCOL_SHA256 = (
    "4a9b09b82e82efa4940602a73a25cdd"
    "aa05361af4c5edee2ae73f46860996aa2"
)

EXPECTED_IFOREST_SHA256 = (
    "9ad52cf66ffff12f423959db37ff9ed7"
    "c5d7292d620023c531217c42c5040dd1"
)

EXPECTED_FORMAL825_ARCHIVE_SHA256 = (
    "5d08fc3b8265789819b0f49c1f93a9c7"
    "9c8fe44ce7ec4aec8f1fa603a08c241b"
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
BOOTSTRAP_SEED = 20260731

EXPECTED_ELIGIBLE_COUNT = 825
EXPECTED_PHYSICAL_COUNT = 706
EXPECTED_CONTROL_COUNT = 119
EXPECTED_TOTAL_PROPOSALS = 2475
EXPECTED_LOCALIZATION_ROWS = 2118


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def interval_iou(
    proposal_start,
    proposal_end,
    event_start,
    event_end,
):
    intersection = max(
        0,
        min(
            proposal_end,
            event_end,
        )
        - max(
            proposal_start,
            event_start,
        ),
    )

    union = (
        max(
            proposal_end,
            event_end,
        )
        - min(
            proposal_start,
            event_start,
        )
    )

    if union <= 0:
        return 0.0

    return float(
        intersection / union
    )


def compute_metrics(frame):
    if len(frame) == 0:
        raise RuntimeError(
            "Cannot compute localization metrics "
            "for an empty group."
        )

    return {
        "sample_count":
            int(
                frame[
                    "sample_id"
                ].nunique()
            ),

        "proposal_count":
            int(len(frame)),

        "mean_iou":
            float(
                frame["iou"].mean()
            ),

        "median_iou":
            float(
                frame["iou"].median()
            ),

        "event_recall_iou_ge_0_3":
            float(
                frame[
                    "iou_ge_0_3"
                ].mean()
            ),

        "event_recall_iou_ge_0_5":
            float(
                frame[
                    "iou_ge_0_5"
                ].mean()
            ),

        "mean_abs_start_error":
            float(
                frame[
                    "absolute_start_error"
                ].mean()
            ),

        "median_abs_start_error":
            float(
                frame[
                    "absolute_start_error"
                ].median()
            ),

        "mean_abs_end_error":
            float(
                frame[
                    "absolute_end_error"
                ].mean()
            ),

        "median_abs_end_error":
            float(
                frame[
                    "absolute_end_error"
                ].median()
            ),

        "mean_abs_center_error":
            float(
                frame[
                    "absolute_center_error"
                ].mean()
            ),

        "median_abs_center_error":
            float(
                frame[
                    "absolute_center_error"
                ].median()
            ),

        "exact_start_rate":
            float(
                frame[
                    "exact_start"
                ].mean()
            ),

        "proposal_overlap_rate":
            float(
                frame[
                    "iou_positive"
                ].mean()
            ),
    }


def bootstrap_group(
    frame,
    repetitions,
    seed,
):
    frame = (
        frame
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    if (
        frame["sample_id"].nunique()
        != len(frame)
    ):
        raise RuntimeError(
            "Bootstrap group contains duplicate sample IDs."
        )

    values = {
        "mean_iou":
            frame[
                "iou"
            ].to_numpy(dtype=float),

        "event_recall_iou_ge_0_3":
            frame[
                "iou_ge_0_3"
            ].to_numpy(dtype=float),

        "event_recall_iou_ge_0_5":
            frame[
                "iou_ge_0_5"
            ].to_numpy(dtype=float),

        "mean_abs_start_error":
            frame[
                "absolute_start_error"
            ].to_numpy(dtype=float),

        "mean_abs_end_error":
            frame[
                "absolute_end_error"
            ].to_numpy(dtype=float),

        "mean_abs_center_error":
            frame[
                "absolute_center_error"
            ].to_numpy(dtype=float),
    }

    sample_count = len(frame)

    rng = np.random.default_rng(
        seed
    )

    indices = rng.integers(
        0,
        sample_count,
        size=(
            repetitions,
            sample_count,
        ),
    )

    draws = {}

    for metric, metric_values in (
        values.items()
    ):
        draws[metric] = (
            metric_values[
                indices
            ].mean(axis=1)
        )

    return draws


def percentile_interval(draws):
    draws = np.asarray(
        draws,
        dtype=float,
    )

    return {
        "lower_95":
            float(
                np.quantile(
                    draws,
                    0.025,
                )
            ),

        "upper_95":
            float(
                np.quantile(
                    draws,
                    0.975,
                )
            ),
    }


required_paths = [
    PLAN_PATH,
    CAUSAL_MANIFEST_PATH,
    PROTOCOL_PATH,
    PROTOCOL_LOCK_PATH,
    PROPOSAL_ROOT,
    PROPOSAL_MANIFEST_PATH,
    IFOREST_MODEL_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)


if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Final evaluation output already exists; "
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


plan = load_json(
    PLAN_PATH
)

causal = load_json(
    CAUSAL_MANIFEST_PATH
)

protocol = load_json(
    PROTOCOL_PATH
)

protocol_lock = load_json(
    PROTOCOL_LOCK_PATH
)

proposal_manifest = load_json(
    PROPOSAL_MANIFEST_PATH
)


if (
    plan.get("plan_sha256")
    != EXPECTED_PLAN_SHA256
):
    raise RuntimeError(
        "Formal plan SHA256 mismatch."
    )

if (
    sha256_file(PROTOCOL_PATH)
    != EXPECTED_PROTOCOL_SHA256
):
    raise RuntimeError(
        "Locked detector protocol SHA256 mismatch."
    )

if protocol.get("status") != "LOCKED":
    raise RuntimeError(
        "Detector protocol is not LOCKED."
    )

if protocol_lock.get("status") != "PASS":
    raise RuntimeError(
        "Detector protocol lock is not PASS."
    )

if (
    protocol_lock.get("protocol_sha256")
    != EXPECTED_PROTOCOL_SHA256
):
    raise RuntimeError(
        "Protocol-lock hash mismatch."
    )

if (
    sha256_file(IFOREST_MODEL_PATH)
    != EXPECTED_IFOREST_SHA256
):
    raise RuntimeError(
        "Frozen IsolationForest SHA256 mismatch."
    )

if proposal_manifest.get("status") != "PASS":
    raise RuntimeError(
        "Proposal-generation manifest is not PASS."
    )

if (
    proposal_manifest.get(
        "eligible_sample_count"
    )
    != EXPECTED_ELIGIBLE_COUNT
):
    raise RuntimeError(
        "Proposal eligible count is not 825."
    )

if (
    proposal_manifest.get(
        "total_proposal_count"
    )
    != EXPECTED_TOTAL_PROPOSALS
):
    raise RuntimeError(
        "Proposal count is not 2475."
    )

if (
    proposal_manifest.get(
        "protocol_sha256"
    )
    != EXPECTED_PROTOCOL_SHA256
):
    raise RuntimeError(
        "Proposal manifest protocol hash mismatch."
    )

if (
    proposal_manifest.get(
        "formal825_parent_archive_sha256"
    )
    != EXPECTED_FORMAL825_ARCHIVE_SHA256
):
    raise RuntimeError(
        "Proposal manifest parent archive mismatch."
    )

if (
    proposal_manifest
    .get("determinism", {})
    .get("proposal_replay")
    != "2475/2475 exact"
):
    raise RuntimeError(
        "Proposal replay is not 2475/2475 exact."
    )

if (
    proposal_manifest
    .get("determinism", {})
    .get("feature_replay")
    != "2475/2475 exact"
):
    raise RuntimeError(
        "Proposal feature replay is not exact."
    )


proposal_file_records = {
    record["relative_path"]:
        record
    for record in proposal_manifest.get(
        "files",
        [],
    )
}

for relative_path, record in (
    proposal_file_records.items()
):
    path = ROOT / relative_path

    if not path.exists():
        raise FileNotFoundError(path)

    if (
        sha256_file(path)
        != record["sha256"]
    ):
        raise RuntimeError(
            "Proposal source file changed: "
            f"{relative_path}"
        )


samples = plan.get(
    "samples",
    [],
)

if len(samples) != 840:
    raise RuntimeError(
        "Formal plan sample count is not 840."
    )

sample_by_id = {
    sample["sample_id"]:
        sample
    for sample in samples
}


eligible_ids = sorted(
    record["sample_id"]
    for record in causal.get(
        "records",
        [],
    )
    if record.get(
        "causal_v2_status"
    ) == "PASS"
)

if len(eligible_ids) != 825:
    raise RuntimeError(
        "Causal-valid sample count is not 825."
    )

if len(set(eligible_ids)) != 825:
    raise RuntimeError(
        "Causal-valid sample IDs are not unique."
    )


sample_order_hash = sha256_bytes(
    json.dumps(
        eligible_ids,
        separators=(",", ":"),
    ).encode("utf-8")
)

if (
    proposal_manifest.get(
        "eligible_sample_order_sha256"
    )
    != sample_order_hash
):
    raise RuntimeError(
        "Proposal sample-order hash mismatch."
    )


ground_truth_rows = []

for sample_id in eligible_ids:
    sample = sample_by_id.get(
        sample_id
    )

    if sample is None:
        raise RuntimeError(
            f"Sample missing from plan: {sample_id}"
        )

    label = str(
        sample["label"]
    )

    localization_eligible = (
        label in PHYSICAL_CLASSES
    )

    event_start = (
        int(sample["event_start"])
        if localization_eligible
        else None
    )

    event_end = (
        int(sample["event_end"])
        if localization_eligible
        else None
    )

    if localization_eligible:
        if not (
            0
            <= event_start
            < event_end
            <= 80
        ):
            raise RuntimeError(
                f"{sample_id}: invalid event interval."
            )

    ground_truth_rows.append(
        {
            "sample_id":
                sample_id,

            "label":
                label,

            "severity":
                str(
                    sample["severity"]
                ),

            "channel_model":
                str(
                    sample["channel_model"]
                ),

            "event_start":
                event_start,

            "event_end":
                event_end,

            "event_length":
                (
                    event_end
                    - event_start
                    if localization_eligible
                    else None
                ),

            "localization_eligible":
                localization_eligible,
        }
    )


ground_truth = pd.DataFrame(
    ground_truth_rows
)


physical_count = int(
    ground_truth[
        "localization_eligible"
    ].sum()
)

control_count = int(
    (
        ~ground_truth[
            "localization_eligible"
        ]
    ).sum()
)


if physical_count != EXPECTED_PHYSICAL_COUNT:
    raise RuntimeError(
        f"Physical count={physical_count}, expected 706."
    )

if control_count != EXPECTED_CONTROL_COUNT:
    raise RuntimeError(
        f"Control count={control_count}, expected 119."
    )


evaluation_rows = []
proposal_hashes_before = {}


for detector in DETECTORS:
    proposal_path = (
        PROPOSAL_ROOT
        / detector
        / "proposals.csv"
    )

    if not proposal_path.exists():
        raise FileNotFoundError(
            proposal_path
        )

    proposal_hashes_before[
        detector
    ] = sha256_file(
        proposal_path
    )

    proposals = pd.read_csv(
        proposal_path
    )

    if len(proposals) != 825:
        raise RuntimeError(
            f"{detector}: proposal rows={len(proposals)}."
        )

    if proposals[
        "sample_id"
    ].nunique() != 825:
        raise RuntimeError(
            f"{detector}: sample IDs are not unique."
        )

    if sorted(
        proposals["sample_id"].tolist()
    ) != eligible_ids:
        raise RuntimeError(
            f"{detector}: sample-ID alignment mismatch."
        )

    if not (
        proposals[
            "detector"
        ] == detector
    ).all():
        raise RuntimeError(
            f"{detector}: detector column mismatch."
        )

    if not (
        proposals[
            "proposal_end"
        ]
        - proposals[
            "proposal_start"
        ]
        == 24
    ).all():
        raise RuntimeError(
            f"{detector}: proposal length mismatch."
        )

    merged = proposals.merge(
        ground_truth,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    if merged[
        "label"
    ].isna().any():
        raise RuntimeError(
            f"{detector}: missing ground truth after merge."
        )

    for row in merged.itertuples(
        index=False
    ):
        base = {
            "formal_index":
                int(
                    row.formal_index
                ),

            "sample_id":
                str(
                    row.sample_id
                ),

            "detector":
                detector,

            "label":
                str(
                    row.label
                ),

            "severity":
                str(
                    row.severity
                ),

            "channel_model":
                str(
                    row.channel_model
                ),

            "proposal_start":
                int(
                    row.proposal_start
                ),

            "proposal_end":
                int(
                    row.proposal_end
                ),

            "proposal_length":
                int(
                    row.proposal_length
                ),

            "max_frame_score":
                float(
                    row.max_frame_score
                ),

            "proposal_mean_score":
                float(
                    row.proposal_mean_score
                ),

            "localization_eligible":
                bool(
                    row.localization_eligible
                ),
        }

        if not row.localization_eligible:
            base.update(
                {
                    "event_start":
                        None,

                    "event_end":
                        None,

                    "event_length":
                        None,

                    "iou":
                        None,

                    "iou_positive":
                        None,

                    "iou_ge_0_3":
                        None,

                    "iou_ge_0_5":
                        None,

                    "absolute_start_error":
                        None,

                    "absolute_end_error":
                        None,

                    "absolute_center_error":
                        None,

                    "exact_start":
                        None,
                }
            )

            evaluation_rows.append(
                base
            )

            continue

        event_start = int(
            row.event_start
        )

        event_end = int(
            row.event_end
        )

        proposal_start = int(
            row.proposal_start
        )

        proposal_end = int(
            row.proposal_end
        )

        iou = interval_iou(
            proposal_start,
            proposal_end,
            event_start,
            event_end,
        )

        proposal_center = (
            proposal_start
            + proposal_end
        ) / 2.0

        event_center = (
            event_start
            + event_end
        ) / 2.0

        base.update(
            {
                "event_start":
                    event_start,

                "event_end":
                    event_end,

                "event_length":
                    event_end
                    - event_start,

                "iou":
                    iou,

                "iou_positive":
                    bool(
                        iou > 0.0
                    ),

                "iou_ge_0_3":
                    bool(
                        iou >= 0.3
                    ),

                "iou_ge_0_5":
                    bool(
                        iou >= 0.5
                    ),

                "absolute_start_error":
                    abs(
                        proposal_start
                        - event_start
                    ),

                "absolute_end_error":
                    abs(
                        proposal_end
                        - event_end
                    ),

                "absolute_center_error":
                    abs(
                        proposal_center
                        - event_center
                    ),

                "exact_start":
                    bool(
                        proposal_start
                        == event_start
                    ),
            }
        )

        evaluation_rows.append(
            base
        )


evaluation = pd.DataFrame(
    evaluation_rows
)


if len(evaluation) != 2475:
    raise RuntimeError(
        f"Evaluation row count={len(evaluation)}."
    )


localization = evaluation[
    evaluation[
        "localization_eligible"
    ]
].copy()


if len(localization) != EXPECTED_LOCALIZATION_ROWS:
    raise RuntimeError(
        "Localization evaluation row count "
        f"is {len(localization)}, expected 2118."
    )

if localization[
    "iou"
].isna().any():
    raise RuntimeError(
        "Physical localization rows contain missing IoU."
    )


overall_rows = []
stratified_rows = []


for detector in DETECTORS:
    frame = localization[
        localization[
            "detector"
        ] == detector
    ]

    metrics = compute_metrics(
        frame
    )

    overall_rows.append(
        {
            "detector":
                detector,

            "dimension":
                "overall",

            "group":
                "all_physical",

            **metrics,
        }
    )


group_specs = [
    (
        "label",
        ["detector", "label"],
    ),
    (
        "severity",
        ["detector", "severity"],
    ),
    (
        "channel",
        ["detector", "channel_model"],
    ),
]


for dimension, columns in group_specs:
    grouped = localization.groupby(
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

        metrics = compute_metrics(
            frame
        )

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

                **{
                    column:
                        key_map[column]
                    for column
                    in columns
                },

                **metrics,
            }
        )


overall = pd.DataFrame(
    overall_rows
)

stratified = pd.DataFrame(
    stratified_rows
)


bootstrap_ci_rows = []
bootstrap_draw_arrays = {}


bootstrap_groups = []


for detector in DETECTORS:
    bootstrap_groups.append(
        (
            "overall",
            detector,
            "all_physical",
            localization[
                localization[
                    "detector"
                ] == detector
            ],
        )
    )

    for severity in [
        "mild",
        "moderate",
        "severe",
    ]:
        bootstrap_groups.append(
            (
                "severity",
                detector,
                severity,
                localization[
                    (
                        localization[
                            "detector"
                        ] == detector
                    )
                    & (
                        localization[
                            "severity"
                        ] == severity
                    )
                ],
            )
        )


for group_index, (
    dimension,
    detector,
    group_name,
    frame,
) in enumerate(
    bootstrap_groups
):
    if len(frame) == 0:
        raise RuntimeError(
            "Empty bootstrap group: "
            f"{detector}/{group_name}"
        )

    draws = bootstrap_group(
        frame,
        BOOTSTRAP_REPETITIONS,
        BOOTSTRAP_SEED
        + group_index,
    )

    point_metrics = compute_metrics(
        frame
    )

    safe_group = (
        f"{dimension}__"
        f"{detector}__"
        f"{group_name}"
    )

    for metric, metric_draws in (
        draws.items()
    ):
        interval = percentile_interval(
            metric_draws
        )

        bootstrap_ci_rows.append(
            {
                "dimension":
                    dimension,

                "detector":
                    detector,

                "group":
                    group_name,

                "sample_count":
                    int(
                        frame[
                            "sample_id"
                        ].nunique()
                    ),

                "metric":
                    metric,

                "point_estimate":
                    float(
                        point_metrics[
                            metric
                        ]
                    ),

                "lower_95":
                    interval[
                        "lower_95"
                    ],

                "upper_95":
                    interval[
                        "upper_95"
                    ],

                "bootstrap_repetitions":
                    BOOTSTRAP_REPETITIONS,

                "bootstrap_seed":
                    BOOTSTRAP_SEED
                    + group_index,

                "resampling_unit":
                    "sample_id",
            }
        )

        bootstrap_draw_arrays[
            f"{safe_group}__{metric}"
        ] = metric_draws


bootstrap_ci = pd.DataFrame(
    bootstrap_ci_rows
)


evaluation_path = (
    TEMP_ROOT
    / "proposal_evaluations.csv"
)

overall_path = (
    TEMP_ROOT
    / "proposal_metrics_overall.csv"
)

stratified_path = (
    TEMP_ROOT
    / "proposal_metrics_stratified.csv"
)

bootstrap_ci_path = (
    TEMP_ROOT
    / "proposal_bootstrap_confidence_intervals.csv"
)

bootstrap_draws_path = (
    TEMP_ROOT
    / "proposal_bootstrap_draws.npz"
)

summary_path = (
    TEMP_ROOT
    / "summary.json"
)


evaluation.to_csv(
    evaluation_path,
    index=False,
)

overall.to_csv(
    overall_path,
    index=False,
)

stratified.to_csv(
    stratified_path,
    index=False,
)

bootstrap_ci.to_csv(
    bootstrap_ci_path,
    index=False,
)

np.savez_compressed(
    bootstrap_draws_path,
    **bootstrap_draw_arrays,
)


summary = {
    "schema":
        "phyguard.sionna.formal825."
        "upstream_detector_evaluation.v1",

    "status":
        "PASS",

    "protocol_sha256":
        EXPECTED_PROTOCOL_SHA256,

    "proposal_manifest_sha256":
        sha256_file(
            PROPOSAL_MANIFEST_PATH
        ),

    "evaluation_scope": {
        "causal_valid_sample_count":
            825,

        "physical_localization_sample_count":
            706,

        "control_sample_count":
            119,

        "detector_count":
            3,

        "proposal_count":
            2475,

        "physical_localization_evaluation_count":
            2118,
    },

    "interval_semantics":
        "Left-closed, right-open intervals.",

    "localization_metrics":
        [
            "IoU",
            "event recall at IoU >= 0.3",
            "event recall at IoU >= 0.5",
            "absolute start error",
            "absolute end error",
            "absolute center error",
        ],

    "overall_metrics":
        overall.to_dict(
            orient="records"
        ),

    "bootstrap": {
        "repetitions":
            BOOTSTRAP_REPETITIONS,

        "base_seed":
            BOOTSTRAP_SEED,

        "resampling_unit":
            "sample_id",

        "groups":
            [
                "detector overall",
                "detector x severity",
            ],
    },

    "methodological_boundary": {
        "ground_truth_labels_opened":
            True,

        "ground_truth_event_intervals_opened":
            True,

        "ground_truth_used_for_evaluation_only":
            True,

        "proposal_generation_repeated":
            False,

        "proposal_files_modified":
            False,

        "detector_parameter_tuning":
            False,

        "candidate_window_tuning":
            False,

        "post_result_detector_selection":
            False,

        "downstream_diagnostic_predictions_generated":
            False,

        "control_localization_metrics_computed":
            False,

        "all_three_detectors_reported":
            True,
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


for detector in DETECTORS:
    proposal_path = (
        PROPOSAL_ROOT
        / detector
        / "proposals.csv"
    )

    if (
        sha256_file(proposal_path)
        != proposal_hashes_before[
            detector
        ]
    ):
        raise RuntimeError(
            f"{detector}: proposal CSV changed "
            "during evaluation."
        )


if (
    sha256_file(IFOREST_MODEL_PATH)
    != EXPECTED_IFOREST_SHA256
):
    raise RuntimeError(
        "Frozen IsolationForest changed "
        "during evaluation."
    )

if (
    sha256_file(PROTOCOL_PATH)
    != EXPECTED_PROTOCOL_SHA256
):
    raise RuntimeError(
        "Locked protocol changed "
        "during evaluation."
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
        "upstream_detector_evaluation_manifest.v1",

    "status":
        "PASS",

    "protocol_sha256":
        EXPECTED_PROTOCOL_SHA256,

    "proposal_manifest_path":
        str(
            PROPOSAL_MANIFEST_PATH
        ),

    "proposal_manifest_sha256":
        sha256_file(
            PROPOSAL_MANIFEST_PATH
        ),

    "eligible_sample_count":
        825,

    "physical_localization_sample_count":
        706,

    "control_sample_count":
        119,

    "detector_count":
        3,

    "proposal_count":
        2475,

    "localization_evaluation_count":
        2118,

    "bootstrap": {
        "repetitions":
            BOOTSTRAP_REPETITIONS,

        "base_seed":
            BOOTSTRAP_SEED,

        "resampling_unit":
            "sample_id",
    },

    "source_integrity": {
        "proposal_files_unchanged":
            True,

        "protocol_unchanged":
            True,

        "frozen_isolation_forest_unchanged":
            True,
    },

    "methodological_boundary": {
        "labels_used_for_proposal_generation":
            False,

        "event_intervals_used_for_proposal_generation":
            False,

        "labels_used_for_evaluation":
            True,

        "event_intervals_used_for_evaluation":
            True,

        "detector_parameter_tuning":
            False,

        "candidate_window_tuning":
            False,

        "detector_selection":
            False,

        "downstream_predictions_generated":
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
print("eligible_sample_count: 825")
print("physical_localization_sample_count: 706")
print("control_sample_count: 119")
print("detector_count: 3")
print("proposal_count: 2475")
print("localization_evaluation_count: 2118")
print(
    "bootstrap_repetitions:",
    BOOTSTRAP_REPETITIONS,
)
print(
    "ground_truth_used_for_evaluation_only:",
    True,
)
print(
    "proposal_files_unchanged:",
    True,
)
print(
    "detector_parameter_tuning:",
    False,
)
print(
    "candidate_window_tuning:",
    False,
)
print(
    "downstream_predictions_generated:",
    False,
)


print("\nOVERALL DETECTOR METRICS")

print(
    overall[
        [
            "detector",
            "sample_count",
            "mean_iou",
            "median_iou",
            "event_recall_iou_ge_0_3",
            "event_recall_iou_ge_0_5",
            "mean_abs_start_error",
            "mean_abs_end_error",
            "mean_abs_center_error",
        ]
    ].to_string(
        index=False
    )
)


print("\nDETECTOR × SEVERITY METRICS")

severity_view = stratified[
    stratified[
        "dimension"
    ] == "severity"
][
    [
        "detector",
        "severity",
        "sample_count",
        "mean_iou",
        "event_recall_iou_ge_0_3",
        "event_recall_iou_ge_0_5",
        "mean_abs_start_error",
        "mean_abs_end_error",
    ]
]

print(
    severity_view.to_string(
        index=False
    )
)


print("\nDETECTOR × LABEL METRICS")

label_view = stratified[
    stratified[
        "dimension"
    ] == "label"
][
    [
        "detector",
        "label",
        "sample_count",
        "mean_iou",
        "event_recall_iou_ge_0_3",
        "event_recall_iou_ge_0_5",
        "mean_abs_start_error",
        "mean_abs_end_error",
    ]
]

print(
    label_view.to_string(
        index=False
    )
)


print("\nDETECTOR × CHANNEL METRICS")

channel_view = stratified[
    stratified[
        "dimension"
    ] == "channel"
][
    [
        "detector",
        "channel_model",
        "sample_count",
        "mean_iou",
        "event_recall_iou_ge_0_3",
        "event_recall_iou_ge_0_5",
        "mean_abs_start_error",
        "mean_abs_end_error",
    ]
]

print(
    channel_view.to_string(
        index=False
    )
)


print("\nOVERALL BOOTSTRAP 95% INTERVALS")

overall_ci = bootstrap_ci[
    bootstrap_ci[
        "dimension"
    ] == "overall"
]

print(
    overall_ci[
        [
            "detector",
            "metric",
            "point_estimate",
            "lower_95",
            "upper_95",
        ]
    ].to_string(
        index=False
    )
)


print("\noutput_root:", OUTPUT_ROOT)
print("summary:", OUTPUT_ROOT / "summary.json")
print("manifest:", OUTPUT_MANIFEST_PATH)

print(
    "\nFORMAL825_LOCKED_UPSTREAM_"
    "PROPOSAL_EVALUATION_PASS"
)
