import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest


ROOT = Path("/root/phyguard_revision")

R0_ROOT = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
)

DETECTOR_SCRIPT = (
    R0_ROOT
    / "scripts"
    / "detector_proposals.py"
)

FEATURE_SCRIPT = (
    R0_ROOT
    / "scripts"
    / "features.py"
)

FORMAL_PLAN = (
    ROOT
    / "configs"
    / "sionna_formal_test840_plan_v1.json"
)

CAUSAL_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_causal_validation_v2.json"
)

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "sionna_formal825_upstream_detector_protocol_v1.json"
)

PROTOCOL_LOCK = (
    ROOT
    / "manifests"
    / "sionna_formal825_upstream_detector_protocol_lock_v1.json"
)

PROTOCOL_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal825_upstream_detector_protocol_locked.zip"
)

FORMAL825_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal825_locked_reconstruction_final.zip"
)

IFOREST_MODEL = (
    ROOT
    / "artifacts"
    / "frozen_upstream_detectors"
    / "formal825_source_fitted_isolation_forest_v1.joblib"
)

DATA_ROOT = (
    ROOT
    / "data"
    / "sionna_formal_test840"
)

LEGACY_RESULT_ROOT = (
    ROOT
    / "results"
    / "sionna_formal_test840"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "sionna_formal825_upstream_detector_proposals_v1"
)

TEMP_ROOT = Path(
    str(OUTPUT_ROOT) + "_tmp"
)

OUTPUT_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal825_upstream_detector_proposals_v1.json"
)

EXPECTED_PLAN_SHA256 = (
    "b767fe07cd17eef0c42ec58cfe801732"
    "d38f7124ed5859915195e21223083437"
)

EXPECTED_PROTOCOL_SHA256 = (
    "4a9b09b82e82efa4940602a73a25cdd"
    "aa05361af4c5edee2ae73f46860996aa2"
)

EXPECTED_PROTOCOL_ARCHIVE_SHA256 = (
    "a0aecfcb798a4c6113992abb39686c4d"
    "9140e03bc6a5c3ee9e4d6946c62d9053"
)

EXPECTED_FORMAL825_ARCHIVE_SHA256 = (
    "5d08fc3b8265789819b0f49c1f93a9c7"
    "9c8fe44ce7ec4aec8f1fa603a08c241b"
)

EXPECTED_IFOREST_SHA256 = (
    "9ad52cf66ffff12f423959db37ff9ed7"
    "c5d7292d620023c531217c42c5040dd1"
)

EXPECTED_DETECTOR_SCRIPT_SHA256 = (
    "f38672d30a769aebeb0b0e156f187dce"
    "b9daa7317fe97fc32faaec0186f54311"
)

EXPECTED_FEATURE_SCRIPT_SHA256 = (
    "b665765bbdb317f0568d77a5004e4107"
    "5e4a2e037fc5f6caa8102f6c932fc4f7"
)

EXPECTED_ORIGINAL_EVIDENCE_SHA256 = (
    "2a003bc8ecce4243309b297be36740669"
    "c04a8e7dfa27384a5f2de33ec3cc297"
)

DETECTORS = [
    "robust_energy",
    "pca_reconstruction",
    "isolation_forest",
]

REFERENCE_PREFIX = 24
CANDIDATE_WINDOW = 24
EARLIEST_START = 24


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


def import_module(path, name):
    directory = str(path.parent)

    if directory not in sys.path:
        sys.path.insert(
            0,
            directory,
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import module: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


def aggregate_original_evidence(samples):
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

            LEGACY_RESULT_ROOT
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


required = [
    DETECTOR_SCRIPT,
    FEATURE_SCRIPT,
    FORMAL_PLAN,
    CAUSAL_MANIFEST,
    PROTOCOL_PATH,
    PROTOCOL_LOCK,
    PROTOCOL_ARCHIVE,
    FORMAL825_ARCHIVE,
    IFOREST_MODEL,
]

for path in required:
    if not path.exists():
        raise FileNotFoundError(path)


if sklearn.__version__ != "1.8.0":
    raise RuntimeError(
        f"scikit-learn={sklearn.__version__}, "
        "expected 1.8.0."
    )


if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
    raise RuntimeError(
        "Locked upstream protocol SHA256 mismatch."
    )

if (
    sha256_file(PROTOCOL_ARCHIVE)
    != EXPECTED_PROTOCOL_ARCHIVE_SHA256
):
    raise RuntimeError(
        "Locked upstream protocol archive SHA256 mismatch."
    )

if (
    sha256_file(FORMAL825_ARCHIVE)
    != EXPECTED_FORMAL825_ARCHIVE_SHA256
):
    raise RuntimeError(
        "Formal825 final archive SHA256 mismatch."
    )

if sha256_file(IFOREST_MODEL) != EXPECTED_IFOREST_SHA256:
    raise RuntimeError(
        "Source-fitted IsolationForest SHA256 mismatch."
    )

if (
    sha256_file(DETECTOR_SCRIPT)
    != EXPECTED_DETECTOR_SCRIPT_SHA256
):
    raise RuntimeError(
        "detector_proposals.py SHA256 mismatch."
    )

if (
    sha256_file(FEATURE_SCRIPT)
    != EXPECTED_FEATURE_SCRIPT_SHA256
):
    raise RuntimeError(
        "features.py SHA256 mismatch."
    )


protocol = load_json(
    PROTOCOL_PATH
)

protocol_lock = load_json(
    PROTOCOL_LOCK
)

plan = load_json(
    FORMAL_PLAN
)

causal = load_json(
    CAUSAL_MANIFEST
)


if protocol.get("status") != "LOCKED":
    raise RuntimeError(
        "Upstream detector protocol is not LOCKED."
    )

if protocol_lock.get("status") != "PASS":
    raise RuntimeError(
        "Upstream detector protocol lock is not PASS."
    )

if (
    protocol_lock.get("protocol_sha256")
    != EXPECTED_PROTOCOL_SHA256
):
    raise RuntimeError(
        "Protocol-lock manifest hash mismatch."
    )

if (
    protocol_lock
    .get(
        "source_fitted_isolation_forest",
        {},
    )
    .get("sha256")
    != EXPECTED_IFOREST_SHA256
):
    raise RuntimeError(
        "Protocol-lock IsolationForest hash mismatch."
    )

if plan.get("plan_sha256") != EXPECTED_PLAN_SHA256:
    raise RuntimeError(
        "Formal Test-840 plan SHA256 mismatch."
    )

if causal.get("causal_v2_passed_count") != 825:
    raise RuntimeError(
        "Expected 825 causal-valid samples."
    )


all_samples = plan.get(
    "samples",
    [],
)

if len(all_samples) != 840:
    raise RuntimeError(
        "Formal plan sample count is not 840."
    )


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
        "Eligible Formal825 ID count is not 825."
    )

if len(set(eligible_ids)) != 825:
    raise RuntimeError(
        "Eligible Formal825 IDs are not unique."
    )


original_evidence_hash_before, evidence_count_before = (
    aggregate_original_evidence(
        all_samples
    )
)

if (
    original_evidence_hash_before
    != EXPECTED_ORIGINAL_EVIDENCE_SHA256
):
    raise RuntimeError(
        "Original evidence aggregate SHA256 mismatch."
    )

if evidence_count_before != 2520:
    raise RuntimeError(
        "Original evidence file count is not 2520."
    )


detector_module = import_module(
    DETECTOR_SCRIPT,
    "phyguard_formal825_proposal_detector",
)

feature_module = import_module(
    FEATURE_SCRIPT,
    "phyguard_formal825_proposal_features",
)


if list(
    detector_module.DETECTORS
) != DETECTORS:
    raise RuntimeError(
        "Frozen detector order mismatch."
    )


proposal_for_sequence = (
    detector_module.proposal_for_sequence
)

build_one = feature_module.build_one


source_iforest = joblib.load(
    IFOREST_MODEL
)

if not isinstance(
    source_iforest,
    IsolationForest,
):
    raise RuntimeError(
        "Frozen upstream model is not IsolationForest."
    )

if int(source_iforest.n_features_in_) != 10:
    raise RuntimeError(
        "IsolationForest feature dimension is not 10."
    )

if len(source_iforest.estimators_) != 200:
    raise RuntimeError(
        "IsolationForest estimator count is not 200."
    )


if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Final proposal output already exists; "
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


metadata_rows = []
sequence_cache = {}


for formal_index, sample_id in enumerate(
    eligible_ids
):
    sequence_path = (
        DATA_ROOT
        / sample_id
        / "sequence.npz"
    )

    if not sequence_path.exists():
        raise FileNotFoundError(
            sequence_path
        )

    with np.load(
        sequence_path,
        allow_pickle=False,
    ) as archive:
        if "kpi_sequence" not in archive.files:
            raise RuntimeError(
                f"{sample_id}: kpi_sequence missing."
            )

        sequence = np.asarray(
            archive["kpi_sequence"],
            dtype=np.float32,
        )

    if sequence.shape != (80, 10):
        raise RuntimeError(
            f"{sample_id}: sequence shape={sequence.shape}."
        )

    if not np.isfinite(sequence).all():
        raise RuntimeError(
            f"{sample_id}: sequence contains NaN or Inf."
        )

    sequence_cache[
        sample_id
    ] = sequence

    metadata_rows.append(
        {
            "formal_index":
                formal_index,

            "sample_id":
                sample_id,

            "causal_v2_status":
                "PASS",

            "sequence_relative_path":
                str(
                    sequence_path.relative_to(
                        ROOT
                    )
                ),

            "sequence_sha256":
                sha256_file(
                    sequence_path
                ),
        }
    )


metadata = pd.DataFrame(
    metadata_rows
)

metadata.to_csv(
    TEMP_ROOT / "metadata.csv",
    index=False,
)


detector_summaries = {}
proposal_count_total = 0
deterministic_proposal_count = 0
deterministic_feature_count = 0


for detector in DETECTORS:
    print(
        "generating_detector:",
        detector,
    )

    detector_dir = (
        TEMP_ROOT / detector
    )

    detector_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    proposal_rows = []
    score_rows = []
    raw_rows = []
    physical_rows = []


    for formal_index, sample_id in enumerate(
        eligible_ids
    ):
        sequence = sequence_cache[
            sample_id
        ]

        global_iforest = (
            source_iforest
            if detector == "isolation_forest"
            else None
        )

        start_1, end_1, score_1 = (
            proposal_for_sequence(
                sequence,
                detector,
                0,
                window=CANDIDATE_WINDOW,
                global_iforest=global_iforest,
            )
        )

        start_2, end_2, score_2 = (
            proposal_for_sequence(
                sequence,
                detector,
                0,
                window=CANDIDATE_WINDOW,
                global_iforest=global_iforest,
            )
        )

        score_1 = np.asarray(
            score_1,
            dtype=np.float64,
        )

        score_2 = np.asarray(
            score_2,
            dtype=np.float64,
        )

        if (
            start_1 != start_2
            or end_1 != end_2
            or not np.array_equal(
                score_1,
                score_2,
            )
        ):
            raise RuntimeError(
                f"{detector}/{sample_id}: "
                "proposal replay mismatch."
            )

        if score_1.shape != (80,):
            raise RuntimeError(
                f"{detector}/{sample_id}: "
                f"score shape={score_1.shape}."
            )

        if not np.isfinite(score_1).all():
            raise RuntimeError(
                f"{detector}/{sample_id}: "
                "score contains NaN or Inf."
            )

        if not (
            EARLIEST_START
            <= int(start_1)
            <= 56
        ):
            raise RuntimeError(
                f"{detector}/{sample_id}: "
                f"invalid proposal start={start_1}."
            )

        if int(end_1) - int(start_1) != 24:
            raise RuntimeError(
                f"{detector}/{sample_id}: "
                "proposal length is not 24."
            )

        if int(end_1) > 80:
            raise RuntimeError(
                f"{detector}/{sample_id}: "
                "proposal exceeds sequence."
            )


        raw_1, physical_1 = build_one(
            sequence,
            int(start_1),
            int(end_1),
            eps=1e-6,
        )

        raw_2, physical_2 = build_one(
            sequence,
            int(start_1),
            int(end_1),
            eps=1e-6,
        )

        raw_1 = np.asarray(
            raw_1,
            dtype=np.float32,
        )

        raw_2 = np.asarray(
            raw_2,
            dtype=np.float32,
        )

        physical_1 = np.asarray(
            physical_1,
            dtype=np.float32,
        )

        physical_2 = np.asarray(
            physical_2,
            dtype=np.float32,
        )

        if raw_1.shape != (60,):
            raise RuntimeError(
                f"{detector}/{sample_id}: "
                f"raw shape={raw_1.shape}."
            )

        if physical_1.shape != (19,):
            raise RuntimeError(
                f"{detector}/{sample_id}: "
                f"physical shape={physical_1.shape}."
            )

        if not np.array_equal(
            raw_1,
            raw_2,
        ):
            raise RuntimeError(
                f"{detector}/{sample_id}: "
                "raw feature replay mismatch."
            )

        if not np.array_equal(
            physical_1,
            physical_2,
        ):
            raise RuntimeError(
                f"{detector}/{sample_id}: "
                "physical feature replay mismatch."
            )

        if not (
            np.isfinite(raw_1).all()
            and np.isfinite(
                physical_1
            ).all()
        ):
            raise RuntimeError(
                f"{detector}/{sample_id}: "
                "features contain NaN or Inf."
            )


        proposal_mean_score = float(
            score_1[
                int(start_1):
                int(end_1)
            ].mean()
        )

        proposal_rows.append(
            {
                "formal_index":
                    formal_index,

                "sample_id":
                    sample_id,

                "detector":
                    detector,

                "proposal_start":
                    int(start_1),

                "proposal_end":
                    int(end_1),

                "proposal_length":
                    int(end_1)
                    - int(start_1),

                "max_frame_score":
                    float(
                        score_1.max()
                    ),

                "proposal_mean_score":
                    proposal_mean_score,

                "proposal_generation_seed_argument":
                    0,

                "formal_label_used":
                    False,

                "formal_event_interval_used":
                    False,
            }
        )

        score_rows.append(
            score_1
        )

        raw_rows.append(
            raw_1
        )

        physical_rows.append(
            physical_1
        )

        proposal_count_total += 1
        deterministic_proposal_count += 1
        deterministic_feature_count += 1


    proposals = pd.DataFrame(
        proposal_rows
    )

    scores = np.stack(
        score_rows
    ).astype(np.float64)

    raw = np.stack(
        raw_rows
    ).astype(np.float32)

    physical = np.stack(
        physical_rows
    ).astype(np.float32)

    combined = np.concatenate(
        [
            raw,
            physical,
        ],
        axis=1,
    ).astype(np.float32)


    if len(proposals) != 825:
        raise RuntimeError(
            f"{detector}: proposal count "
            f"is {len(proposals)}, expected 825."
        )

    if scores.shape != (825, 80):
        raise RuntimeError(
            f"{detector}: score shape={scores.shape}."
        )

    if raw.shape != (825, 60):
        raise RuntimeError(
            f"{detector}: raw shape={raw.shape}."
        )

    if physical.shape != (825, 19):
        raise RuntimeError(
            f"{detector}: physical shape={physical.shape}."
        )

    if combined.shape != (825, 79):
        raise RuntimeError(
            f"{detector}: combined shape={combined.shape}."
        )


    proposals.to_csv(
        detector_dir / "proposals.csv",
        index=False,
    )

    np.savez_compressed(
        detector_dir / "scores.npz",
        scores=scores,
        sample_ids=np.asarray(
            eligible_ids,
            dtype="U64",
        ),
    )

    np.savez_compressed(
        detector_dir / "features.npz",
        raw=raw,
        physical=physical,
        combined=combined,
        sample_ids=np.asarray(
            eligible_ids,
            dtype="U64",
        ),
    )


    start_counts = (
        proposals[
            "proposal_start"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    detector_summaries[
        detector
    ] = {
        "proposal_count":
            825,

        "score_shape":
            [825, 80],

        "raw_feature_shape":
            [825, 60],

        "physical_feature_shape":
            [825, 19],

        "combined_feature_shape":
            [825, 79],

        "proposal_start_min":
            int(
                proposals[
                    "proposal_start"
                ].min()
            ),

        "proposal_start_max":
            int(
                proposals[
                    "proposal_start"
                ].max()
            ),

        "proposal_start_mean":
            float(
                proposals[
                    "proposal_start"
                ].mean()
            ),

        "unique_proposal_start_count":
            int(
                proposals[
                    "proposal_start"
                ].nunique()
            ),

        "proposal_start_counts":
            {
                str(key): int(value)
                for key, value
                in start_counts.items()
            },

        "mean_max_frame_score":
            float(
                proposals[
                    "max_frame_score"
                ].mean()
            ),

        "mean_proposal_window_score":
            float(
                proposals[
                    "proposal_mean_score"
                ].mean()
            ),

        "formal_label_used":
            False,

        "formal_event_interval_used":
            False,
    }


summary = {
    "schema":
        "phyguard.sionna.formal825."
        "upstream_detector_proposal_generation.v1",

    "status":
        "PASS",

    "protocol_sha256":
        EXPECTED_PROTOCOL_SHA256,

    "protocol_archive_sha256":
        EXPECTED_PROTOCOL_ARCHIVE_SHA256,

    "formal825_parent_archive_sha256":
        EXPECTED_FORMAL825_ARCHIVE_SHA256,

    "source_fitted_isolation_forest_sha256":
        EXPECTED_IFOREST_SHA256,

    "detector_script_sha256":
        EXPECTED_DETECTOR_SCRIPT_SHA256,

    "feature_script_sha256":
        EXPECTED_FEATURE_SCRIPT_SHA256,

    "eligible_sample_count":
        825,

    "detectors":
        DETECTORS,

    "proposal_count_per_detector":
        825,

    "total_proposal_count":
        proposal_count_total,

    "deterministic_proposal_replay":
        f"{deterministic_proposal_count}/2475",

    "deterministic_feature_replay":
        f"{deterministic_feature_count}/2475",

    "candidate_interval_contract": {
        "reference_prefix_frames":
            REFERENCE_PREFIX,

        "window_frames":
            CANDIDATE_WINDOW,

        "earliest_start":
            EARLIEST_START,

        "latest_start":
            56,

        "tie_breaking":
            "earliest maximum",
    },

    "detector_summaries":
        detector_summaries,

    "methodological_boundary": {
        "formal_labels_opened":
            False,

        "formal_labels_used":
            False,

        "formal_event_intervals_opened":
            False,

        "formal_event_intervals_used":
            False,

        "proposal_iou_computed":
            False,

        "diagnostic_predictions_generated":
            False,

        "detector_parameter_tuning":
            False,

        "candidate_window_tuning":
            False,

        "detector_selection":
            False,

        "all_three_detectors_retained":
            True,
    },
}


(TEMP_ROOT / "summary.json").write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


original_evidence_hash_after, evidence_count_after = (
    aggregate_original_evidence(
        all_samples
    )
)

if (
    original_evidence_hash_after
    != original_evidence_hash_before
):
    raise RuntimeError(
        "Original evidence changed "
        "during proposal generation."
    )

if evidence_count_after != evidence_count_before:
    raise RuntimeError(
        "Original evidence count changed."
    )

if sha256_file(IFOREST_MODEL) != EXPECTED_IFOREST_SHA256:
    raise RuntimeError(
        "Frozen IsolationForest changed "
        "during proposal generation."
    )

if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
    raise RuntimeError(
        "Locked protocol changed "
        "during proposal generation."
    )


TEMP_ROOT.rename(
    OUTPUT_ROOT
)


output_files = sorted(
    path
    for path in OUTPUT_ROOT.rglob("*")
    if path.is_file()
)


file_records = [
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
]


sample_order_hash = sha256_bytes(
    json.dumps(
        eligible_ids,
        separators=(",", ":"),
    ).encode("utf-8")
)


manifest = {
    "schema":
        "phyguard.sionna.formal825."
        "upstream_detector_proposals_manifest.v1",

    "status":
        "PASS",

    "protocol_sha256":
        EXPECTED_PROTOCOL_SHA256,

    "protocol_archive_sha256":
        EXPECTED_PROTOCOL_ARCHIVE_SHA256,

    "formal825_parent_archive_sha256":
        EXPECTED_FORMAL825_ARCHIVE_SHA256,

    "source_fitted_isolation_forest_sha256":
        EXPECTED_IFOREST_SHA256,

    "eligible_sample_count":
        825,

    "eligible_sample_order_sha256":
        sample_order_hash,

    "detectors":
        DETECTORS,

    "proposal_count_per_detector":
        825,

    "total_proposal_count":
        2475,

    "feature_count":
        2475,

    "determinism": {
        "proposal_replay":
            "2475/2475 exact",

        "feature_replay":
            "2475/2475 exact",
    },

    "original_evidence": {
        "file_count":
            2520,

        "aggregate_sha256":
            EXPECTED_ORIGINAL_EVIDENCE_SHA256,

        "unchanged":
            True,
    },

    "formal_information_boundary": {
        "labels_opened":
            False,

        "labels_used":
            False,

        "event_intervals_opened":
            False,

        "event_intervals_used":
            False,

        "ground_truth_metrics_computed":
            False,
    },

    "methodological_boundary": {
        "formal_test_model_fitting":
            False,

        "detector_parameter_tuning":
            False,

        "candidate_window_tuning":
            False,

        "post_result_detector_selection":
            False,

        "all_detector_outputs_preserved":
            True,

        "downstream_predictions_generated":
            False,
    },

    "detector_summaries":
        detector_summaries,

    "files":
        file_records,
}


OUTPUT_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_MANIFEST.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print("status: PASS")
print("eligible_sample_count: 825")
print(
    "detectors:",
    DETECTORS,
)
print(
    "proposal_count_per_detector: 825"
)
print("total_proposal_count: 2475")
print(
    "deterministic_proposal_replay: "
    "2475/2475 exact"
)
print(
    "deterministic_feature_replay: "
    "2475/2475 exact"
)
print(
    "formal_labels_opened:",
    False,
)
print(
    "formal_event_intervals_opened:",
    False,
)
print(
    "ground_truth_metrics_computed:",
    False,
)
print(
    "diagnostic_predictions_generated:",
    False,
)
print(
    "original_evidence_unchanged:",
    True,
)


print("\nDETECTOR GENERATION SUMMARY")

for detector in DETECTORS:
    value = detector_summaries[
        detector
    ]

    print(
        detector,
        "| proposal_count=",
        value[
            "proposal_count"
        ],
        "| start_min=",
        value[
            "proposal_start_min"
        ],
        "| start_max=",
        value[
            "proposal_start_max"
        ],
        "| start_mean=",
        value[
            "proposal_start_mean"
        ],
        "| unique_starts=",
        value[
            "unique_proposal_start_count"
        ],
    )


print("\noutput_root:", OUTPUT_ROOT)
print("manifest:", OUTPUT_MANIFEST)

print(
    "\nFORMAL825_LOCKED_UPSTREAM_"
    "PROPOSAL_GENERATION_PASS"
)
