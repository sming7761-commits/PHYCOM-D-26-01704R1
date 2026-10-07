import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


ROOT = Path("/root/phyguard_revision")

PLAN_PATH = (
    ROOT / "configs"
    / "sionna_formal_test840_plan_v1.json"
)

CAUSAL_MANIFEST_PATH = (
    ROOT / "manifests"
    / "sionna_formal_test840_causal_validation_v2.json"
)

PREFLIGHT_PATH = (
    ROOT / "manifests"
    / "formal_test840_locked_reconstruction_preflight.json"
)

DATA_ROOT = (
    ROOT / "data"
    / "sionna_formal_test840"
)

LEGACY_RESULT_ROOT = (
    ROOT / "results"
    / "sionna_formal_test840"
)

OUTPUT_ROOT = (
    ROOT / "results"
    / "sionna_formal_test840_locked_phyguard"
)

OUTPUT_MANIFEST = (
    ROOT / "manifests"
    / "sionna_formal_test840_locked_reconstruction_inference.json"
)

R0_ROOT = (
    ROOT / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
)

FEATURE_PATH = (
    R0_ROOT / "scripts" / "features.py"
)

EVALUATION_COMMON_PATH = (
    R0_ROOT / "scripts"
    / "evaluation_common.py"
)

MODEL_DIR = (
    R0_ROOT / "results" / "full"
    / "evaluation_suite" / "models"
)

EXPECTED_PLAN_SHA256 = (
    "b767fe07cd17eef0c42ec58cfe801732"
    "d38f7124ed5859915195e21223083437"
)

EXPECTED_POLICY_SHA256 = (
    "510b0ba97393a63a199d65414c2b44660"
    "e7140c1696582202b1603c7197f879d"
)

EXPECTED_FORMAL_ARCHIVE_SHA256 = (
    "b6c5f400b7d55949fb1e1f39c0c092d"
    "a8f780eed100866f30176e26e462ff2f5"
)

EXPECTED_EVIDENCE_SHA256 = (
    "2a003bc8ecce4243309b297be36740669"
    "c04a8e7dfa27384a5f2de33ec3cc297"
)

EXPECTED_FEATURE_SHA256 = (
    "b665765bbdb317f0568d77a5004e4107"
    "5e4a2e037fc5f6caa8102f6c932fc4f7"
)

EXPECTED_EVALUATION_COMMON_SHA256 = (
    "afa1f3387c272dee075f72f481e01257"
    "2813892976c54dc860901114f8080cdd"
)

EXPECTED_MODELS = [
    {
        "repeat": 0,
        "filename": "repeat_0_phyguard.joblib",
        "sha256":
            "6877df3e027ec8e7b1964e835c706c96"
            "f54a4a9bdbfcc5b230914773cc7332ff",
        "tau_g": 0.70,
        "tau_c": 0.35,
    },
    {
        "repeat": 1,
        "filename": "repeat_1_phyguard.joblib",
        "sha256":
            "2a009ff1f590500d930e3eec0b6209697"
            "8ba73d1b1eaf2b5fc3c320c5a70f627",
        "tau_g": 0.35,
        "tau_c": 0.45,
    },
    {
        "repeat": 2,
        "filename": "repeat_2_phyguard.joblib",
        "sha256":
            "c875168c7d92449cc08335920723df562"
            "4bd066365c39eb36fab184111055e75",
        "tau_g": 0.40,
        "tau_c": 0.45,
    },
    {
        "repeat": 3,
        "filename": "repeat_3_phyguard.joblib",
        "sha256":
            "ee74feb3e43e34f9149eaeee9797b949"
            "43275344c84b919a973f30fadc17ea99",
        "tau_g": 0.50,
        "tau_c": 0.40,
    },
    {
        "repeat": 4,
        "filename": "repeat_4_phyguard.joblib",
        "sha256":
            "eb1319f18489287ae2aef3313824613e"
            "30108a95bd6e70d096e3c0d45bd84f2d",
        "tau_g": 0.50,
        "tau_c": 0.45,
    },
]

PHYSICAL_CLASSES = [
    "adaptation_mismatch",
    "blockage",
    "interference",
    "mobility",
]

PHYSICAL_SET = set(PHYSICAL_CLASSES)

CONTROL_SET = {
    "normal",
    "nonphysical_goodput",
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


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def import_module(path, name):
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


def plain(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def clean_mapping(value):
    return {
        str(key): plain(item)
        for key, item in value.items()
    }


for required in (
    PLAN_PATH,
    CAUSAL_MANIFEST_PATH,
    PREFLIGHT_PATH,
    FEATURE_PATH,
    EVALUATION_COMMON_PATH,
):
    if not required.exists():
        raise FileNotFoundError(required)


if sklearn.__version__ != "1.8.0":
    raise RuntimeError(
        f"scikit-learn={sklearn.__version__}, "
        "expected 1.8.0."
    )


plan = load_json(PLAN_PATH)
causal = load_json(CAUSAL_MANIFEST_PATH)
preflight = load_json(PREFLIGHT_PATH)


if plan.get("plan_sha256") != EXPECTED_PLAN_SHA256:
    raise RuntimeError(
        "Formal plan SHA256 mismatch."
    )

if (
    causal.get(
        "causal_validation_policy_sha256"
    )
    != EXPECTED_POLICY_SHA256
):
    raise RuntimeError(
        "Causal-policy SHA256 mismatch."
    )

if causal.get("causal_v2_passed_count") != 825:
    raise RuntimeError(
        "Expected 825 causal-valid samples."
    )

if preflight.get("status") != "PASS":
    raise RuntimeError(
        "Locked reconstruction preflight "
        "is not PASS."
    )

if (
    preflight.get("model_count")
    != 5
):
    raise RuntimeError(
        "Preflight model count is not five."
    )

if (
    preflight
    .get("formal_archive", {})
    .get("sha256")
    != EXPECTED_FORMAL_ARCHIVE_SHA256
):
    raise RuntimeError(
        "Preflight formal archive hash mismatch."
    )

if sha256_file(FEATURE_PATH) != EXPECTED_FEATURE_SHA256:
    raise RuntimeError(
        "Feature-builder SHA256 mismatch."
    )

if (
    sha256_file(EVALUATION_COMMON_PATH)
    != EXPECTED_EVALUATION_COMMON_SHA256
):
    raise RuntimeError(
        "evaluation_common.py SHA256 mismatch."
    )


all_samples = plan.get("samples", [])

if len(all_samples) != 840:
    raise RuntimeError(
        "Formal plan does not contain 840 samples."
    )

sample_by_id = {
    sample["sample_id"]: sample
    for sample in all_samples
}


eligible_causal_records = [
    record
    for record in causal.get("records", [])
    if record.get("causal_v2_status")
    == "PASS"
]

if len(eligible_causal_records) != 825:
    raise RuntimeError(
        "Eligible causal-record count "
        "is not 825."
    )


eligible_samples = []

for record in eligible_causal_records:
    sample_id = record["sample_id"]

    if sample_id not in sample_by_id:
        raise RuntimeError(
            f"Sample missing from plan: {sample_id}"
        )

    eligible_samples.append(
        sample_by_id[sample_id]
    )


eligible_samples.sort(
    key=lambda value: value["sample_id"]
)

eligible_ids = [
    sample["sample_id"]
    for sample in eligible_samples
]

if len(set(eligible_ids)) != 825:
    raise RuntimeError(
        "Eligible sample IDs are not unique."
    )


evidence_hash_before, evidence_count = (
    aggregate_original_evidence(
        all_samples
    )
)

if evidence_hash_before != EXPECTED_EVIDENCE_SHA256:
    raise RuntimeError(
        "Original evidence SHA256 mismatch."
    )

if evidence_count != 2520:
    raise RuntimeError(
        "Original evidence count is not 2520."
    )


feature_module = import_module(
    FEATURE_PATH,
    "phyguard_formal825_features",
)

evaluation_module = import_module(
    EVALUATION_COMMON_PATH,
    "phyguard_formal825_evaluation_common",
)

build_one = feature_module.build_one
selective_metrics = (
    evaluation_module.selective_metrics
)
ABSTAIN = str(evaluation_module.ABSTAIN)


raw_rows = []
physical_rows = []
metadata_rows = []


for index, sample in enumerate(
    eligible_samples
):
    sample_id = sample["sample_id"]

    sequence_path = (
        DATA_ROOT
        / sample_id
        / "sequence.npz"
    )

    with np.load(
        sequence_path,
        allow_pickle=False,
    ) as archive:
        sequence = np.asarray(
            archive["kpi_sequence"],
            dtype=np.float32,
        )

    if sequence.shape != (80, 10):
        raise RuntimeError(
            f"{sample_id}: sequence shape "
            f"{sequence.shape}."
        )

    if not np.isfinite(sequence).all():
        raise RuntimeError(
            f"{sample_id}: non-finite sequence."
        )

    start = int(sample["event_start"])
    end = int(sample["event_end"])

    raw_1, physical_1 = build_one(
        sequence,
        start,
        end,
        eps=1e-6,
    )

    raw_2, physical_2 = build_one(
        sequence,
        start,
        end,
        eps=1e-6,
    )

    raw_1 = np.asarray(
        raw_1,
        dtype=np.float32,
    )

    physical_1 = np.asarray(
        physical_1,
        dtype=np.float32,
    )

    raw_2 = np.asarray(
        raw_2,
        dtype=np.float32,
    )

    physical_2 = np.asarray(
        physical_2,
        dtype=np.float32,
    )

    if raw_1.shape != (60,):
        raise RuntimeError(
            f"{sample_id}: raw feature "
            f"shape={raw_1.shape}."
        )

    if physical_1.shape != (19,):
        raise RuntimeError(
            f"{sample_id}: physical feature "
            f"shape={physical_1.shape}."
        )

    if not np.array_equal(
        raw_1,
        raw_2,
    ):
        raise RuntimeError(
            f"{sample_id}: raw features "
            "are not deterministic."
        )

    if not np.array_equal(
        physical_1,
        physical_2,
    ):
        raise RuntimeError(
            f"{sample_id}: physical features "
            "are not deterministic."
        )

    if not (
        np.isfinite(raw_1).all()
        and np.isfinite(physical_1).all()
    ):
        raise RuntimeError(
            f"{sample_id}: non-finite features."
        )

    raw_rows.append(raw_1)
    physical_rows.append(physical_1)

    metadata_rows.append(
        {
            "formal_index": index,
            "sample_id": sample_id,
            "label": sample["label"],
            "severity": sample["severity"],
            "channel_model":
                sample["channel_model"],
            "seed": int(sample["seed"]),
            "event_start": start,
            "event_end": end,
            "causal_v2_status": "PASS",
        }
    )


raw = np.stack(
    raw_rows,
).astype(np.float32)

physical = np.stack(
    physical_rows,
).astype(np.float32)

combined = np.concatenate(
    [raw, physical],
    axis=1,
).astype(np.float32)


if raw.shape != (825, 60):
    raise RuntimeError(
        f"Raw matrix shape={raw.shape}."
    )

if physical.shape != (825, 19):
    raise RuntimeError(
        f"Physical matrix shape={physical.shape}."
    )

if combined.shape != (825, 79):
    raise RuntimeError(
        f"Combined matrix shape={combined.shape}."
    )


metadata = pd.DataFrame(
    metadata_rows
)

truth = metadata[
    "label"
].to_numpy(dtype=object)

physical_truth_mask = np.isin(
    truth,
    PHYSICAL_CLASSES,
)

control_truth_mask = np.isin(
    truth,
    list(CONTROL_SET),
)


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

features_path = (
    OUTPUT_ROOT / "features.npz"
)

metadata_path = (
    OUTPUT_ROOT / "metadata.csv"
)

predictions_path = (
    OUTPUT_ROOT / "predictions.csv"
)

per_repeat_path = (
    OUTPUT_ROOT / "per_repeat_metrics.csv"
)

per_label_path = (
    OUTPUT_ROOT
    / "per_repeat_label_metrics.csv"
)

model_contract_path = (
    OUTPUT_ROOT / "model_contract.csv"
)

summary_path = (
    OUTPUT_ROOT / "summary.json"
)

audit_path = (
    OUTPUT_ROOT / "audit.json"
)


np.savez_compressed(
    features_path,
    raw=raw,
    physical=physical,
    combined=combined,
)

metadata.to_csv(
    metadata_path,
    index=False,
)


prediction_rows = []
repeat_metric_rows = []
label_metric_rows = []
model_contract_rows = []

model_hashes_before = {}


for expected in EXPECTED_MODELS:
    repeat = expected["repeat"]

    model_path = (
        MODEL_DIR
        / expected["filename"]
    )

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model_hash_before = sha256_file(
        model_path
    )

    model_hashes_before[
        repeat
    ] = model_hash_before

    if (
        model_hash_before
        != expected["sha256"]
    ):
        raise RuntimeError(
            f"Repeat {repeat} model hash mismatch."
        )

    bundle = joblib.load(model_path)

    if set(bundle) != {
        "gate",
        "resolver",
    }:
        raise RuntimeError(
            f"Repeat {repeat}: bundle keys mismatch."
        )

    gate = bundle["gate"]
    resolver = bundle["resolver"]

    if int(gate.n_features_in_) != 19:
        raise RuntimeError(
            f"Repeat {repeat}: Gate dimension mismatch."
        )

    if int(resolver.n_features_in_) != 79:
        raise RuntimeError(
            f"Repeat {repeat}: Resolver dimension mismatch."
        )

    resolver_classes = [
        str(value)
        for value in resolver.classes_
    ]

    if resolver_classes != PHYSICAL_CLASSES:
        raise RuntimeError(
            f"Repeat {repeat}: Resolver class order mismatch."
        )

    gate_probability_1 = (
        gate.predict_proba(
            physical
        )[:, 1]
    )

    resolver_probability_1 = (
        resolver.predict_proba(
            combined
        )
    )

    gate_probability_2 = (
        gate.predict_proba(
            physical
        )[:, 1]
    )

    resolver_probability_2 = (
        resolver.predict_proba(
            combined
        )
    )

    if not np.array_equal(
        gate_probability_1,
        gate_probability_2,
    ):
        raise RuntimeError(
            f"Repeat {repeat}: Gate inference "
            "is not deterministic."
        )

    if not np.array_equal(
        resolver_probability_1,
        resolver_probability_2,
    ):
        raise RuntimeError(
            f"Repeat {repeat}: Resolver inference "
            "is not deterministic."
        )

    mechanism_index = (
        resolver_probability_1.argmax(
            axis=1
        )
    )

    mechanism_label = np.asarray(
        resolver.classes_,
        dtype=object,
    )[mechanism_index]

    mechanism_confidence = (
        resolver_probability_1.max(
            axis=1
        )
    )

    selected_mask = (
        (
            gate_probability_1
            >= expected["tau_g"]
        )
        & (
            mechanism_confidence
            >= expected["tau_c"]
        )
    )

    prediction = np.where(
        selected_mask,
        mechanism_label,
        ABSTAIN,
    ).astype(object)

    repeat_metrics = clean_mapping(
        selective_metrics(
            truth,
            prediction,
        )
    )

    physical_exact = float(
        np.mean(
            prediction[
                physical_truth_mask
            ]
            == truth[
                physical_truth_mask
            ]
        )
    )

    physical_selection = float(
        np.mean(
            selected_mask[
                physical_truth_mask
            ]
        )
    )

    control_abstention = float(
        np.mean(
            prediction[
                control_truth_mask
            ]
            == ABSTAIN
        )
    )

    normal_mask = (
        truth == "normal"
    )

    nonphysical_mask = (
        truth
        == "nonphysical_goodput"
    )

    repeat_row = {
        "repeat": repeat,
        "n": len(truth),
        "selected_count":
            int(selected_mask.sum()),
        "abstained_count":
            int(
                len(truth)
                - selected_mask.sum()
            ),
        "physical_exact_diagnosis_rate":
            physical_exact,
        "physical_selection_rate":
            physical_selection,
        "control_abstention_rate":
            control_abstention,
        "normal_abstention_rate":
            float(
                np.mean(
                    prediction[
                        normal_mask
                    ]
                    == ABSTAIN
                )
            ),
        "nonphysical_abstention_rate":
            float(
                np.mean(
                    prediction[
                        nonphysical_mask
                    ]
                    == ABSTAIN
                )
            ),
        "tau_g": expected["tau_g"],
        "tau_c": expected["tau_c"],
        **repeat_metrics,
    }

    repeat_metric_rows.append(
        repeat_row
    )


    for label in sorted(
        set(truth.tolist())
    ):
        mask = truth == label

        label_selected = (
            selected_mask[mask]
        )

        label_prediction = (
            prediction[mask]
        )

        label_truth = truth[mask]

        selected_count = int(
            label_selected.sum()
        )

        selected_accuracy = None

        if selected_count > 0:
            selected_accuracy = float(
                np.mean(
                    label_prediction[
                        label_selected
                    ]
                    == label_truth[
                        label_selected
                    ]
                )
            )

        if label in PHYSICAL_SET:
            task_success = (
                label_prediction
                == label_truth
            )
        else:
            task_success = (
                label_prediction
                == ABSTAIN
            )

        label_metric_rows.append(
            {
                "repeat": repeat,
                "label": label,
                "prediction_count":
                    int(mask.sum()),
                "selected_count":
                    selected_count,
                "abstained_count":
                    int(
                        mask.sum()
                        - selected_count
                    ),
                "selection_rate":
                    float(
                        label_selected.mean()
                    ),
                "abstention_rate":
                    float(
                        1.0
                        - label_selected.mean()
                    ),
                "selected_accuracy":
                    selected_accuracy,
                "exact_label_rate":
                    float(
                        np.mean(
                            label_prediction
                            == label_truth
                        )
                    ),
                "task_success_rate":
                    float(
                        np.mean(task_success)
                    ),
                "mean_gate_probability":
                    float(
                        np.mean(
                            gate_probability_1[
                                mask
                            ]
                        )
                    ),
                "mean_mechanism_confidence":
                    float(
                        np.mean(
                            mechanism_confidence[
                                mask
                            ]
                        )
                    ),
            }
        )


    for index in range(
        len(metadata)
    ):
        row = metadata.iloc[index]

        probability_map = {
            resolver_classes[
                class_index
            ]:
                float(
                    resolver_probability_1[
                        index,
                        class_index,
                    ]
                )
            for class_index in range(
                len(resolver_classes)
            )
        }

        prediction_rows.append(
            {
                "repeat": repeat,
                "formal_index":
                    int(row.formal_index),
                "sample_id":
                    row.sample_id,
                "truth":
                    row.label,
                "prediction":
                    str(prediction[index]),
                "selected":
                    bool(selected_mask[index]),
                "correct_exact":
                    bool(
                        prediction[index]
                        == truth[index]
                    ),
                "task_success":
                    bool(
                        (
                            prediction[index]
                            == truth[index]
                        )
                        if truth[index]
                        in PHYSICAL_SET
                        else (
                            prediction[index]
                            == ABSTAIN
                        )
                    ),
                "severity":
                    row.severity,
                "channel_model":
                    row.channel_model,
                "seed":
                    int(row.seed),
                "event_start":
                    int(row.event_start),
                "event_end":
                    int(row.event_end),
                "tau_g":
                    expected["tau_g"],
                "tau_c":
                    expected["tau_c"],
                "gate_probability":
                    float(
                        gate_probability_1[
                            index
                        ]
                    ),
                "mechanism_confidence":
                    float(
                        mechanism_confidence[
                            index
                        ]
                    ),
                "joint_confidence":
                    float(
                        gate_probability_1[
                            index
                        ]
                        * mechanism_confidence[
                            index
                        ]
                    ),
                "resolver_adaptation_mismatch":
                    probability_map[
                        "adaptation_mismatch"
                    ],
                "resolver_blockage":
                    probability_map[
                        "blockage"
                    ],
                "resolver_interference":
                    probability_map[
                        "interference"
                    ],
                "resolver_mobility":
                    probability_map[
                        "mobility"
                    ],
            }
        )


    model_hash_after = sha256_file(
        model_path
    )

    if model_hash_after != model_hash_before:
        raise RuntimeError(
            f"Repeat {repeat}: model file changed."
        )

    model_contract_rows.append(
        {
            "repeat": repeat,
            "model_path":
                str(model_path),
            "model_sha256":
                model_hash_before,
            "gate_feature_dimension":
                int(gate.n_features_in_),
            "resolver_feature_dimension":
                int(
                    resolver.n_features_in_
                ),
            "resolver_classes":
                "|".join(
                    resolver_classes
                ),
            "tau_g":
                expected["tau_g"],
            "tau_c":
                expected["tau_c"],
            "prediction_count":
                len(truth),
            "deterministic_replay":
                True,
            "model_file_unchanged":
                True,
        }
    )


predictions = pd.DataFrame(
    prediction_rows
)

per_repeat = pd.DataFrame(
    repeat_metric_rows
)

per_label = pd.DataFrame(
    label_metric_rows
)

model_contract = pd.DataFrame(
    model_contract_rows
)


if len(predictions) != 4125:
    raise RuntimeError(
        f"Prediction count={len(predictions)}, "
        "expected 4125."
    )

if len(per_repeat) != 5:
    raise RuntimeError(
        "Per-repeat metric count is not five."
    )

if len(per_label) != 30:
    raise RuntimeError(
        "Per-repeat label metric count "
        "is not 30."
    )


predictions.to_csv(
    predictions_path,
    index=False,
)

per_repeat.to_csv(
    per_repeat_path,
    index=False,
)

per_label.to_csv(
    per_label_path,
    index=False,
)

model_contract.to_csv(
    model_contract_path,
    index=False,
)


pooled_truth = predictions[
    "truth"
].to_numpy(dtype=object)

pooled_prediction = predictions[
    "prediction"
].to_numpy(dtype=object)

pooled_metrics = clean_mapping(
    selective_metrics(
        pooled_truth,
        pooled_prediction,
    )
)


core_metrics = [
    "selected_accuracy",
    "coverage",
    "false_specific_rate",
    "macro_f1",
    "balanced_accuracy",
    "physical_exact_diagnosis_rate",
    "physical_selection_rate",
    "control_abstention_rate",
    "normal_abstention_rate",
    "nonphysical_abstention_rate",
]


metric_summary = {}

for metric in core_metrics:
    if metric not in per_repeat.columns:
        continue

    values = pd.to_numeric(
        per_repeat[metric],
        errors="coerce",
    ).to_numpy(dtype=float)

    finite = values[
        np.isfinite(values)
    ]

    metric_summary[metric] = {
        "count": int(len(finite)),
        "mean": (
            float(np.mean(finite))
            if len(finite)
            else None
        ),
        "std": (
            float(
                np.std(
                    finite,
                    ddof=1,
                )
            )
            if len(finite) > 1
            else 0.0
        ),
        "minimum": (
            float(np.min(finite))
            if len(finite)
            else None
        ),
        "maximum": (
            float(np.max(finite))
            if len(finite)
            else None
        ),
    }


pooled_label_summary = []

for label in sorted(
    predictions["truth"].unique()
):
    subset = predictions[
        predictions["truth"] == label
    ]

    selected = subset[
        "selected"
    ].astype(bool)

    pooled_label_summary.append(
        {
            "label": label,
            "prediction_count":
                int(len(subset)),
            "selected_count":
                int(selected.sum()),
            "selection_rate":
                float(selected.mean()),
            "abstention_rate":
                float(1.0 - selected.mean()),
            "exact_label_rate":
                float(
                    subset[
                        "correct_exact"
                    ].mean()
                ),
            "task_success_rate":
                float(
                    subset[
                        "task_success"
                    ].mean()
                ),
            "mean_gate_probability":
                float(
                    subset[
                        "gate_probability"
                    ].mean()
                ),
            "mean_mechanism_confidence":
                float(
                    subset[
                        "mechanism_confidence"
                    ].mean()
                ),
        }
    )


summary = {
    "schema":
        "phyguard.sionna.formal825."
        "locked_reconstruction.summary.v1",

    "status": "PASS",

    "provenance_class":
        "revision-time locked reconstruction",

    "formal_scope": {
        "preregistered_count": 840,
        "artifact_qc_eligible_count": 839,
        "causal_valid_count": 825,
        "model_repeat_count": 5,
        "prediction_count": 4125,
    },

    "feature_shapes": {
        "raw": list(raw.shape),
        "physical": list(
            physical.shape
        ),
        "combined": list(
            combined.shape
        ),
    },

    "per_repeat_metric_summary":
        metric_summary,

    "pooled_metrics":
        pooled_metrics,

    "pooled_label_summary":
        pooled_label_summary,

    "label_counts":
        dict(
            Counter(
                metadata["label"]
            )
        ),

    "severity_counts":
        dict(
            Counter(
                metadata["severity"]
            )
        ),

    "channel_counts":
        dict(
            Counter(
                metadata[
                    "channel_model"
                ]
            )
        ),

    "methodological_boundary": {
        "model_training":
            False,
        "model_weight_update":
            False,
        "feature_selection":
            False,
        "threshold_tuning":
            False,
        "threshold_calibration":
            False,
        "failed_sample_replacement":
            False,
        "formal_labels_used_for_prediction":
            False,
        "labels_used_for_evaluation_only":
            True,
        "oracle_event_intervals_from_locked_plan":
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


evidence_hash_after, evidence_count_after = (
    aggregate_original_evidence(
        all_samples
    )
)

if evidence_hash_after != evidence_hash_before:
    raise RuntimeError(
        "Original evidence changed "
        "during locked inference."
    )

if evidence_count_after != evidence_count:
    raise RuntimeError(
        "Original evidence count changed."
    )


for expected in EXPECTED_MODELS:
    repeat = expected["repeat"]

    model_path = (
        MODEL_DIR
        / expected["filename"]
    )

    if (
        sha256_file(model_path)
        != model_hashes_before[repeat]
    ):
        raise RuntimeError(
            f"Repeat {repeat}: model changed "
            "after inference."
        )


audit = {
    "schema":
        "phyguard.sionna.formal825."
        "locked_reconstruction.audit.v1",

    "status": "PASS",

    "sklearn_version":
        sklearn.__version__,

    "formal_plan_sha256":
        EXPECTED_PLAN_SHA256,

    "causal_policy_sha256":
        EXPECTED_POLICY_SHA256,

    "formal_archive_sha256":
        EXPECTED_FORMAL_ARCHIVE_SHA256,

    "feature_builder": {
        "path": str(FEATURE_PATH),
        "sha256":
            EXPECTED_FEATURE_SHA256,
        "eps": 1e-6,
    },

    "evaluation_common": {
        "path":
            str(
                EVALUATION_COMMON_PATH
            ),
        "sha256":
            EXPECTED_EVALUATION_COMMON_SHA256,
    },

    "model_count": 5,

    "model_contract":
        model_contract_rows,

    "deterministic_feature_replay":
        "825/825",

    "deterministic_prediction_replay":
        "4125/4125",

    "original_evidence": {
        "file_count":
            evidence_count,
        "aggregate_sha256":
            evidence_hash_before,
        "unchanged":
            True,
    },

    "safety": {
        "training_performed":
            False,
        "threshold_tuning_performed":
            False,
        "calibration_performed":
            False,
        "feature_selection_performed":
            False,
        "models_modified":
            False,
        "original_sequences_modified":
            False,
        "original_metadata_modified":
            False,
        "original_reports_modified":
            False,
    },
}

audit_path.write_text(
    json.dumps(
        audit,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


output_files = [
    features_path,
    metadata_path,
    predictions_path,
    per_repeat_path,
    per_label_path,
    model_contract_path,
    summary_path,
    audit_path,
]

file_records = [
    {
        "relative_path":
            str(path.relative_to(ROOT)),
        "sha256":
            sha256_file(path),
        "size_bytes":
            path.stat().st_size,
    }
    for path in output_files
]


manifest = {
    "schema":
        "phyguard.sionna.formal825."
        "locked_reconstruction.inference.v1",

    "status": "PASS",

    "provenance_class":
        "revision-time locked reconstruction",

    "eligible_sample_count": 825,

    "model_count": 5,

    "prediction_count": 4125,

    "formal_plan_sha256":
        EXPECTED_PLAN_SHA256,

    "causal_policy_sha256":
        EXPECTED_POLICY_SHA256,

    "formal_archive_sha256":
        EXPECTED_FORMAL_ARCHIVE_SHA256,

    "feature_builder_sha256":
        EXPECTED_FEATURE_SHA256,

    "evaluation_common_sha256":
        EXPECTED_EVALUATION_COMMON_SHA256,

    "model_hashes": {
        str(record["repeat"]):
            record["model_sha256"]
        for record in model_contract_rows
    },

    "thresholds": [
        {
            "repeat":
                record["repeat"],
            "tau_g":
                record["tau_g"],
            "tau_c":
                record["tau_c"],
        }
        for record in model_contract_rows
    ],

    "feature_shapes": {
        "raw": [825, 60],
        "physical": [825, 19],
        "combined": [825, 79],
    },

    "determinism": {
        "feature_replay":
            "825/825",
        "prediction_replay":
            "4125/4125",
    },

    "original_evidence": {
        "file_count": 2520,
        "aggregate_sha256":
            EXPECTED_EVIDENCE_SHA256,
        "unchanged":
            True,
    },

    "methodological_boundary": {
        "training_performed":
            False,
        "threshold_tuning_performed":
            False,
        "threshold_calibration_performed":
            False,
        "feature_selection_performed":
            False,
        "model_selection_performed":
            False,
        "formal_results_used_to_modify_pipeline":
            False,
    },

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
print("sklearn_version:", sklearn.__version__)
print("eligible_sample_count: 825")
print("model_count: 5")
print("prediction_count: 4125")
print(
    "feature_shapes:",
    {
        "raw": list(raw.shape),
        "physical":
            list(physical.shape),
        "combined":
            list(combined.shape),
    },
)
print(
    "deterministic_feature_replay: 825/825"
)
print(
    "deterministic_prediction_replay: 4125/4125"
)
print(
    "original_evidence_unchanged:",
    True,
)

print("\nPER-REPEAT METRICS")
print(
    per_repeat.to_string(
        index=False
    )
)

print("\nMEAN ± STD")

for metric, values in (
    metric_summary.items()
):
    print(
        metric,
        f"{values['mean']:.8f}",
        "+/-",
        f"{values['std']:.8f}",
    )

print("\nPOOLED LABEL SUMMARY")

for row in pooled_label_summary:
    print(row)

print("\nsummary:", summary_path)
print("audit:", audit_path)
print("manifest:", OUTPUT_MANIFEST)
print("predictions:", predictions_path)

print(
    "\nSIONNA_FORMAL825_LOCKED_"
    "RECONSTRUCTION_INFERENCE_PASS"
)
