import hashlib
import importlib.util
import inspect
import json
import zipfile
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn


ROOT = Path("/root/phyguard_revision")

EXTERNAL_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna24_external_validation_final.zip"
)

FORMAL_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal_test840_causal_final.zip"
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

R0_ROOT = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
    / "PhyGuard_R0_Reconstruction_Checkpoint2"
    / "phyguard_rebuild"
)

FEATURE_PATH = (
    R0_ROOT
    / "scripts"
    / "features.py"
)

EVALUATION_COMMON_PATH = (
    R0_ROOT
    / "scripts"
    / "evaluation_common.py"
)

MODEL_DIR = (
    R0_ROOT
    / "results"
    / "full"
    / "evaluation_suite"
    / "models"
)

OUTPUT_MANIFEST = (
    ROOT
    / "manifests"
    / "formal_test840_locked_reconstruction_preflight.json"
)

EXPECTED_EXTERNAL_ARCHIVE_SHA256 = (
    "238980731f09b758ef834e4961ba1427"
    "665a352aac279f164f02efba5a033aad"
)

EXPECTED_FORMAL_ARCHIVE_SHA256 = (
    "b6c5f400b7d55949fb1e1f39c0c092d"
    "a8f780eed100866f30176e26e462ff2f5"
)

EXPECTED_FEATURE_SHA256 = (
    "b665765bbdb317f0568d77a5004e4107"
    "5e4a2e037fc5f6caa8102f6c932fc4f7"
)

EXPECTED_EVALUATION_COMMON_SHA256 = (
    "afa1f3387c272dee075f72f481e01257"
    "2813892976c54dc860901114f8080cdd"
)

EXPECTED_THRESHOLD_MEMBER_SHA256 = (
    "cfdf67e558196294557e33b9f736c606"
    "c53aeea2ea0b532b0f08222806423107"
)

EXPECTED_THRESHOLD_ROWS = [
    {
        "repeat": 0,
        "tau_g": 0.70,
        "tau_c": 0.35,
    },
    {
        "repeat": 1,
        "tau_g": 0.35,
        "tau_c": 0.45,
    },
    {
        "repeat": 2,
        "tau_g": 0.40,
        "tau_c": 0.45,
    },
    {
        "repeat": 3,
        "tau_g": 0.50,
        "tau_c": 0.40,
    },
    {
        "repeat": 4,
        "tau_g": 0.50,
        "tau_c": 0.45,
    },
]

EXPECTED_MODELS = [
    {
        "repeat": 0,
        "filename": "repeat_0_phyguard.joblib",
        "sha256":
            "6877df3e027ec8e7b1964e835c706c96"
            "f54a4a9bdbfcc5b230914773cc7332ff",
    },
    {
        "repeat": 1,
        "filename": "repeat_1_phyguard.joblib",
        "sha256":
            "2a009ff1f590500d930e3eec0b6209697"
            "8ba73d1b1eaf2b5fc3c320c5a70f627",
    },
    {
        "repeat": 2,
        "filename": "repeat_2_phyguard.joblib",
        "sha256":
            "c875168c7d92449cc08335920723df562"
            "4bd066365c39eb36fab184111055e75",
    },
    {
        "repeat": 3,
        "filename": "repeat_3_phyguard.joblib",
        "sha256":
            "ee74feb3e43e34f9149eaeee9797b949"
            "43275344c84b919a973f30fadc17ea99",
    },
    {
        "repeat": 4,
        "filename": "repeat_4_phyguard.joblib",
        "sha256":
            "eb1319f18489287ae2aef3313824613e"
            "30108a95bd6e70d096e3c0d45bd84f2d",
    },
]

EXPECTED_CLASSES = [
    "adaptation_mismatch",
    "blockage",
    "interference",
    "mobility",
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


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


for path in (
    EXTERNAL_ARCHIVE,
    FORMAL_ARCHIVE,
    FORMAL_PLAN,
    CAUSAL_MANIFEST,
    FEATURE_PATH,
    EVALUATION_COMMON_PATH,
):
    if not path.exists():
        raise FileNotFoundError(path)


errors = []

external_archive_hash = sha256_file(
    EXTERNAL_ARCHIVE
)

formal_archive_hash = sha256_file(
    FORMAL_ARCHIVE
)

feature_hash_before = sha256_file(
    FEATURE_PATH
)

evaluation_common_hash_before = (
    sha256_file(
        EVALUATION_COMMON_PATH
    )
)


if (
    external_archive_hash
    != EXPECTED_EXTERNAL_ARCHIVE_SHA256
):
    errors.append(
        "External-validation archive SHA256 mismatch."
    )

if (
    formal_archive_hash
    != EXPECTED_FORMAL_ARCHIVE_SHA256
):
    errors.append(
        "Formal-test archive SHA256 mismatch."
    )

if feature_hash_before != EXPECTED_FEATURE_SHA256:
    errors.append(
        "Feature-builder SHA256 mismatch."
    )

if (
    evaluation_common_hash_before
    != EXPECTED_EVALUATION_COMMON_SHA256
):
    errors.append(
        "evaluation_common.py SHA256 mismatch."
    )

if sklearn.__version__ != "1.8.0":
    errors.append(
        "scikit-learn version is "
        f"{sklearn.__version__}, expected 1.8.0."
    )


threshold_member_name = (
    "manifests/"
    "source_fixed_thresholds_locked.json"
)

audit_member_name = (
    "results/"
    "sionna24_locked_phyguard/"
    "audit.json"
)


with zipfile.ZipFile(
    EXTERNAL_ARCHIVE,
    mode="r",
) as archive:
    bad_file = archive.testzip()

    if bad_file is not None:
        errors.append(
            f"External archive ZIP failure: {bad_file}"
        )

    names = set(archive.namelist())

    for member in (
        threshold_member_name,
        audit_member_name,
    ):
        if member not in names:
            errors.append(
                f"Missing archive member: {member}"
            )

    threshold_bytes = archive.read(
        threshold_member_name
    )

    audit_bytes = archive.read(
        audit_member_name
    )


threshold_member_hash = sha256_bytes(
    threshold_bytes
)

if (
    threshold_member_hash
    != EXPECTED_THRESHOLD_MEMBER_SHA256
):
    errors.append(
        "Frozen threshold-member SHA256 mismatch."
    )


threshold_document = json.loads(
    threshold_bytes.decode("utf-8")
)

threshold_rows = []

for item in sorted(
    threshold_document.get(
        "thresholds",
        [],
    ),
    key=lambda value: int(
        value["repeat"]
    ),
):
    threshold_rows.append(
        {
            "repeat":
                int(item["repeat"]),
            "tau_g":
                float(item["tau_g"]),
            "tau_c":
                float(item["tau_c"]),
        }
    )

if threshold_rows != EXPECTED_THRESHOLD_ROWS:
    errors.append(
        "Frozen threshold rows do not match "
        "the five locked operating points."
    )


external_audit = json.loads(
    audit_bytes.decode("utf-8")
)

archive_model_contract = (
    external_audit.get(
        "model_contract",
        [],
    )
)

if len(archive_model_contract) != 5:
    errors.append(
        "External archive model-contract "
        "count is not five."
    )


plan = load_json(FORMAL_PLAN)

causal = load_json(
    CAUSAL_MANIFEST
)

plan_by_id = {
    sample["sample_id"]: sample
    for sample in plan.get(
        "samples",
        [],
    )
}

eligible_records = [
    record
    for record in causal.get(
        "records",
        [],
    )
    if record.get(
        "causal_v2_status"
    ) == "PASS"
]

if len(eligible_records) != 825:
    errors.append(
        "Formal causal-valid record count "
        f"is {len(eligible_records)}, expected 825."
    )


invalid_intervals = []
missing_plan_ids = []

for record in eligible_records:
    sample_id = record["sample_id"]

    sample = plan_by_id.get(
        sample_id
    )

    if sample is None:
        missing_plan_ids.append(
            sample_id
        )
        continue

    start = sample.get(
        "event_start"
    )

    end = sample.get(
        "event_end"
    )

    if not (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= 80
    ):
        invalid_intervals.append(
            {
                "sample_id": sample_id,
                "event_start": start,
                "event_end": end,
            }
        )


if missing_plan_ids:
    errors.append(
        "Causal-valid sample IDs missing "
        "from formal plan."
    )

if invalid_intervals:
    errors.append(
        "Invalid candidate intervals found "
        "among causal-valid samples."
    )


label_counts = Counter(
    record["label"]
    for record in eligible_records
)

severity_counts = Counter(
    record["severity"]
    for record in eligible_records
)

channel_counts = Counter(
    record["channel_model"]
    for record in eligible_records
)


spec = importlib.util.spec_from_file_location(
    "phyguard_locked_features_preflight",
    FEATURE_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        "Unable to construct feature-module spec."
    )

feature_module = (
    importlib.util.module_from_spec(
        spec
    )
)

spec.loader.exec_module(
    feature_module
)

build_one = getattr(
    feature_module,
    "build_one",
)

signature = inspect.signature(
    build_one
)

eps_default = (
    signature
    .parameters["eps"]
    .default
)

if float(eps_default) != 1e-6:
    errors.append(
        "build_one eps default is not 1e-6."
    )


feature_probe = np.zeros(
    (80, 10),
    dtype=np.float32,
)

raw_probe, physical_probe = (
    build_one(
        feature_probe,
        24,
        48,
    )
)

combined_probe = np.concatenate(
    [
        raw_probe,
        physical_probe,
    ]
)

if raw_probe.shape != (60,):
    errors.append(
        f"Raw feature shape={raw_probe.shape}, "
        "expected (60,)."
    )

if physical_probe.shape != (19,):
    errors.append(
        "Physical feature shape="
        f"{physical_probe.shape}, expected (19,)."
    )

if combined_probe.shape != (79,):
    errors.append(
        "Combined feature shape="
        f"{combined_probe.shape}, expected (79,)."
    )

if not (
    np.isfinite(raw_probe).all()
    and np.isfinite(
        physical_probe
    ).all()
):
    errors.append(
        "Feature-builder probe produced "
        "non-finite values."
    )


model_records = []

for expected in EXPECTED_MODELS:
    repeat = expected["repeat"]

    model_path = (
        MODEL_DIR
        / expected["filename"]
    )

    if not model_path.exists():
        errors.append(
            f"Missing model file: {model_path}"
        )

        continue

    model_hash_before = sha256_file(
        model_path
    )

    if (
        model_hash_before
        != expected["sha256"]
    ):
        errors.append(
            f"Repeat {repeat} model SHA256 mismatch."
        )

    bundle = joblib.load(
        model_path
    )

    if set(bundle) != {
        "gate",
        "resolver",
    }:
        errors.append(
            f"Repeat {repeat} bundle keys mismatch."
        )

        continue

    gate = bundle["gate"]
    resolver = bundle["resolver"]

    gate_dimension = int(
        gate.n_features_in_
    )

    resolver_dimension = int(
        resolver.n_features_in_
    )

    resolver_classes = [
        str(value)
        for value in resolver.classes_
    ]

    if gate_dimension != 19:
        errors.append(
            f"Repeat {repeat} Gate dimension "
            f"is {gate_dimension}, expected 19."
        )

    if resolver_dimension != 79:
        errors.append(
            f"Repeat {repeat} Resolver dimension "
            f"is {resolver_dimension}, expected 79."
        )

    if resolver_classes != EXPECTED_CLASSES:
        errors.append(
            f"Repeat {repeat} Resolver classes mismatch."
        )

    gate_input = np.zeros(
        (1, 19),
        dtype=np.float32,
    )

    resolver_input = np.zeros(
        (1, 79),
        dtype=np.float32,
    )

    gate_probability_1 = (
        gate.predict_proba(
            gate_input
        )
    )

    gate_probability_2 = (
        gate.predict_proba(
            gate_input
        )
    )

    resolver_probability_1 = (
        resolver.predict_proba(
            resolver_input
        )
    )

    resolver_probability_2 = (
        resolver.predict_proba(
            resolver_input
        )
    )

    if gate_probability_1.shape != (
        1,
        2,
    ):
        errors.append(
            f"Repeat {repeat} Gate output shape mismatch."
        )

    if resolver_probability_1.shape != (
        1,
        4,
    ):
        errors.append(
            f"Repeat {repeat} Resolver output shape mismatch."
        )

    if not np.array_equal(
        gate_probability_1,
        gate_probability_2,
    ):
        errors.append(
            f"Repeat {repeat} Gate probe "
            "is not deterministic."
        )

    if not np.array_equal(
        resolver_probability_1,
        resolver_probability_2,
    ):
        errors.append(
            f"Repeat {repeat} Resolver probe "
            "is not deterministic."
        )

    model_hash_after = sha256_file(
        model_path
    )

    if model_hash_after != model_hash_before:
        errors.append(
            f"Repeat {repeat} model file "
            "changed during read-only loading."
        )

    model_records.append(
        {
            "repeat": repeat,
            "path": str(model_path),
            "sha256": model_hash_before,
            "size_bytes":
                model_path.stat().st_size,
            "bundle_keys":
                sorted(bundle),
            "gate_feature_dimension":
                gate_dimension,
            "resolver_feature_dimension":
                resolver_dimension,
            "resolver_classes":
                resolver_classes,
            "tau_g":
                EXPECTED_THRESHOLD_ROWS[
                    repeat
                ]["tau_g"],
            "tau_c":
                EXPECTED_THRESHOLD_ROWS[
                    repeat
                ]["tau_c"],
            "gate_probe_probability":
                gate_probability_1[
                    0
                ].tolist(),
            "resolver_probe_probability":
                resolver_probability_1[
                    0
                ].tolist(),
            "deterministic_probe":
                True,
            "file_unchanged":
                model_hash_after
                == model_hash_before,
        }
    )


feature_hash_after = sha256_file(
    FEATURE_PATH
)

evaluation_common_hash_after = (
    sha256_file(
        EVALUATION_COMMON_PATH
    )
)

if feature_hash_after != feature_hash_before:
    errors.append(
        "Feature-builder file changed "
        "during preflight."
    )

if (
    evaluation_common_hash_after
    != evaluation_common_hash_before
):
    errors.append(
        "evaluation_common.py changed "
        "during preflight."
    )


output = {
    "schema":
        "phyguard.formal_test840."
        "locked_reconstruction_preflight.v1",

    "status":
        "PASS"
        if not errors
        else "FAIL",

    "sklearn_version":
        sklearn.__version__,

    "external_archive": {
        "path":
            str(EXTERNAL_ARCHIVE),
        "sha256":
            external_archive_hash,
        "threshold_member":
            threshold_member_name,
        "threshold_member_sha256":
            threshold_member_hash,
    },

    "formal_archive": {
        "path":
            str(FORMAL_ARCHIVE),
        "sha256":
            formal_archive_hash,
    },

    "feature_builder": {
        "path":
            str(FEATURE_PATH),
        "sha256":
            feature_hash_before,
        "eps_default":
            float(eps_default),
        "raw_dimension":
            int(raw_probe.size),
        "physical_dimension":
            int(
                physical_probe.size
            ),
        "combined_dimension":
            int(combined_probe.size),
        "file_unchanged":
            feature_hash_after
            == feature_hash_before,
    },

    "evaluation_common": {
        "path":
            str(
                EVALUATION_COMMON_PATH
            ),
        "sha256":
            evaluation_common_hash_before,
        "file_unchanged":
            evaluation_common_hash_after
            == evaluation_common_hash_before,
    },

    "locked_thresholds":
        threshold_rows,

    "model_count":
        len(model_records),

    "models":
        model_records,

    "formal_scope": {
        "eligible_count":
            len(eligible_records),
        "all_candidate_intervals_valid":
            not invalid_intervals,
        "missing_plan_ids":
            missing_plan_ids,
        "invalid_intervals":
            invalid_intervals,
        "label_counts":
            dict(label_counts),
        "severity_counts":
            dict(severity_counts),
        "channel_counts":
            dict(channel_counts),
    },

    "safety": {
        "formal_sequences_opened":
            False,
        "formal_predictions_generated":
            False,
        "models_updated":
            False,
        "training_performed":
            False,
        "feature_selection_performed":
            False,
        "threshold_tuning_performed":
            False,
        "calibration_performed":
            False,
    },

    "errors":
        errors,
}

OUTPUT_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_MANIFEST.write_text(
    json.dumps(
        output,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print("status:", output["status"])
print(
    "sklearn_version:",
    sklearn.__version__,
)
print(
    "external_archive_sha256:",
    external_archive_hash,
)
print(
    "formal_archive_sha256:",
    formal_archive_hash,
)
print(
    "threshold_member_sha256:",
    threshold_member_hash,
)
print(
    "threshold_rows:",
    threshold_rows,
)
print(
    "feature_builder_sha256:",
    feature_hash_before,
)
print(
    "evaluation_common_sha256:",
    evaluation_common_hash_before,
)
print(
    "feature_dimensions:",
    {
        "raw": int(raw_probe.size),
        "physical":
            int(physical_probe.size),
        "combined":
            int(combined_probe.size),
    },
)
print(
    "model_count:",
    len(model_records),
)
print(
    "formal_eligible_count:",
    len(eligible_records),
)
print(
    "all_candidate_intervals_valid:",
    not invalid_intervals,
)
print(
    "label_counts:",
    dict(label_counts),
)
print(
    "severity_counts:",
    dict(severity_counts),
)
print(
    "channel_counts:",
    dict(channel_counts),
)


print("\nMODEL RECORDS")

for record in model_records:
    print(
        "repeat=",
        record["repeat"],
        "sha256=",
        record["sha256"],
        "gate_dim=",
        record[
            "gate_feature_dimension"
        ],
        "resolver_dim=",
        record[
            "resolver_feature_dimension"
        ],
        "classes=",
        record["resolver_classes"],
        "tau_g=",
        record["tau_g"],
        "tau_c=",
        record["tau_c"],
        "deterministic=",
        record[
            "deterministic_probe"
        ],
    )


if errors:
    print("\nERRORS")

    for error in errors:
        print("-", error)

    raise RuntimeError(
        "Locked reconstruction preflight failed."
    )


print("\nmanifest:", OUTPUT_MANIFEST)

print(
    "\nFORMAL_TEST840_LOCKED_"
    "RECONSTRUCTION_PREFLIGHT_PASS"
)
