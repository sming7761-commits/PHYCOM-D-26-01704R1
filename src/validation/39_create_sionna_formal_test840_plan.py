import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

SOURCE_PLAN_PATH = (
    ROOT
    / "configs"
    / "sionna_prevalidation210_plan_v1.json"
)

SMOKE_PLAN_PATH = (
    ROOT
    / "configs"
    / "smoke24_plan_v1.json"
)

POLICY_PATH = (
    ROOT
    / "configs"
    / "sionna_causal_validation_policy_v2.json"
)

PREVALIDATION_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_prevalidation210_causal_v2_final.zip"
)

OUTPUT_PLAN_JSON = (
    ROOT
    / "configs"
    / "sionna_formal_test840_plan_v1.json"
)

OUTPUT_PLAN_CSV = (
    ROOT
    / "configs"
    / "sionna_formal_test840_plan_v1.csv"
)

OUTPUT_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_plan_lock.json"
)

EXPECTED_SOURCE_PLAN_SHA256 = (
    "3e67a71b555c3d32dbfc39c2e2066cfa"
    "87eda523b2265c1af276c42d8978362f"
)

EXPECTED_POLICY_SHA256 = (
    "510b0ba97393a63a199d65414c2b44660"
    "e7140c1696582202b1603c7197f879d"
)

EXPECTED_PREVALIDATION_ARCHIVE_SHA256 = (
    "da8d6dfe0e5fa443bae14cf6501012b8"
    "5982e8bb02221c9e1b5caa76d128461f"
)

CHANNELS = [
    ("CDL-A", "CDLA"),
    ("CDL-B", "CDLB"),
    ("CDL-C", "CDLC"),
]

STRATA = [
    ("normal", "control"),

    ("interference", "mild"),
    ("interference", "moderate"),
    ("interference", "severe"),

    ("blockage", "mild"),
    ("blockage", "moderate"),
    ("blockage", "severe"),

    ("mobility", "mild"),
    ("mobility", "moderate"),
    ("mobility", "severe"),

    ("adaptation_mismatch", "mild"),
    ("adaptation_mismatch", "moderate"),
    ("adaptation_mismatch", "severe"),

    ("nonphysical_goodput", "control"),
]

REPLICATES_PER_STRATUM = 20
SEED_BASE = 8_100_000

EXPECTED_LABEL_COUNTS = {
    "normal": 60,
    "interference": 180,
    "blockage": 180,
    "mobility": 180,
    "adaptation_mismatch": 180,
    "nonphysical_goodput": 60,
}

EXPECTED_CHANNEL_COUNTS = {
    "CDL-A": 280,
    "CDL-B": 280,
    "CDL-C": 280,
}

EXPECTED_SEVERITY_COUNTS = {
    "control": 120,
    "mild": 240,
    "moderate": 240,
    "severe": 240,
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


def sha256_json(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def label_token(label):
    return label.upper()


for required in (
    SOURCE_PLAN_PATH,
    SMOKE_PLAN_PATH,
    POLICY_PATH,
    PREVALIDATION_ARCHIVE,
):
    if not required.exists():
        raise FileNotFoundError(required)


source_plan = json.loads(
    SOURCE_PLAN_PATH.read_text(
        encoding="utf-8"
    )
)

smoke_plan = json.loads(
    SMOKE_PLAN_PATH.read_text(
        encoding="utf-8"
    )
)

policy = json.loads(
    POLICY_PATH.read_text(
        encoding="utf-8"
    )
)


if (
    source_plan.get("plan_sha256")
    != EXPECTED_SOURCE_PLAN_SHA256
):
    raise RuntimeError(
        "Source Prevalidation-210 plan hash changed."
    )

if (
    policy.get("policy_sha256")
    != EXPECTED_POLICY_SHA256
):
    raise RuntimeError(
        "Locked causal-validation policy hash changed."
    )

if policy.get("status") != "LOCKED":
    raise RuntimeError(
        "Causal-validation policy is not LOCKED."
    )

archive_sha256 = sha256_file(
    PREVALIDATION_ARCHIVE
)

if (
    archive_sha256
    != EXPECTED_PREVALIDATION_ARCHIVE_SHA256
):
    raise RuntimeError(
        "Frozen Prevalidation-210 archive hash changed."
    )


source_samples = source_plan.get(
    "samples",
    []
)

if len(source_samples) != 210:
    raise RuntimeError(
        f"Expected 210 source samples, "
        f"found {len(source_samples)}."
    )


templates = {}

for sample in source_samples:
    key = (
        sample["channel_model"],
        sample["label"],
        sample["severity"],
    )

    current = templates.get(key)

    if (
        current is None
        or int(sample.get("replicate", 999))
        < int(current.get("replicate", 999))
    ):
        templates[key] = sample


expected_template_keys = {
    (
        channel_model,
        label,
        severity,
    )
    for channel_model, _ in CHANNELS
    for label, severity in STRATA
}

missing_template_keys = (
    expected_template_keys
    - set(templates)
)

if missing_template_keys:
    raise RuntimeError(
        "Missing source templates: "
        + str(sorted(missing_template_keys))
    )


samples = []

for channel_index, (
    channel_model,
    channel_token,
) in enumerate(CHANNELS):

    for stratum_index, (
        label,
        severity,
    ) in enumerate(STRATA):

        template = templates[
            (
                channel_model,
                label,
                severity,
            )
        ]

        for replicate in range(
            1,
            REPLICATES_PER_STRATUM + 1,
        ):
            sample = copy.deepcopy(
                template
            )

            if severity == "control":
                sample_id = (
                    f"FT840_{channel_token}_"
                    f"{label_token(label)}_"
                    f"R{replicate:02d}"
                )
            else:
                sample_id = (
                    f"FT840_{channel_token}_"
                    f"{label_token(label)}_"
                    f"{severity.upper()}_"
                    f"R{replicate:02d}"
                )

            seed = (
                SEED_BASE
                + channel_index * 100_000
                + stratum_index * 1_000
                + replicate
            )

            sample["sample_id"] = sample_id
            sample["seed"] = seed
            sample["replicate"] = replicate

            sample["channel_model"] = (
                channel_model
            )

            sample["label"] = label
            sample["severity"] = severity

            sample["dataset_role"] = (
                "disjoint_formal_test"
            )

            sample["formal_test"] = True

            sample[
                "causal_validation_policy_version"
            ] = policy["version"]

            sample[
                "causal_validation_policy_sha256"
            ] = EXPECTED_POLICY_SHA256

            sample[
                "eligible_for_training"
            ] = False

            sample[
                "eligible_for_threshold_tuning"
            ] = False

            sample[
                "eligible_for_model_selection"
            ] = False

            sample[
                "eligible_for_feature_selection"
            ] = False

            sample[
                "eligible_for_calibration"
            ] = False

            sample[
                "policy_modification_allowed"
            ] = False

            sample[
                "prevalidation_overlap_allowed"
            ] = False

            sample["formal_test_stratum"] = (
                f"{channel_model}|"
                f"{label}|"
                f"{severity}"
            )

            if label == "nonphysical_goodput":
                source_normal_id = (
                    f"FT840_{channel_token}_"
                    f"NORMAL_R{replicate:02d}"
                )

                sample[
                    "source_normal_id"
                ] = source_normal_id

                sample[
                    "paired_normal_id"
                ] = source_normal_id

            else:
                sample.pop(
                    "source_normal_id",
                    None,
                )

                sample.pop(
                    "paired_normal_id",
                    None,
                )

            samples.append(sample)


if len(samples) != 840:
    raise RuntimeError(
        f"Formal plan count={len(samples)}, "
        "expected 840."
    )


sample_ids = [
    sample["sample_id"]
    for sample in samples
]

seeds = [
    int(sample["seed"])
    for sample in samples
]

if len(set(sample_ids)) != 840:
    raise RuntimeError(
        "Formal-test sample IDs are not unique."
    )

if len(set(seeds)) != 840:
    raise RuntimeError(
        "Formal-test seeds are not unique."
    )


prevalidation_ids = {
    sample["sample_id"]
    for sample in source_samples
}

prevalidation_seeds = {
    int(sample["seed"])
    for sample in source_samples
}

smoke_samples = smoke_plan.get(
    "samples",
    []
)

smoke_ids = {
    sample["sample_id"]
    for sample in smoke_samples
    if "sample_id" in sample
}

smoke_seeds = {
    int(sample["seed"])
    for sample in smoke_samples
    if "seed" in sample
}


prevalidation_id_overlap = sorted(
    set(sample_ids)
    & prevalidation_ids
)

prevalidation_seed_overlap = sorted(
    set(seeds)
    & prevalidation_seeds
)

smoke_id_overlap = sorted(
    set(sample_ids)
    & smoke_ids
)

smoke_seed_overlap = sorted(
    set(seeds)
    & smoke_seeds
)


if prevalidation_id_overlap:
    raise RuntimeError(
        "Formal/prevalidation sample-ID overlap: "
        + str(prevalidation_id_overlap)
    )

if prevalidation_seed_overlap:
    raise RuntimeError(
        "Formal/prevalidation seed overlap: "
        + str(prevalidation_seed_overlap)
    )

if smoke_id_overlap:
    raise RuntimeError(
        "Formal/Smoke-24 sample-ID overlap: "
        + str(smoke_id_overlap)
    )

if smoke_seed_overlap:
    raise RuntimeError(
        "Formal/Smoke-24 seed overlap: "
        + str(smoke_seed_overlap)
    )


formal_id_set = set(sample_ids)

for sample in samples:
    if sample["label"] != "nonphysical_goodput":
        continue

    source_normal_id = sample[
        "source_normal_id"
    ]

    if source_normal_id not in formal_id_set:
        raise RuntimeError(
            f"{sample['sample_id']}: "
            f"paired normal missing: "
            f"{source_normal_id}"
        )


label_counts = Counter(
    sample["label"]
    for sample in samples
)

channel_counts = Counter(
    sample["channel_model"]
    for sample in samples
)

severity_counts = Counter(
    sample["severity"]
    for sample in samples
)

stratum_counts = Counter(
    sample["formal_test_stratum"]
    for sample in samples
)


if dict(label_counts) != EXPECTED_LABEL_COUNTS:
    raise RuntimeError(
        f"Unexpected label counts: "
        f"{dict(label_counts)}"
    )

if dict(channel_counts) != EXPECTED_CHANNEL_COUNTS:
    raise RuntimeError(
        f"Unexpected channel counts: "
        f"{dict(channel_counts)}"
    )

if dict(severity_counts) != EXPECTED_SEVERITY_COUNTS:
    raise RuntimeError(
        f"Unexpected severity counts: "
        f"{dict(severity_counts)}"
    )

if len(stratum_counts) != 42:
    raise RuntimeError(
        f"Stratum count={len(stratum_counts)}, "
        "expected 42."
    )

if set(stratum_counts.values()) != {20}:
    raise RuntimeError(
        "Each formal-test stratum must contain "
        "exactly 20 samples."
    )


formal_plan = copy.deepcopy(
    source_plan
)

formal_plan.pop(
    "plan_sha256",
    None,
)

formal_plan["schema"] = (
    "phyguard.sionna."
    "formal_test840.plan.v1"
)

formal_plan["protocol_name"] = (
    "PhyGuard Sionna Formal Test-840"
)

formal_plan["status"] = (
    "PLAN_CREATED_POLICY_LOCKED"
)

formal_plan["dataset_role"] = (
    "disjoint_formal_test"
)

formal_plan["sample_count"] = 840

formal_plan["replicates_per_stratum"] = (
    REPLICATES_PER_STRATUM
)

formal_plan["channel_models"] = [
    channel_model
    for channel_model, _ in CHANNELS
]

formal_plan["condition_count_per_channel"] = (
    len(STRATA)
)

formal_plan["stratum_count"] = (
    len(stratum_counts)
)

formal_plan["seed_namespace"] = {
    "base": SEED_BASE,
    "minimum": min(seeds),
    "maximum": max(seeds),
    "unique_seed_count": len(set(seeds)),
}

formal_plan["source_prevalidation"] = {
    "plan_sha256":
        EXPECTED_SOURCE_PLAN_SHA256,

    "frozen_archive_sha256":
        EXPECTED_PREVALIDATION_ARCHIVE_SHA256,

    "used_for_policy_development":
        True,

    "sample_id_overlap_count":
        len(prevalidation_id_overlap),

    "seed_overlap_count":
        len(prevalidation_seed_overlap),
}

formal_plan["smoke24_overlap"] = {
    "sample_id_overlap_count":
        len(smoke_id_overlap),

    "seed_overlap_count":
        len(smoke_seed_overlap),
}

formal_plan["causal_validation_policy"] = {
    "version":
        policy["version"],

    "sha256":
        EXPECTED_POLICY_SHA256,

    "locked_before_formal_generation":
        True,

    "modification_allowed_during_formal_test":
        False,
}

formal_plan["methodological_boundary"] = {
    "eligible_for_training":
        False,

    "eligible_for_threshold_tuning":
        False,

    "eligible_for_model_selection":
        False,

    "eligible_for_feature_selection":
        False,

    "eligible_for_calibration":
        False,

    "formal_test_results_may_not_modify_policy":
        True,
}

formal_plan["samples"] = samples

formal_plan["plan_sha256"] = (
    sha256_json(formal_plan)
)


OUTPUT_PLAN_JSON.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PLAN_JSON.write_text(
    json.dumps(
        formal_plan,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


with OUTPUT_PLAN_CSV.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "sample_id",
            "channel_model",
            "label",
            "severity",
            "replicate",
            "seed",
            "event_start",
            "event_end",
            "source_normal_id",
            "formal_test_stratum",
        ],
    )

    writer.writeheader()

    for sample in samples:
        writer.writerow(
            {
                "sample_id":
                    sample["sample_id"],

                "channel_model":
                    sample["channel_model"],

                "label":
                    sample["label"],

                "severity":
                    sample["severity"],

                "replicate":
                    sample["replicate"],

                "seed":
                    sample["seed"],

                "event_start":
                    sample.get("event_start"),

                "event_end":
                    sample.get("event_end"),

                "source_normal_id":
                    sample.get(
                        "source_normal_id",
                        "",
                    ),

                "formal_test_stratum":
                    sample[
                        "formal_test_stratum"
                    ],
            }
        )


manifest = {
    "schema":
        "phyguard.sionna."
        "formal_test840.plan_lock.v1",

    "status": "PASS",

    "sample_count": 840,

    "plan_path":
        str(OUTPUT_PLAN_JSON),

    "plan_csv_path":
        str(OUTPUT_PLAN_CSV),

    "plan_sha256":
        formal_plan["plan_sha256"],

    "source_prevalidation_plan_sha256":
        EXPECTED_SOURCE_PLAN_SHA256,

    "source_prevalidation_archive_sha256":
        EXPECTED_PREVALIDATION_ARCHIVE_SHA256,

    "causal_validation_policy_sha256":
        EXPECTED_POLICY_SHA256,

    "label_counts":
        dict(label_counts),

    "channel_counts":
        dict(channel_counts),

    "severity_counts":
        dict(severity_counts),

    "stratum_count":
        len(stratum_counts),

    "samples_per_stratum":
        20,

    "seed_minimum":
        min(seeds),

    "seed_maximum":
        max(seeds),

    "unique_seed_count":
        len(set(seeds)),

    "prevalidation_sample_id_overlap_count":
        len(prevalidation_id_overlap),

    "prevalidation_seed_overlap_count":
        len(prevalidation_seed_overlap),

    "smoke24_sample_id_overlap_count":
        len(smoke_id_overlap),

    "smoke24_seed_overlap_count":
        len(smoke_seed_overlap),

    "paired_nonphysical_controls":
        60,

    "policy_locked_before_generation":
        True,

    "policy_modification_allowed":
        False,

    "training_allowed":
        False,

    "threshold_tuning_allowed":
        False,

    "model_selection_allowed":
        False,

    "feature_selection_allowed":
        False,

    "calibration_allowed":
        False,
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
print("sample_count: 840")
print(
    "plan_sha256:",
    formal_plan["plan_sha256"],
)
print("stratum_count:", len(stratum_counts))
print("samples_per_stratum: 20")
print("unique_sample_ids:", len(set(sample_ids)))
print("unique_seeds:", len(set(seeds)))
print("seed_range:", [min(seeds), max(seeds)])
print(
    "prevalidation_sample_id_overlap_count:",
    len(prevalidation_id_overlap),
)
print(
    "prevalidation_seed_overlap_count:",
    len(prevalidation_seed_overlap),
)
print(
    "smoke24_sample_id_overlap_count:",
    len(smoke_id_overlap),
)
print(
    "smoke24_seed_overlap_count:",
    len(smoke_seed_overlap),
)

print("\nLABEL COUNTS")
for key, value in sorted(label_counts.items()):
    print(key, value)

print("\nCHANNEL COUNTS")
for key, value in sorted(channel_counts.items()):
    print(key, value)

print("\nSEVERITY COUNTS")
for key, value in sorted(severity_counts.items()):
    print(key, value)

print("\nplan_json:", OUTPUT_PLAN_JSON)
print("plan_csv:", OUTPUT_PLAN_CSV)
print("manifest:", OUTPUT_MANIFEST)

print(
    "\nSIONNA_FORMAL_TEST840_PLAN_CREATE_PASS"
)
