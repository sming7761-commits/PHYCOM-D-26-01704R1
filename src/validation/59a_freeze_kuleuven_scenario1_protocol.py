import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

INVENTORY_PATH = (
    ROOT
    / "artifacts"
    / "public_real_measurement_package_structure_audit_v1"
    / "kuleuven_file_inventory.csv"
)

PILOT_SUMMARY_PATH = (
    ROOT
    / "artifacts"
    / "kuleuven_scenario1_pilot_decode_v1"
    / "summary.json"
)

PILOT_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_scenario1_pilot_decode_v1.json"
)

DOCUMENTATION_PATH = (
    ROOT
    / "artifacts"
    / "public_real_measurement_semantic_audit_v1"
    / "kuleuven_documentation_combined.txt"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_scenario1_external_validation_protocol_v1"
)

FILE_MANIFEST_PATH = OUTPUT_ROOT / "scenario1_file_manifest.csv"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "kuleuven_scenario1_external_validation_protocol_v1.json"
)

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_scenario1_external_validation_protocol_v1.json"
)

EXPECTED_SIZE = 12_800_000

PATTERN = re.compile(
    r"^comm_sym2000_int10_10s_"
    r"scen1_activity(?P<activity>\d{2})_"
    r"rep(?P<repetition>[1-5])\.bin$",
    flags=re.IGNORECASE,
)

POSITIVE_ACTIVITIES = {
    1, 2, 5, 6, 7, 8,
    9, 10, 13, 14, 15, 16,
    17, 18, 21, 22,
    25, 26, 29, 30,
}

NEGATIVE_ACTIVITIES = {
    3, 4,
    11, 12,
    19, 20,
    23, 24,
    27, 28,
    31, 32,
}

ACTIVITY_DESCRIPTIONS = {
    1: "Walk straight crossing the link from c to f, slow speed",
    2: "Walk straight crossing the link from c to f, fast speed",
    3: "Walk from c to m and U-turn before blocking, slow speed",
    4: "Walk from c to m and U-turn before blocking, fast speed",
    5: "Walk straight crossing the link from f to c, slow speed",
    6: "Walk straight crossing the link from f to c, fast speed",
    7: "Start blocking at m, walk m to c, U-turn, return to m, slow",
    8: "Start blocking at m, walk m to c, U-turn, return to m, fast",
    9: "Walk straight crossing the link from b to e, slow speed",
    10: "Walk straight crossing the link from b to e, fast speed",
    11: "Walk from b to i and U-turn before blocking, slow speed",
    12: "Walk from b to i and U-turn before blocking, fast speed",
    13: "Walk straight crossing the link from e to b, slow speed",
    14: "Walk straight crossing the link from e to b, fast speed",
    15: "Start blocking at i, walk i to b, U-turn, return to i, slow",
    16: "Start blocking at i, walk i to b, U-turn, return to i, fast",
    17: "Walk straight crossing the link from a to f, slow speed",
    18: "Walk straight crossing the link from a to f, fast speed",
    19: "Walk from a to i and U-turn before blocking, slow speed",
    20: "Walk from a to i and U-turn before blocking, fast speed",
    21: "Walk straight crossing the link from f to a, slow speed",
    22: "Walk straight crossing the link from f to a, fast speed",
    23: "Walk from f to i and U-turn before blocking, slow speed",
    24: "Walk from f to i and U-turn before blocking, fast speed",
    25: "Walk straight crossing the link from d to c, slow speed",
    26: "Walk straight crossing the link from d to c, fast speed",
    27: "Walk from d to i and U-turn before blocking, slow speed",
    28: "Walk from d to i and U-turn before blocking, fast speed",
    29: "Walk straight crossing the link from c to d, slow speed",
    30: "Walk straight crossing the link from c to d, fast speed",
    31: "Walk from c to i and U-turn before blocking, slow speed",
    32: "Walk from c to i and U-turn before blocking, fast speed",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


for path in [
    INVENTORY_PATH,
    PILOT_SUMMARY_PATH,
    PILOT_MANIFEST_PATH,
    DOCUMENTATION_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(path)


if load_json(PILOT_SUMMARY_PATH).get("status") != "PASS":
    raise RuntimeError("Stage 58I summary is not PASS.")

if load_json(PILOT_MANIFEST_PATH).get("status") != "PASS":
    raise RuntimeError("Stage 58I manifest is not PASS.")

if OUTPUT_ROOT.exists() or PROTOCOL_PATH.exists():
    raise RuntimeError(
        "Stage 59A output already exists; refusing to overwrite."
    )


with INVENTORY_PATH.open(
    "r",
    encoding="utf-8",
    newline="",
) as stream:
    inventory = list(csv.DictReader(stream))


locked_rows = []

for row in inventory:
    filename = str(row.get("filename") or "")
    match = PATTERN.match(filename)

    if match is None:
        continue

    activity = int(match.group("activity"))
    repetition = int(match.group("repetition"))

    if activity in POSITIVE_ACTIVITIES:
        label = "BLOCKAGE_EVENT_PRESENT"
        label_id = 1
        rationale = (
            "Author-described trajectory crosses the Tx-Rx link "
            "or begins with the Tx-Rx link blocked."
        )
    elif activity in NEGATIVE_ACTIVITIES:
        label = "NO_LINK_BLOCKAGE"
        label_id = 0
        rationale = (
            "Author-described trajectory explicitly turns "
            "before blocking the Tx-Rx link."
        )
    else:
        raise RuntimeError(f"Unassigned activity: {activity}")

    size_bytes = int(row.get("size_bytes") or 0)

    if size_bytes != EXPECTED_SIZE:
        raise RuntimeError(
            f"Unexpected size for {filename}: {size_bytes}"
        )

    if str(row.get("restricted")).lower() not in {
        "", "false", "0", "none"
    }:
        raise RuntimeError(f"Restricted file encountered: {filename}")

    locked_rows.append(
        {
            "sequence_id": f"KU_S1_A{activity:02d}_R{repetition}",
            "filename": filename,
            "file_id": int(row["file_id"]),
            "directory_label": row.get("directory_label"),
            "scenario": 1,
            "activity": activity,
            "repetition": repetition,
            "size_bytes": size_bytes,
            "sequence_label": label,
            "sequence_label_id": label_id,
            "activity_description": ACTIVITY_DESCRIPTIONS[activity],
            "label_rationale": rationale,
            "selection_basis": "official Scenario 1 activity description",
        }
    )


locked_rows.sort(
    key=lambda item: (
        item["activity"],
        item["repetition"],
    )
)


if len(locked_rows) != 160:
    raise RuntimeError(
        f"Expected 160 Scenario 1 files, found {len(locked_rows)}."
    )


pair_counts = Counter(
    (row["activity"], row["repetition"])
    for row in locked_rows
)

if len(pair_counts) != 160 or any(
    count != 1 for count in pair_counts.values()
):
    raise RuntimeError("Scenario/activity/repetition uniqueness failed.")


for activity in range(1, 33):
    repetitions = {
        row["repetition"]
        for row in locked_rows
        if row["activity"] == activity
    }

    if repetitions != {1, 2, 3, 4, 5}:
        raise RuntimeError(
            f"Incomplete repetitions for activity {activity}: "
            f"{sorted(repetitions)}"
        )


positive_count = sum(
    row["sequence_label_id"] == 1
    for row in locked_rows
)

negative_count = sum(
    row["sequence_label_id"] == 0
    for row in locked_rows
)

if positive_count != 100 or negative_count != 60:
    raise RuntimeError(
        f"Unexpected label counts: {positive_count}/{negative_count}"
    )


OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

with FILE_MANIFEST_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(locked_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(locked_rows)


protocol = {
    "schema":
        "phyguard.kuleuven."
        "scenario1_external_validation_protocol.v1",

    "created_at_utc":
        datetime.now(timezone.utc).isoformat(),

    "status":
        "LOCKED_BEFORE_FULL_SCENARIO1_DOWNLOAD",

    "dataset_role":
        "mechanism-specific external real-measurement validation",

    "task":
        "sequence-level presence of a physical link-blockage event",

    "not_claimed_tasks": [
        "per-symbol blockage localization ground truth",
        "continuous blockage over the full ten-second sequence",
        "four-class PhyGuard diagnosis",
        "end-to-end anomaly localization",
    ],

    "population": {
        "scenario": 1,
        "activity_count": 32,
        "repetitions_per_activity": 5,
        "sequence_count": 160,
        "positive_sequence_count": 100,
        "negative_sequence_count": 60,
        "total_download_size_bytes": sum(
            row["size_bytes"] for row in locked_rows
        ),
    },

    "label_contract": {
        "positive_label": "BLOCKAGE_EVENT_PRESENT",
        "positive_activities": sorted(POSITIVE_ACTIVITIES),
        "negative_label": "NO_LINK_BLOCKAGE",
        "negative_activities": sorted(NEGATIVE_ACTIVITIES),
        "source": (
            "Official Description of Comms Data activity definitions"
        ),
        "label_independent_of_model_outputs": True,
        "label_independent_of_signal_thresholds": True,
    },

    "evaluation_boundary": {
        "all_160_sequences_reserved_for_external_evaluation": True,
        "model_fitting_on_kuleuven_data": False,
        "threshold_selection_on_kuleuven_data": False,
        "performance_based_file_selection": False,
        "candidate_interval_policy": (
            "UNRESOLVED_PENDING_LOCKED_SIGNAL_AUDIT"
        ),
        "feature_adapter_policy": (
            "UNRESOLVED_PENDING_LOCKED_SIGNAL_AUDIT"
        ),
    },

    "binary_contract": {
        "shape": [2000, 100, 2, 8],
        "complex_samples_per_file": 3_200_000,
        "bytes_per_complex_sample": 4,
        "storage": (
            "big-endian int16 imaginary followed by "
            "big-endian int16 real"
        ),
        "scale_divisor": 16384.0,
    },

    "file_manifest_path": str(FILE_MANIFEST_PATH),
    "file_manifest_sha256": sha256_file(FILE_MANIFEST_PATH),

    "source_assets": {
        "inventory_sha256": sha256_file(INVENTORY_PATH),
        "documentation_sha256": sha256_file(DOCUMENTATION_PATH),
        "pilot_summary_sha256": sha256_file(PILOT_SUMMARY_PATH),
        "pilot_manifest_sha256": sha256_file(PILOT_MANIFEST_PATH),
    },

    "methodological_boundary": {
        "full_scenario1_downloaded": False,
        "models_trained": False,
        "model_predictions_computed": False,
        "classification_performance_computed": False,
        "thresholds_selected": False,
        "event_intervals_inferred": False,
        "existing_sionna_assets_modified": False,
        "existing_locked_models_modified": False,
    },
}


PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)

PROTOCOL_PATH.write_text(
    json.dumps(
        protocol,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


summary = {
    "schema":
        "phyguard.kuleuven."
        "scenario1_external_validation_protocol_summary.v1",

    "status": "PASS",
    "protocol_path": str(PROTOCOL_PATH),
    "protocol_sha256": sha256_file(PROTOCOL_PATH),
    "file_manifest_path": str(FILE_MANIFEST_PATH),
    "file_manifest_sha256": sha256_file(FILE_MANIFEST_PATH),
    "sequence_count": 160,
    "positive_sequence_count": positive_count,
    "negative_sequence_count": negative_count,
    "total_download_size_bytes": sum(
        row["size_bytes"] for row in locked_rows
    ),
    "full_scenario1_downloaded": False,
    "models_trained": False,
    "performance_computed": False,
}

SUMMARY_PATH.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)


manifest = {
    "schema":
        "phyguard.kuleuven."
        "scenario1_external_validation_protocol_manifest.v1",

    "status": "PASS",
    "protocol_sha256": sha256_file(PROTOCOL_PATH),
    "summary_sha256": sha256_file(SUMMARY_PATH),
    "file_manifest_sha256": sha256_file(FILE_MANIFEST_PATH),
    "sequence_count": 160,
    "positive_sequence_count": positive_count,
    "negative_sequence_count": negative_count,
    "full_scenario1_downloaded": False,
    "models_trained": False,
    "performance_computed": False,
}

MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)


print("status: PASS")
print("protocol_status: LOCKED_BEFORE_FULL_SCENARIO1_DOWNLOAD")
print("sequence_count:", len(locked_rows))
print("positive_sequence_count:", positive_count)
print("negative_sequence_count:", negative_count)
print(
    "total_download_size_bytes:",
    sum(row["size_bytes"] for row in locked_rows),
)
print("all_sequences_external_evaluation: True")
print("full_scenario1_downloaded: False")
print("models_trained: False")
print("performance_computed: False")
print("thresholds_selected: False")
print("event_intervals_inferred: False")
print("protocol:", PROTOCOL_PATH)
print("file_manifest:", FILE_MANIFEST_PATH)
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)
print("\nKULEUVEN_SCENARIO1_EXTERNAL_VALIDATION_PROTOCOL_V1_PASS")
