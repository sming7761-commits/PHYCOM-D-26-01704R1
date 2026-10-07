#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "physics_sensitivity_protocol_v1.json"
)

PROTOCOL_ARTIFACT_ROOT = (
    ROOT
    / "artifacts"
    / "physics_sensitivity_protocol_v1"
)

PROTOCOL_AMENDMENT_ROOT = (
    ROOT
    / "artifacts"
    / "physics_sensitivity_protocol_phi_assertion_no_import_amendment_v1"
)

FIXED_MODEL_RESULT_ROOT = (
    ROOT
    / "artifacts"
    / "physics_sensitivity_fixed_model_results_v1"
)

FIXED_MODEL_AMENDMENT_ROOT = (
    ROOT
    / "artifacts"
    / "physics_sensitivity_fixed_model_protocol_amendment_v1"
)

FOREST_COMPATIBILITY_ROOT = (
    ROOT
    / "artifacts"
    / "physics_sensitivity_forest_probability_compatibility_amendment_v1"
)

UNCERTAINTY_ROOT = (
    ROOT
    / "artifacts"
    / "physics_sensitivity_uncertainty_v1"
)

MANUSCRIPT_ASSET_ROOT = (
    ROOT
    / "artifacts"
    / "physics_sensitivity_manuscript_assets_v1"
)

RESULT_ROOT = ROOT / "results"

ARCHIVE_PATH = (
    RESULT_ROOT
    / "phyguard_physics_sensitivity_final.zip"
)

SHA_PATH = (
    RESULT_ROOT
    / "phyguard_physics_sensitivity_final.sha256"
)

FINAL_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "phyguard_physics_sensitivity_final.json"
)

SOURCE_RESULT_MANIFEST = (
    ROOT
    / "manifests"
    / "physics_sensitivity_fixed_model_results_v1.json"
)

SOURCE_UNCERTAINTY_MANIFEST = (
    ROOT
    / "manifests"
    / "physics_sensitivity_uncertainty_v1.json"
)

KEY_RESULT_PATH = (
    MANUSCRIPT_ASSET_ROOT
    / "key_results.json"
)

FREEZE_SUMMARY_PATH = (
    MANUSCRIPT_ASSET_ROOT
    / "freeze_summary.json"
)

EXCLUDED_DEBUG_PATH = (
    MANUSCRIPT_ASSET_ROOT
    / "excluded_debug_artifacts.json"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def copy_file(source: Path, target: Path) -> None:
    require(
        source.exists(),
        f"Required manuscript source is missing: {source}",
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        target,
    )


for path in [
    PROTOCOL_PATH,
    PROTOCOL_ARTIFACT_ROOT / "summary.json",
    FIXED_MODEL_RESULT_ROOT / "summary.json",
    FIXED_MODEL_RESULT_ROOT / "aggregate_metrics.csv",
    FIXED_MODEL_RESULT_ROOT / "threshold_policy_aggregate_metrics.csv",
    FIXED_MODEL_RESULT_ROOT / "default_reproduction_audit.json",
    FIXED_MODEL_AMENDMENT_ROOT / "summary.json",
    FOREST_COMPATIBILITY_ROOT / "summary.json",
    UNCERTAINTY_ROOT / "summary.json",
    UNCERTAINTY_ROOT / "paired_formula_uncertainty.csv",
    UNCERTAINTY_ROOT / "factor_operating_ranges.csv",
    UNCERTAINTY_ROOT / "main_table_formula_factor_ranges.csv",
    UNCERTAINTY_ROOT / "main_table_threshold_policy_sa_0p90.csv",
    UNCERTAINTY_ROOT / "supplementary_all_13_formula_configs.csv",
    UNCERTAINTY_ROOT / "supplementary_all_9_threshold_policies.csv",
    UNCERTAINTY_ROOT / "manuscript_results_draft.txt",
    UNCERTAINTY_ROOT / "reviewer_response_draft.txt",
    UNCERTAINTY_ROOT / "claim_boundary.json",
    SOURCE_RESULT_MANIFEST,
    SOURCE_UNCERTAINTY_MANIFEST,
]:
    require(
        path.exists(),
        f"Required final input is missing: {path}",
    )

for path in [
    MANUSCRIPT_ASSET_ROOT,
    ARCHIVE_PATH,
    SHA_PATH,
    FINAL_MANIFEST_PATH,
]:
    require(
        not path.exists(),
        f"Final freeze output already exists: {path}",
    )


protocol = json.loads(
    PROTOCOL_PATH.read_text(
        encoding="utf-8"
    )
)

protocol_summary = json.loads(
    (
        PROTOCOL_ARTIFACT_ROOT
        / "summary.json"
    ).read_text(
        encoding="utf-8"
    )
)

fixed_summary = json.loads(
    (
        FIXED_MODEL_RESULT_ROOT
        / "summary.json"
    ).read_text(
        encoding="utf-8"
    )
)

default_audit = json.loads(
    (
        FIXED_MODEL_RESULT_ROOT
        / "default_reproduction_audit.json"
    ).read_text(
        encoding="utf-8"
    )
)

uncertainty_summary = json.loads(
    (
        UNCERTAINTY_ROOT
        / "summary.json"
    ).read_text(
        encoding="utf-8"
    )
)

claim_boundary = json.loads(
    (
        UNCERTAINTY_ROOT
        / "claim_boundary.json"
    ).read_text(
        encoding="utf-8"
    )
)


require(
    protocol.get("status")
    == "LOCKED_BEFORE_PHYSICS_SENSITIVITY_EXECUTION",
    "Sensitivity protocol is not locked.",
)

require(
    protocol_summary.get("status") == "PASS",
    "Protocol summary is not PASS.",
)

require(
    fixed_summary.get("status") == "PASS",
    "Fixed-model sensitivity summary is not PASS.",
)

require(
    uncertainty_summary.get("status") == "PASS",
    "Uncertainty summary is not PASS.",
)

require(
    default_audit.get("prediction_match") is True,
    "DEFAULT 1,080-prediction reproduction is not PASS.",
)

require(
    fixed_summary.get("formula_variant_count") == 13,
    "Expected 13 formula variants.",
)

require(
    fixed_summary.get("fixed_model_inference_task_count") == 65,
    "Expected 65 fixed-model inference tasks.",
)

require(
    fixed_summary.get("training_task_count") == 0,
    "Fixed-model sensitivity must have zero training tasks.",
)

require(
    fixed_summary.get("threshold_policy_count") == 9,
    "Expected nine threshold policies.",
)

require(
    fixed_summary.get("threshold_policy_evaluation_count") == 45,
    "Expected 45 threshold-policy evaluations.",
)

require(
    fixed_summary.get("source_models_modified") is False,
    "Frozen source models were marked as modified.",
)

require(
    fixed_summary.get("test_used_for_threshold_selection") is False,
    "Test data was marked as used for threshold selection.",
)

require(
    uncertainty_summary.get("bootstrap_replicates") == 10000,
    "Expected 10,000 bootstrap replicates.",
)

require(
    uncertainty_summary.get("cluster_count") == 726,
    "Expected 726 unique sequence clusters.",
)

require(
    uncertainty_summary.get("paired_comparison_count") == 84,
    "Expected 84 paired comparisons.",
)

require(
    uncertainty_summary.get("default_prediction_match") is True,
    "Uncertainty analysis did not retain DEFAULT reproduction.",
)

require(
    uncertainty_summary.get(
        "confirmatory_significance_claim_allowed"
    )
    is False,
    "Confirmatory significance claim boundary is not locked.",
)

require(
    claim_boundary.get("trainable_physics_weights_claimed") is False,
    "Claim boundary incorrectly claims trainable physics weights.",
)

require(
    claim_boundary.get(
        "validation_constraint_claimed_as_test_guarantee"
    )
    is False,
    "Claim boundary incorrectly treats validation as a test guarantee.",
)


MANUSCRIPT_ASSET_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

copy_map = {
    UNCERTAINTY_ROOT
    / "main_table_formula_factor_ranges.csv":
        MANUSCRIPT_ASSET_ROOT
        / "main_table_formula_factor_ranges.csv",

    UNCERTAINTY_ROOT
    / "main_table_threshold_policy_sa_0p90.csv":
        MANUSCRIPT_ASSET_ROOT
        / "main_table_threshold_policy_sa_0p90.csv",

    UNCERTAINTY_ROOT
    / "supplementary_all_13_formula_configs.csv":
        MANUSCRIPT_ASSET_ROOT
        / "supplementary_all_13_formula_configs.csv",

    UNCERTAINTY_ROOT
    / "supplementary_all_9_threshold_policies.csv":
        MANUSCRIPT_ASSET_ROOT
        / "supplementary_all_9_threshold_policies.csv",

    UNCERTAINTY_ROOT
    / "paired_formula_uncertainty.csv":
        MANUSCRIPT_ASSET_ROOT
        / "paired_formula_uncertainty.csv",

    UNCERTAINTY_ROOT
    / "manuscript_results_draft.txt":
        MANUSCRIPT_ASSET_ROOT
        / "manuscript_results_draft.txt",

    UNCERTAINTY_ROOT
    / "reviewer_response_draft.txt":
        MANUSCRIPT_ASSET_ROOT
        / "reviewer_response_draft.txt",

    UNCERTAINTY_ROOT
    / "claim_boundary.json":
        MANUSCRIPT_ASSET_ROOT
        / "claim_boundary.json",
}

for source, target in copy_map.items():
    copy_file(
        source,
        target,
    )


key_results = {
    "schema": (
        "phyguard.physics_sensitivity_key_results.v1"
    ),
    "status": "PASS",
    "method": (
        "Fixed-model sensitivity of the revision-time locked "
        "PhyGuard reconstruction."
    ),
    "default_reproduction": {
        "prediction_match": True,
        "prediction_row_count": 1080,
        "metric_max_absolute_differences": (
            default_audit.get(
                "metric_max_absolute_differences"
            )
        ),
    },
    "execution": {
        "formula_variant_count": 13,
        "fixed_model_inference_task_count": 65,
        "training_task_count": 0,
        "threshold_policy_count": 9,
        "threshold_policy_evaluation_count": 45,
    },
    "uncertainty": {
        "bootstrap_replicates": 10000,
        "cluster_count": 726,
        "paired_comparison_count": 84,
        "exact_signflip_minimum_nonzero_two_sided_p": 0.0625,
        "confirmatory_significance_claim_allowed": False,
    },
    "overall_nondefault_delta_envelope": (
        uncertainty_summary.get(
            "overall_nondefault_delta_envelope"
        )
    ),
    "integrity": {
        "source_models_modified": False,
        "source_features_modified": False,
        "source_results_modified": False,
        "test_used_for_threshold_selection": False,
        "all_formula_configs_reported": True,
        "all_threshold_policies_reported": True,
    },
    "formal_hardware": (
        "NVIDIA GeForce RTX 3080 10 GB"
    ),
}

KEY_RESULT_PATH.write_text(
    json.dumps(
        key_results,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


excluded_debug_candidates = [
    ROOT
    / "artifacts"
    / "physics_sensitivity_results_v1",

    ROOT
    / "artifacts"
    / "physics_sensitivity_results_v1_pre_float32_execution_quarantine_v1",

    ROOT
    / "artifacts"
    / "physics_sensitivity_fixed_model_results_v1_pre_forest_probability_hotfix_quarantine_v1",

    ROOT
    / "artifacts"
    / "physics_sensitivity_fixed_model_protocol_amendment_v1_pre_forest_probability_hotfix_quarantine",

    ROOT
    / "artifacts"
    / "physics_sensitivity_float32_feature_audit_amendment_v1",

    ROOT
    / "artifacts"
    / "physics_sensitivity_float32_execution_amendment_v1",

    ROOT
    / "artifacts"
    / "physics_sensitivity_exact_source_forensic_audit_v1",
]

excluded_debug = {
    "schema": (
        "phyguard.physics_sensitivity_excluded_debug_artifacts.v1"
    ),
    "status": "PASS",
    "reason": (
        "These paths belong to failed pre-execution compatibility "
        "or refit attempts and are excluded from the formal result "
        "archive. No non-default sensitivity conclusion was accepted "
        "from them."
    ),
    "paths": [
        {
            "path": str(
                path.relative_to(
                    ROOT
                )
            ),
            "exists": path.exists(),
        }
        for path in excluded_debug_candidates
    ],
}

EXCLUDED_DEBUG_PATH.write_text(
    json.dumps(
        excluded_debug,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


freeze_summary = {
    "schema": (
        "phyguard.physics_sensitivity_final_freeze_summary.v1"
    ),
    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "status": "PASS",
    "protocol_status": (
        protocol.get("status")
    ),
    "formula_variant_count": 13,
    "fixed_model_inference_task_count": 65,
    "training_task_count": 0,
    "threshold_policy_count": 9,
    "threshold_policy_evaluation_count": 45,
    "bootstrap_replicates": 10000,
    "cluster_count": 726,
    "paired_comparison_count": 84,
    "default_prediction_match": True,
    "source_models_modified": False,
    "test_used_for_threshold_selection": False,
    "confirmatory_significance_claim_allowed": False,
    "formal_hardware": (
        "NVIDIA GeForce RTX 3080 10 GB"
    ),
}

FREEZE_SUMMARY_PATH.write_text(
    json.dumps(
        freeze_summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


formal_roots = [
    PROTOCOL_PATH,
    PROTOCOL_ARTIFACT_ROOT,
    FIXED_MODEL_RESULT_ROOT,
    FIXED_MODEL_AMENDMENT_ROOT,
    FOREST_COMPATIBILITY_ROOT,
    UNCERTAINTY_ROOT,
    MANUSCRIPT_ASSET_ROOT,
    SOURCE_RESULT_MANIFEST,
    SOURCE_UNCERTAINTY_MANIFEST,
]

if PROTOCOL_AMENDMENT_ROOT.exists():
    formal_roots.append(
        PROTOCOL_AMENDMENT_ROOT
    )


script_candidates = [
    ROOT
    / "scripts"
    / "61c_freeze_physics_sensitivity_protocol.py",

    ROOT
    / "scripts"
    / "61d4_run_fixed_model_physics_sensitivity.py",

    ROOT
    / "scripts"
    / "61d4_check_fixed_model_physics_sensitivity.py",

    ROOT
    / "scripts"
    / "61d5_patch_forest_probability_compatibility.py",

    ROOT
    / "scripts"
    / "61e_run_physics_sensitivity_uncertainty.py",

    ROOT
    / "scripts"
    / "61e_check_physics_sensitivity_uncertainty.py",
]

for path in script_candidates:
    require(
        path.exists(),
        f"Required final script is missing: {path}",
    )

formal_roots.extend(
    script_candidates
)


archive_files: list[Path] = []

for root in formal_roots:
    if root.is_file():
        archive_files.append(
            root
        )

    elif root.is_dir():
        archive_files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
        )

    else:
        raise RuntimeError(
            f"Formal archive root is missing: {root}"
        )


unique_files = sorted(
    set(
        archive_files
    ),
    key=lambda path: str(
        path.relative_to(
            ROOT
        )
    ),
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

with zipfile.ZipFile(
    ARCHIVE_PATH,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=9,
) as archive:
    for path in unique_files:
        archive.write(
            path,
            path.relative_to(
                ROOT
            ),
        )


archive_sha256 = sha256_file(
    ARCHIVE_PATH
)

SHA_PATH.write_text(
    (
        archive_sha256
        + "  "
        + ARCHIVE_PATH.name
        + "\n"
    ),
    encoding="utf-8",
)


file_manifest = [
    {
        "path": str(
            path.relative_to(
                ROOT
            )
        ),
        "size_bytes": (
            path.stat().st_size
        ),
        "sha256": sha256_file(
            path
        ),
    }
    for path in unique_files
]

final_manifest = {
    "schema": (
        "phyguard.physics_sensitivity_final_manifest.v1"
    ),
    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "status": "PASS",
    "archive_path": str(
        ARCHIVE_PATH
    ),
    "archive_size_bytes": (
        ARCHIVE_PATH.stat().st_size
    ),
    "archive_sha256": archive_sha256,
    "archived_file_count": len(
        file_manifest
    ),
    "protocol": {
        "formula_variant_count": 13,
        "threshold_policy_count": 9,
    },
    "execution": {
        "fixed_model_inference_task_count": 65,
        "training_task_count": 0,
        "threshold_policy_evaluation_count": 45,
    },
    "uncertainty": {
        "bootstrap_replicates": 10000,
        "cluster_count": 726,
        "paired_comparison_count": 84,
    },
    "integrity": {
        "default_prediction_match": True,
        "source_models_modified": False,
        "test_used_for_threshold_selection": False,
        "all_formula_configs_reported": True,
        "all_threshold_policies_reported": True,
        "confirmatory_significance_claim_allowed": False,
        "failed_debug_artifacts_excluded": True,
    },
    "formal_hardware": (
        "NVIDIA GeForce RTX 3080 10 GB"
    ),
    "files": file_manifest,
}

FINAL_MANIFEST_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL_MANIFEST_PATH.write_text(
    json.dumps(
        final_manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print("status: PASS")
print("formula_variant_count: 13")
print("fixed_model_inference_task_count: 65")
print("training_task_count: 0")
print("threshold_policy_count: 9")
print("threshold_policy_evaluation_count: 45")
print("bootstrap_replicates: 10000")
print("cluster_count: 726")
print("paired_comparison_count: 84")
print("default_prediction_match: True")
print("source_models_modified: False")
print("test_used_for_threshold_selection: False")
print(
    "confirmatory_significance_claim_allowed: False"
)
print(
    "formal_hardware: NVIDIA GeForce RTX 3080 10 GB"
)
print("archived_file_count:", len(file_manifest))
print("archive:", ARCHIVE_PATH)
print(
    "archive_size_bytes:",
    ARCHIVE_PATH.stat().st_size,
)
print("archive_sha256:", archive_sha256)
print("sha256_file:", SHA_PATH)
print("manifest:", FINAL_MANIFEST_PATH)
print(
    "manuscript_assets:",
    MANUSCRIPT_ASSET_ROOT,
)
print()
print(
    "PHYSICS_SENSITIVITY_FINAL_FREEZE_PASS"
)
