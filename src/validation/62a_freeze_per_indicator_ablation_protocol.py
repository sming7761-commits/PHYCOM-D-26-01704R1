#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

LOCKED_RECONSTRUCTION_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal825_locked_reconstruction_final.zip"
)

CAUSAL_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_sionna_formal_test840_causal_final.zip"
)

RELIABILITY_ARCHIVE = (
    ROOT
    / "results"
    / "phyguard_cross_system_reliability_statistics_v2_final.zip"
)

SIONNA_IDS_PATH = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_metadata_contract_v1"
    / "sionna_analysis825_ids.txt"
)

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "per_indicator_ablation_protocol_v1.json"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "per_indicator_ablation_protocol_v1"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)

TOKEN_AUDIT_PATH = (
    OUTPUT_ROOT
    / "code_token_audit.csv"
)

READY_TEXT_PATH = (
    OUTPUT_ROOT
    / "PROTOCOL_READY.md"
)

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "per_indicator_ablation_protocol_v1.json"
)

EXPECTED_SHA256 = {
    str(LOCKED_RECONSTRUCTION_ARCHIVE):
        "5d08fc3b8265789819b0f49c1f93a9c79c8fe44ce7ec4aec8f1fa603a08c241b",

    str(CAUSAL_ARCHIVE):
        "b6c5f400b7d55949fb1e1f39c0c092da8f780eed100866f30176e26e462ff2f5",

    str(RELIABILITY_ARCHIVE):
        "c6d9ee842f5a490160b92ab34f786be2e2c4267402e986c616a17d577fcfc359",

    str(SIONNA_IDS_PATH):
        "a8aa74f95df897a57c82f60ea0d2f23353eaa023ec0446b079dd8a88d7de462a",
}

SCAN_ROOTS = [
    ROOT / "scripts",
    (
        ROOT
        / "artifacts"
        / "uploaded_phyguard_packages"
        / "PhyGuard_R0_Reconstruction_Checkpoint2"
        / "phyguard_rebuild"
    ),
]

TOKENS = [
    "per_proxy",
    "bler_post_ldpc",
    "ber",
    "ber_pre_ldpc",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def scan_code_tokens() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    allowed_suffixes = {
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".txt",
        ".md",
        ".csv",
    }

    for scan_root in SCAN_ROOTS:
        if not scan_root.exists():
            continue

        for path in sorted(
            scan_root.rglob("*")
        ):
            if not path.is_file():
                continue

            if path.suffix.lower() not in allowed_suffixes:
                continue

            if path.stat().st_size > 2_000_000:
                continue

            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception:
                continue

            for token in TOKENS:
                count = text.count(token)

                if count > 0:
                    rows.append(
                        {
                            "token":
                                token,

                            "relative_path":
                                str(
                                    path.relative_to(
                                        ROOT
                                    )
                                ),

                            "count":
                                count,
                        }
                    )

    return rows


required_paths = [
    LOCKED_RECONSTRUCTION_ARCHIVE,
    CAUSAL_ARCHIVE,
    RELIABILITY_ARCHIVE,
    SIONNA_IDS_PATH,
]

for path in required_paths:
    require(
        path.exists(),
        f"Required frozen asset is missing: {path}",
    )

    actual_sha256 = sha256_file(
        path
    )

    require(
        actual_sha256
        == EXPECTED_SHA256[str(path)],
        (
            "Frozen-asset checksum mismatch:\n"
            f"path={path}\n"
            f"expected={EXPECTED_SHA256[str(path)]}\n"
            f"actual={actual_sha256}"
        ),
    )


analysis_ids = [
    line.strip()
    for line in SIONNA_IDS_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

require(
    len(analysis_ids) == 825,
    (
        "Expected exactly 825 frozen Sionna IDs, "
        f"found {len(analysis_ids)}."
    ),
)

require(
    len(set(analysis_ids)) == 825,
    "Frozen Sionna IDs are not unique.",
)


for path in [
    PROTOCOL_PATH,
    OUTPUT_ROOT,
    MANIFEST_PATH,
]:
    require(
        not path.exists(),
        (
            "Protocol output already exists; "
            f"refusing to overwrite: {path}"
        ),
    )


token_rows = scan_code_tokens()

token_hit_counts = {
    token:
        sum(
            int(row["count"])
            for row in token_rows
            if row["token"] == token
        )
    for token in TOKENS
}

for token in TOKENS:
    require(
        token_hit_counts[token] > 0,
        (
            "Required code/schema token was not found: "
            f"{token}"
        ),
    )


status = (
    "LOCKED_BEFORE_PER_INDICATOR_ABLATION_EXECUTION"
)

variants = [
    {
        "variant_id":
            "FULL",

        "raw_packet_block_error_feature":
            True,

        "physics_packet_block_error_term":
            True,

        "ber_feature_and_terms":
            "UNCHANGED",

        "description":
            (
                "Locked full reconstructed pipeline. "
                "The source-domain raw indicator is per_proxy; "
                "the coded Sionna counterpart is bler_post_ldpc."
            ),
    },
    {
        "variant_id":
            "NO_PER_RAW",

        "raw_packet_block_error_feature":
            False,

        "physics_packet_block_error_term":
            True,

        "ber_feature_and_terms":
            "UNCHANGED",

        "description":
            (
                "Remove per_proxy/bler_post_ldpc only from the "
                "raw statistical/model-input branch. Preserve "
                "the corresponding physics-score term."
            ),
    },
    {
        "variant_id":
            "NO_PER_SCORE",

        "raw_packet_block_error_feature":
            True,

        "physics_packet_block_error_term":
            False,

        "ber_feature_and_terms":
            "UNCHANGED",

        "description":
            (
                "Retain per_proxy/bler_post_ldpc in the raw "
                "statistical/model-input branch, but set its "
                "physics-score contribution to zero."
            ),
    },
    {
        "variant_id":
            "NO_ERROR_INDICATOR",

        "raw_packet_block_error_feature":
            False,

        "physics_packet_block_error_term":
            False,

        "ber_feature_and_terms":
            "UNCHANGED",

        "description":
            (
                "Remove the packet/block-error indicator from "
                "both raw and physics-score paths. BER remains "
                "unchanged. This is the combined 2x2 factorial "
                "corner and is not a removal of the whole BER "
                "evidence family."
            ),
    },
]


protocol = {
    "schema":
        "phyguard.per_indicator_ablation_protocol.v1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        status,

    "purpose":
        (
            "Determine whether PhyGuard's selective-diagnosis "
            "performance is materially dependent on the "
            "revision-time locked 48-bit uncoded packet-error "
            "proxy, while preserving the BER evidence channel."
        ),

    "scientific_question":
        (
            "Does the packet/block-error indicator contribute "
            "through the raw statistical branch, the "
            "physics-score branch, both branches, or neither?"
        ),

    "factorial_design": {
        "design":
            "2x2",

        "factor_raw_packet_block_error_feature":
            [
                "PRESENT",
                "ABSENT",
            ],

        "factor_physics_packet_block_error_term":
            [
                "PRESENT",
                "ABSENT",
            ],

        "variants":
            variants,

        "fixed_channel":
            (
                "BER is retained unchanged in every variant. "
                "Source name: ber. Sionna name: ber_pre_ldpc."
            ),

        "semantic_mapping": {
            "source_packet_error_indicator":
                "per_proxy",

            "sionna_packet_error_indicator":
                "bler_post_ldpc",

            "source_bit_error_indicator":
                "ber",

            "sionna_bit_error_indicator":
                "ber_pre_ldpc",
        },
    },

    "frozen_assets": {
        "locked_reconstruction_archive": {
            "path":
                str(
                    LOCKED_RECONSTRUCTION_ARCHIVE.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                EXPECTED_SHA256[
                    str(
                        LOCKED_RECONSTRUCTION_ARCHIVE
                    )
                ],
        },

        "causal_archive": {
            "path":
                str(
                    CAUSAL_ARCHIVE.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                EXPECTED_SHA256[
                    str(
                        CAUSAL_ARCHIVE
                    )
                ],
        },

        "cross_system_reliability_archive": {
            "path":
                str(
                    RELIABILITY_ARCHIVE.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                EXPECTED_SHA256[
                    str(
                        RELIABILITY_ARCHIVE
                    )
                ],
        },

        "sionna_analysis_membership": {
            "path":
                str(
                    SIONNA_IDS_PATH.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                EXPECTED_SHA256[
                    str(
                        SIONNA_IDS_PATH
                    )
                ],

            "sample_count":
                825,
        },
    },

    "execution_contract": {
        "pipeline":
            (
                "Use the same revision-time locked "
                "reconstruction pipeline and the same frozen "
                "825-sample Sionna evaluation path. Alter only "
                "the two preregistered packet/block-error masks."
            ),

        "repeat_bundles":
            5,

        "model_training":
            (
                "Refit the gate and resolver separately for "
                "each factorial variant using the exact same "
                "training/validation memberships and random "
                "seeds as FULL."
            ),

        "threshold_selection":
            (
                "Any gate, confidence, or abstention threshold "
                "must be selected using validation data only. "
                "No test-set threshold adjustment is allowed."
            ),

        "feature_dimensionality":
            (
                "When a raw feature is absent, remove it from "
                "both training and inference rather than "
                "replacing it with its test-set mean. Feature "
                "order for all retained channels must remain "
                "unchanged."
            ),

        "score_masking":
            (
                "When the packet/block-error physics term is "
                "absent, set only that term's contribution to "
                "zero. Do not renormalize remaining weights "
                "unless the frozen implementation already "
                "performs a deterministic normalization that "
                "is identical across all variants."
            ),

        "no_model_selection":
            (
                "All four variants must be executed and "
                "reported. No variant may be dropped based on "
                "performance."
            ),
    },

    "primary_endpoint": {
        "metric":
            "selected_accuracy",

        "comparison":
            (
                "For each ablation, compute paired absolute "
                "difference: ablation minus FULL."
            ),

        "aggregation":
            (
                "Report mean and standard deviation across the "
                "five locked repeat bundles, plus a paired "
                "sample-cluster bootstrap interval."
            ),

        "bootstrap_replicates":
            10000,

        "bootstrap_seed_source":
            62011,

        "bootstrap_seed_sionna":
            62021,

        "practical_retention_margin":
            -0.05,

        "retention_rule":
            (
                "The ablation retains the primary endpoint "
                "when the lower bound of the paired 95% "
                "bootstrap interval is greater than -0.05."
            ),

        "interpretation":
            (
                "The -0.05 margin is a preregistered practical "
                "retention threshold, not a regulatory or "
                "clinical non-inferiority claim."
            ),
    },

    "secondary_endpoints": [
        "physical_selection_rate",
        "false_specific_rate",
        "macro_f1",
        "balanced_accuracy",
        "physical_exact_match",
        "control_abstention",
        "coverage",
    ],

    "mandatory_outputs": [
        "variant-level metrics for all five repeat bundles",
        "paired per-sample prediction table",
        "paired deltas versus FULL",
        "95% paired-bootstrap intervals",
        "2x2 factorial main effects",
        "2x2 interaction effect",
        "class-wise selected accuracy",
        "mechanism-wise physical exact match",
        "control abstention",
        "training time",
        "inference time",
        "retained feature names",
        "active physics-score terms",
    ],

    "factorial_effects": {
        "raw_path_main_effect":
            (
                "Average contrast between raw indicator "
                "PRESENT and ABSENT across score-term levels."
            ),

        "score_path_main_effect":
            (
                "Average contrast between physics score term "
                "PRESENT and ABSENT across raw-feature levels."
            ),

        "interaction":
            (
                "Difference-in-differences between raw-feature "
                "and score-term factors."
            ),

        "reporting":
            (
                "Report effect sizes and paired-bootstrap "
                "intervals; do not use the interaction to "
                "select or suppress variants."
            ),
    },

    "classification_rules": {
        "PACKET_BLOCK_ERROR_INDICATOR_NOT_ESSENTIAL":
            (
                "NO_ERROR_INDICATOR retains selected accuracy "
                "within the -0.05 margin on both the locked "
                "reconstruction evaluation and frozen Sionna "
                "evaluation."
            ),

        "PATH_SPECIFIC_DEPENDENCE":
            (
                "NO_ERROR_INDICATOR fails retention, but "
                "exactly one of NO_PER_RAW or NO_PER_SCORE "
                "retains the primary endpoint on both systems."
            ),

        "DISTRIBUTED_DEPENDENCE":
            (
                "NO_ERROR_INDICATOR fails retention, while "
                "both single-path ablations retain the primary "
                "endpoint on both systems."
            ),

        "STRONG_INDICATOR_DEPENDENCE":
            (
                "NO_ERROR_INDICATOR and both single-path "
                "ablations fail the retention rule on either "
                "system."
            ),

        "MIXED_OR_INCONCLUSIVE":
            (
                "Any result pattern not covered above, "
                "including materially different source and "
                "Sionna conclusions."
            ),
    },

    "claim_boundaries": [
        (
            "The experiment evaluates dependence on the "
            "packet/block-error indicator, not calibration "
            "equivalence between per_proxy and coded BLER."
        ),
        (
            "BER remains present in all four variants; "
            "NO_ERROR_INDICATOR must not be described as "
            "removing every error-related KPI."
        ),
        (
            "A retained ablation supports architectural "
            "robustness, not proof that the removed signal is "
            "physically meaningless."
        ),
        (
            "A performance decline supports dependence, not "
            "proof of data leakage or self-fulfilling labels."
        ),
    ],

    "prohibited_posthoc_actions": [
        "changing the -0.05 retention margin",
        "changing bootstrap seeds or replicate count",
        "retuning thresholds on test data",
        "renormalizing score weights only for favorable variants",
        "dropping a variant",
        "redefining NO_ERROR_INDICATOR after observing results",
        "removing BER from only selected variants",
        "changing the frozen 825-sample membership",
        "changing label definitions",
        "changing repeat memberships or random seeds",
        "reporting only the better of source and Sionna results",
    ],

    "methodological_boundary": {
        "kpi_values_opened":
            False,

        "models_trained":
            False,

        "predictions_generated":
            False,

        "performance_metrics_computed":
            False,

        "thresholds_selected":
            False,

        "test_results_viewed":
            False,

        "code_and_schema_tokens_audited":
            True,
    },
}


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

PROTOCOL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

PROTOCOL_PATH.write_text(
    json.dumps(
        protocol,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


with TOKEN_AUDIT_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "token",
            "relative_path",
            "count",
        ],
    )

    writer.writeheader()
    writer.writerows(
        token_rows
    )


summary = {
    "schema":
        "phyguard.per_indicator_ablation_protocol_summary.v1",

    "status":
        status,

    "design":
        "2x2 factorial",

    "variants":
        [
            variant["variant_id"]
            for variant in variants
        ],

    "source_packet_error_indicator":
        "per_proxy",

    "sionna_packet_error_indicator":
        "bler_post_ldpc",

    "ber_retained_in_all_variants":
        True,

    "frozen_sionna_sample_count":
        825,

    "repeat_bundles":
        5,

    "primary_endpoint":
        "selected_accuracy",

    "practical_retention_margin":
        -0.05,

    "bootstrap_replicates":
        10000,

    "token_hit_counts":
        token_hit_counts,

    "protocol_path":
        str(
            PROTOCOL_PATH.relative_to(
                ROOT
            )
        ),

    "protocol_sha256":
        sha256_file(
            PROTOCOL_PATH
        ),

    "token_audit_path":
        str(
            TOKEN_AUDIT_PATH.relative_to(
                ROOT
            )
        ),

    "token_audit_sha256":
        sha256_file(
            TOKEN_AUDIT_PATH
        ),

    "methodological_boundary":
        protocol[
            "methodological_boundary"
        ],
}

SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


ready_text = """# Stage 62A — Packet/block-error indicator ablation protocol

Status: **LOCKED BEFORE EXECUTION**

## Fixed 2×2 design

| Variant | Raw packet/block-error feature | Physics-score term | BER |
|---|---:|---:|---|
| FULL | present | present | unchanged |
| NO_PER_RAW | absent | present | unchanged |
| NO_PER_SCORE | present | absent | unchanged |
| NO_ERROR_INDICATOR | absent | absent | unchanged |

`NO_ERROR_INDICATOR` means the packet/block-error indicator is absent from both
paths. It does **not** remove BER.

## Primary endpoint

Selected accuracy, reported as paired absolute difference from FULL. Practical
retention is declared only when the lower bound of the paired 95% bootstrap
interval exceeds -0.05. The five locked repeat bundles and frozen 825-sample
Sionna membership must be reused.

## Scientific interpretation

The factorial design separates dependence through the raw statistical branch
from dependence through the physics-score branch. It directly tests whether
the revision-time 48-bit uncoded packet-error proxy is an essential component,
while preserving the independently measured BER evidence channel.

## Locked boundary

No KPI values were opened, no models were trained, no predictions were
generated, and no performance results were viewed when this protocol was
created.
"""

READY_TEXT_PATH.write_text(
    ready_text,
    encoding="utf-8",
)


manifest = {
    "schema":
        "phyguard.per_indicator_ablation_protocol_manifest.v1",

    "status":
        status,

    "protocol_sha256":
        sha256_file(
            PROTOCOL_PATH
        ),

    "summary_sha256":
        sha256_file(
            SUMMARY_PATH
        ),

    "token_audit_sha256":
        sha256_file(
            TOKEN_AUDIT_PATH
        ),

    "ready_text_sha256":
        sha256_file(
            READY_TEXT_PATH
        ),

    "locked_assets":
        {
            str(
                path.relative_to(ROOT)
            ):
                EXPECTED_SHA256[
                    str(path)
                ]
            for path in required_paths
        },

    "methodological_boundary":
        protocol[
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


print("status:", status)
print("design: 2x2 factorial")
print(
    "variants:",
    ", ".join(
        variant["variant_id"]
        for variant in variants
    ),
)
print("ber_retained_in_all_variants: True")
print("frozen_sionna_sample_count: 825")
print("repeat_bundles: 5")
print("primary_endpoint: selected_accuracy")
print("practical_retention_margin: -0.05")
print("bootstrap_replicates: 10000")
print("token_hit_counts:", token_hit_counts)
print("kpi_values_opened: False")
print("models_trained: False")
print("predictions_generated: False")
print("performance_metrics_computed: False")
print("protocol:", PROTOCOL_PATH)
print("summary:", SUMMARY_PATH)
print("token_audit:", TOKEN_AUDIT_PATH)
print("ready_text:", READY_TEXT_PATH)
print("manifest:", MANIFEST_PATH)
print()
print(
    "PER_INDICATOR_ABLATION_PROTOCOL_V1_PASS"
)
