#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

RESULT_ROOT = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_statistics_v2"
)

PROTOCOL_ROOT = (
    ROOT
    / "artifacts"
    / "cross_system_reliability_statistical_protocol_v2"
)

PROTOCOL_CONFIG = (
    ROOT
    / "configs"
    / "cross_system_reliability_statistical_protocol_v2.json"
)

PROTOCOL_MANIFEST = (
    ROOT
    / "manifests"
    / "cross_system_reliability_statistical_protocol_v2.json"
)

RESULT_MANIFEST = (
    ROOT
    / "manifests"
    / "cross_system_reliability_statistics_v2.json"
)

EXECUTION_SCRIPT = (
    ROOT
    / "scripts"
    / "60e2v2_run_cross_system_reliability_statistics.py"
)

PROTOCOL_AMENDMENT_SCRIPT = (
    ROOT
    / "scripts"
    / "60e1a_amend_zenodo_continuous_mcs_protocol.py"
)

EXECUTION_LOG = (
    ROOT
    / "logs"
    / "60e2v2_run_cross_system_reliability_statistics.log"
)

OUTPUT_ZIP = (
    ROOT
    / "results"
    / "phyguard_cross_system_reliability_statistics_v2_final.zip"
)

OUTPUT_SHA = OUTPUT_ZIP.with_suffix(
    ".zip.sha256"
)

EXPECTED_RESULT_FILES = [
    "source_sequence_effects.csv",
    "sionna_sequence_effects.csv",
    "sionna_subgroup_results.csv",
    "zenodo_file_effects.csv",
    "zenodo_secondary_results.csv",
    "statistical_distributions.npz",
    "summary.json",
]

EXPECTED = {
    "source": {
        "eligible": 1037,
        "median": -0.517916,
        "ci_low": -0.539107,
        "ci_high": -0.500990,
        "support": True,
    },
    "sionna": {
        "eligible": 541,
        "median": -0.214320,
        "ci_low": -0.250040,
        "ci_high": -0.174045,
        "support": True,
    },
    "zenodo": {
        "eligible_files": 10,
        "aggregate": -0.238660,
        "ci_low": -0.322807,
        "ci_high": -0.122197,
        "exact_p": 0.00390625,
        "negative_files": 9,
        "support": True,
    },
    "classification":
        "FULL_DIRECTIONAL_SUPPORT",
}


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def close(
    actual: float,
    expected: float,
    tolerance: float = 5e-7,
) -> bool:
    return abs(actual - expected) <= tolerance


def extract_section(
    text: str,
    start: str,
    end: str,
) -> str:
    pattern = re.compile(
        re.escape(start)
        + r"\n(.*?)\n"
        + re.escape(end),
        flags=re.DOTALL,
    )

    match = pattern.search(text)

    if match is None:
        raise RuntimeError(
            f"Could not extract log section: {start}"
        )

    return match.group(1)


def parse_value(
    section: str,
    key: str,
) -> str:
    match = re.search(
        rf"^{re.escape(key)}:\s*(.+?)\s*$",
        section,
        flags=re.MULTILINE,
    )

    if match is None:
        raise RuntimeError(
            f"Missing key in log section: {key}"
        )

    return match.group(1)


required_paths = [
    RESULT_ROOT,
    PROTOCOL_ROOT,
    PROTOCOL_CONFIG,
    PROTOCOL_MANIFEST,
    RESULT_MANIFEST,
    EXECUTION_SCRIPT,
    PROTOCOL_AMENDMENT_SCRIPT,
    EXECUTION_LOG,
]

for path in required_paths:
    require(
        path.exists(),
        f"Required path is missing: {path}",
    )

for filename in EXPECTED_RESULT_FILES:
    path = RESULT_ROOT / filename
    require(
        path.exists(),
        f"Expected result file is missing: {path}",
    )

require(
    not OUTPUT_ZIP.exists(),
    f"Final archive already exists: {OUTPUT_ZIP}",
)

require(
    not OUTPUT_SHA.exists(),
    f"Final SHA file already exists: {OUTPUT_SHA}",
)


log_text = EXECUTION_LOG.read_text(
    encoding="utf-8",
    errors="replace",
)

require(
    "CROSS_SYSTEM_RELIABILITY_STATISTICS_V2_PASS"
    in log_text,
    "Stage 60E2-v2 PASS marker is absent from the execution log.",
)


source_section = extract_section(
    log_text,
    "SOURCE RESULT",
    "SIONNA RESULT",
)

sionna_section = extract_section(
    log_text,
    "SIONNA RESULT",
    "SIONNA SUBGROUPS",
)

zenodo_section = extract_section(
    log_text,
    "ZENODO RESULT",
    "ZENODO FILE EFFECTS",
)

cross_section = extract_section(
    log_text,
    "CROSS-SYSTEM RESULT",
    "summary:",
)


source_actual = {
    "eligible":
        int(
            parse_value(
                source_section,
                "eligible_sequence_count",
            )
        ),

    "median":
        float(
            parse_value(
                source_section,
                "median_effect",
            )
        ),

    "support":
        (
            parse_value(
                source_section,
                "support_rule_satisfied",
            )
            == "True"
        ),
}

source_ci = re.search(
    r"bootstrap_ci95:\s*\('([^']+)',\s*'([^']+)'\)",
    source_section,
)

require(
    source_ci is not None,
    "Source bootstrap interval was not found.",
)

source_actual["ci_low"] = float(
    source_ci.group(1)
)
source_actual["ci_high"] = float(
    source_ci.group(2)
)


sionna_actual = {
    "eligible":
        int(
            parse_value(
                sionna_section,
                "eligible_sequence_count",
            )
        ),

    "median":
        float(
            parse_value(
                sionna_section,
                "median_effect",
            )
        ),

    "support":
        (
            parse_value(
                sionna_section,
                "support_rule_satisfied",
            )
            == "True"
        ),
}

sionna_ci = re.search(
    r"bootstrap_ci95:\s*\('([^']+)',\s*'([^']+)'\)",
    sionna_section,
)

require(
    sionna_ci is not None,
    "Sionna bootstrap interval was not found.",
)

sionna_actual["ci_low"] = float(
    sionna_ci.group(1)
)
sionna_actual["ci_high"] = float(
    sionna_ci.group(2)
)


zenodo_actual = {
    "eligible_files":
        int(
            parse_value(
                zenodo_section,
                "eligible_file_count",
            )
        ),

    "aggregate":
        float(
            parse_value(
                zenodo_section,
                "aggregate_effect",
            )
        ),

    "exact_p":
        float(
            parse_value(
                zenodo_section,
                "exact_sign_flip_p",
            )
        ),

    "negative_files":
        int(
            parse_value(
                zenodo_section,
                "negative_file_count",
            )
        ),

    "support":
        (
            parse_value(
                zenodo_section,
                "support_rule_satisfied",
            )
            == "True"
        ),
}

zenodo_ci = re.search(
    r"bootstrap_ci95:\s*\('([^']+)',\s*'([^']+)'\)",
    zenodo_section,
)

require(
    zenodo_ci is not None,
    "Zenodo bootstrap interval was not found.",
)

zenodo_actual["ci_low"] = float(
    zenodo_ci.group(1)
)
zenodo_actual["ci_high"] = float(
    zenodo_ci.group(2)
)


classification = parse_value(
    cross_section,
    "classification",
)


require(
    source_actual["eligible"]
    == EXPECTED["source"]["eligible"],
    "Unexpected source eligible count.",
)

require(
    close(
        source_actual["median"],
        EXPECTED["source"]["median"],
    ),
    "Unexpected source median effect.",
)

require(
    close(
        source_actual["ci_low"],
        EXPECTED["source"]["ci_low"],
    )
    and close(
        source_actual["ci_high"],
        EXPECTED["source"]["ci_high"],
    ),
    "Unexpected source confidence interval.",
)

require(
    source_actual["support"] is True,
    "Source support rule is not satisfied.",
)


require(
    sionna_actual["eligible"]
    == EXPECTED["sionna"]["eligible"],
    "Unexpected Sionna eligible count.",
)

require(
    close(
        sionna_actual["median"],
        EXPECTED["sionna"]["median"],
    ),
    "Unexpected Sionna median effect.",
)

require(
    close(
        sionna_actual["ci_low"],
        EXPECTED["sionna"]["ci_low"],
    )
    and close(
        sionna_actual["ci_high"],
        EXPECTED["sionna"]["ci_high"],
    ),
    "Unexpected Sionna confidence interval.",
)

require(
    sionna_actual["support"] is True,
    "Sionna support rule is not satisfied.",
)


require(
    zenodo_actual["eligible_files"]
    == EXPECTED["zenodo"]["eligible_files"],
    "Unexpected Zenodo eligible file count.",
)

require(
    close(
        zenodo_actual["aggregate"],
        EXPECTED["zenodo"]["aggregate"],
    ),
    "Unexpected Zenodo aggregate effect.",
)

require(
    close(
        zenodo_actual["ci_low"],
        EXPECTED["zenodo"]["ci_low"],
    )
    and close(
        zenodo_actual["ci_high"],
        EXPECTED["zenodo"]["ci_high"],
    ),
    "Unexpected Zenodo confidence interval.",
)

require(
    close(
        zenodo_actual["exact_p"],
        EXPECTED["zenodo"]["exact_p"],
    ),
    "Unexpected Zenodo exact p-value.",
)

require(
    zenodo_actual["negative_files"]
    == EXPECTED["zenodo"]["negative_files"],
    "Unexpected Zenodo negative-file count.",
)

require(
    zenodo_actual["support"] is True,
    "Zenodo support rule is not satisfied.",
)

require(
    classification
    == EXPECTED["classification"],
    "Cross-system classification mismatch.",
)


summary = json.loads(
    (
        RESULT_ROOT
        / "summary.json"
    ).read_text(
        encoding="utf-8"
    )
)

zenodo_rows = read_csv(
    RESULT_ROOT
    / "zenodo_file_effects.csv"
)

secondary_rows = read_csv(
    RESULT_ROOT
    / "zenodo_secondary_results.csv"
)

subgroup_rows = read_csv(
    RESULT_ROOT
    / "sionna_subgroup_results.csv"
)

require(
    len(zenodo_rows) == 10,
    "Zenodo file-effect table must contain ten rows.",
)

negative_count_from_csv = sum(
    1
    for row in zenodo_rows
    if float(row["primary_effect"]) < 0
)

require(
    negative_count_from_csv == 9,
    "Zenodo CSV negative-file count mismatch.",
)


freeze_record = {
    "schema":
        "phyguard.cross_system_reliability_statistics_freeze.v2",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "FROZEN",

    "classification":
        classification,

    "source":
        source_actual,

    "sionna":
        sionna_actual,

    "zenodo":
        zenodo_actual,

    "claim_boundary": {
        "supported":
            (
                "Trend-level directional reliability evidence "
                "is supported across the revision reconstruction, "
                "coded Sionna simulation, and real O-RAN telemetry."
            ),

        "not_supported":
            (
                "The analysis does not establish numerical "
                "calibration equivalence between the uncoded "
                "48-bit packet-error proxy and coded BLER."
            ),

        "mechanism_heterogeneity":
            (
                "Sionna aggregate support is mechanism-dependent: "
                "interference and blockage show negative median "
                "effects, whereas mobility and adaptation mismatch "
                "do not."
            ),

        "controls":
            (
                "Control sequences were ineligible for the "
                "within-MCS endpoint and must not be described "
                "as either positive or negative support."
            ),

        "zenodo_file_heterogeneity":
            (
                "Nine of ten Zenodo recordings show the expected "
                "negative effect; one recording shows a positive "
                "effect."
            ),
    },

    "frozen_source_files": [],
}


manuscript_text = f"""# Cross-system reliability validation

## Frozen statistical result

The preregistered directional endpoint was supported in all three systems. In
the revision-time source reconstruction, the median within-MCS association
between SINR and the uncoded 48-bit packet-error proxy was
{source_actual['median']:.3f} across {source_actual['eligible']} eligible
sequences, with a 95% stratified-bootstrap interval of
[{source_actual['ci_low']:.3f}, {source_actual['ci_high']:.3f}]. In the coded
Sionna evaluation, the corresponding SINR–post-LDPC-BLER association was
{sionna_actual['median']:.3f} across {sionna_actual['eligible']} eligible
sequences, with a 95% interval of
[{sionna_actual['ci_low']:.3f}, {sionna_actual['ci_high']:.3f}].

For the ten real O-RAN telemetry recordings, `phy_mcs` was a continuous or
aggregated telemetry quantity rather than an integer index. Following the
encoding amendment locked before inspecting the SINR–BLER association, we used
a within-file partial Spearman correlation controlling continuous `phy_mcs`.
The equal-weight Fisher-z aggregate was {zenodo_actual['aggregate']:.3f}
(95% file-cluster bootstrap interval
[{zenodo_actual['ci_low']:.3f}, {zenodo_actual['ci_high']:.3f}];
exact one-sided sign-flip p={zenodo_actual['exact_p']:.8f}), with the expected
negative direction in {zenodo_actual['negative_files']} of 10 recordings.

## Required interpretation boundary

These findings support the **directional, trend-level reliability role** of the
error indicator across reconstructed uncoded telemetry, coded simulation, and
real O-RAN measurements. They do **not** establish numerical calibration
equivalence between the 48-bit uncoded packet-error proxy and coded BLER.

The coded Sionna result is heterogeneous by mechanism. Interference and
blockage showed negative median effects, while mobility and adaptation mismatch
did not. Control sequences were ineligible for the within-MCS endpoint and
therefore must not be counted as either supporting or contradicting the
directional claim.

## Reviewer-response wording

We agree that the 48-bit quantity should not be presented as a calibrated coded
PER. We therefore relabel it as a revision-time locked **uncoded 48-bit
packet-error proxy** and added a preregistered cross-system directional
validation. The expected negative SINR–error relationship was supported in the
source reconstruction, in coded Sionna post-LDPC BLER, and in ten real O-RAN
telemetry recordings after controlling continuous MCS. The real-data aggregate
partial Spearman effect was {zenodo_actual['aggregate']:.3f}
(95% CI [{zenodo_actual['ci_low']:.3f}, {zenodo_actual['ci_high']:.3f}]),
with 9/10 recordings in the expected direction and an exact one-sided
sign-flip p-value of {zenodo_actual['exact_p']:.8f}. We have limited the claim
accordingly: the proxy captures a reproducible trend-level reliability signal,
but it is not numerically interchangeable with coded BLER.
"""


with tempfile.TemporaryDirectory(
    prefix="stage60e3_",
    dir=str(ROOT / "results"),
) as temporary_directory:
    stage = Path(temporary_directory)

    (stage / "artifacts").mkdir(
        parents=True,
        exist_ok=True,
    )

    (stage / "configs").mkdir(
        parents=True,
        exist_ok=True,
    )

    (stage / "manifests").mkdir(
        parents=True,
        exist_ok=True,
    )

    (stage / "scripts").mkdir(
        parents=True,
        exist_ok=True,
    )

    (stage / "logs").mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copytree(
        RESULT_ROOT,
        stage
        / "artifacts"
        / RESULT_ROOT.name,
    )

    shutil.copytree(
        PROTOCOL_ROOT,
        stage
        / "artifacts"
        / PROTOCOL_ROOT.name,
    )

    for source, destination in [
        (
            PROTOCOL_CONFIG,
            stage / "configs" / PROTOCOL_CONFIG.name,
        ),
        (
            PROTOCOL_MANIFEST,
            stage / "manifests" / PROTOCOL_MANIFEST.name,
        ),
        (
            RESULT_MANIFEST,
            stage / "manifests" / RESULT_MANIFEST.name,
        ),
        (
            EXECUTION_SCRIPT,
            stage / "scripts" / EXECUTION_SCRIPT.name,
        ),
        (
            PROTOCOL_AMENDMENT_SCRIPT,
            stage / "scripts" / PROTOCOL_AMENDMENT_SCRIPT.name,
        ),
        (
            EXECUTION_LOG,
            stage / "logs" / EXECUTION_LOG.name,
        ),
    ]:
        shutil.copy2(
            source,
            destination,
        )

    (stage / "FREEZE_SUMMARY.json").write_text(
        json.dumps(
            freeze_record,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (stage / "MANUSCRIPT_READY_RESULTS.md").write_text(
        manuscript_text,
        encoding="utf-8",
    )

    checksums = []

    for path in sorted(stage.rglob("*")):
        if path.is_file():
            relative = path.relative_to(stage)
            digest = sha256_file(path)

            checksums.append(
                f"{digest}  {relative.as_posix()}"
            )

            freeze_record[
                "frozen_source_files"
            ].append(
                {
                    "path":
                        relative.as_posix(),

                    "sha256":
                        digest,

                    "size_bytes":
                        path.stat().st_size,
                }
            )

    (stage / "FREEZE_SUMMARY.json").write_text(
        json.dumps(
            freeze_record,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    checksums = []

    for path in sorted(stage.rglob("*")):
        if (
            path.is_file()
            and path.name != "SHA256SUMS.txt"
        ):
            checksums.append(
                f"{sha256_file(path)}  "
                f"{path.relative_to(stage).as_posix()}"
            )

    (stage / "SHA256SUMS.txt").write_text(
        "\n".join(checksums) + "\n",
        encoding="utf-8",
    )

    OUTPUT_ZIP.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        OUTPUT_ZIP,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    path.relative_to(stage),
                )


archive_sha256 = sha256_file(
    OUTPUT_ZIP
)

OUTPUT_SHA.write_text(
    f"{archive_sha256}  {OUTPUT_ZIP.name}\n",
    encoding="utf-8",
)


print("classification:", classification)
print(
    "source:",
    source_actual,
)
print(
    "sionna:",
    sionna_actual,
)
print(
    "zenodo:",
    zenodo_actual,
)
print(
    "zenodo_negative_files_from_csv:",
    negative_count_from_csv,
)
print(
    "archive:",
    OUTPUT_ZIP,
)
print(
    "archive_size_bytes:",
    OUTPUT_ZIP.stat().st_size,
)
print(
    "archive_sha256:",
    archive_sha256,
)
print(
    "sha_file:",
    OUTPUT_SHA,
)
print()
print(
    "CROSS_SYSTEM_RELIABILITY_STATISTICS_"
    "V2_FINAL_FREEZE_PASS"
)
