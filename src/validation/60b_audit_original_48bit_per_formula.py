import hashlib
import json
import re
import zipfile
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "original_48bit_per_formula_audit_v1"
)

HITS_PATH = OUTPUT_ROOT / "ranked_formula_hits.txt"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "original_48bit_per_formula_audit_v1.json"
)

KPI_CONTRACT_SOURCE = (
    ROOT
    / "scripts"
    / "01_lock_kpi_contract.py"
)

TEXT_EXTENSIONS = {
    ".py", ".txt", ".tex", ".md", ".json",
    ".yaml", ".yml", ".toml", ".ipynb",
}

SEARCH_ROOTS = [
    ROOT / "scripts",
    ROOT / "configs",
    ROOT / "src",
    ROOT / "paper_assets",
    ROOT / "manuscript",
    ROOT / "artifacts" / "uploaded_phyguard_packages",
]

SKIP_PART_TOKENS = {
    "downloads",
    "quarantine",
    "__pycache__",
    ".git",
    "kuleuven_scenario1_locked_download_v1",
    "public_real_measurement_package_structure_audit_v1_failed",
}

PATTERN = re.compile(
    r"("
    r"48[_ -]?(?:bit|bits)|"
    r"packet[_ -]?(?:error|length|size|bits)|"
    r"\bper\b|"
    r"\bber\b|"
    r"reshape\s*\([^)]*48|"
    r"1\s*-\s*\([^)]*ber[^)]*\)\s*\*\*\s*48|"
    r"1\s*-\s*\([^)]*ber[^)]*\)\s*\^\s*48|"
    r"np\.any|"
    r"\.any\s*\("
    r")",
    flags=re.IGNORECASE,
)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def skipped(path):
    return any(
        token in path.parts
        for token in SKIP_PART_TOKENS
    )


def score_context(text):
    lower = text.lower()
    score = 0

    has_48 = "48" in lower
    has_packet = "packet" in lower
    has_per = re.search(r"\bper\b", lower) is not None
    has_ber = re.search(r"\bber\b", lower) is not None

    if has_48 and has_packet:
        score += 60

    if has_48 and has_per:
        score += 45

    if has_packet and has_per:
        score += 30

    if "reshape" in lower and "48" in lower:
        score += 40

    if ("np.any" in lower or ".any(" in lower) and has_packet:
        score += 35

    if "1-" in lower.replace(" ", "") and has_ber and "48" in lower:
        score += 50

    if "packet_error" in lower or "packet error" in lower:
        score += 25

    if "surrogate" in lower:
        score += 15

    if any(
        token in lower
        for token in [
            "def ",
            "return ",
            "per =",
            "per=",
            "packet_len",
            "packet_size",
        ]
    ):
        score += 12

    return score


def collect_hits_from_lines(origin, lines, source_type):
    hits = []

    for index, line in enumerate(lines):
        if PATTERN.search(line) is None:
            continue

        start = max(0, index - 5)
        end = min(len(lines), index + 6)

        context_lines = lines[start:end]
        context_text = "\n".join(context_lines)
        score = score_context(context_text)

        hits.append(
            {
                "score": score,
                "source_type": source_type,
                "origin": origin,
                "match_line": index + 1,
                "context_start": start + 1,
                "context_end": end,
                "context": [
                    {
                        "line_number": line_index + 1,
                        "text": lines[line_index],
                    }
                    for line_index in range(start, end)
                ],
            }
        )

    return hits


if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Stage 60B output already exists; refusing to overwrite."
    )

if not KPI_CONTRACT_SOURCE.exists():
    raise FileNotFoundError(KPI_CONTRACT_SOURCE)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)


hits = []
plain_file_count = 0
zip_file_count = 0
zip_member_count = 0


for search_root in SEARCH_ROOTS:
    if not search_root.exists():
        continue

    for path in search_root.rglob("*"):
        if not path.is_file():
            continue

        if skipped(path):
            continue

        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        if path.stat().st_size > 25 * 1024 * 1024:
            continue

        try:
            lines = path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except Exception:
            continue

        plain_file_count += 1

        hits.extend(
            collect_hits_from_lines(
                origin=str(path.relative_to(ROOT)),
                lines=lines,
                source_type="plain_text_file",
            )
        )


zip_candidates = []

uploaded_root = (
    ROOT
    / "artifacts"
    / "uploaded_phyguard_packages"
)

if uploaded_root.exists():
    zip_candidates.extend(
        uploaded_root.rglob("*.zip")
    )

zip_candidates.extend(
    ROOT.glob("*.zip")
)


seen_zip_paths = set()

for zip_path in sorted(zip_candidates):
    resolved = str(zip_path.resolve())

    if resolved in seen_zip_paths:
        continue

    seen_zip_paths.add(resolved)

    if zip_path.stat().st_size > 500 * 1024 * 1024:
        continue

    try:
        archive = zipfile.ZipFile(
            zip_path,
            mode="r",
        )
    except Exception:
        continue

    zip_file_count += 1

    with archive:
        for member in archive.infolist():
            member_path = Path(member.filename)

            if member.is_dir():
                continue

            if member_path.suffix.lower() not in TEXT_EXTENSIONS:
                continue

            if member.file_size > 25 * 1024 * 1024:
                continue

            if any(
                token in member_path.parts
                for token in SKIP_PART_TOKENS
            ):
                continue

            try:
                raw = archive.read(member)
                text = raw.decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                continue

            zip_member_count += 1

            hits.extend(
                collect_hits_from_lines(
                    origin=(
                        str(zip_path.relative_to(ROOT))
                        + "::"
                        + member.filename
                    ),
                    lines=text.splitlines(),
                    source_type="zip_member",
                )
            )


hits.sort(
    key=lambda item: (
        -item["score"],
        item["origin"],
        item["match_line"],
    )
)


deduplicated_hits = []
seen_contexts = set()

for hit in hits:
    key = (
        hit["origin"],
        hit["context_start"],
        hit["context_end"],
    )

    if key in seen_contexts:
        continue

    seen_contexts.add(key)
    deduplicated_hits.append(hit)


top_hits = deduplicated_hits[:120]


with HITS_PATH.open(
    "w",
    encoding="utf-8",
) as stream:
    for rank, hit in enumerate(
        top_hits,
        start=1,
    ):
        stream.write("=" * 100 + "\n")
        stream.write(
            f"HIT {rank:03d} | score={hit['score']} "
            f"| type={hit['source_type']} "
            f"| {hit['origin']}:{hit['match_line']}\n"
        )
        stream.write("=" * 100 + "\n")

        for row in hit["context"]:
            marker = (
                ">>>"
                if row["line_number"] == hit["match_line"]
                else "   "
            )

            stream.write(
                f"{marker} {row['line_number']:05d}: "
                f"{row['text']}\n"
            )

        stream.write("\n")


contract_lines = KPI_CONTRACT_SOURCE.read_text(
    encoding="utf-8",
    errors="replace",
).splitlines()

contract_context = [
    {
        "line_number": index + 1,
        "text": contract_lines[index],
    }
    for index in range(
        max(0, 114),
        min(len(contract_lines), 145),
    )
]


strong_formula_hits = [
    hit
    for hit in top_hits
    if hit["score"] >= 70
]


summary = {
    "schema":
        "phyguard.original_48bit_per_formula_audit.v1",

    "status":
        "PASS",

    "plain_text_file_count":
        plain_file_count,

    "zip_file_count":
        zip_file_count,

    "zip_text_member_count":
        zip_member_count,

    "deduplicated_hit_count":
        len(deduplicated_hits),

    "reported_hit_count":
        len(top_hits),

    "strong_formula_hit_count":
        len(strong_formula_hits),

    "formula_resolution_status":
        (
            "CANDIDATE_IMPLEMENTATION_FOUND"
            if strong_formula_hits
            else "EXECUTABLE_FORMULA_NOT_YET_FOUND"
        ),

    "kpi_contract_source":
        str(
            KPI_CONTRACT_SOURCE.relative_to(ROOT)
        ),

    "kpi_contract_context_lines_115_145":
        contract_context,

    "hits_path":
        str(HITS_PATH),

    "hits_sha256":
        sha256_file(HITS_PATH),

    "methodological_boundary": {
        "zenodo_data_rows_read":
            False,

        "zenodo_values_analyzed":
            False,

        "surrogate_formula_changed":
            False,

        "formula_inferred_without_source":
            False,

        "models_trained":
            False,

        "performance_computed":
            False,

        "thresholds_selected":
            False,
    },
}


SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


manifest = {
    "schema":
        "phyguard.original_48bit_per_formula_audit_manifest.v1",

    "status":
        "PASS",

    "summary_sha256":
        sha256_file(SUMMARY_PATH),

    "hits_sha256":
        sha256_file(HITS_PATH),

    "formula_resolution_status":
        summary["formula_resolution_status"],

    "methodological_boundary":
        summary["methodological_boundary"],
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


print("status: PASS")
print("plain_text_file_count:", plain_file_count)
print("zip_file_count:", zip_file_count)
print("zip_text_member_count:", zip_member_count)
print(
    "deduplicated_hit_count:",
    len(deduplicated_hits),
)
print(
    "strong_formula_hit_count:",
    len(strong_formula_hits),
)
print(
    "formula_resolution_status:",
    summary["formula_resolution_status"],
)
print("zenodo_data_rows_read: False")
print("zenodo_values_analyzed: False")
print("formula_inferred_without_source: False")
print("models_trained: False")
print("performance_computed: False")
print("thresholds_selected: False")

print("\nKPI CONTRACT SOURCE LINES 115-145")
for row in contract_context:
    print(
        f"{row['line_number']:04d}: "
        f"{row['text']}"
    )

print("\nTOP FORMULA CANDIDATES")
for hit in top_hits[:40]:
    print(
        f"score={hit['score']:03d}",
        "|",
        hit["source_type"],
        "|",
        f"{hit['origin']}:{hit['match_line']}",
    )

    for row in hit["context"]:
        if (
            "48" in row["text"]
            or "packet" in row["text"].lower()
            or re.search(
                r"\b(?:per|ber)\b",
                row["text"],
                flags=re.IGNORECASE,
            )
        ):
            print(
                f"    {row['line_number']:05d}: "
                f"{row['text'].strip()}"
            )

print("\nhits:", HITS_PATH)
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)
print(
    "\nORIGINAL_48BIT_PER_FORMULA_AUDIT_V1_PASS"
)
