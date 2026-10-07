import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/root/phyguard_revision")

SEMANTIC_ROOT = (
    ROOT
    / "artifacts"
    / "public_real_measurement_semantic_audit_v1"
)

DOCUMENTATION_PATH = (
    SEMANTIC_ROOT
    / "kuleuven_documentation_combined.txt"
)

INVENTORY_PATH = (
    ROOT
    / "artifacts"
    / "public_real_measurement_package_structure_audit_v1"
    / "kuleuven_file_inventory.csv"
)

PARENT_MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "public_real_measurement_semantic_audit_v1.json"
)

OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "kuleuven_label_binary_contract_audit_v1"
)

MATCHED_LINES_PATH = (
    OUTPUT_ROOT
    / "documentation_relevant_lines.txt"
)

FILENAME_MATRIX_PATH = (
    OUTPUT_ROOT
    / "filename_scenario_activity_matrix.csv"
)

UNMATCHED_BINARIES_PATH = (
    OUTPUT_ROOT
    / "unmatched_binary_filenames.txt"
)

SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "kuleuven_label_binary_contract_audit_v1.json"
)


KEYWORDS = [
    "scenario",
    "activity",
    "blockage",
    "human",
    "binary",
    "format",
    "byte",
    "float",
    "double",
    "complex",
    "real",
    "imaginary",
    "iq",
    "ofdm",
    "subcarrier",
    "symbol",
    "channel",
    "measurement",
    "duration",
    "sampling",
    "converter",
    "parse",
]


FILENAME_PATTERN = re.compile(
    r"^comm_sym(?P<symbols>\d+)"
    r"_int(?P<interval>\d+)"
    r"_(?P<duration>\d+)s"
    r"_scen(?P<scenario>\d+)"
    r"_activity(?P<activity>\d+)"
    r"_rep(?P<repetition>\d+)"
    r"\.bin$",
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


required_paths = [
    DOCUMENTATION_PATH,
    INVENTORY_PATH,
    PARENT_MANIFEST_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)


parent_manifest = json.loads(
    PARENT_MANIFEST_PATH.read_text(
        encoding="utf-8"
    )
)

if parent_manifest.get("status") != "PASS":
    raise RuntimeError(
        "Parent semantic audit is not PASS."
    )


if OUTPUT_ROOT.exists():
    raise RuntimeError(
        "Stage 58F output already exists; "
        "refusing to overwrite it."
    )

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)


documentation = DOCUMENTATION_PATH.read_text(
    encoding="utf-8",
    errors="replace",
)

lines = documentation.splitlines()

relevant_blocks = []

used_line_numbers = set()


for line_index, line in enumerate(
    lines,
    start=1,
):
    lower = line.lower()

    if not any(
        keyword in lower
        for keyword in KEYWORDS
    ):
        continue

    start = max(
        1,
        line_index - 3,
    )

    end = min(
        len(lines),
        line_index + 3,
    )

    block_numbers = set(
        range(
            start,
            end + 1,
        )
    )

    if block_numbers.issubset(
        used_line_numbers
    ):
        continue

    block = []

    for number in range(
        start,
        end + 1,
    ):
        block.append(
            f"{number:04d}: "
            f"{lines[number - 1]}"
        )

    relevant_blocks.append(
        "\n".join(block)
    )

    used_line_numbers.update(
        block_numbers
    )


MATCHED_LINES_PATH.write_text(
    "\n\n"
    + (
        "\n\n"
        + "=" * 88
        + "\n\n"
    ).join(
        relevant_blocks
    )
    + "\n",
    encoding="utf-8",
)


with INVENTORY_PATH.open(
    "r",
    encoding="utf-8",
    newline="",
) as stream:
    inventory = list(
        csv.DictReader(stream)
    )


binary_rows = [
    row
    for row in inventory
    if Path(
        row["filename"]
    ).suffix.lower() == ".bin"
]


parsed_rows = []

unmatched_names = []


for row in binary_rows:
    filename = str(
        row["filename"]
    )

    match = FILENAME_PATTERN.match(
        filename
    )

    if match is None:
        unmatched_names.append(
            filename
        )
        continue

    values = match.groupdict()

    parsed_rows.append(
        {
            "filename":
                filename,

            "directory_label":
                row.get(
                    "directory_label"
                ),

            "file_id":
                row.get(
                    "file_id"
                ),

            "size_bytes":
                int(
                    row.get(
                        "size_bytes"
                    )
                    or 0
                ),

            "symbols":
                int(
                    values["symbols"]
                ),

            "interval":
                int(
                    values["interval"]
                ),

            "duration_seconds":
                int(
                    values["duration"]
                ),

            "scenario":
                int(
                    values["scenario"]
                ),

            "activity":
                int(
                    values["activity"]
                ),

            "repetition":
                int(
                    values["repetition"]
                ),
        }
    )


UNMATCHED_BINARIES_PATH.write_text(
    "\n".join(
        sorted(
            unmatched_names
        )
    )
    + (
        "\n"
        if unmatched_names
        else ""
    ),
    encoding="utf-8",
)


combination_counter = Counter(
    (
        row["scenario"],
        row["activity"],
    )
    for row in parsed_rows
)

repetition_sets = defaultdict(
    set
)

size_sets = defaultdict(
    set
)

symbol_sets = defaultdict(
    set
)

interval_sets = defaultdict(
    set
)

duration_sets = defaultdict(
    set
)


for row in parsed_rows:
    key = (
        row["scenario"],
        row["activity"],
    )

    repetition_sets[key].add(
        row["repetition"]
    )

    size_sets[key].add(
        row["size_bytes"]
    )

    symbol_sets[key].add(
        row["symbols"]
    )

    interval_sets[key].add(
        row["interval"]
    )

    duration_sets[key].add(
        row["duration_seconds"]
    )


matrix_rows = []

for scenario, activity in sorted(
    combination_counter
):
    key = (
        scenario,
        activity,
    )

    matrix_rows.append(
        {
            "scenario":
                scenario,

            "activity":
                activity,

            "file_count":
                combination_counter[key],

            "repetitions":
                ",".join(
                    str(value)
                    for value in sorted(
                        repetition_sets[key]
                    )
                ),

            "unique_size_count":
                len(
                    size_sets[key]
                ),

            "minimum_size_bytes":
                min(
                    size_sets[key]
                ),

            "maximum_size_bytes":
                max(
                    size_sets[key]
                ),

            "symbols":
                ",".join(
                    str(value)
                    for value in sorted(
                        symbol_sets[key]
                    )
                ),

            "interval":
                ",".join(
                    str(value)
                    for value in sorted(
                        interval_sets[key]
                    )
                ),

            "duration_seconds":
                ",".join(
                    str(value)
                    for value in sorted(
                        duration_sets[key]
                    )
                ),
        }
    )


with FILENAME_MATRIX_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "scenario",
            "activity",
            "file_count",
            "repetitions",
            "unique_size_count",
            "minimum_size_bytes",
            "maximum_size_bytes",
            "symbols",
            "interval",
            "duration_seconds",
        ],
    )

    writer.writeheader()

    writer.writerows(
        matrix_rows
    )


scenario_counts = Counter(
    row["scenario"]
    for row in parsed_rows
)

activity_counts = Counter(
    (
        row["scenario"],
        row["activity"],
    )
    for row in parsed_rows
)

repetition_counts = Counter(
    row["repetition"]
    for row in parsed_rows
)

symbols_values = sorted(
    {
        row["symbols"]
        for row in parsed_rows
    }
)

interval_values = sorted(
    {
        row["interval"]
        for row in parsed_rows
    }
)

duration_values = sorted(
    {
        row["duration_seconds"]
        for row in parsed_rows
    }
)


documentation_signals = {
    keyword:
        sum(
            keyword in line.lower()
            for line in lines
        )
    for keyword in KEYWORDS
}


summary = {
    "schema":
        "phyguard.kuleuven."
        "label_binary_contract_audit.v1",

    "status":
        "PASS",

    "parent_manifest_path":
        str(
            PARENT_MANIFEST_PATH
        ),

    "parent_manifest_sha256":
        sha256_file(
            PARENT_MANIFEST_PATH
        ),

    "documentation": {
        "line_count":
            len(lines),

        "relevant_context_block_count":
            len(
                relevant_blocks
            ),

        "keyword_line_counts":
            documentation_signals,

        "activity_mapping_resolved":
            False,

        "binary_layout_resolved":
            False,

        "resolution_status":
            (
                "PENDING_MANUAL_REVIEW_OF_EXTRACTED_"
                "DOCUMENTATION_CONTEXTS"
            ),
    },

    "filename_contract": {
        "binary_inventory_count":
            len(
                binary_rows
            ),

        "filename_pattern_match_count":
            len(
                parsed_rows
            ),

        "filename_pattern_unmatched_count":
            len(
                unmatched_names
            ),

        "scenario_counts":
            {
                str(key):
                    int(value)
                for key, value
                in sorted(
                    scenario_counts.items()
                )
            },

        "scenario_activity_pair_count":
            len(
                activity_counts
            ),

        "repetition_counts":
            {
                str(key):
                    int(value)
                for key, value
                in sorted(
                    repetition_counts.items()
                )
            },

        "symbols_values":
            symbols_values,

        "interval_values":
            interval_values,

        "duration_values":
            duration_values,
    },

    "methodological_boundary": {
        "binary_measurement_files_opened":
            False,

        "models_trained":
            False,

        "labels_constructed":
            False,

        "performance_computed":
            False,

        "thresholds_selected":
            False,

        "datasets_selected_by_performance":
            False,

        "existing_sionna_assets_modified":
            False,

        "existing_locked_models_modified":
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


output_files = sorted(
    path
    for path in OUTPUT_ROOT.rglob("*")
    if path.is_file()
)


manifest = {
    "schema":
        "phyguard.kuleuven."
        "label_binary_contract_audit_manifest.v1",

    "status":
        "PASS",

    "binary_inventory_count":
        len(
            binary_rows
        ),

    "filename_pattern_match_count":
        len(
            parsed_rows
        ),

    "filename_pattern_unmatched_count":
        len(
            unmatched_names
        ),

    "scenario_activity_pair_count":
        len(
            activity_counts
        ),

    "methodological_boundary":
        summary[
            "methodological_boundary"
        ],

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
print("binary_measurement_files_opened: False")
print("models_trained: False")
print("labels_constructed: False")
print("performance_computed: False")
print("thresholds_selected: False")
print("datasets_selected_by_performance: False")

print("\nKU FILENAME CONTRACT")
print(
    "binary_inventory_count:",
    len(
        binary_rows
    ),
)
print(
    "filename_pattern_match_count:",
    len(
        parsed_rows
    ),
)
print(
    "filename_pattern_unmatched_count:",
    len(
        unmatched_names
    ),
)
print(
    "scenario_counts:",
    dict(
        sorted(
            scenario_counts.items()
        )
    ),
)
print(
    "scenario_activity_pair_count:",
    len(
        activity_counts
    ),
)
print(
    "repetition_counts:",
    dict(
        sorted(
            repetition_counts.items()
        )
    ),
)
print(
    "symbols_values:",
    symbols_values,
)
print(
    "interval_values:",
    interval_values,
)
print(
    "duration_values:",
    duration_values,
)

print("\nKU SCENARIO-ACTIVITY MATRIX")

for row in matrix_rows:
    print(
        f"scenario={row['scenario']}",
        f"activity={row['activity']}",
        f"files={row['file_count']}",
        f"repetitions={row['repetitions']}",
        f"size_range="
        f"{row['minimum_size_bytes']}-"
        f"{row['maximum_size_bytes']}",
    )

print("\nRELEVANT DOCUMENTATION CONTEXTS")
print(
    MATCHED_LINES_PATH.read_text(
        encoding="utf-8"
    )
)

print("\noutput_root:", OUTPUT_ROOT)
print(
    "documentation_contexts:",
    MATCHED_LINES_PATH,
)
print(
    "scenario_activity_matrix:",
    FILENAME_MATRIX_PATH,
)
print(
    "unmatched_binaries:",
    UNMATCHED_BINARIES_PATH,
)
print("summary:", SUMMARY_PATH)
print("manifest:", MANIFEST_PATH)

print(
    "\nKULEUVEN_LABEL_BINARY_"
    "CONTRACT_AUDIT_PASS"
)
