import hashlib
import io
import json
import re
import zipfile
from pathlib import Path


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

CAUSAL_MANIFEST = (
    ROOT
    / "manifests"
    / "sionna_formal_test840_causal_validation_v2.json"
)

OUTPUT_MANIFEST = (
    ROOT
    / "manifests"
    / "locked_reconstruction_contract_audit.json"
)

EXPECTED_EXTERNAL_ARCHIVE_SHA256 = (
    "238980731f09b758ef834e4961ba1427665a352aac279f164f02efba5a033aad"
)

EXPECTED_FORMAL_ARCHIVE_SHA256 = (
    "b6c5f400b7d55949fb1e1f39c0c092da8f780eed100866f30176e26e462ff2f5"
)

TEXT_EXTENSIONS = {
    ".json",
    ".py",
    ".txt",
    ".csv",
    ".md",
    ".yaml",
    ".yml",
}

SEARCH_TERMS = [
    "revision-time",
    "reconstruction",
    "locked",
    "gate",
    "resolver",
    "feature",
    "threshold",
    "selection",
    "abstain",
    "bundle",
    "predict",
    "sklearn",
    "19",
    "79",
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


def recursive_summary(value, path="$", output=None):
    if output is None:
        output = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"

            lower_key = str(key).lower()

            if any(
                token in lower_key
                for token in (
                    "threshold",
                    "feature",
                    "coef",
                    "intercept",
                    "class",
                    "label",
                    "scaler",
                    "gate",
                    "resolver",
                    "bundle",
                    "version",
                    "model",
                )
            ):
                if isinstance(child, list):
                    output.append(
                        {
                            "path": child_path,
                            "type": "list",
                            "length": len(child),
                            "preview": child[:5],
                        }
                    )

                elif isinstance(child, dict):
                    output.append(
                        {
                            "path": child_path,
                            "type": "dict",
                            "keys": list(child)[:15],
                        }
                    )

                else:
                    output.append(
                        {
                            "path": child_path,
                            "type": type(child).__name__,
                            "value": child,
                        }
                    )

            recursive_summary(
                child,
                child_path,
                output,
            )

    elif isinstance(value, list):
        if len(value) in {5, 19, 79}:
            output.append(
                {
                    "path": path,
                    "type": "list",
                    "length": len(value),
                    "preview": value[:5],
                }
            )

        for index, child in enumerate(value):
            if isinstance(child, (dict, list)):
                recursive_summary(
                    child,
                    f"{path}[{index}]",
                    output,
                )

    return output


for required in (
    EXTERNAL_ARCHIVE,
    FORMAL_ARCHIVE,
    CAUSAL_MANIFEST,
):
    if not required.exists():
        raise FileNotFoundError(required)


external_hash = sha256_file(
    EXTERNAL_ARCHIVE
)

formal_hash = sha256_file(
    FORMAL_ARCHIVE
)

if external_hash != EXPECTED_EXTERNAL_ARCHIVE_SHA256:
    raise RuntimeError(
        "Smoke-24 external-validation archive SHA256 mismatch."
    )

if formal_hash != EXPECTED_FORMAL_ARCHIVE_SHA256:
    raise RuntimeError(
        "Formal Test-840 final archive SHA256 mismatch."
    )


causal = load_json(
    CAUSAL_MANIFEST
)

if causal.get("causal_v2_passed_count") != 825:
    raise RuntimeError(
        "Formal causal-valid sample count is not 825."
    )

eligible_ids = [
    record["sample_id"]
    for record in causal.get("records", [])
    if record.get("causal_v2_status") == "PASS"
]

if len(eligible_ids) != 825:
    raise RuntimeError(
        f"Eligible-ID count={len(eligible_ids)}, expected 825."
    )


inventory = []
json_summaries = {}
script_matches = {}
candidate_model_entries = []


with zipfile.ZipFile(
    EXTERNAL_ARCHIVE,
    mode="r",
) as archive:
    bad_file = archive.testzip()

    if bad_file is not None:
        raise RuntimeError(
            f"External archive ZIP failure: {bad_file}"
        )

    for info in archive.infolist():
        if info.is_dir():
            continue

        data = archive.read(info.filename)

        suffix = Path(
            info.filename
        ).suffix.lower()

        record = {
            "entry": info.filename,
            "size_bytes": len(data),
            "sha256": sha256_bytes(data),
            "suffix": suffix,
        }

        inventory.append(record)

        lower_name = info.filename.lower()

        if (
            any(
                token in lower_name
                for token in (
                    "model",
                    "gate",
                    "resolver",
                    "bundle",
                    "classifier",
                    "pipeline",
                    "reconstruction",
                    "locked",
                )
            )
            or suffix in {
                ".joblib",
                ".pkl",
                ".pickle",
                ".npz",
            }
        ):
            candidate_model_entries.append(
                record
            )

        if suffix not in TEXT_EXTENSIONS:
            continue

        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                continue

        if suffix == ".json":
            try:
                value = json.loads(text)

                summary = recursive_summary(
                    value
                )

                if summary:
                    json_summaries[
                        info.filename
                    ] = summary

            except json.JSONDecodeError:
                pass

        matching_lines = []

        for line_number, line in enumerate(
            text.splitlines(),
            start=1,
        ):
            lower_line = line.lower()

            matched_terms = [
                term
                for term in SEARCH_TERMS
                if term.lower() in lower_line
            ]

            if matched_terms:
                matching_lines.append(
                    {
                        "line_number":
                            line_number,
                        "matched_terms":
                            matched_terms,
                        "text":
                            line[:500],
                    }
                )

        if matching_lines:
            script_matches[
                info.filename
            ] = matching_lines


output = {
    "schema":
        "phyguard.locked_reconstruction."
        "contract_audit.v1",

    "status": "PASS",

    "external_validation_archive": {
        "path": str(EXTERNAL_ARCHIVE),
        "sha256": external_hash,
        "entry_count": len(inventory),
        "zip_test": "PASS",
    },

    "formal_test_archive": {
        "path": str(FORMAL_ARCHIVE),
        "sha256": formal_hash,
    },

    "formal_evaluation_scope": {
        "causal_valid_sample_count": 825,
        "eligible_sample_ids_sha256":
            sha256_bytes(
                json.dumps(
                    sorted(eligible_ids),
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
    },

    "inventory":
        inventory,

    "candidate_model_entries":
        candidate_model_entries,

    "json_contract_summaries":
        json_summaries,

    "text_matches":
        script_matches,

    "safety": {
        "models_loaded": False,
        "pickle_deserialized": False,
        "formal_samples_modified": False,
        "training_performed": False,
        "threshold_tuning_performed": False,
        "model_selection_performed": False,
    },
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


print("status: PASS")
print(
    "external_archive_sha256:",
    external_hash,
)
print(
    "external_archive_entry_count:",
    len(inventory),
)
print(
    "formal_archive_sha256:",
    formal_hash,
)
print(
    "formal_causal_valid_count:",
    len(eligible_ids),
)
print(
    "candidate_model_entry_count:",
    len(candidate_model_entries),
)


print("\nZIP INVENTORY")

for record in inventory:
    print(
        record["entry"],
        "| size=",
        record["size_bytes"],
        "| sha256=",
        record["sha256"],
    )


print("\nCANDIDATE MODEL ENTRIES")

if not candidate_model_entries:
    print("<none by filename; inspect JSON/script summaries>")
else:
    for record in candidate_model_entries:
        print(
            record["entry"],
            "| suffix=",
            record["suffix"],
            "| sha256=",
            record["sha256"],
        )


print("\nJSON CONTRACT SUMMARY")

if not json_summaries:
    print("<none>")
else:
    for filename, entries in json_summaries.items():
        print(f"\n--- {filename} ---")

        for entry in entries:
            print(
                json.dumps(
                    entry,
                    ensure_ascii=False,
                )
            )


print("\nSCRIPT/TEXT MATCHES")

if not script_matches:
    print("<none>")
else:
    for filename, entries in script_matches.items():
        print(f"\n--- {filename} ---")

        for entry in entries:
            print(
                f"L{entry['line_number']:04d}",
                "|",
                entry["text"],
            )


print("\nmanifest:", OUTPUT_MANIFEST)

print(
    "\nLOCKED_RECONSTRUCTION_"
    "CONTRACT_AUDIT_PASS"
)
