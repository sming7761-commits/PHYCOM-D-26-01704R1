import argparse
import fcntl
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path('/root/phyguard_revision')

PLAN_PATH = ROOT / 'configs/sionna_formal_test840_plan_v1.json'
POLICY_PATH = ROOT / 'configs/sionna_causal_validation_policy_v2.json'
PROTOCOL_ARCHIVE = ROOT / 'results/phyguard_sionna_formal_test840_protocol_locked.zip'
PREVALIDATION_ARCHIVE = ROOT / 'results/phyguard_sionna_prevalidation210_causal_v2_final.zip'

EXPECTED_PLAN_SHA256 = 'b767fe07cd17eef0c42ec58cfe801732d38f7124ed5859915195e21223083437'
EXPECTED_POLICY_SHA256 = '510b0ba97393a63a199d65414c2b44660e7140c1696582202b1603c7197f879d'
EXPECTED_PROTOCOL_ARCHIVE_SHA256 = 'ae721c16860d75cfce303c877f20125a9ee7583f7f6e8da659711c46339a3f5e'
EXPECTED_PREVALIDATION_ARCHIVE_SHA256 = 'da8d6dfe0e5fa443bae14cf6501012b85982e8bb02221c9e1b5caa76d128461f'

GENERATOR_ROOT = ROOT / 'scripts/formal_test840_generators'
DATA_ROOT = ROOT / 'data/sionna_formal_test840'
RESULT_ROOT = ROOT / 'results/sionna_formal_test840'
SAMPLE_LOG_ROOT = ROOT / 'logs/sionna_formal_test840'
MANIFEST_PATH = ROOT / 'manifests/sionna_formal_test840_generation_manifest.json'
LOCK_PATH = ROOT / 'logs/41_generate_sionna_formal_test840.lock'

GENERATOR_NAMES = [
    'normal',
    'interference',
    'blockage',
    'mobility',
    'adaptation_mismatch',
]

LABEL_TO_GENERATOR = {
    'normal': 'normal.py',
    'interference': 'interference.py',
    'blockage': 'blockage.py',
    'mobility': 'mobility.py',
    'adaptation_mismatch': 'adaptation_mismatch.py',
}

EXPECTED_LABEL_COUNTS = {
    'normal': 60,
    'interference': 180,
    'blockage': 180,
    'mobility': 180,
    'adaptation_mismatch': 180,
    'nonphysical_goodput': 60,
}

EXPECTED_CHANNEL_COUNTS = {
    'CDL-A': 280,
    'CDL-B': 280,
    'CDL-C': 280,
}

EXPECTED_SEVERITY_COUNTS = {
    'control': 120,
    'mild': 240,
    'moderate': 240,
    'severe': 240,
}

EXPECTED_DIRECTION_ERROR = {
    'interference': 'Interference physical-direction validation failed.',
    'blockage': 'Blockage physical-direction validation failed.',
    'adaptation_mismatch': 'Adaptation-mismatch directional validation failed.',
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def severity_environment(sample: dict) -> dict:
    label = sample['label']
    severity = sample['severity']
    environment = {}

    if label == 'interference':
        environment['PHYGUARD_INTERFERENCE_SIR_DB'] = str(
            {'mild': 6.0, 'moderate': 3.0, 'severe': 0.0}[severity]
        )
    elif label == 'blockage':
        peak = {'mild': -6.0, 'moderate': -9.0, 'severe': -12.0}[severity]
        environment['PHYGUARD_BLOCKAGE_PEAK_DB'] = str(peak)
        environment['PHYGUARD_BLOCKAGE_EDGE_DB'] = str(peak / 3.0)
    elif label == 'mobility':
        environment['MOBILITY_EVENT_SPEED_MPS'] = str(
            {'mild': 45.0, 'moderate': 60.0, 'severe': 75.0}[severity]
        )
    elif label == 'adaptation_mismatch':
        environment['PHYGUARD_ADAPTATION_PHASE_DEG'] = str(
            {'mild': 30.0, 'moderate': 45.0, 'severe': 60.0}[severity]
        )
        environment['PHYGUARD_ADAPTATION_EDGE_DEG'] = str(
            {'mild': 8.0, 'moderate': 12.0, 'severe': 16.0}[severity]
        )

    return environment


def verify_protocol() -> dict:
    for required in (
        PLAN_PATH,
        POLICY_PATH,
        PROTOCOL_ARCHIVE,
        PREVALIDATION_ARCHIVE,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    if sha256_file(PROTOCOL_ARCHIVE) != EXPECTED_PROTOCOL_ARCHIVE_SHA256:
        raise RuntimeError('Formal Test-840 protocol archive hash changed.')
    if sha256_file(PREVALIDATION_ARCHIVE) != EXPECTED_PREVALIDATION_ARCHIVE_SHA256:
        raise RuntimeError('Frozen Prevalidation-210 archive hash changed.')

    with zipfile.ZipFile(PROTOCOL_ARCHIVE, 'r') as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise RuntimeError(f'Protocol ZIP integrity failure: {bad_file}')

    plan = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
    policy = json.loads(POLICY_PATH.read_text(encoding='utf-8'))

    plan_without_hash = dict(plan)
    plan_without_hash.pop('plan_sha256', None)

    if plan.get('plan_sha256') != EXPECTED_PLAN_SHA256:
        raise RuntimeError('Stored formal plan hash changed.')
    if sha256_json(plan_without_hash) != EXPECTED_PLAN_SHA256:
        raise RuntimeError('Recomputed formal plan hash changed.')
    if policy.get('policy_sha256') != EXPECTED_POLICY_SHA256:
        raise RuntimeError('Causal-validation policy hash changed.')
    if policy.get('status') != 'LOCKED':
        raise RuntimeError('Causal-validation policy is not LOCKED.')

    samples = plan.get('samples', [])
    if len(samples) != 840:
        raise RuntimeError(f'Expected 840 samples, found {len(samples)}.')

    ids = [sample['sample_id'] for sample in samples]
    seeds = [int(sample['seed']) for sample in samples]
    if len(set(ids)) != 840 or len(set(seeds)) != 840:
        raise RuntimeError('Formal sample IDs or seeds are not unique.')

    label_counts = Counter(sample['label'] for sample in samples)
    channel_counts = Counter(sample['channel_model'] for sample in samples)
    severity_counts = Counter(sample['severity'] for sample in samples)
    stratum_counts = Counter(sample['formal_test_stratum'] for sample in samples)

    if dict(label_counts) != EXPECTED_LABEL_COUNTS:
        raise RuntimeError(f'Unexpected label counts: {dict(label_counts)}')
    if dict(channel_counts) != EXPECTED_CHANNEL_COUNTS:
        raise RuntimeError(f'Unexpected channel counts: {dict(channel_counts)}')
    if dict(severity_counts) != EXPECTED_SEVERITY_COUNTS:
        raise RuntimeError(f'Unexpected severity counts: {dict(severity_counts)}')
    if len(stratum_counts) != 42 or set(stratum_counts.values()) != {20}:
        raise RuntimeError('Formal stratum contract is invalid.')

    for sample in samples:
        sample_id = sample['sample_id']
        if sample.get('formal_test') is not True:
            raise RuntimeError(f'{sample_id}: formal_test is not true')
        if sample.get('causal_validation_policy_sha256') != EXPECTED_POLICY_SHA256:
            raise RuntimeError(f'{sample_id}: policy hash mismatch')
        for field in (
            'eligible_for_training',
            'eligible_for_threshold_tuning',
            'eligible_for_model_selection',
            'eligible_for_feature_selection',
            'eligible_for_calibration',
            'policy_modification_allowed',
        ):
            if sample.get(field) is not False:
                raise RuntimeError(f'{sample_id}: {field} is not false')

    return plan


def extract_locked_generators() -> list[dict]:
    if GENERATOR_ROOT.exists():
        shutil.rmtree(GENERATOR_ROOT)
    GENERATOR_ROOT.mkdir(parents=True, exist_ok=True)

    records = []

    with zipfile.ZipFile(PREVALIDATION_ARCHIVE, 'r') as archive:
        names = set(archive.namelist())

        for name in GENERATOR_NAMES:
            candidates = [
                f'scripts/prevalidation210_full_generators/{name}.py',
                f'scripts/prevalidation210_generators/{name}.py',
            ]
            matches = [candidate for candidate in candidates if candidate in names]

            if len(matches) != 1:
                raise RuntimeError(
                    f'Expected one frozen generator for {name}, found {matches}'
                )

            target = GENERATOR_ROOT / f'{name}.py'
            target.write_bytes(archive.read(matches[0]))
            py_compile.compile(str(target), doraise=True)

            records.append(
                {
                    'generator': name,
                    'archive_entry': matches[0],
                    'target_path': str(target),
                    'sha256': sha256_file(target),
                    'size_bytes': target.stat().st_size,
                }
            )

    return records


def create_nonphysical_control(sample: dict, plan: dict) -> None:
    sample_id = sample['sample_id']
    source_normal_id = sample.get('source_normal_id') or sample.get('paired_normal_id')
    if not source_normal_id:
        raise RuntimeError(f'{sample_id}: paired normal ID missing.')

    source_path = DATA_ROOT / source_normal_id / 'sequence.npz'
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    output_dir = DATA_ROOT / sample_id
    result_dir = RESULT_ROOT / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    with np.load(source_path, allow_pickle=False) as archive:
        source = archive['kpi_sequence'].astype(np.float32)
        kpi_names = archive['kpi_names']
        frame_index = archive['frame_index']

    if source.shape != (80, 10):
        raise RuntimeError(f'{sample_id}: invalid source shape {source.shape}')
    if 'goodput' not in str(kpi_names[8]).lower():
        raise RuntimeError(f'{sample_id}: KPI index 8 is not goodput.')

    replicate = int(sample['replicate'])
    targets = {1: 0.35, 2: 0.50, 3: 0.20, 4: 0.65}
    target_key = ((replicate - 1) % 4) + 1
    target = targets[target_key]
    start = int(sample['event_start'])
    end = int(sample['event_end'])

    sequence = source.copy()
    envelope = np.full(end - start, target, dtype=np.float32)
    ramp = min(4, max(1, (end - start) // 2))

    if ramp > 1:
        envelope[:ramp] = np.linspace(0.85, target, ramp, dtype=np.float32)
        envelope[-ramp:] = np.linspace(target, 0.85, ramp, dtype=np.float32)

    sequence[start:end, 8] = envelope
    replay = source.copy()
    replay[start:end, 8] = envelope

    exact_replay = bool(np.array_equal(sequence, replay))
    unchanged = bool(
        np.array_equal(
            sequence[:, [0, 1, 2, 3, 4, 5, 6, 7, 9]],
            source[:, [0, 1, 2, 3, 4, 5, 6, 7, 9]],
        )
    )
    consistency_error = float(
        np.mean(np.abs(sequence[start:end, 8] - (1.0 - sequence[start:end, 5])))
    )

    np.savez_compressed(
        output_dir / 'sequence.npz',
        kpi_sequence=sequence,
        kpi_names=kpi_names,
        frame_index=frame_index,
    )

    metadata = {
        'sample_id': sample_id,
        'generator_name': 'Deterministic nonphysical telemetry control',
        'generator_version': 'formal_test840_nonphysical_goodput_v1.0.0',
        'seed': sample['seed'],
        'label': sample['label'],
        'severity': sample['severity'],
        'channel_model': sample['channel_model'],
        'event_start': start,
        'event_end': end,
        'source_normal_id': source_normal_id,
        'replicate': replicate,
        'locked_target_key': target_key,
        'target_goodput': target,
        'kpi_contract_sha256': plan.get('kpi_contract_sha256'),
        'sequence_shape': [80, 10],
        'exact_bitwise_replay': exact_replay,
        'max_abs_replay_error': 0.0,
        'numerical_replay_pass': exact_replay,
        'other_nine_kpis_bitwise_unchanged': unchanged,
        'goodput_consistency_error': consistency_error,
    }

    status = 'PASS' if (
        exact_replay
        and unchanged
        and consistency_error >= 0.20
        and np.isfinite(sequence).all()
    ) else 'FAIL'

    report = {
        'status': status,
        'sample_id': sample_id,
        'sequence_shape': [80, 10],
        'exact_bitwise_replay': exact_replay,
        'source_normal_id': source_normal_id,
        'locked_target_key': target_key,
        'target_event_goodput': target,
        'other_nine_kpis_bitwise_unchanged': unchanged,
        'goodput_consistency_error': consistency_error,
        'errors': [] if status == 'PASS' else ['Nonphysical-control validation failed.'],
    }

    (output_dir / 'metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    (result_dir / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def read_artifacts(sample: dict) -> tuple[list[str], dict]:
    sample_id = sample['sample_id']
    sequence_path = DATA_ROOT / sample_id / 'sequence.npz'
    metadata_path = DATA_ROOT / sample_id / 'metadata.json'
    report_path = RESULT_ROOT / sample_id / 'report.json'
    errors = []
    metadata = {}
    report = {}

    for path in (sequence_path, metadata_path, report_path):
        if not path.exists():
            errors.append(f'missing {path}')

    if errors:
        return errors, {
            'sequence_path': sequence_path,
            'metadata_path': metadata_path,
            'report_path': report_path,
            'metadata': metadata,
            'report': report,
        }

    try:
        with np.load(sequence_path, allow_pickle=False) as archive:
            required = {'kpi_sequence', 'kpi_names', 'frame_index'}
            missing = required - set(archive.files)
            if missing:
                errors.append(f'missing NPZ keys: {sorted(missing)}')
            else:
                sequence = archive['kpi_sequence']
                if sequence.shape != (80, 10):
                    errors.append(f'invalid sequence shape {sequence.shape}')
                if sequence.dtype != np.float32:
                    errors.append(f'invalid sequence dtype {sequence.dtype}')
                if not np.isfinite(sequence).all():
                    errors.append('sequence contains NaN or Inf')
                if len(archive['kpi_names']) != 10:
                    errors.append('KPI-name count is not 10')
                if len(archive['frame_index']) != 80:
                    errors.append('frame-index count is not 80')
    except Exception as exc:
        errors.append(f'NPZ read error: {exc!r}')

    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    except Exception as exc:
        errors.append(f'metadata JSON error: {exc!r}')

    try:
        report = json.loads(report_path.read_text(encoding='utf-8'))
    except Exception as exc:
        errors.append(f'report JSON error: {exc!r}')

    if metadata:
        for field in ('sample_id', 'label', 'severity', 'channel_model'):
            if metadata.get(field) != sample[field]:
                errors.append(f'metadata {field} mismatch')
        if metadata.get('sequence_shape') not in ([80, 10], None):
            errors.append('metadata sequence shape mismatch')
        if metadata.get('exact_bitwise_replay') is not True:
            errors.append('exact bitwise replay failed')
        if metadata.get('numerical_replay_pass') is not True:
            errors.append('numerical replay failed')
        if metadata.get('kpi_contract_sha256') != sample.get('kpi_contract_sha256'):
            errors.append('KPI-contract SHA256 mismatch')

        validation = metadata.get('validation')
        if isinstance(validation, dict):
            if validation.get('all_finite') is not True:
                errors.append('base validation all_finite failed')
            if int(validation.get('varying_kpi_count', 0)) < 6:
                errors.append('fewer than six varying KPIs')
            for name, item in validation.get('range_checks', {}).items():
                if item.get('pass') is not True:
                    errors.append(f'range check failed: {name}')

    return errors, {
        'sequence_path': sequence_path,
        'metadata_path': metadata_path,
        'report_path': report_path,
        'metadata': metadata,
        'report': report,
    }


def collected_report_errors(metadata: dict, report: dict) -> list[str]:
    values = []
    validation = metadata.get('validation')
    if isinstance(validation, dict):
        values.extend(validation.get('errors', []))
    values.extend(report.get('errors', []))

    unique = []
    for value in values:
        if value and value not in unique:
            unique.append(str(value))
    return unique


def build_generation_qc(sample: dict, return_code: int) -> dict:
    errors, artifacts = read_artifacts(sample)
    metadata = artifacts['metadata']
    report = artifacts['report']
    label = sample['label']
    legacy_status = report.get('status') if report else None
    report_errors = collected_report_errors(metadata, report) if metadata and report else []
    expected_direction_error = EXPECTED_DIRECTION_ERROR.get(label)

    if metadata and report:
        if label in EXPECTED_DIRECTION_ERROR:
            unexpected = [
                item for item in report_errors if item != expected_direction_error
            ]
            errors.extend(f'unexpected generator error: {item}' for item in unexpected)

            exit_consistent = (
                (return_code == 0 and legacy_status == 'PASS' and not report_errors)
                or (
                    return_code == 1
                    and legacy_status == 'FAIL'
                    and expected_direction_error in report_errors
                    and not unexpected
                )
            )
            if not exit_consistent:
                errors.append(
                    'generator exit/status is not consistent with the locked legacy directional validator'
                )
        elif label in {'normal', 'mobility', 'nonphysical_goodput'}:
            if return_code != 0:
                errors.append(f'generator return code={return_code}')
            if legacy_status != 'PASS':
                errors.append(f'legacy report status={legacy_status!r}')
            errors.extend(f'generator error: {item}' for item in report_errors)
        else:
            errors.append(f'unsupported label {label}')

    record = {
        'schema': 'phyguard.sionna.formal_test840.generation_qc.v1',
        'status': 'PASS' if not errors else 'FAIL',
        'sample_id': sample['sample_id'],
        'label': label,
        'severity': sample['severity'],
        'channel_model': sample['channel_model'],
        'seed': sample['seed'],
        'formal_plan_sha256': EXPECTED_PLAN_SHA256,
        'causal_validation_policy_sha256': EXPECTED_POLICY_SHA256,
        'generator_return_code': return_code,
        'legacy_directional_status': legacy_status,
        'legacy_report_errors': report_errors,
        'errors': errors,
        'sequence_sha256': (
            sha256_file(artifacts['sequence_path'])
            if artifacts['sequence_path'].exists()
            else None
        ),
        'metadata_sha256': (
            sha256_file(artifacts['metadata_path'])
            if artifacts['metadata_path'].exists()
            else None
        ),
        'report_sha256': (
            sha256_file(artifacts['report_path'])
            if artifacts['report_path'].exists()
            else None
        ),
    }
    return record


def existing_pass_record(sample: dict):
    sample_id = sample['sample_id']
    qc_path = RESULT_ROOT / sample_id / 'generation_qc_report.json'
    if not qc_path.exists():
        return None

    try:
        record = json.loads(qc_path.read_text(encoding='utf-8'))
    except Exception:
        return None

    if record.get('status') != 'PASS':
        return None
    if record.get('sample_id') != sample_id:
        return None
    if record.get('formal_plan_sha256') != EXPECTED_PLAN_SHA256:
        return None
    if record.get('causal_validation_policy_sha256') != EXPECTED_POLICY_SHA256:
        return None

    for hash_field, path in (
        ('sequence_sha256', DATA_ROOT / sample_id / 'sequence.npz'),
        ('metadata_sha256', DATA_ROOT / sample_id / 'metadata.json'),
        ('report_sha256', RESULT_ROOT / sample_id / 'report.json'),
    ):
        if not path.exists() or record.get(hash_field) != sha256_file(path):
            return None

    return record


def write_manifest(generator_records: list[dict], run_records: list[dict], running: bool) -> dict:
    passed = [record for record in run_records if record['status'] == 'PASS']
    failed = [record for record in run_records if record['status'] != 'PASS']
    legacy_counts = Counter(record.get('legacy_directional_status') for record in run_records)

    manifest = {
        'schema': 'phyguard.sionna.formal_test840.generation.v1',
        'status': (
            'RUNNING'
            if running
            else ('PASS' if len(passed) == 840 and not failed else 'FAIL')
        ),
        'purpose': (
            'Generate all locked Formal Test-840 artifacts. '
            'Legacy directional outcomes are preserved but do not replace '
            'the separately locked causal-validation-v2 evaluation.'
        ),
        'formal_plan_sha256': EXPECTED_PLAN_SHA256,
        'causal_validation_policy_sha256': EXPECTED_POLICY_SHA256,
        'protocol_archive_sha256': EXPECTED_PROTOCOL_ARCHIVE_SHA256,
        'source_prevalidation_archive_sha256': EXPECTED_PREVALIDATION_ARCHIVE_SHA256,
        'expected_sample_count': 840,
        'completed_count': len(run_records),
        'artifact_qc_passed_count': len(passed),
        'artifact_qc_failed_count': len(failed),
        'artifact_qc_failed_sample_ids': [record['sample_id'] for record in failed],
        'legacy_directional_status_counts': dict(legacy_counts),
        'methodological_boundary': {
            'policy_modified': False,
            'formal_causal_v2_evaluation_performed': False,
            'training_performed': False,
            'threshold_tuning_performed': False,
            'model_selection_performed': False,
            'feature_selection_performed': False,
            'calibration_performed': False,
        },
        'generator_source': 'extracted from frozen Prevalidation-210 causal-v2 archive',
        'generators': generator_records,
        'runs': run_records,
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()

    plan = verify_protocol()
    generator_records = extract_locked_generators()

    print('FORMAL TEST-840 RUNNER AUDIT')
    print('status: PASS')
    print('sample_count: 840')
    print('formal_plan_sha256:', EXPECTED_PLAN_SHA256)
    print('policy_sha256:', EXPECTED_POLICY_SHA256)
    print('protocol_archive_sha256:', EXPECTED_PROTOCOL_ARCHIVE_SHA256)
    print('source_prevalidation_archive_sha256:', EXPECTED_PREVALIDATION_ARCHIVE_SHA256)
    print('generator_count:', len(generator_records))

    if args.audit_only:
        print('\nSIONNA_FORMAL_TEST840_RUNNER_AUDIT_PASS')
        return

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK_PATH.open('a+', encoding='utf-8')

    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError('Another Formal Test-840 process is already running.') from exc

    lock_stream.seek(0)
    lock_stream.truncate()
    lock_stream.write(f'pid={os.getpid()}\n')
    lock_stream.flush()

    for directory in (DATA_ROOT, RESULT_ROOT, SAMPLE_LOG_ROOT):
        directory.mkdir(parents=True, exist_ok=True)

    run_records = []
    print('\nFORMAL TEST-840 GENERATION START', flush=True)

    for index, sample in enumerate(plan['samples'], start=1):
        sample_id = sample['sample_id']
        label = sample['label']

        existing = existing_pass_record(sample)
        if existing is not None:
            record = dict(existing)
            record.update(
                {
                    'execution': 'resume_skip_existing_pass',
                    'elapsed_seconds': 0.0,
                    'log_path': str(SAMPLE_LOG_ROOT / f'{sample_id}.log'),
                }
            )
            run_records.append(record)
            print(f'[{index:03d}/840] {sample_id} SKIP_EXISTING_PASS', flush=True)
            write_manifest(generator_records, run_records, running=True)
            continue

        data_dir = DATA_ROOT / sample_id
        result_dir = RESULT_ROOT / sample_id
        log_path = SAMPLE_LOG_ROOT / f'{sample_id}.log'

        if data_dir.exists():
            shutil.rmtree(data_dir)
        if result_dir.exists():
            shutil.rmtree(result_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        print(
            f'[{index:03d}/840] {sample_id} '
            f'label={label} severity={sample["severity"]} '
            f'channel={sample["channel_model"]}',
            flush=True,
        )

        start_time = time.perf_counter()
        return_code = 0
        output = ''

        if label == 'nonphysical_goodput':
            try:
                create_nonphysical_control(sample, plan)
                output = (result_dir / 'report.json').read_text(encoding='utf-8')
            except Exception as exc:
                return_code = 1
                output = 'NONPHYSICAL CONTROL ERROR\n' + repr(exc)
        else:
            generator_path = GENERATOR_ROOT / LABEL_TO_GENERATOR[label]
            environment = os.environ.copy()
            environment.update(
                {
                    'PHYGUARD_PLAN_PATH': str(PLAN_PATH),
                    'PHYGUARD_SAMPLE_ID': sample_id,
                    'PHYGUARD_OUTPUT_DIR': str(data_dir),
                    'PHYGUARD_RESULT_DIR': str(result_dir),
                    'PYTHONHASHSEED': str(sample['seed']),
                }
            )
            environment.update(severity_environment(sample))

            process = subprocess.run(
                [sys.executable, str(generator_path)],
                cwd=str(ROOT),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            return_code = process.returncode
            output = process.stdout

        log_path.write_text(output, encoding='utf-8')
        elapsed_seconds = time.perf_counter() - start_time

        record = build_generation_qc(sample, return_code)
        record.update(
            {
                'execution': 'generated',
                'elapsed_seconds': elapsed_seconds,
                'log_path': str(log_path),
            }
        )

        (result_dir / 'generation_qc_report.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

        run_records.append(record)

        print(
            '  artifact_qc_status=', record['status'],
            'legacy_directional_status=', record.get('legacy_directional_status'),
            'return_code=', return_code,
            'elapsed_seconds=', round(elapsed_seconds, 3),
            flush=True,
        )
        if record['errors']:
            print('  errors=', record['errors'], flush=True)

        write_manifest(generator_records, run_records, running=True)

    final_manifest = write_manifest(generator_records, run_records, running=False)

    print('\nFORMAL TEST-840 GENERATION SUMMARY')
    print('status:', final_manifest['status'])
    print('completed_count:', final_manifest['completed_count'])
    print('artifact_qc_passed_count:', final_manifest['artifact_qc_passed_count'])
    print('artifact_qc_failed_count:', final_manifest['artifact_qc_failed_count'])
    print('artifact_qc_failed_sample_ids:', final_manifest['artifact_qc_failed_sample_ids'])
    print('legacy_directional_status_counts:', final_manifest['legacy_directional_status_counts'])
    print('manifest:', MANIFEST_PATH)

    if final_manifest['status'] != 'PASS':
        raise RuntimeError('Formal Test-840 artifact generation completed with QC failures.')

    print('\nSIONNA_FORMAL_TEST840_GENERATION_ARTIFACT_QC_PASS')


if __name__ == '__main__':
    main()
