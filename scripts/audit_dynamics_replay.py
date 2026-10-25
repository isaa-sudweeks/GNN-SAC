"""Read trusted full checkpoints and report conservative dynamics-data eligibility.

This is an audit, not a replay converter or an authorization to fit a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any

SOURCE_TASKS = frozenset({
    'tetrahedron', 'henneberg_n8_2tube_187', 'henneberg_n6_2tube_1',
    'usevitch_210272254_p1', 'henneberg_n5_1tube_1', 'henneberg_n6_1tube_2',
    'henneberg_n8_3tube_64', 'usevitch_60243677150_p1',
})
PROVENANCE_FIELDS = ('episode_id', 'env_id', 'timestep', 'truncated')


def valid_indices(capacity: int, size: int, index: int) -> list[int]:
    """Return valid physical rows oldest first, rejecting inconsistent rings."""
    if not (capacity > 0 and 0 <= size <= capacity and 0 <= index < capacity):
        raise ValueError('Invalid capacity, size, or write index')
    if size < capacity and index != size:
        raise ValueError('Partially filled ring must have idx == size')
    start = index if size == capacity else 0
    return [(start + offset) % capacity for offset in range(size)]


def tensor_summary(value: Any) -> dict:
    import torch
    value = torch.as_tensor(value).detach().cpu()
    finite = torch.isfinite(value)
    result = {'shape': list(value.shape), 'finite': bool(finite.all())}
    if value.numel() and finite.any():
        good = value[finite].double()
        result.update(min=float(good.min()), max=float(good.max()), mean=float(good.mean()))
    return result


def audit_buffer(buffer: dict, *, seed: int, sample_count: int) -> dict:
    """Inspect seeded rows; never infer episode identity from row adjacency."""
    import torch
    rows = valid_indices(*(int(buffer[k]) for k in ('capacity', 'size', 'idx')))
    chosen = sorted(random.Random(seed).sample(rows, min(len(rows), sample_count)))
    version = int(buffer.get('format_version', 0))
    if version in (3, 4):
        fields = buffer['replay_buffer']['_storage']['_storage']
        get = lambda key, row: fields[key][row]
        names = {'obs': 'obs_x', 'next_obs': 'next_obs_x', 'action': 'action',
                 'reward': 'reward', 'terminated': 'terminated'}
        available = sorted(str(k) for k in fields.keys())
        graph_fields = sorted(str(k) for k in (buffer.get('static') or {}).keys())
    else:
        get = lambda key, row: buffer[key][row]
        names = {k: k for k in ('obs', 'next_obs', 'action', 'reward', 'terminated')}
        available = sorted(buffer.keys())
        graph_fields = []
        for key in names:
            if len(buffer[key]) != int(buffer['capacity']):
                raise ValueError(f'{key} length differs from capacity')
    summaries, failures, action_values = [], [], []
    for row in chosen:
        entry = {'index': row}
        try:
            for key, name in names.items():
                value = get(name, row)
                if key in ('obs', 'next_obs') and hasattr(value, 'x'):
                    if not graph_fields:
                        graph_fields = sorted(value.keys())
                    value = value.x
                entry[key] = tensor_summary(value)
            if entry['obs']['shape'] != entry['next_obs']['shape']:
                failures.append({'index': row, 'reason': 'observation shapes differ'})
            if not all(entry[k]['finite'] for k in names):
                failures.append({'index': row, 'reason': 'nonfinite field'})
            action_values.append(torch.as_tensor(get(names['action'], row)).reshape(-1).float())
            summaries.append(entry)
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            failures.append({'index': row, 'reason': str(error)})
    action = torch.cat(action_values) if action_values else torch.empty(0)
    provenance = {k: k in available for k in PROVENANCE_FIELDS}
    return {
        'format_version': version, 'capacity': int(buffer['capacity']), 'size': len(rows),
        'write_index': int(buffer['idx']), 'sample_seed': seed, 'sampled_indices': chosen,
        'checked_rows': len(summaries), 'sample_failures': failures,
        'stored_fields': available, 'graph_fields': graph_fields,
        'action_sample': tensor_summary(action), 'provenance_fields': provenance,
        'trajectory_eligible': False,
        'one_step_status': 'candidate_pending_state_and_physics_audit' if summaries and not failures else 'insufficient',
        'limits': ['Only sampled rows checked for finite values and shape alignment.',
                   'No claim of temporal continuity, independent episodes, or sufficient dynamic state.',
                   'Action values describe saved commands; actuator routing/clipping remains to verify.'],
        'rows': summaries,
    }


def classify_label_contract(config: dict) -> dict:
    """Classify saved settings for a clean-state nominal-native fitting contract.

    This is a settings screen, not proof of the historical collector executable,
    clean labels, simulator parity, or practical prediction sufficiency.
    """
    randomization = bool(config.get('domain_randomization', False))
    params = config.get('domain_randomization_params') or {}

    def active(name: str) -> bool:
        setting = params.get(name) or {}
        return (randomization and bool(setting.get('enabled', False))
                and float(setting.get('std', 0.0)) > 0)

    observation_noise = active('observation_noise')
    action_noise = active('action_noise')
    backend = config.get('sim_backend', config.get('mujoco_backend'))
    reasons = []
    if observation_noise:
        reasons.append('Saved settings enable observation noise; replay has no separate clean observation labels.')
    if backend != 'mujoco':
        reasons.append('Native MuJoCo equivalence is unverified for the saved collection backend.')
    return {
        'contract': 'clean_observable_state_nominal_native',
        'saved_collection_backend': backend,
        'observation_noise_enabled': observation_noise,
        'action_noise_enabled': action_noise,
        'clean_label_status': 'not_certified_noisy_saved_settings' if observation_noise else 'requires_collection_trace',
        'nominal_native_eligible': False,
        'fresh_nominal_fallback_justified': bool(reasons),
        'reasons': reasons,
        'hidden_targets_required': False,
        'action_semantics': 'Owning trainer stores post-noise/projected normalized node commands; historical executable and environment routing still require provenance.',
        'limits': ['Noise configuration is not a direct measurement of row corruption.',
                   'Action noise alone does not invalidate commands saved after noise.',
                   'Missing hidden actuator targets alone does not reject observable-input fitting.'],
    }


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def audit_checkpoint(path: Path, *, seed: int = 0, sample_count: int = 128) -> dict:
    """Load an explicitly selected trusted checkpoint, filtering tasks before rows."""
    import torch
    started = time.perf_counter()
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    config = checkpoint.get('config', {})
    replay = checkpoint.get('buffer', {})
    buffers = replay.get('buffers')
    if buffers is None:
        tasks = config.get('truss_topologies') or [config.get('truss_topology')]
        if isinstance(tasks, str):
            tasks = [tasks]
        if len(tasks) != 1 or tasks[0] is None:
            raise ValueError('Single replay task identity is ambiguous; supply an identified specialist')
        buffers = {tasks[0]: replay}
    reports, excluded = {}, []
    for task, buffer in buffers.items():
        # Older specialists use the generic task key, so resolve it only when
        # the saved config independently names exactly one topology.
        topology = str(task).split(':')[-1]
        if topology == 'truss-graph':
            configured = config.get('truss_topologies') or [config.get('truss_topology')]
            if isinstance(configured, str):
                configured = [configured]
            if len(configured) == 1:
                topology = configured[0]
        if topology not in SOURCE_TASKS:
            excluded.append(task)
            continue
        reports[topology] = audit_buffer(buffer, seed=seed, sample_count=sample_count)
        reports[topology]['stored_task_key'] = task
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (subprocess.SubprocessError, OSError):
        revision = None
    return {'checkpoint': str(path.resolve()), 'checkpoint_bytes': path.stat().st_size,
            'checkpoint_sha256': digest(path), 'audit_revision': revision,
            'torch_version': torch.__version__, 'config': config, 'tasks': reports,
            'excluded_without_row_inspection': excluded, 'elapsed_seconds': time.perf_counter() - started,
            'full_checkpoint_keys': sorted(checkpoint.keys()),
            'label_contract': classify_label_contract(config)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.sample_count <= 0:
        parser.error('--sample-count must be positive')
    report = audit_checkpoint(args.checkpoint, seed=args.seed, sample_count=args.sample_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
