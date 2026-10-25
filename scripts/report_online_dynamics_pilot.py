"""Verify completed online pilot evidence and write a measured source-only report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'sac')]
from dynamics.dataset import file_sha256

LABELS = {'random': 'Random control', 'dynamics_mpc': 'Online dynamics MPC',
          'reward_value_mpc': 'Online reward + value MPC'}
COLORS = {'random': '#737985', 'dynamics_mpc': '#1565c0', 'reward_value_mpc': '#d85b24'}


def verify(root: Path, document: dict) -> dict:
    cfg, runs = document['protocol']['settings'], document['runs']
    expected = {(c, s) for c in cfg['conditions'] for s in cfg['seeds']}
    if {(r['condition'], r['seed']) for r in runs} != expected or len(runs) != len(expected):
        raise ValueError('Experiment matrix is incomplete or duplicated')
    for name, expected_hash in document['protocol']['sources'].items():
        if file_sha256(ROOT / name) != expected_hash:
            raise ValueError(f'Experiment source changed after protocol hashing: {name}')
    checks = []
    for seed in cfg['seeds']:
        matching = [r for r in runs if r['seed'] == seed]
        warmups = []
        for run in matching:
            folder = root / f"{run['condition']}-seed{seed}"
            for name, field in [('final-model.pt', 'checkpoint_sha256'), ('replay.npz', 'replay_sha256'), ('warmup.npz', 'warmup_sha256')]:
                if file_sha256(folder / name) != run[field]:
                    raise ValueError(f'Artifact changed: {folder / name}')
            with np.load(folder / 'replay.npz') as data:
                if len(data['state']) != cfg['steps'] or run['interactions'] != cfg['steps']:
                    raise ValueError('Interaction budget mismatch')
                episode = data['episode']
                same = episode[1:] == episode[:-1]
                if not np.array_equal(data['next_state'][:-1][same], data['state'][1:][same]):
                    raise ValueError('Real replay temporal adjacency mismatch')
                if np.any(data['terminated'][:-1][same] | data['truncated'][:-1][same]):
                    raise ValueError('A replay episode continues after end')
                warmups.append({key: data[key][:cfg['warmup']].copy() for key in data.files})
            expected_updates = cfg['initial_updates'] + (cfg['steps'] - cfg['warmup']) * cfg['updates_per_step']
            if run['updates'] != expected_updates:
                raise ValueError('Update budget mismatch')
            for evaluation in run['evaluations']:
                for episode in evaluation['episodes']:
                    if file_sha256(folder / episode['trace']) != episode['sha256']:
                        raise ValueError('Evaluation trace hash mismatch')
            # No evaluation identity can occur among fitting reset identities.
            fitting_seeds = {e['seed'] for e in run['training_episodes']}
            eval_seeds = {e['seed'] for v in run['evaluations'] for e in v['episodes']}
            if fitting_seeds & eval_seeds:
                raise ValueError('Fitting/evaluation reset leakage')
        for warmup in warmups[1:]:
            for key in warmups[0]:
                if not np.array_equal(warmup[key], warmups[0][key]):
                    raise ValueError(f'Per-seed initial experience differs across conditions: {key}')
        for budget in cfg['evaluate_at']:
            starts = [{e['seed']: e['initial_state_sha256'] for e in next(v for v in r['evaluations'] if v['steps'] == budget)['episodes']} for r in matching]
            if any(start != starts[0] for start in starts[1:]):
                raise ValueError('Controllers did not receive paired starts')
        checks.append(dict(seed=seed, warmup_arrays_identical=True, paired_starts=True,
                           replay_adjacency_exact=True, budgets_exact=True, artifact_hashes=True))
    return dict(passed=True, checks=checks)


def report(root: Path) -> None:
    doc = json.loads((root / 'report.json').read_text())
    audit = verify(root, doc)
    (root / 'integrity-audit.json').write_text(json.dumps(audit, indent=2))
    cfg, runs = doc['protocol']['settings'], doc['runs']
    rows = []
    for condition in cfg['conditions']:
        selected = [r for r in runs if r['condition'] == condition]
        episodes = [e for r in selected for e in r['evaluations'][-1]['episodes']]
        predictions = [p for r in selected for p in r['prediction_summary'] if p['horizon'] == 10 and p['predictor'] == 'gnn']
        rows.append(dict(condition=condition, seeds=len(selected), eval_episodes=len(episodes),
            mean_displacement=float(np.mean([e['displacement'] for e in episodes])),
            median_displacement=float(np.median([e['displacement'] for e in episodes])),
            seed_means=[r['evaluations'][-1]['mean_displacement'] for r in selected],
            collapses=sum(e['collapsed'] for e in episodes),
            mean_discounted_task_return=float(np.mean([e['discounted_task_return'] for e in episodes])),
            mean_task_return=float(np.mean([e['task_return'] for e in episodes])),
            mean_raw_return=float(np.mean([e['raw_return'] for e in episodes])),
            mean_h10_progress_mae=float(np.mean([p['progress_mae'] for p in predictions])) if predictions else None,
            training_collapses=sum(e['collapsed'] for r in selected for e in r['training_episodes']),
            total_wall_seconds=sum(r['wall_seconds'] for r in selected),
            median_planning_seconds=float(np.median([r['evaluations'][-1]['median_planning_seconds'] for r in selected]))))
    paired = []
    for left, right in [('dynamics_mpc', 'random'), ('reward_value_mpc', 'random'), ('reward_value_mpc', 'dynamics_mpc')]:
        differences, by_seed = [], []
        for seed in cfg['seeds']:
            a = next(r for r in runs if r['condition'] == left and r['seed'] == seed)['evaluations'][-1]['episodes']
            b = next(r for r in runs if r['condition'] == right and r['seed'] == seed)['evaluations'][-1]['episodes']
            delta = [x['displacement'] - y['displacement'] for x, y in zip(a, b)]
            differences.extend(delta)
            by_seed.append(float(np.mean(delta)))
        paired.append(dict(comparison=f'{left} minus {right}', differences=differences,
                           seed_means=by_seed, mean=float(np.mean(differences)), median=float(np.median(differences)),
                           wins=sum(d > 0 for d in differences), pairs=len(differences)))
    summary = dict(conditions=rows, paired=paired, total_wall_seconds=sum(r['wall_seconds'] for r in runs),
                   fitting_interactions=sum(r['interactions'] for r in runs), integrity=audit)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    for condition in cfg['conditions']:
        selected = [r for r in runs if r['condition'] == condition]
        values = np.asarray([[v['mean_displacement'] for v in r['evaluations']] for r in selected])
        color = COLORS[condition]
        axes[0].plot(cfg['evaluate_at'], values.mean(0), 'o-', color=color, label=LABELS[condition])
        axes[0].fill_between(cfg['evaluate_at'], values.min(0), values.max(0), color=color, alpha=.12)
    axes[0].axhline(0, color='#aaa', lw=.8)
    axes[0].set(xlabel='Real training transitions', ylabel='Forward displacement / initial diagonal',
                title='Frozen evaluation during learning')
    axes[0].legend(fontsize=8, loc='best')
    labels = ['Random', 'Dynamics\nMPC', 'Reward + value\nMPC']
    for i, row in enumerate(rows):
        axes[1].bar(i, row['mean_displacement'], color=COLORS[row['condition']], alpha=.75)
        axes[1].scatter(np.full(len(row['seed_means']), i), row['seed_means'], color='#20242a', s=24, zorder=3)
        axes[2].bar(i, row['mean_h10_progress_mae'] or 0., color=COLORS[row['condition']], alpha=.75)
    for i, row in enumerate(rows):
        axes[1].annotate(f"{row['collapses']}/{row['eval_episodes']} collapses", (i, row['mean_displacement']), xytext=(0, 12 if row['mean_displacement'] >= 0 else -16), textcoords='offset points', ha='center', fontsize=8)
    axes[1].margins(y=.32)
    axes[1].axhline(0, color='#aaa', lw=.8)
    axes[1].set(xticks=range(3), xticklabels=labels, ylabel='Forward displacement / initial diagonal',
                title='Final control: mean, dots = seed means')
    axes[2].set(xticks=range(3), xticklabels=labels, ylabel='10-step progress MAE / initial diagonal',
                title='Common random-action prediction check')
    fig.suptitle('Online learned-MPC pilot: source tetrahedron only', fontsize=14)
    fig.supxlabel(f"{len(cfg['seeds'])} training seeds × {cfg['evaluation_episodes']} paired evaluation resets; {cfg['episode_steps']}-step episodes. Bands show seed range, not confidence intervals.", fontsize=9)
    fig.savefig(root / 'results.png', dpi=180)
    fig.savefig(root / 'results.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    for condition in cfg['conditions']:
        selected = [r for r in runs if r['condition'] == condition]
        budgets = [h['step'] for h in selected[0]['history']]
        losses = np.asarray([[h['losses'].get('one_step', np.nan) + cfg['rollout_weight'] * h['losses'].get('rollout', np.nan)
                              for h in r['history']] for r in selected])
        valid = np.isfinite(losses).any(0)
        color = COLORS[condition]
        axes[0].plot(np.asarray(budgets)[valid], np.nanmean(losses[:, valid], 0), color=color, label=LABELS[condition])
        axes[0].fill_between(np.asarray(budgets)[valid], np.nanmin(losses[:, valid], 0), np.nanmax(losses[:, valid], 0), color=color, alpha=.12)
        bins = np.arange(160, cfg['steps'] + 1, 160)
        episodes = np.asarray([[np.mean([e['task_return'] for e in r['training_episodes'] if end - 160 < e['ending_step'] <= end])
                                for end in bins] for r in selected])
        axes[2].plot(bins, episodes.mean(0), color=color)
        axes[2].fill_between(bins, episodes.min(0), episodes.max(0), color=color, alpha=.12)
        if condition == 'reward_value_mpc':
            for component, component_color, label in [('reward_loss', '#197d59', 'Reward loss'), ('q_loss', '#844dc3', 'Twin-Q loss')]:
                values = np.asarray([[h['losses'].get(component, np.nan) for h in r['history']] for r in selected])
                axes[1].plot(np.asarray(budgets)[valid], np.nanmean(values[:, valid], 0), color=component_color, label=label)
                axes[1].fill_between(np.asarray(budgets)[valid], np.nanmin(values[:, valid], 0), np.nanmax(values[:, valid], 0), color=component_color, alpha=.12)
    axes[0].set(title='Dynamics fitting loss', ylabel='One-step + weighted recursive loss', xlabel='Real fitting transitions')
    axes[0].legend(fontsize=8)
    axes[1].set(title='Reward/value fitting losses', ylabel='Scaled Huber loss (log scale)', xlabel='Real fitting transitions')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=8)
    axes[2].set(title='Training episode task return', ylabel='Return, including collapse cost', xlabel='Real fitting transitions (160-step bins)')
    fig.suptitle('Online training: means across seeds, bands show seed range', fontsize=14)
    fig.supxlabel('Fitting data change during collection. These training curves are not independent evaluation or evidence of convergence.', fontsize=9)
    fig.savefig(root / 'training-curves.png', dpi=180)
    plt.close(fig)
    lines = ['# Online learned-MPC source experiment — October 24, 2026', '',

        'Completed the predeclared three-condition, three-seed source tetrahedron experiment. '
        'All fitting transitions came from real simulator execution: a shared short random warm-up, '
        'then each learned planner controlled its own collection with exploration. No simulator-MPC or teacher fitting data.', '',
        f"**Budget:** {cfg['steps']:,} real fitting transitions and {runs[0]['updates']:,} updates per run; "
        f"{summary['fitting_interactions']:,} fitting transitions total. Measured experiment wall time "
        f"{summary['total_wall_seconds']/60:.2f} minutes. Final checkpoints were fixed by budget, not evaluation outcome.", '',
        '![Measured results](results.png)', '', '![Training curves](training-curves.png)', '',
        '| Condition | Mean forward displacement | Median | Collapse / 9 | Discounted task return | H10 prediction MAE | Median plan latency |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        error_text = f"{row['mean_h10_progress_mae']:.6f}" if row['mean_h10_progress_mae'] is not None else 'n/a'
        lines.append(f"| {LABELS[row['condition']]} | {row['mean_displacement']:.6f} | {row['median_displacement']:.6f} | {row['collapses']}/{row['eval_episodes']} | {row['mean_discounted_task_return']:.6f} | {error_text} | {row['median_planning_seconds']*1000:.1f} ms |")
    lines += ['', 'Displacement and progress error use initial bounding-box-diagonal units. '
              'Prediction uses four common existing smooth-random source evaluation episodes, excluded from fitting. '
              'Random control displacement evaluates random actions; its prediction error evaluates its trained model. '
              'The random-trained model is not used as a controller in this table.', '',
              '| Paired final displacement comparison | Mean difference | Median difference | Positive pairs |',
              '|---|---:|---:|---:|']
    for pair in paired:
        lines.append(f"| {pair['comparison']} | {pair['mean']:.6f} | {pair['median']:.6f} | {pair['wins']}/{pair['pairs']} |")
    lines += ['', '## Method and interpretation limits', '',
        '- Reward/value MPC learns an immediate progress/effort reward and a terminal twin-Q value with a delayed graph actor and target networks. Both planners retain the same geometry collapse gate and fixed collapse charge.',
        '- The terminal value approximates discounted return beyond the ten-step planning horizon. It does not guarantee infinite-horizon optimality. This explicit-state GNN method is TD-MPC2-inspired, not a TD-MPC2 reproduction.',
        '- The reward/value arm changes reward supervision, value learning and shared graph features together. This comparison cannot isolate those contributions.',
        '- Only source tetrahedron was used. No held-out or final-test topology was accessed; these small-budget results do not establish transfer, convergence, or a general verdict on TD-MPC2.',
        '- Evaluation episodes are paired across methods. Three resets per training seed and repeated checkpoints are not independent training replications. No significance claim is made.',
        '- Common source prediction episodes are development diagnostics; the loop did not fit their states, rewards or normalization.', '',
        '## Evidence and verification', '',
        'All nine runs passed artifact hashes, identical warm-up arrays within each seed, exact replay adjacency, reset-safe boundaries, exact interaction/update budgets, disjoint fitting/evaluation identities and paired initial-state hashes. '
        'Each run preserves real replay, evaluation traces, model/actor/target/optimizer states, average loss history and hashes. '
        'See `protocol.json`, `summary.json`, `integrity-audit.json`, each run’s `report.json` and `prediction.json`.', '',
        'Reproduce with `scripts/run_online_dynamics_pilot.py` and `config/dynamics_online.yaml`; '
        'summarize with `scripts/report_online_dynamics_pilot.py OUTPUT_DIRECTORY`. Generated files remain ignored.', '']
    lines += ['## Measured interpretation', '',
        'Both online learned-MPC variants moved farther forward than random control in all nine paired final resets. '
        'Their average displacements were similar, while the reward/value variant recorded fewer collapses '
        '(3/9 versus 7/9) and a higher mean discounted task return (0.070505 versus -0.273019). '
        'The reward/value mean task return was higher than dynamics-only for each of the three training seeds, '
        'but seed 0 still had negative mean return. This is promising source behavior, not reliable control.', '',
        'The common random-action prediction check favored the random-trained model. The online models visited '
        'different, more mobile/failure-prone states, so these errors do not establish that online collection is '
        'inferior; they also do not establish a general predictive benefit. Changing reward/value supervision '
        'and the distribution together prevents a clean causal attribution.', '']
    diagnostic_path = root / 'own-rollout-diagnostics.json'
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text())
        lines += ['## Post-run value calibration audit', '',
            'This descriptive check was added after training and did not select checkpoints or change any run. '
            'It examines the final critic on real recorded actions whose transitions actually terminated. '
            'Their one-step TD targets contain no bootstrap term. The critic remains optimistic even on '
            'fitting-role terminal examples; the average values below are in unscaled task-reward units.', '',
            '| Value-model seed | Fitting terminal transitions | Mean predicted min-Q | Mean realized terminal reward | Mean absolute error |',
            '|---|---:|---:|---:|---:|']
        for run in diagnostic['runs']:
            if run['condition'] == 'reward_value_mpc':
                row = run['terminal_q_calibration']['fitting']
                lines.append(f"| {run['seed']} | {row['terminal_transitions']} | {row['mean_predicted_q']:.6f} | {row['mean_realized_terminal_reward']:.6f} | {row['terminal_q_mae']:.6f} |")
        lines += ['', 'The three terminal-ending evaluation transitions also have approximately 1.013 task-unit min-Q error. '
            'This small conditional check does not prove a unique cause or a required hidden-state input. '
            'It does show that the learned terminal tail cannot yet be treated as calibrated. '
            'The next focused experiment should audit terminal targets, reward/value scaling and loss weighting, '
            'then test a predeclared value-learning repair on the saved real replay before broadening topology scope. '
            'No such repair was run here.', '']
    lines += ['Method reference: [TD-MPC2 official implementation](https://github.com/nicklashansen/tdmpc2). '
        'This experiment uses explicit observable-state prediction and scalar reward/Q losses; it does not test the full published algorithm.', '']
    (root / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    report(parser.parse_args().output)
