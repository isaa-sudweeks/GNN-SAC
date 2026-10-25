"""Summarize the fixed sampling ablation and paired frozen-control experiment."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ARMS = ('original', 'uniform', 'balanced')
TOPOLOGIES = ('tetrahedron', 'henneberg_n6_2tube_1', 'henneberg_n8_3tube_64')


def prediction_summary(rows: list[dict[str, Any]], cohort: str) -> dict[str, Any]:
    cells = [r for r in rows if r['horizon'] == 10 and r['cohort'] == cohort and r['windows']]
    count = sum(r['windows'] for r in cells)
    failures = sum(r['true_collapses'] for r in cells)
    return dict(windows=count, true_collapses=failures,
        any_false_negatives=sum(r['any_imagined']['false_negatives'] for r in cells),
        any_false_positives=sum(r['any_imagined']['false_positives'] for r in cells),
        endpoint_false_negatives=sum(r['endpoint']['false_negatives'] for r in cells),
        nonfinite_windows=sum(r['nonfinite_windows'] for r in cells),
        label_disagreements=sum(r['true_geometry_label_disagreements'] for r in cells),
        progress_mae=sum(r['progress_mae'] * r['windows'] for r in cells) / count if count else None,
        position_rmse=(sum(r['position_rmse'] ** 2 * r['windows'] for r in cells) / count) ** .5 if count else None,
        velocity_rmse=(sum(r['velocity_rmse'] ** 2 * r['windows'] for r in cells) / count) ** .5 if count else None)


def ratio(numerator: int, denominator: int) -> str:
    return f'{numerator}/{denominator}' if denominator else 'n/a (empty cohort)'


def build(run: Path) -> tuple[str, dict[str, Any]]:
    completion = json.loads((run / 'completion.json').read_text())
    if not completion['complete']:
        raise ValueError('Cannot report an incomplete campaign as completed')
    artifact_audit = json.loads((run / 'independent-verification.json').read_text())
    forward_audit = json.loads((run / 'frozen-forward-verification/verification.json').read_text())
    if not artifact_audit['passed'] or not forward_audit['passed']:
        raise ValueError('Final reporting requires both successful completion audits')
    controls, predictions, rankings, by_seed, sampling = defaultdict(list), defaultdict(list), defaultdict(list), [], []
    for seed in (0, 1, 2):
        for arm in ARMS:
            folder = run / (f'prepare-seed{seed}' if arm == 'original' else f'{arm}-seed{seed}')
            report = json.loads((folder / 'report.json').read_text())
            if report['status'] != 'complete':
                raise ValueError('Missing complete group report')
            for topology, cell in report['topologies'].items():
                episodes = cell['evaluation']
                controls[(topology, arm)].extend(episodes)
                predictions[(topology, arm)].extend(cell['diagnostic']['predictions'])
                rankings[(topology, arm)].extend(cell['diagnostic']['rankings'])
                by_seed.append(dict(seed=seed, topology=topology, arm=arm, episodes=episodes,
                    collapses=sum(r['collapsed'] for r in episodes),
                    displacement=float(np.mean([r['displacement'] for r in episodes])),
                    discounted_return=float(np.mean([r['discounted_return'] for r in episodes]))))
                if arm != 'original':
                    sampling.append(dict(seed=seed, topology=topology, arm=arm,
                        counts=cell['sampling']['counts'], availability=cell['availability']))
    control_summary, prediction_cells, ranking_cells, pairs = [], [], [], []
    ordering = lambda key: (TOPOLOGIES.index(key[0]), ARMS.index(key[1]))
    for topology, arm in sorted(controls, key=ordering):
        episodes = controls[(topology, arm)]
        control_summary.append(dict(topology=topology, arm=arm, episodes=len(episodes),
            collapses=sum(r['collapsed'] for r in episodes),
            displacement=float(np.mean([r['displacement'] for r in episodes])),
            discounted_return=float(np.mean([r['discounted_return'] for r in episodes])),
            task_return=float(np.mean([r['task_return'] for r in episodes])),
            raw_return=float(np.mean([r['raw_return'] for r in episodes])),
            median_episode_planning_seconds=float(np.median([r['median_planning_seconds'] for r in episodes]))))
        for cohort in ('terminal_ending', 'precollapse', 'ordinary'):
            prediction_cells.append(dict(topology=topology, arm=arm, cohort=cohort,
                **prediction_summary(predictions[(topology, arm)], cohort)))
        cells = rankings[(topology, arm)]
        informative = [r['pooled']['spearman'] for r in cells if r['pooled']['informative']]
        ranking_cells.append(dict(topology=topology, arm=arm, states=len(cells),
            selected_collapses=sum(r['pooled']['selected_collapse'] for r in cells),
            mean_regret=float(np.mean([r['pooled']['regret'] for r in cells])),
            median_spearman=float(np.median(informative)) if informative else None,
            false_negatives=sum(r['pooled']['false_negatives'] for r in cells),
            false_positives=sum(r['pooled']['false_positives'] for r in cells),
            true_failures=sum(r['pooled']['true_failures'] for r in cells),
            true_survivors=sum(r['pooled']['candidates'] - r['pooled']['true_failures'] for r in cells)))
    for seed in (0, 1, 2):
        topologies = TOPOLOGIES
        for topology in topologies:
            keyed = {row['arm']: row for row in by_seed if row['seed'] == seed and row['topology'] == topology}
            original = {r['seed']: r for r in keyed['original']['episodes']}
            uniform = {r['seed']: r for r in keyed['uniform']['episodes']}
            balanced = {r['seed']: r for r in keyed['balanced']['episodes']}
            if original.keys() != uniform.keys() or uniform.keys() != balanced.keys():
                raise ValueError('Reset identities do not pair')
            differences = []
            for reset, first in uniform.items():
                second = balanced[reset]
                if len({first['initial_state_sha256'], second['initial_state_sha256'], original[reset]['initial_state_sha256']}) != 1:
                    raise ValueError('Actual reset states do not pair')
                differences.append(dict(reset_seed=reset,
                    collapse_difference=int(second['collapsed']) - int(first['collapsed']),
                    displacement_difference=second['displacement'] - first['displacement'],
                    return_difference=second['discounted_return'] - first['discounted_return']))
            pairs.append(dict(seed=seed, topology=topology, episodes=len(differences),
                collapse_difference=sum(r['collapse_difference'] for r in differences),
                displacement_difference=float(np.mean([r['displacement_difference'] for r in differences])),
                return_difference=float(np.mean([r['return_difference'] for r in differences])), pairs=differences))
    measured = dict(completion=completion, control=control_summary, prediction=prediction_cells,
        ranking=ranking_cells, paired_by_seed=pairs, sampling=sampling, by_seed=by_seed)
    all_reports = [json.loads(path.read_text()) for path in run.glob('*-seed*/report.json')]
    aggregate_wall_minutes = sum(row['elapsed_seconds'] for row in all_reports) / 60
    nonfinite = sum(row.get('nonfinite_windows', 0) for cells in predictions.values() for row in cells)
    disagreements = sum(row.get('true_geometry_label_disagreements', 0) for cells in predictions.values() for row in cells)
    measured['aggregate_worker_wall_minutes'] = aggregate_wall_minutes
    keyed_control = {(r['topology'], r['arm']): r for r in control_summary}
    broad_gain = all(keyed_control[(t, 'balanced')]['collapses'] <= keyed_control[(t, 'uniform')]['collapses']
                     and keyed_control[(t, 'balanced')]['displacement'] >= keyed_control[(t, 'uniform')]['displacement']
                     for t in TOPOLOGIES)
    conclusion = ('Balanced replay improved or preserved both survival and mean movement across source means.'
                  if broad_gain else 'Failure-balanced replay did not provide a consistent source-control improvement at this budget.')
    lines = ['# Failure-balanced replay and frozen source control — October25,2026', '',
        '**' + conclusion + '**', '',
        f'Experiments18–19 completed in {completion["elapsed_seconds"] / 60:.2f} elapsed minutes. '
        'Three original shared-model training seeds, three source topologies, six continuation jobs; '
        '36,000 added shared Adam updates. No new fitting interactions or unseen robots. '
        'The user reserved Experiment3; no model-free/transfer comparison was executed.', '',
        f'Summed worker wall time including preparation, fitting/checkpoints and evaluation: '
        f'{aggregate_wall_minutes:.2f} minutes with three concurrent workers. Post-run audit '
        'and reporting are additional. There were no campaign failures or operational restarts.', '',
        'Plan: [reviewed protocol](../../plans/failure_balanced_replay.md). Notebook: '
        '`My vault/Research/06 Experiment Notebook/`, Experiments18 and19.', '',
        '## Measured interpretation', '',
        'The six-node source improves in two seeds and regresses in the third. Tetrahedron '
        'mean movement is lower in all three seeds under balanced replay; eight-node '
        'discounted task return is lower in all three. Across the three source means, '
        'balanced sampling is not a general repair. These are descriptive three-seed results, '
        'not a significance claim or a verdict on other sampling fractions, budgets or losses.', '',
        'Both continuation arms have28/45 total control collapses. Pooling would conceal '
        'the six-node improvement and regressions on the other sources. The original '
        'reference has27/45, so extra fixed-replay fitting itself is not a uniform gain.', '',
        'The eight-node fixed-pool selected-collapse metric improves from8/9 (uniform) to4/9 '
        '(balanced), but closed-loop collapses rise from14/15 to15/15. That improvement '
        'does not establish useful control: the pool comes from the original policy and '
        'open-loop branch outcomes differ from the repaired controller’s replanning/state '
        'distribution. Six-node terminal sensitivity ties at6/8 missed H10 events, while '
        'eight-node sensitivity is worse for balanced replay (5/8 missed versus2/8). '
        'Tetrahedron validation has no terminal-ending H10 cohort, so its failure sensitivity '
        'is inconclusive. Ordinary-state progress error is higher for balanced than uniform '
        'on each source mean. No unique cause of the control differences is identified.', '',
        'Retain these results as a test of the failure-emphasis hypothesis; do not adopt '
        '50/50 sampling as the general default on this evidence. No further repair or '
        'Experiment3 was launched.', '',
        '## Frozen closed-loop outcomes', '',
        '| Source | Model | Collapses /15 | Mean movement | Mean discounted task return |',
        '|---|---|---:|---:|---:|']
    for row in control_summary:
        lines.append(f"| {row['topology']} | {row['arm']} | {row['collapses']} | {row['displacement']:.6f} | {row['discounted_return']:.6f} |")
    lines += ['', 'Movement is forward unique-physical-node mean displacement divided by initial diagonal. '
        'Every row has three model seeds and five independent paired resets per seed. '
        'Episodes last at most80 controls/16 simulated seconds, with H10/2-second CEM planning. '
        'Artificial limits preserve their truncation labels; all true collapses are retained.', '',
        '## Paired balanced minus uniform effects, every seed', '',
        '| Source | Seed | Collapse difference /5 | Movement difference | Discounted return difference |',
        '|---|---:|---:|---:|---:|']
    for row in pairs:
        lines.append(f"| {row['topology']} | {row['seed']} | {row['collapse_difference']:+d} | {row['displacement_difference']:+.6f} | {row['return_difference']:+.6f} |")
    lines += ['', '## H10 collapse-sensitive validation', '',
        '| Source | Model | Terminal misses (any imagined collapse) | Terminal misses (endpoint only) | Ordinary false positives | Ordinary progress MAE |',
        '|---|---|---:|---:|---:|---:|']
    for topology, arm in sorted(predictions, key=ordering):
        terminal = next(r for r in prediction_cells if r['topology'] == topology and r['arm'] == arm and r['cohort'] == 'terminal_ending')
        ordinary = next(r for r in prediction_cells if r['topology'] == topology and r['arm'] == arm and r['cohort'] == 'ordinary')
        error = ordinary['progress_mae']
        lines.append(f"| {topology} | {arm} | {ratio(terminal['any_false_negatives'], terminal['true_collapses'])} | "
            f"{ratio(terminal['endpoint_false_negatives'], terminal['true_collapses'])} | "
            f"{ratio(ordinary['any_false_positives'], ordinary['windows'] - ordinary['true_collapses'])} | "
            f"{error:.6f} |" if error is not None else f'| {topology} | {arm} | empty | empty | empty | n/a |')
    lines += ['', 'Terminal, precollapse and ordinary cohorts are defined separately at each horizon. '
        'Primary H10 sensitivity uses any imagined collapse, matching planner stopping; endpoint '
        'sensitivity is also retained. All models see the same27 new original-policy validation '
        'episodes, excluded from fitting and normalization. Empty cohorts are inconclusive. '
        'Overlapping windows are not independent examples. Per-cell H1/5/10 state errors and '
        'true-geometry/terminal-label agreement are retained in each group report.', '',
        f'Across the recorded validation cells, nonfinite windows={nonfinite} and '
        f'true-geometry/terminal-label disagreements={disagreements}; repeated cells across '
        'models/horizons are not independent event counts.', '',
        '## Identical fixed validation candidate pools', '',
        '| Source | Model | Selected collapse /9 | Mean regret | Median Spearman | Missed collapsing candidates | False alarms |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in ranking_cells:
        correlation = f"{row['median_spearman']:.4f}" if row['median_spearman'] is not None else 'uninformative'
        lines.append(f"| {row['topology']} | {row['arm']} | {row['selected_collapses']} | {row['mean_regret']:.6f} | {correlation} | "
            f"{ratio(row['false_negatives'], row['true_failures'])} | {ratio(row['false_positives'], row['true_survivors'])} |")
    lines += ['', 'Each validation state has41 candidates:32 original-model CEM proposals plus zero '
        'and eight smooth-random sequences. Twenty-seven pools/1,107 branches were fixed before '
        'fitting. These are a common original-policy distribution, not the repaired models’ '
        'refined proposal distributions. Unique-sequence sensitivity is retained. Branches '
        'execute open-loop while deployed MPC replans after each first action.', '',
        '## Failure-adjacent fitting availability', '',
        '| Source | Seed | Failure-adjacent transitions / all | Failure-adjacent valid windows / all |',
        '|---|---:|---:|---:|']
    for row in sampling:
        if row['arm'] != 'uniform':
            continue
        one, window = row['availability']['one_step'], row['availability']['recursive_positions']
        lines.append(f"| {row['topology']} | {row['seed']} | {one['failure']}/{one['failure'] + one['ordinary']} | "
            f"{window['failure']}/{window['failure'] + window['ordinary']} |")
    lines += ['', 'These pools are identical across continuation arms within seed. Full exposure '
        'counts and any empty-stratum fallbacks are retained by arm/source/seed.', '',
        '## Fitting intervention and interpretation', '',
        'Both arms start from exactly the same20,000-transition/source shared model, frozen '
        'normalization and Adam state within seed. Each receives2,000 extra batches/source '
        '(6,000 shared updates), ending at66,000 Adam steps and22,000 updates per view. '
        'Each source/arm/seed gets64,000 one-step draws and64,000 five-step windows. '
        'Balanced sampling draws16 failure-adjacent and16 complement examples per batch '
        'when both strata exist; uniform retains the original distribution. Loss, model, '
        'learning rate, clipping and target update are unchanged. No importance correction '
        'is applied: this intentionally changes the effective objective distribution.', '',
        'Failure-adjacent means endpoint distance0..10 to an observed true terminal, with '
        'episode-safe recursive windows. The complement is not proven safe and includes '
        'budget-cut episodes. Availability, actual exposure counts and fallback calls are '
        'reported by source/seed in measured-summary.json and fitting reports. Training '
        'losses come from different sampling distributions and are not a common validation metric.', '',
        'This is a fixed-replay continuation test of one sampling intervention, not a '
        'from-scratch online replication. Source validation is assessed at the fixed final '
        'checkpoint; no best-validation selection or conditional suppression of control '
        'results. Survival gains must be read with movement/return and false alarms. '
        'No binary repair-pass threshold or significance claim is inferred from three seeds.', '',
        '## Artifacts and verification', '',
        f'Raw run: `{run.resolve()}`. `protocol.json` hashes source/config/checkpoints; '
        '`prepared-data-manifest.json` freezes validation and original-reference evidence '
        'before fitting. Each fitting job retains full shared optimizer/model/target and '
        'sampler restart state, averaged losses, final diagnostics and control traces. '
        '`completion.json` records elapsed time and no Experiment3 execution. '
        'Large generated artifacts remain ignored.', '',
        'Twenty-one affected tests pass, including exact uniform/native fitting parity, '
        'full-round model/Adam/target/sampler restart, stale-progress recovery and actual '
        'simulator reset/boundary/restoration smoke. The independent artifact audit passes '
        '14,386 checks, including67 source/input and147 frozen-data hashes, exact sampling '
        'RNG/exposure accounting, paired resets and raw task-reward recomputation. A separate '
        'one-thread no_grad frozen-forward audit passes15,488 checks, reproducing729 '
        'prediction cells and3,321 candidate score predictions without simulator execution '
        'or optimizer steps. Both audit scripts/results are preserved with the outputs.', '']
    return '\n'.join(lines), measured


def curves(run: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    topologies = ['tetrahedron', 'henneberg_n6_2tube_1', 'henneberg_n8_3tube_64']
    figure = make_subplots(rows=3, cols=1, subplot_titles=topologies, shared_xaxes=True)
    colors = ['#2563eb', '#d97706', '#059669']
    for seed in (0, 1, 2):
        for arm in ('uniform', 'balanced'):
            history = json.loads((run / f'{arm}-seed{seed}' / 'report.json').read_text())['history']
            for index, topology in enumerate(topologies):
                figure.add_trace(go.Scatter(x=[r['round'] for r in history],
                    y=[r['losses'][topology]['loss'] for r in history], mode='lines',
                    name=f'{arm}, seed{seed}', legendgroup=f'{arm}-{seed}', showlegend=index == 0,
                    line=dict(color=colors[seed], dash='solid' if arm == 'uniform' else 'dash')),
                    row=index + 1, col=1)
    figure.update_layout(title='Replay continuation losses — different sampling distributions',
                         height=900, template='plotly_white')
    figure.update_xaxes(title_text='Additional updates per source', row=3, col=1)
    figure.update_yaxes(title_text='Mean fitting loss')
    figure.write_html(run / 'learning-curves.html', include_plotlyjs=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    text, measured = build(args.run)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(text)
    (args.run / 'report.md').write_text(text)
    (args.run / 'measured-summary.json').write_text(json.dumps(measured, indent=2, allow_nan=False))
    curves(args.run)
    (args.run / 'report-provenance.json').write_text(json.dumps(dict(
        reporting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        report_sha256=hashlib.sha256(text.encode()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
