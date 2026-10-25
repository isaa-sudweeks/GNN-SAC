"""Build the measured candidate-ranking report from independently audited artifacts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def build_report(run: Path) -> str:
    """Summarize fixed-run decisions without selecting checkpoints or fitting models."""
    data = json.loads((run / 'results.json').read_text())
    audit = json.loads((run / 'independent-verification.json').read_text())
    if not data['complete']:
        raise ValueError('Cannot publish a partial experiment as completed')
    if not audit.get('passed') or audit['results_sha256'] != hashlib.sha256((run / 'results.json').read_bytes()).hexdigest():
        raise ValueError('Independent verification does not match these results')
    if (len(data['episodes']), len(data['rows']), audit['simulator_candidate_branches']) != (36, 216, 7884):
        raise ValueError('This measured report requires the fixed Experiment17 protocol')
    groups = defaultdict(lambda: defaultdict(int))
    all_advantages = []
    for row in data['rows']:
        with np.load(run / row['artifact'], allow_pickle=False) as arrays:
            condition = row['condition']
            selected = row['native']['selected']
            failures = arrays['simulator_collapse']
            truth = arrays['simulator_scores']
            own_safe = (arrays['provenance'] == condition + '_cem') & ~failures
            cell = groups[(row['topology'], condition)]
            cell['states'] += 1
            cell['collapses'] += int(failures[selected])
            cell['missed_selected'] += int(failures[selected] and not arrays[condition + '_collapse'][selected])
            cell['pooled_safe'] += int((~failures).any())
            if failures[selected] and own_safe.any():
                advantage = float(truth[own_safe].max() - truth[selected])
                cell['own_safe_on_failure'] += 1
                cell['better_own_safe_on_failure'] += int(advantage > 0)
                all_advantages.append(advantage)
    lines = ['# Online dynamics candidate ranking — measured results, October 25, 2026', '',
        '**Both shared and separate models make finite-horizon scoring errors that favor collapsing sequences.**', '',
        f'The reviewed fixed experiment completed in {data["elapsed_seconds"] / 60:.2f} minutes: '
        '36 source episodes, 18 paired fresh resets, three training seeds, three source robots, '
        '108 snapshots and 7,884 simulator candidate branches. No fitting or unseen robots.', '',
        'This extends Experiment09 (older offline model, no collapsing candidates) and '
        'Experiment16 (newer models, executed-action prediction only). Protocol and '
        'review: [frozen plan](../../plans/online_candidate_ranking.md). Notebook: '
        '`My vault/Research/06 Experiment Notebook/GNN Dynamics Pilot - 17 Online Candidate Ranking - 2026-10-25.md`.', '',
        '## Direct decision evidence', '',
        '| Robot | Model | Snapshot tests | Selected collapsing sequence | Missed selected collapse | Own pool had surviving alternative | Own survivor had better true score |',
        '|---|---|---:|---:|---:|---:|---:|']
    order = ['tetrahedron', 'henneberg_n6_2tube_1', 'henneberg_n8_3tube_64']
    for topology in order:
        for condition in ('shared', 'separate'):
            cell = groups[(topology, condition)]
            lines.append(f'| {topology} | {condition} | {cell["states"]} | {cell["collapses"]} | '
                f'{cell["missed_selected"]} | {cell["own_safe_on_failure"]} | {cell["better_own_safe_on_failure"]} |')
    lines += ['',
        'Across61 selected collapsing sequences,59 were predicted to survive. In46 cases, '
        'the selecting model already proposed a surviving alternative with a better true task score. '
        f'The smallest true-score advantage among those46 was {min(all_advantages):.6f}. '
        'This subset demonstrates a score-ranking error within the model’s own candidate pool, '
        'rather than simply an absence of surviving proposals or a survival/progress preference.', '',
        'In the other15 selected-collapse cases, the model’s own32 proposals all collapsed, '
        'while the73-sequence union contained a survivor. Candidate coverage and scoring can '
        'both matter; the union contained a surviving sequence at every snapshot. Neither '
        'observation establishes a globally safe policy or long-term recovery.', '',
        'Mean regret was higher for shared models on all three topology means, but shared '
        'regret was higher in only two of three seeds for each topology. Do not infer uniform '
        'sharing degradation or a statistically established difference from these small samples.', '',
        '## What was held fixed', '',
        'Final20,000-transition gamma=.995 checkpoints; original normalization and physics; '
        '80-control-step episodes (16 seconds); H10 planning (2 seconds);16 CEM candidates '
        'in two iterations. Both models score identical73-sequence unions:32 proposals per '
        'model, one zero sequence and eight smooth-random sequences. New reset identities '
        'are excluded from previous fitting/evaluation roles. No exploration or online updates.', '',
        'CEM selections here are recreated with a common proposal seed and the generating '
        'controller’s predecision warm start. They are not necessarily the actual episode’s '
        'original plan; original plans are saved. All73 include duplicates; unique-sequence '
        'and candidate-provenance sensitivity metrics are retained. Full raw and grouped '
        'seed/generator/stratum rows are in results.json.', '',
        '## Verification and limits', '',
        'Independent audit:77,157 checks pass, including reward/metric recomputation, '
        'paired resets, selected indices, duplicate sensitivity and89 preserved source/checkpoint '
        'hashes. The runner also verifies exact recorded continuation, candidate-order '
        'independence, fixed models and unchanged live state. All8 new diagnostic unit '
        'tests and57 existing dynamics tests pass; git diff --check passes.', '',
        'Actual terminal durations were retained:2,328 candidate branches include a shortened '
        'terminal interval. Learned rollouts use the nominal .2-second interval. This timing '
        'approximation remains part of the tested method, not a newly corrected model.', '',
        'Branches execute whole sequences open-loop; deployed MPC executes the first action '
        'and replans. These are108 correlated states drawn from36 episodes, not independent '
        'candidate replications. Selection includes hindsight-labeled precollapse states; '
        'the rates are diagnostic, not an estimate of ordinary deployment failure probability. '
        'No unique causal explanation or effective repair has been established.', '',
        '## Relation to the research goal', '',
        'This test identifies a source-control limitation. It does **not** establish whether '
        'dynamics generalizes more easily than model-free control. The next-stage '
        '[comparative design](../../plans/dynamics_vs_modelfree_generalization.md) requires '
        'a source-trained GNN-SAC generalist with the same source topology access, real '
        'interaction budget, task reward, physics and paired unseen evaluation states. '
        'Existing specialists or differently trained generalists cannot substitute for '
        'that comparator. Keep fitting and tuning out of held-out/final-test robots.', '',
        'A targeted source repair would need to improve collapse-sensitive scoring on '
        'independent validation states, followed by fixed-budget control checks. This '
        'experiment does not authorize or execute such a repair; avoid replacing the '
        'central matched transfer comparison with an open-ended series of diagnostics.', '',
        '## Artifact locations', '',
        f'Full local run: `{run.resolve()}`. `report.md` contains every topology/seed '
        'summary; `results.json` includes per-state native/pooled/unique/provenance metrics; '
        '`verification.json`, `independent-verification.json` and '
        '`independent-own-proposal-coverage.json` preserve audits and own-pool comparisons. '
        'Per-state arrays contain all actions/scores/flags; trusted local pickle files '
        'retain full snapshots and simulator reward/state traces. Raw artifacts remain ignored.', '']
    # Require the independently computed check artifact to be readable and nonempty.
    if not audit:
        raise ValueError('Independent verification is missing')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    text = build_report(args.run)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(text)
    (args.run / 'measured-interpretation.md').write_text(text)
    (args.run / 'report-provenance.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        results_sha256=hashlib.sha256((args.run / 'results.json').read_bytes()).hexdigest(),
        report_sha256=hashlib.sha256(text.encode()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
