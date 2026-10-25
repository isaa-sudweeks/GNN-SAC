"""Post-run descriptive errors on each frozen controller's own evaluation traces.

These diagnostics cannot select weights or alter the predeclared training protocol.
The controllers visit different states, so this is not a common-distribution ranking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'sac')]
from dynamics.online import OnlineLearner
from dynamics.training import evaluate_recursive


def diagnose(root: Path) -> dict:
    document = json.loads((root / 'report.json').read_text())
    cfg = document['protocol']['settings']
    torch.set_num_threads(1)
    reports = []
    for run in document['runs']:
        folder = root / f"{run['condition']}-seed{run['seed']}"
        metadata = json.loads((folder / 'metadata.json').read_text())
        episodes = []
        for index, record in enumerate(run['evaluations'][-1]['episodes']):
            with np.load(folder / record['trace']) as data:
                episodes.append({**metadata, **{k: data[k] for k in data.files},
                    'role': 'evaluation', 'episode_id': index, 'topology': cfg['topology']})
        learner = OnlineLearner(metadata, episodes[0]['state'][0], cfg, run['seed'], run['condition'] == 'reward_value_mpc')
        saved = torch.load(folder / 'final-model.pt', weights_only=False, map_location='cpu')
        learner.model.load_state_dict(saved['model'])
        learner.actor.load_state_dict(saved['actor'])
        reward_errors, progress_errors, false_negatives, failures, false_positives, total = [], [], 0, 0, 0, 0
        with torch.no_grad():
            for episode in episodes:
                state = torch.tensor(episode['state'], dtype=torch.float32)
                action = torch.tensor(episode['action'], dtype=torch.float32)
                predicted, move = learner.model.dynamics.step(state, action, learner.graph)
                prediction = learner.model.task(state, action, learner.graph)[0] * cfg['reward_scale']
                reward_errors.extend((prediction - torch.tensor(episode['base_reward'])).abs().tolist())
                progress_errors.extend((move[:, 0] - torch.tensor(episode['progress'][:, 0])).abs().div(metadata['initial_diagonal']).tolist())
                collapse = learner.collapsed(predicted).numpy()
                truth = np.asarray(episode['terminated'], dtype=bool)
                failures += int(truth.sum())
                false_negatives += int((truth & ~collapse).sum())
                false_positives += int((~truth & collapse).sum())
                total += len(truth)
        recursive = evaluate_recursive(learner.model.dynamics, episodes, [1, 5, 10, 20])
        recursive['claim_boundary'] = 'Frozen own-controller source evaluation errors; descriptive; no fitting or selection and no shared state distribution across methods'
        (folder / 'own-rollout-prediction.json').write_text(json.dumps(recursive, indent=2))
        calibration = {}
        if learner.use_value:
            for role, arrays in [('fitting', dict(np.load(folder / 'replay.npz'))),
                                 ('evaluation', {k: np.concatenate([e[k] for e in episodes]) for k in ('state', 'action', 'terminated', 'reward')})]:
                terminal = np.asarray(arrays['terminated'], dtype=bool)
                if terminal.any():
                    with torch.no_grad():
                        _, q1, q2 = learner.model.task(torch.tensor(arrays['state'][terminal], dtype=torch.float32),
                            torch.tensor(arrays['action'][terminal], dtype=torch.float32), learner.graph)
                        predicted_q = torch.minimum(q1, q2).numpy() * cfg['reward_scale']
                    observed = np.asarray(arrays['reward'])[terminal]
                    calibration[role] = dict(terminal_transitions=int(terminal.sum()),
                        mean_predicted_q=float(predicted_q.mean()), mean_realized_terminal_reward=float(observed.mean()),
                        terminal_q_mae=float(np.abs(predicted_q - observed).mean()),
                        mean_optimism=float((predicted_q - observed).mean()))
                else:
                    calibration[role] = dict(terminal_transitions=0)
        reports.append(dict(condition=run['condition'], seed=run['seed'], transitions=total,
            one_step_progress_mae=float(np.mean(progress_errors)),
            learned_base_reward_mae=float(np.mean(reward_errors)) if learner.use_value else None,
            true_terminal_transitions=failures, one_step_collapse_false_negatives=false_negatives,
            one_step_collapse_false_positives=false_positives,
            terminal_q_calibration=calibration,
            horizon10=[r for r in recursive['summary'] if r['horizon'] == 10 and r['predictor'] == 'gnn']))
    result = dict(runs=reports, status='descriptive post-run diagnostics; no additional training or controller selection')
    (root / 'own-rollout-diagnostics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    diagnose(parser.parse_args().output)
