"""Fit and evaluate the source-only observable-input dynamics baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sac")]
from dynamics.dataset import load_dataset
from dynamics.training import evaluate_recursive, fit_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'outputs/dynamics-pilot/source-training-2026-10-23')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/dynamics-pilot/model-source-2026-10-23')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/dynamics_model.yaml')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite run: {args.output}')
    settings = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    fitting = load_dataset(args.data, roles=['fitting'])
    validation = load_dataset(args.data, roles=['validation'])
    model, fit = fit_model(fitting, validation, settings, args.output)
    (args.output / 'data-provenance.json').write_text(json.dumps(dict(
        dataset_root=str(args.data.resolve()), manifest_sha256=hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest(),
        fitting_files=[e['file'] for e in fitting], validation_files=[e['file'] for e in validation]), indent=2))
    evaluation = load_dataset(args.data, roles=['evaluation'])
    reports = {}
    for role, episodes in [('validation', validation), ('evaluation', evaluation)]:
        if episodes:
            report = evaluate_recursive(model, episodes, settings['horizons'], settings['device'])
            report['revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            report['checkpoint_sha256'] = fit['checkpoint_sha256']
            reports[role] = report
            (args.output / f'{role}-recursive.json').write_text(json.dumps(report, indent=2))
    if 'validation' in reports and 'evaluation' in reports:
        horizon = 10
        choices = {}
        validation_baselines = {name: [r['progress_mae'] for r in reports['validation']['summary']
            if r['horizon'] == horizon and r['predictor'] == name] for name in ('persistence', 'constant_velocity')}
        global_errors = {name: sum(values)/len(values) for name, values in validation_baselines.items()}
        global_choice = min(global_errors, key=global_errors.get)
        for row in reports['validation']['summary']:
            if row['horizon'] == horizon and row['predictor'] in ('persistence', 'constant_velocity'):
                topology = row['topology']
                if topology not in choices or row['progress_mae'] < choices[topology]['validation_progress_mae']:
                    choices[topology] = dict(predictor=row['predictor'], validation_progress_mae=row['progress_mae'])
        screens = []
        for topology, choice in choices.items():
            selected = [r for r in reports['evaluation']['summary'] if r['topology'] == topology and r['horizon'] == horizon]
            by_predictor = {r['predictor']: r for r in selected}
            baseline = by_predictor[choice['predictor']]['progress_mae']
            gnn = by_predictor['gnn']['progress_mae']
            screens.append(dict(topology=topology, horizon=horizon, **choice,
                evaluation_baseline_progress_mae=baseline, evaluation_gnn_progress_mae=gnn,
                absolute_improvement=baseline-gnn,
                relative_improvement=(baseline-gnn)/baseline if baseline > 1e-8 else None,
                passes_25_percent_screen=bool(baseline > 1e-8 and gnn <= .75*baseline and not by_predictor['gnn']['nonfinite_windows'])))
        (args.output / 'source-prediction-screen.json').write_text(json.dumps(dict(
            rule='baseline chosen per source topology on validation horizon10, frozen before source evaluation interpretation; 25% progress MAE reduction provisional source screen',
            global_baseline=dict(predictor=global_choice, topology_balanced_validation_progress_mae=global_errors,
                rule='choose once on topology-balanced validation horizon10; freeze for unseen-topology evaluation'),
            screens=screens, all_sources_pass=all(r['passes_25_percent_screen'] for r in screens)), indent=2))
    print(json.dumps(fit, indent=2))

if __name__ == '__main__':
    main()
