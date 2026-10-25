"""Compare fixed online-MPC discount runs by rescoring saved real evaluation traces.

No model update, simulator step or new controller evaluation is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'sac')]
from scripts.run_online_dynamics_long import nested_equal
from scripts.collect_dynamics_training_data import write_json


def discounted_trace_return(rewards: np.ndarray, gamma: float) -> float:
    """Rescore every saved step including true-collapse charges and stopped tails."""
    rewards=np.asarray(rewards,dtype=np.float64)
    if rewards.ndim!=1 or not np.isfinite(rewards).all() or not 0<gamma<=1:
        raise ValueError('Expected finite one-dimensional rewards and 0 < gamma <= 1')
    return float(np.dot(rewards,np.power(gamma,np.arange(len(rewards)))))


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def compare(baseline: Path, rerun: Path) -> dict:
    """Verify pairing and show per-seed outcomes using common score discounts."""
    torch.set_num_threads(1)
    roots=[baseline,rerun]; documents=[json.loads((root/'report.json').read_text()) for root in roots]
    configs=[d['protocol']['settings'] for d in documents]
    if [key for key in configs[0] if configs[0][key]!=configs[1][key]]!=['gamma']:
        raise ValueError('Comparison must differ only in gamma')
    if [cfg['gamma'] for cfg in configs]!=[.995,.95]:
        raise ValueError('Expected predeclared .995 baseline and .95 rerun')
    cfg=configs[0];checks={};rows=[]; traces={}; digests={}
    expected={(condition,seed) for condition in cfg['conditions'] for seed in cfg['seeds']}
    for root,document in zip(roots,documents):
        policy_gamma=document['protocol']['settings']['gamma']
        runs={(r['condition'],r['seed']):r for r in document['runs']}
        if set(runs)!=expected or any(r['status']!='complete' for r in runs.values()):
            raise ValueError('All declared groups must complete')
        checks[f'campaign-audit:{policy_gamma}']=json.loads((root/'integrity-audit.json').read_text())['passed']
        for (condition,seed),run in runs.items():
            folder=root/f'{condition}-seed{seed}'
            for topology in cfg['topologies']:
                evaluations=[('monitoring',e,folder/topology) for e in run['evaluations'][topology]]
                evaluations.append(('fresh',run['fresh'][topology],folder/topology/'fresh-final'))
                for role,evaluation,trace_folder in evaluations:
                    for episode_index,episode in enumerate(evaluation['episodes']):
                        file=trace_folder/episode['trace']
                        digest=hashlib.sha256(file.read_bytes()).hexdigest()
                        checks[f'trace:{policy_gamma}:{condition}:{seed}:{topology}:{role}:{evaluation["steps"]}:{episode_index}']=digest==episode['sha256']
                        digests[str(file)]=digest
                        with np.load(file) as arrays:
                            rewards=arrays['reward'].copy()
                        identity=(condition,seed,topology,role,evaluation['steps'],episode_index)
                        traces[policy_gamma,identity]=episode
                        for score_gamma in [.95,.995,1.]:
                            score=discounted_trace_return(rewards,score_gamma)
                            if score_gamma==policy_gamma:
                                checks[f'return:{policy_gamma}:{identity}']=bool(np.isclose(score,episode['discounted_task_return'],rtol=1e-12,atol=1e-12))
                            rows.append(dict(policy_gamma=policy_gamma,score_gamma=score_gamma,condition=condition,seed=seed,
                                topology=topology,role=role,step=evaluation['steps'],episode=episode_index,reset_seed=episode['seed'],
                                discounted_task_return=score,displacement=episode['displacement'],collapsed=int(episode['collapsed']),transitions=episode['transitions']))
    identities={identity for policy,identity in traces}
    for identity in identities:
        a,b=traces[.995,identity],traces[.95,identity]
        checks['paired-start:'+str(identity)]=a['seed']==b['seed'] and a['initial_state_sha256']==b['initial_state_sha256']
    for condition,seed in expected:
        a_folder=baseline/f'{condition}-seed{seed}';b_folder=rerun/f'{condition}-seed{seed}'
        a=torch.load(a_folder/'initial-dynamics.pt',map_location='cpu',weights_only=False)
        b=torch.load(b_folder/'initial-dynamics.pt',map_location='cpu',weights_only=False)
        checks[f'initial:{condition}:{seed}']=nested_equal(a,b)
        checks[f'normalization:{condition}:{seed}']=json.loads((a_folder/'normalization.json').read_text())==json.loads((b_folder/'normalization.json').read_text())
        for topology in cfg['topologies']:
            with np.load(a_folder/topology/'warmup.npz') as first,np.load(b_folder/topology/'warmup.npz') as second:
                checks[f'warmup:{condition}:{seed}:{topology}']=all(np.array_equal(first[k],second[k]) for k in first.files)
    output=rerun/'discount-comparison';output.mkdir(exist_ok=True)
    audit=dict(passed=all(checks.values()),checks=checks,source_trace_hashes=digests)
    write_json(output/'integrity-audit.json',audit)
    if not audit['passed']:raise ValueError([key for key,value in checks.items() if not value])
    save_csv(output/'rescored-episodes.csv',rows)
    keyed={(r['policy_gamma'],r['score_gamma'],r['condition'],r['seed'],r['topology'],r['role'],r['step'],r['episode']):r for r in rows}
    paired=[]
    for row in rows:
        if row['policy_gamma']!=.95:continue
        original=keyed[(.995,row['score_gamma'],row['condition'],row['seed'],row['topology'],row['role'],row['step'],row['episode'])]
        paired.append(dict(score_gamma=row['score_gamma'],condition=row['condition'],seed=row['seed'],topology=row['topology'],role=row['role'],step=row['step'],episode=row['episode'],
            return_difference=row['discounted_task_return']-original['discounted_task_return'],displacement_difference=row['displacement']-original['displacement'],collapse_difference=row['collapsed']-original['collapsed']))
    save_csv(output/'paired-policy-gamma095-minus-gamma0995.csv',paired)
    figure=make_subplots(rows=3,cols=3,row_titles=cfg['topologies'],column_titles=['Return scored at .95','Return scored at .995','Undiscounted return'])
    lines=['# Discount-only source dynamics-MPC comparison','',
        'Both campaigns are complete. Planning discount changes from .995 to .95; all other settings, initial weights, random warmups and fitting-only normalization match. Frozen checkpoint resets pair exactly. Returns below compare controllers under the same scoring convention using their saved executed reward traces. No new fitting or simulator evaluation.',
        '', '[Common-discount learning curves](learning-curves.html)',
        '', '## Fresh final gamma=.95 controller minus gamma=.995 controller, every topology/seed','',
        '| Topology | Condition | Seed | Return difference scored .95 | Return difference scored .995 | Undiscounted difference | Movement difference | Collapse difference / 5 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for ti,topology in enumerate(cfg['topologies'],1):
        for condition in cfg['conditions']:
            for seed in cfg['seeds']:
                values=[r for r in paired if r['topology']==topology and r['condition']==condition and r['seed']==seed and r['role']=='fresh']
                means={g:float(np.mean([r['return_difference'] for r in values if r['score_gamma']==g])) for g in [.95,.995,1.]}
                unique=[r for r in values if r['score_gamma']==.95]
                lines.append(f"| {topology} | {condition} | {seed} | {means[.95]:+.6f} | {means[.995]:+.6f} | {means[1.]:+.6f} | {np.mean([r['displacement_difference'] for r in unique]):+.6f} | {sum(r['collapse_difference'] for r in unique):+d} |")
                for ci,score_gamma in enumerate([.95,.995,1.],1):
                    for policy_gamma in [.995,.95]:
                        samples=[r for r in rows if r['topology']==topology and r['condition']==condition and r['seed']==seed and r['role']=='monitoring' and r['score_gamma']==score_gamma and r['policy_gamma']==policy_gamma]
                        y=[float(np.mean([r['discounted_task_return'] for r in samples if r['step']==step])) for step in cfg['evaluate_at']]
                        figure.add_trace(go.Scatter(x=cfg['evaluate_at'],y=y,name=f'{condition} seed{seed} planner gamma={policy_gamma}',legendgroup=f'{condition}-{seed}-{policy_gamma}',showlegend=ti==1 and ci==1,
                            line=dict(color=('#1464b4' if condition=='shared' else '#d15e25'),dash=('solid' if policy_gamma==.95 else 'dash')),opacity=[1.,.75,.5][seed]),row=ti,col=ci)
    figure.update_layout(height=1150,width=1650,title='Same-discount scoring of each controller: source comparison, all seeds')
    figure.write_html(output/'learning-curves.html',include_plotlyjs=True)
    lines += ['', '## Limits','',
        'Each controller collects its own real experience. These are paired reset comparisons, not identical training trajectories after warmup. The .95 setting was selected after observing the original campaign and at the user request; this is source development evidence, not independent confirmation or unseen-topology transfer. A discounted-return improvement must be interpreted alongside displacement, collapse counts and undiscounted task return. More aggressive discounting can reduce the numerical charge of a late failure without improving survival. Three seeds and five fresh resets do not establish reliability. Native per-campaign reports contain within-gamma shared-minus-separate effects and prediction/latency tables.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    summary=dict(passed=True,episodes=len(rows)//3,paired_comparisons=len(paired),report=str(output/'report.md'))
    write_json(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--rerun',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(compare(args.baseline,args.rerun),indent=2))
