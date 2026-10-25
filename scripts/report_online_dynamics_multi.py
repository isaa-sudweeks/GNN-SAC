"""Verify and report the predeclared shared versus separate source experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from collections import defaultdict

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'sac')]
from dynamics.dataset import file_sha256
from dynamics.online import OnlineLearner
from dynamics.training import evaluate_recursive
from scripts.collect_dynamics_training_data import write_json
from scripts.run_online_dynamics_long import load_resume,nested_equal


def csv_write(path: Path, rows: list[dict]) -> None:
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report(root: Path) -> dict:
    torch.set_num_threads(1)
    document=json.loads((root/'report.json').read_text())
    cfg=document['protocol']['settings']; runs={(r['condition'],r['seed']):r for r in document['runs']}
    if len(runs)!=6 or any(r['status']!='complete' for r in runs.values()):raise ValueError('Incomplete declared campaign')
    checks={}; rows=[]; paired=[]; predictions=[]; common=[]
    for path,digest in document['protocol']['sources'].items():
        checks['source:'+path]=file_sha256(ROOT/path)==digest
    for seed in cfg['seeds']:
        initial=[]; saved={}; norms=[]
        for condition in cfg['conditions']:
            folder=root/f'{condition}-seed{seed}'
            initial.append(torch.load(folder/'initial-dynamics.pt',weights_only=False))
            saved[condition]=load_resume(folder,cfg)
            norms.append(saved[condition]['statistics'])
        checks[f'initial-seed{seed}']=nested_equal(initial[0],initial[1]) and all(nested_equal(initial[0][cfg['topologies'][0]],v) for v in initial[0].values())
        checks[f'normalization-seed{seed}']=norms[0]==norms[1]
        for topology in cfg['topologies']:
            evaluation_by_condition={}
            for condition in cfg['conditions']:
                run=runs[condition,seed]; folder=root/f'{condition}-seed{seed}'/topology
                state=saved[condition]['topologies'][topology]
                checks[f'budget:{condition}:{seed}:{topology}']=run['interactions'][topology]==cfg['steps'] and run['updates'][topology]==cfg['steps'] and state['updates']==cfg['steps']
                optimizer_steps={int(v['step']) for v in state['optimizer']['state'].values()}
                checks[f'adam:{condition}:{seed}:{topology}']=optimizer_steps=={cfg['steps']*(3 if condition=='shared' else 1)}
                data=state['replay']; episode=data['episode'];same=episode[1:]==episode[:-1]
                checks[f'boundaries:{condition}:{seed}:{topology}']=not bool(((data['terminated'][:-1]|data['truncated'][:-1])&same).any()) and np.array_equal(data['next_state'][:-1][same],data['state'][1:][same])
                checks[f'mask:{condition}:{seed}:{topology}']=bool(np.all(data['action'][:,~np.asarray(state['metadata']['action_mask'],bool)]==0))
                with np.load(folder/'replay.npz') as final:
                    checks[f'replay:{condition}:{seed}:{topology}']=all(np.array_equal(data[k],final[k]) for k in data)
                evaluations=[('monitoring',e,folder) for e in run['evaluations'][topology]]+[('fresh',run['fresh'][topology],folder/'fresh-final')]
                checks[f'checkpoints:{condition}:{seed}:{topology}']=[e['steps'] for role,e,_ in evaluations if role=='monitoring']==cfg['evaluate_at']
                checks[f'roles:{condition}:{seed}:{topology}']=not ({r['seed'] for e in run['evaluations'][topology] for r in e['episodes']} & {r['seed'] for r in run['fresh'][topology]['episodes']})
                training_seeds=set(range(state['runtime']['reset_base'],state['runtime']['reset_base']+state['runtime']['episode']+1))
                evaluation_seeds={row['seed'] for _,e,_ in evaluations for row in e['episodes']}
                checks[f'train-eval-seeds:{condition}:{seed}:{topology}']=not bool(training_seeds & evaluation_seeds)
                normalization=state['model']
                checks[f'frozen-normalization:{condition}:{seed}:{topology}']=all(torch.equal(normalization['dynamics.'+key],torch.tensor(norms[0][stat],dtype=torch.float32)) for key,stat in [('state_mean','mean'),('state_scale','scale'),('delta_scale','delta_scale'),('geometry_scale','geometry_scale')])
                evaluation_by_condition[condition]=evaluations
                for role,evaluation,trace_folder in evaluations:
                    for index,episode in enumerate(evaluation['episodes']):
                        checks[f'trace:{condition}:{seed}:{topology}:{role}:{evaluation["steps"]}:{index}']=file_sha256(trace_folder/episode['trace'])==episode['sha256']
                        rows.append(dict(condition=condition,seed=seed,topology=topology,role=role,step=evaluation['steps'],episode=index,
                            reset_seed=episode['seed'],displacement=episode['displacement'],discounted_task_return=episode['discounted_task_return'],
                            collapsed=int(episode['collapsed']),raw_return=episode['raw_return'],planning_median_seconds=evaluation['median_planning_seconds'],planning_p95_seconds=evaluation['p95_planning_seconds']))
                    for metric in evaluation['prediction']['summary']:
                        predictions.append(dict(condition=condition,seed=seed,role=role,step=evaluation['steps'],**metric))
            for (role,a,af),(other_role,b,bf) in zip(evaluation_by_condition['shared'],evaluation_by_condition['separate']):
                assert role==other_role and a['steps']==b['steps']
                for i,(x,y) in enumerate(zip(a['episodes'],b['episodes'])):
                    checks[f'paired:{seed}:{topology}:{role}:{a["steps"]}:{i}']=x['seed']==y['seed'] and x['initial_state_sha256']==y['initial_state_sha256']
                    paired.append(dict(seed=seed,topology=topology,role=role,step=a['steps'],episode=i,
                        displacement=x['displacement']-y['displacement'],discounted_task_return=x['discounted_task_return']-y['discounted_task_return'],
                        collapses=int(x['collapsed'])-int(y['collapsed']),planning_seconds=a['median_planning_seconds']-b['median_planning_seconds']))
            # Common final action/state distribution: the union of both fresh trace sets.
            episodes=[]
            for condition in cfg['conditions']:
                folder=root/f'{condition}-seed{seed}'/topology/'fresh-final'
                for i,record in enumerate(runs[condition,seed]['fresh'][topology]['episodes']):
                    with np.load(folder/record['trace']) as data:
                        episodes.append({**saved[condition]['topologies'][topology]['metadata'],**dict(data),
                            'role':'evaluation','episode_id':f'{condition}-{i}'})
            for condition in cfg['conditions']:
                state=saved[condition]['topologies'][topology]
                learner=OnlineLearner(state['metadata'],episodes[0]['state'][0],cfg,seed,False)
                learner.model.load_state_dict(state['model'])
                metrics=evaluate_recursive(learner.model.dynamics,episodes,[1,5,10,20])
                common.extend(dict(condition=condition,seed=seed,**r) for r in metrics['summary'] if r['predictor']=='gnn')
            with np.load(root/f'shared-seed{seed}'/topology/'warmup.npz') as a,np.load(root/f'separate-seed{seed}'/topology/'warmup.npz') as b:
                checks[f'warmup:{seed}:{topology}']=all(np.array_equal(a[k],b[k]) for k in a.files)
        del saved
    audit=dict(passed=all(checks.values()),checks=checks)
    write_json(root/'integrity-audit.json',audit)
    if not audit['passed']:raise ValueError([k for k,v in checks.items() if not v])
    csv_write(root/'episodes.csv',rows);csv_write(root/'paired-differences.csv',paired)
    write_json(root/'prediction-errors.json',dict(own=predictions,common_fresh=common))
    curves=make_subplots(rows=3,cols=3,row_titles=cfg['topologies'],column_titles=['Displacement / diagonal','Discounted task return','Collapses / 3'])
    colors={'shared':'#1464b4','separate':'#d15e25'}
    for ti,topology in enumerate(cfg['topologies'],1):
        for condition in cfg['conditions']:
            for seed in cfg['seeds']:
                for col,key in enumerate(['displacement','discounted_task_return','collapsed'],1):
                    values=[r for r in rows if r['condition']==condition and r['seed']==seed and r['topology']==topology and r['role']=='monitoring']
                    y=[float(np.mean([r[key] for r in values if r['step']==step])) if key!='collapsed' else sum(r[key] for r in values if r['step']==step) for step in cfg['evaluate_at']]
                    curves.add_trace(go.Scatter(x=cfg['evaluate_at'],y=y,name=f'{condition} seed {seed}',legendgroup=f'{condition}-{seed}',showlegend=ti==1 and col==1,
                        mode='lines+markers',line=dict(color=colors[condition],dash=['solid','dash','dot'][seed])),row=ti,col=col)
    curves.update_layout(height=1000,width=1400,title='Source topology learning curves: every training seed; fixed monitoring resets')
    curves.write_html(root/'learning-curves.html',include_plotlyjs=True)
    freshfig=make_subplots(rows=1,cols=3,subplot_titles=cfg['topologies'])
    final=[]
    for ti,topology in enumerate(cfg['topologies'],1):
        for seed in cfg['seeds']:
            p=[r for r in paired if r['topology']==topology and r['seed']==seed and r['role']=='fresh']
            delta=float(np.mean([r['discounted_task_return'] for r in p]))
            freshfig.add_trace(go.Scatter(x=[seed]*len(p),y=[r['discounted_task_return'] for r in p],mode='markers',name=f'seed {seed}',showlegend=ti==1),row=1,col=ti)
            final.append(dict(topology=topology,seed=seed,return_difference=delta,displacement_difference=float(np.mean([r['displacement'] for r in p])),collapse_difference=sum(r['collapses'] for r in p)))
        freshfig.add_hline(y=0,line_dash='dash',row=1,col=ti)
    freshfig.update_layout(title='Fresh final episodes: paired shared minus separate task return',height=450,width=1400)
    freshfig.write_html(root/'fresh-final.html',include_plotlyjs=True)
    # Portable SVG exports when Kaleido's browser is available; HTML is always retained.
    export_errors=[]
    for figure,name in [(curves,'learning-curves'),(freshfig,'fresh-final')]:
        try:figure.write_image(root/f'{name}.svg')
        except Exception as error:export_errors.append(str(error))
    write_json(root/'summary.json',dict(final=final,integrity_passed=True,export_errors=export_errors))
    lines=['# Shared versus separate source dynamics-MPC results','',
        'All six groups completed: 360,000 real fitting transitions; 20,000 transitions and updates per topology/condition/seed. All integrity checks pass.',
        '',f"Elapsed campaign time: {document['wall_seconds']/3600:.2f} hours. Summed group time: {sum(r['wall_seconds'] for r in runs.values())/3600:.2f} hours.",
        '', '[Interactive learning curves](learning-curves.html) · [Fresh paired outcomes](fresh-final.html)',
        '', '## Fresh final outcomes by topology and training seed','',
        '| Topology | Seed | Shared return | Separate return | Shared − separate return | Displacement difference | Collapse difference |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in final:
        values={c:[r for r in rows if r['condition']==c and r['seed']==row['seed'] and r['topology']==row['topology'] and r['role']=='fresh'] for c in cfg['conditions']}
        lines.append(f"| {row['topology']} | {row['seed']} | {np.mean([r['discounted_task_return'] for r in values['shared']]):.6f} | {np.mean([r['discounted_task_return'] for r in values['separate']]):.6f} | {row['return_difference']:+.6f} | {row['displacement_difference']:+.6f} | {row['collapse_difference']:+d} |")
    lines += ['', '## Final metrics for every model', '',
        '| Topology | Seed | Model | Movement | Return | Collapses / 5 | Plan median ms | Plan p95 ms | Common H10 progress MAE | Position RMSE m | Velocity RMSE m/s |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for topology in cfg['topologies']:
        for seed in cfg['seeds']:
            for condition in cfg['conditions']:
                values=[r for r in rows if r['topology']==topology and r['seed']==seed and r['condition']==condition and r['role']=='fresh']
                metric=next((r for r in common if r['topology']==topology and r['seed']==seed and r['condition']==condition and r['horizon']==10), dict(progress_mae=float('nan'),position_rmse=float('nan'),velocity_rmse=float('nan')))
                lines.append(f"| {topology} | {seed} | {condition} | {np.mean([r['displacement'] for r in values]):.6f} | {np.mean([r['discounted_task_return'] for r in values]):.6f} | {sum(r['collapsed'] for r in values)} | {values[0]['planning_median_seconds']*1000:.2f} | {values[0]['planning_p95_seconds']*1000:.2f} | {metric['progress_mae']:.6f} | {metric['position_rmse']:.6f} | {metric['velocity_rmse']:.6f} |")
    lines += ['', '## Observed sharing effects', '']
    for topology in cfg['topologies']:
        values=[r for r in final if r['topology']==topology]
        lines.append(f"- {topology}: mean paired fresh return difference {np.mean([r['return_difference'] for r in values]):+.6f}; shared improves the seed mean in {sum(r['return_difference']>0 for r in values)}/3 seeds; collapse-count difference {sum(r['collapse_difference'] for r in values):+d}/15 episodes.")
    successful=all(r['return_difference']>0 and r['collapse_difference']<=0 for r in final)
    lines += ['', 'All nine source/seed pairs improve return without additional collapses in this sample.' if successful else 'The results do not uniformly show improved return without additional collapses across every topology and seed. A blanket no-sacrifice conclusion is unsupported.']
    lines += ['','## Complete per-topology, per-seed evidence','',
        'NaN in an H10 table cell means no episode lasted ten steps; those failures remain in the control metrics. `episodes.csv` records every checkpoint/fresh episode, displacement, return, collapse and planning latency. `paired-differences.csv` records all matched differences. `prediction-errors.json` records own-controller errors at each checkpoint and both final models on the same union of fresh trajectories, with physical position/velocity RMSE and progress MAE at horizons 1/5/10/20. `integrity-audit.json` checks source hashes, exact budgets, Adam counts, replay adjacency, masked actions, paired starts, initialization, fitting warmups and normalization.',
        '', '## Limits and next experiment','',
        'This compares learning across three trained source topologies. It is not evidence of unseen-topology transfer. Three seeds and five fresh resets per seed are too small to establish noninferiority or reliability. Monitoring resets repeat across checkpoints; no checkpoint was selected from their outcomes. Larger shared Adam state receives three times as many aggregate updates as each specialist, while per-topology exposures match. Fixed update order and pooled fitting-only normalization are documented design choices. Own-controller prediction errors have different state distributions; common-union final errors support comparison but are descriptive.',
        '', 'Recommendation: inspect each topology and training seed jointly for return, movement, failures and common-distribution prediction errors before authorizing another experiment. Do not tune or expand training automatically. A follow-up should replicate any topology-specific loss of control under a new predeclared seed set before testing unseen robots.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return dict(audit_passed=True,final=final)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    print(json.dumps(report(parser.parse_args().output),indent=2))
