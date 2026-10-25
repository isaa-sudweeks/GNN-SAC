"""Source-backed report for bounded saved-replay multi-topology diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def aggregate(values: list[dict]) -> dict:
    values=[v for v in values if v['windows']]
    if not values:return dict(windows=0)
    count=sum(v['windows'] for v in values)
    average=lambda key:sum(v[key]*v['windows'] for v in values)/count
    return dict(windows=count,progress_mae=average('progress_mae'),position_rmse=float(np.sqrt(sum(v['position_rmse']**2*v['windows'] for v in values)/count)),
        velocity_rmse=float(np.sqrt(sum(v['velocity_rmse']**2*v['windows'] for v in values)/count)),rigidity_mae=average('rigidity_mae'),
        true_collapses=sum(v['true_collapses'] for v in values),false_negatives=sum(v['endpoint']['false_negatives'] for v in values),
        predicted_any_collapse=sum(v['any_imagined']['predicted_collapses'] for v in values),
        false_positives=sum(v['endpoint']['false_positives'] for v in values),nonfinite=sum(v['nonfinite_windows'] for v in values),
        geometry_label_disagreements=sum(v['true_geometry_label_disagreements'] for v in values))


def report(root: Path) -> dict:
    result=json.loads((root/'results.json').read_text());cfg=result['protocol']['settings']
    predictions=result['predictions'];gradients=result['gradients'];cosines=result['cosines']
    flat=[]
    for row in predictions:
        record={k:v for k,v in row.items() if not isinstance(v,dict)}
        if row['windows']:
            record.update({prefix+'_'+k:v for prefix in ['endpoint','any_imagined'] for k,v in row[prefix].items()})
        flat.append(record)
    keys=list(dict.fromkeys(k for row in flat for k in row))
    with (root/'prediction-cells.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(flat)
    paired=[]
    for role in ['fitting','fresh']:
        for stratum in ['terminal_ending','precollapse','ordinary']:
            for topology in cfg['topologies']:
                for seed in cfg['seeds']:
                    for horizon in [1,5,10,20]:
                        by_condition={condition:aggregate([r for r in predictions if r['role']==role and r['stratum']==stratum and r['topology']==topology and r['seed']==seed and r['horizon']==horizon and r['condition']==condition]) for condition in ['shared','separate']}
                        a,b=by_condition['shared'],by_condition['separate']
                        if a['windows']!=b['windows']:raise ValueError('Unpaired prediction cohorts')
                        paired.append(dict(role=role,stratum=stratum,topology=topology,seed=seed,horizon=horizon,
                            shared=a,separate=b,progress_difference=(a['progress_mae']-b['progress_mae']) if a['windows'] else None))
    (root/'paired-summary.json').write_text(json.dumps(paired,indent=2))
    lines=['# Frozen source dynamics failure and gradient diagnostic','',
        f"Completed the predeclared saved-data diagnostic in {result['wall_seconds']:.2f} seconds using one Torch/inter-op thread. No optimizer steps, simulator execution, new collection, normalization changes, checkpoint selection or unseen topology access occurred.",
        '', 'Input: final fixed 20,000-transition checkpoints, real fitting replay and already-recorded independent fresh-final source trajectories from the gamma=.995 campaign. All input hashes are verified again after diagnosis; every gradient call verifies parameters, buffers, Torch RNG and `.grad` fields remain unchanged.',
        '', '## Fresh terminal-ending H10 windows', '',
        'Both frozen models receive exactly the same executed-action windows from the union of both controllers. Each generator cell contributes at most 32 selected windows; raw per-generator rows remain in `prediction-cells.csv`. Counts below sum selected windows; errors are weighted by windows, not independent episode claims. Empty cells are retained. A false negative means the endpoint model geometry does not predict an actual recorded collapse.',
        '', '| Topology | Seed | Windows | Shared progress MAE | Separate progress MAE | Difference | Shared missed collapse | Separate missed collapse |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in paired:
        if row['role']!='fresh' or row['stratum']!='terminal_ending' or row['horizon']!=10:continue
        a,b=row['shared'],row['separate']
        if not a['windows']:
            lines.append(f"| {row['topology']} | {row['seed']} | 0 | n/a | n/a | n/a | n/a | n/a |")
        else:
            lines.append(f"| {row['topology']} | {row['seed']} | {a['windows']} | {a['progress_mae']:.6f} | {b['progress_mae']:.6f} | {row['progress_difference']:+.6f} | {a['false_negatives']}/{a['true_collapses']} | {b['false_negatives']}/{b['true_collapses']} |")
    lines+=['','## Local fitting gradients at final shared weights','',
        'Four predetermined batches per topology and seed each contain 16 one-step examples and 16 five-step windows from each fitting generator. Cosines compare topology gradients at the same shared parameter vector. Raw gradients are measured before clipping; specialist norms/losses use the same own-topology examples. Negative cosine indicates local first-order opposition on the selected examples, not proven sustained interference or a demonstrated cause of control collapse.',
        '', '| Seed | Topology pair | Mean total cosine | Negative batches | Range |',
        '|---:|---|---:|---:|---|']
    for seed in cfg['seeds']:
        for first,second in [(cfg['topologies'][0],cfg['topologies'][1]),(cfg['topologies'][0],cfg['topologies'][2]),(cfg['topologies'][1],cfg['topologies'][2])]:
            vals=[r['total'] for r in cosines if r['seed']==seed and r['first']==first and r['second']==second]
            lines.append(f"| {seed} | {first} / {second} | {np.mean(vals):+.4f} | {sum(v<0 for v in vals)}/4 | {min(vals):+.4f} to {max(vals):+.4f} |")
    lines+=['','## Matched fitting objective and gradient norms','',
        '| Topology | Seed | Shared loss | Separate loss | Shared total-gradient norm | Separate norm |',
        '|---|---:|---:|---:|---:|---:|']
    for topology in cfg['topologies']:
        for seed in cfg['seeds']:
            vals={c:[r for r in gradients if r['topology']==topology and r['seed']==seed and r['condition']==c] for c in ['shared','separate']}
            lines.append(f"| {topology} | {seed} | {np.mean([r['loss'] for r in vals['shared']]):.4f} | {np.mean([r['loss'] for r in vals['separate']]):.4f} | {np.mean([r['gradient_norms']['total'] for r in vals['shared']]):.4f} | {np.mean([r['gradient_norms']['total'] for r in vals['separate']]):.4f} |")
    lines+=['','## Failure interpretation by topology','']
    observations=[]
    for topology in cfg['topologies']:
        cells=[r for r in paired if r['role']=='fresh' and r['stratum']=='terminal_ending' and r['horizon']==10 and r['topology']==topology and r['shared']['windows']]
        shared_fn=sum(r['shared']['false_negatives'] for r in cells); separate_fn=sum(r['separate']['false_negatives'] for r in cells)
        terminals=sum(r['shared']['true_collapses'] for r in cells)
        worsens=sum(r['progress_difference']>0 for r in cells)
        message=f"{topology}: shared misses {shared_fn}/{terminals} selected fresh H10 collapse endpoints; separate misses {separate_fn}/{terminals}. Shared H10 progress error is greater in {worsens}/{len(cells)} nonempty seed cohorts."
        lines.append('- '+message);observations.append(message)
    fresh_h10=[r for r in paired if r['role']=='fresh' and r['horizon']==10]
    mismatch=sum(r['shared'].get('geometry_label_disagreements',0) for r in fresh_h10)
    nonfinite=sum(r['shared'].get('nonfinite',0)+r['separate'].get('nonfinite',0) for r in fresh_h10)
    lines+=['',f"Recorded-vs-geometric endpoint label disagreements across selected fresh H10 cohorts: {mismatch}. Nonfinite predictions across both frozen models in those cohorts: {nonfinite}.",
        '', 'The control outcomes themselves remain those of the completed campaign: tetrahedron had four additional shared fresh collapses, six-node sharing had four fewer collapses but less movement in all seeds, and both methods collapsed on every eight-node fresh episode. This diagnostic measures executed-action prediction, not scores/ranking of unexecuted CEM candidates or counterfactual action outcomes. It cannot uniquely attribute the failures to model capacity, local gradient conflict, planner ranking, state aliasing, exploration or the discount factor.',
        '', '## Limits and next recommendation', '',
        'Use these results to target a separately predeclared source-only diagnostic or ablation; do not infer unseen transfer. If both models miss near-collapse geometry, repair claims should first be checked on a new source validation manifest with collapse sensitivity and candidate ranking measured separately. A shared-specific fitting or near-collapse error motivates a controlled interference/capacity/update-order comparison, but final-point gradient cosines are insufficient to choose one repair. The concurrently authorized gamma=.95 rerun is independent and must retain its fixed protocol; this diagnostic neither selects nor modifies it.',
        '', 'Fitting errors are in-sample and optimistic. Fresh samples were previously evaluated by the campaign, so these are post hoc explanations, not untouched confirmation. Sampled windows overlap; endpoints and repeated reset episodes are not independent replicates. Three training seeds and four gradient batches do not establish a general optimization law. Generator/cohort counts are explicit in `samples.json` and `prediction-cells.csv`; no numerical or difficult cases were silently dropped.',
        '', '[Gradient cosine plot](gradient-cosines.html) · [Fresh H10 prediction plot](near-collapse-errors.html)',
        '', 'Implementation: `scripts/run_dynamics_multi_diagnostic.py`; tests: `tests/test_dynamics_multi_diagnostic.py`; predeclared protocol: `docs/plans/gnn_dynamics_multi_diagnostic.md`.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    figure=make_subplots(rows=1,cols=3,subplot_titles=['Tetra / six-node','Tetra / eight-node','Six / eight-node'])
    pairs=[(cfg['topologies'][0],cfg['topologies'][1]),(cfg['topologies'][0],cfg['topologies'][2]),(cfg['topologies'][1],cfg['topologies'][2])]
    for col,(first,second) in enumerate(pairs,1):
        for seed in cfg['seeds']:
            vals=[r['total'] for r in cosines if r['seed']==seed and r['first']==first and r['second']==second]
            figure.add_trace(go.Scatter(x=[seed]*len(vals),y=vals,mode='markers',name=f'seed {seed}',legendgroup=str(seed),showlegend=col==1,marker=dict(color=['#1464b4','#d15e25','#16835d'][seed])),row=1,col=col)
        figure.add_hline(y=0,line_dash='dash',row=1,col=col)
    figure.update_xaxes(tickvals=[0,1,2],title_text='Training seed');figure.update_yaxes(range=[-1,1])
    figure.update_layout(title='Local total dynamics-gradient cosines at common frozen shared weights',width=1400,height=450)
    figure.write_html(root/'gradient-cosines.html',include_plotlyjs=True)
    errorfig=make_subplots(rows=1,cols=3,subplot_titles=cfg['topologies'])
    for col,topology in enumerate(cfg['topologies'],1):
        for condition in ['shared','separate']:
            vals=[r for r in paired if r['role']=='fresh' and r['stratum']=='terminal_ending' and r['horizon']==10 and r['topology']==topology and r[condition]['windows']]
            errorfig.add_trace(go.Scatter(x=[r['seed'] for r in vals],y=[r[condition]['progress_mae'] for r in vals],name=condition,legendgroup=condition,showlegend=col==1,mode='lines+markers',line=dict(color='#1464b4' if condition=='shared' else '#d15e25')),row=1,col=col)
    errorfig.update_xaxes(tickvals=[0,1,2],title_text='Training seed')
    errorfig.update_yaxes(title_text='H10 normalized progress MAE')
    errorfig.update_layout(title='Same selected fresh terminal-ending executed-action windows',width=1400,height=450)
    errorfig.write_html(root/'near-collapse-errors.html',include_plotlyjs=True)
    summary=dict(observations=observations,total_negative_cosines=sum(r['total']<0 for r in cosines),total_cosines=len(cosines),
        wall_seconds=result['wall_seconds'],no_mutation=result['no_input_artifact_mutation'] and result['all_gradient_calls_no_mutation'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    print(json.dumps(report(parser.parse_args().output),indent=2))
