"""Verify and report the fixed-method 20k source-only online comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'sac')]
from dynamics.dataset import file_sha256

LABELS={'dynamics_mpc':'Dynamics MPC','reward_value_mpc':'Reward + value MPC','random':'Random control'}
COLORS={'dynamics_mpc':'#1565c0','reward_value_mpc':'#d85b24','random':'#777e87'}


def aggregate(rows: list[dict]) -> dict:
    return dict(episodes=len(rows),mean_displacement=float(np.mean([r['displacement'] for r in rows])),
        median_displacement=float(np.median([r['displacement'] for r in rows])),
        mean_discounted_return=float(np.mean([r['discounted_task_return'] for r in rows])),
        mean_raw_return=float(np.mean([r['raw_return'] for r in rows])),
        collapses=sum(r['collapsed'] for r in rows))


def verify(root: Path, document: dict) -> dict:
    protocol=document['protocol'];cfg=protocol['settings'];runs=document['runs']
    expected={(c,s) for c in cfg['conditions'] for s in cfg['seeds']}
    if len(runs)!=len(expected) or {(r['condition'],r['seed']) for r in runs}!=expected:
        raise ValueError('Full experiment matrix is incomplete')
    for name,digest in protocol['sources'].items():
        if file_sha256(ROOT/name)!=digest:raise ValueError(f'Source changed: {name}')
    for run in runs:
        folder=root/f"{run['condition']}-seed{run['seed']}"
        if run['status']!='complete' or run['interactions']!=cfg['steps'] or run['updates']!=cfg['steps']:
            raise ValueError('Run failed or budget mismatch')
        for filename,key in [('replay.npz','replay_sha256'),('final-model.pt','checkpoint_sha256'),('warmup.npz','warmup_sha256')]:
            if file_sha256(folder/filename)!=run[key]:raise ValueError(f'Hash mismatch {folder/filename}')
        prefix=json.loads((folder/'prefix-comparison.json').read_text())
        if not prefix['passed'] or not all(prefix['checks'].values()):raise ValueError('Short-run prefix mismatch')
        if [v['steps'] for v in run['evaluations']]!=cfg['evaluate_at']:raise ValueError('Monitoring budget mismatch')
        with np.load(folder/'replay.npz') as replay:
            same=replay['episode'][1:]==replay['episode'][:-1]
            if not np.array_equal(replay['next_state'][:-1][same],replay['state'][1:][same]):raise ValueError('Replay adjacency mismatch')
            if np.any(replay['terminated'][:-1][same]|replay['truncated'][:-1][same]):raise ValueError('Replay crosses episode end')
            for key in ['state','next_state','action','progress','reward','base_reward']:
                if not np.isfinite(replay[key]).all():raise ValueError('Nonfinite real replay')
        fit={e['seed'] for e in run['training_episodes']}
        monitor={e['seed'] for v in run['evaluations'] for e in v['episodes']}
        fresh={e['seed'] for e in run['fresh_evaluation']['episodes']}
        if fit&monitor or fit&fresh or monitor&fresh:raise ValueError('Reset-role overlap')
        for subfolder,values in [('',[e for v in run['evaluations'] for e in v['episodes']]),
                                 ('fresh-final',run['fresh_evaluation']['episodes']),('fresh-random',run['fresh_random']['episodes'])]:
            for row in values:
                if file_sha256(folder/subfolder/row['trace'])!=row['sha256']:raise ValueError('Evaluation trace hash mismatch')
    for seed in cfg['seeds']:
        pair=sorted([r for r in runs if r['seed']==seed],key=lambda r:r['condition'])
        for i in range(len(cfg['evaluate_at'])):
            starts=[{e['seed']:e['initial_state_sha256'] for e in r['evaluations'][i]['episodes']} for r in pair]
            if starts[0]!=starts[1]:raise ValueError('Monitoring starts unpaired')
        starts=[{e['seed']:e['initial_state_sha256'] for e in r['fresh_evaluation']['episodes']} for r in pair]
        random_starts=[{e['seed']:e['initial_state_sha256'] for e in r['fresh_random']['episodes']} for r in pair]
        if not (starts[0]==starts[1]==random_starts[0]==random_starts[1]):raise ValueError('Fresh starts unpaired')
        if pair[0]['fresh_random']['episodes']!=pair[1]['fresh_random']['episodes']:
            # Trace hashes/names should be identical as well; latency isn't in episode records.
            raise ValueError('Fresh random controls differ')
        with np.load(root/f"{pair[0]['condition']}-seed{seed}"/'warmup.npz') as a,np.load(root/f"{pair[1]['condition']}-seed{seed}"/'warmup.npz') as b:
            if not all(np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Warmups differ')
    return dict(passed=True,complete_runs=len(runs),exact_short_prefixes=len(runs),paired_monitoring=True,
                paired_fresh=True,role_separation=True,replay_adjacency=True,budgets=True,artifact_and_source_hashes=True)


def report(root: Path) -> dict:
    document=json.loads((root/'report.json').read_text());cfg=document['protocol']['settings'];runs=document['runs']
    audit=verify(root,document)
    monitoring=[];fresh=[];per_seed=[]
    for condition in cfg['conditions']:
        selected=sorted([r for r in runs if r['condition']==condition],key=lambda r:r['seed'])
        for budget in cfg['evaluate_at']:
            values=[e for r in selected for v in r['evaluations'] if v['steps']==budget for e in v['episodes']]
            monitoring.append(dict(condition=condition,steps=budget,**aggregate(values)))
        fresh.append(dict(condition=condition,**aggregate([e for r in selected for e in r['fresh_evaluation']['episodes']])))
        for run in selected:
            per_seed.append(dict(condition=condition,seed=run['seed'],initial=aggregate(run['evaluations'][0]['episodes']),
                                 final_monitoring=aggregate(run['evaluations'][-1]['episodes']),fresh=aggregate(run['fresh_evaluation']['episodes'])))
    random_runs=[r for r in runs if r['condition']==cfg['conditions'][0]]
    fresh.append(dict(condition='random',**aggregate([e for r in random_runs for e in r['fresh_random']['episodes']])))
    elapsed=json.loads((root/'heartbeat.json').read_text())['wall_seconds']
    summary=dict(integrity=audit,monitoring=monitoring,fresh=fresh,per_seed=per_seed,
                 elapsed_seconds=elapsed,aggregate_run_wall_seconds=sum(r['wall_seconds'] for r in runs),
                 interactions=sum(r['interactions'] for r in runs),updates=sum(r['updates'] for r in runs))
    (root/'integrity-audit.json').write_text(json.dumps(audit,indent=2))
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    budgets=np.asarray(cfg['evaluate_at'])
    for condition in cfg['conditions']:
        selected=sorted([r for r in runs if r['condition']==condition],key=lambda r:r['seed'])
        for ax,key,title in [(axes[0,0],'displacement','Forward movement / initial diagonal'),
                              (axes[0,1],'discounted_task_return','Discounted task return'),(axes[1,0],'collapsed','Evaluation collapse fraction')]:
            values=np.asarray([[np.mean([e[key] for e in v['episodes']]) for v in r['evaluations']] for r in selected])
            ax.plot(budgets,values.mean(0),'o-',color=COLORS[condition],label=LABELS[condition])
            ax.fill_between(budgets,values.min(0),values.max(0),color=COLORS[condition],alpha=.12)
            ax.set(ylabel=title,xlabel='Real fitting transitions')
            ax.axhline(0,color='#aaa',lw=.7)
        if condition=='reward_value_mpc':
            values=np.asarray([[v['terminal_calibration'].get('mae',np.nan) for v in r['evaluations']] for r in selected])
            axes[1,1].plot(budgets,np.nanmean(values,0),'o-',color=COLORS[condition])
            axes[1,1].fill_between(budgets,np.nanmin(values,0),np.nanmax(values,0),color=COLORS[condition],alpha=.12)
    axes[0,0].legend(fontsize=9)
    axes[1,1].set(ylabel='Min-Q MAE on real fitting terminal transitions',xlabel='Real fitting transitions')
    axes[1,0].set_ylim(-.04,1.04)
    fig.suptitle('Fixed-method longer online MPC: source tetrahedron',fontsize=14)
    fig.supxlabel('3 training seeds; same 3 monitoring resets per seed. Bands are seed ranges, not confidence intervals.',fontsize=9)
    fig.savefig(root/'learning-curves.png',dpi=180);fig.savefig(root/'learning-curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
    ordered=[next(r for r in fresh if r['condition']==c) for c in ['random']+cfg['conditions']]
    labels=['Random','Dynamics\nMPC','Reward + value\nMPC']
    for ax,key,title in [(axes[0],'mean_displacement','Forward movement / initial diagonal'),
                          (axes[1],'mean_discounted_return','Discounted task return'),(axes[2],'collapses','Collapse count / 15')]:
        ax.bar(range(len(ordered)),[r[key] for r in ordered],color=[COLORS[r['condition']] for r in ordered])
        ax.set(xticks=range(3),xticklabels=labels,ylabel=title);ax.axhline(0,color='#aaa',lw=.7)
    fig.suptitle('Fresh final source evaluation: 20,000 transitions per run',fontsize=14)
    fig.supxlabel('5 fresh paired resets × 3 training seeds; fixed final checkpoint; no fitting or selection on these episodes.',fontsize=9)
    fig.savefig(root/'fresh-final.png',dpi=180);plt.close(fig)
    lines=['# Longer online learned-MPC source results — October 24, 2026','',
        f"Completed all six fixed-method runs: **{summary['interactions']:,} real fitting transitions**, "
        f"{cfg['steps']:,} per condition/seed. Elapsed time {elapsed/60:.1f} minutes; summed per-run wall time "
        f"{summary['aggregate_run_wall_seconds']/3600:.2f} hours. Three independent workers, one Torch thread each.",'',
        'Learning settings were held fixed. All six 1,600-step prefixes match the original short runs exactly, '
        'including model/actor/target/optimizer state, real replay and monitoring evaluation arrays. '
        'No simulator-MPC or teacher generated fitting data. No topology expansion or held-out access occurred.','',
        '![Monitoring learning curves](learning-curves.png)','','## Monitoring checkpoints','',
        '| Method | Real transitions | Mean movement | Mean discounted task return | Collapses |','|---|---:|---:|---:|---:|']
    for r in monitoring:lines.append(f"| {LABELS[r['condition']]} | {r['steps']:,} | {r['mean_displacement']:.6f} | {r['mean_discounted_return']:.6f} | {r['collapses']}/{r['episodes']} |")
    lines+=['','## Fresh final evaluation','','![Fresh final evaluation](fresh-final.png)','',
            '| Method | Mean movement | Median movement | Mean discounted task return | Collapses |','|---|---:|---:|---:|---:|']
    for r in ordered:lines.append(f"| {LABELS[r['condition']]} | {r['mean_displacement']:.6f} | {r['median_displacement']:.6f} | {r['mean_discounted_return']:.6f} | {r['collapses']}/{r['episodes']} |")
    lines+=['','Movement is forward physical-node mean displacement divided by initial bounding-box diagonal. '
        'Task return discounts progress minus routed-command effort and a true-collapse charge; raw environment returns remain in the JSON evidence. '
        'Each episode runs at most 80 control steps, retaining early-collapse outcomes. Fresh reset seeds are disjoint from fitting and monitoring.','',
        '## Per-seed final outcomes','','| Method | Training seed | Monitoring return at 1,600 | Monitoring return at 20,000 | Fresh final return | Fresh collapses |','|---|---:|---:|---:|---:|---:|']
    for r in per_seed:lines.append(f"| {LABELS[r['condition']]} | {r['seed']} | {r['initial']['mean_discounted_return']:.6f} | {r['final_monitoring']['mean_discounted_return']:.6f} | {r['fresh']['mean_discounted_return']:.6f} | {r['fresh']['collapses']}/{r['fresh']['episodes']} |")
    lines+=['','## Verification and limits','',
        'Restart regression tests prove bitwise identical uninterrupted versus mid-episode resumed training and evaluation for both variants. '
        'The final artifact audit verifies exact budgets, source/artifact hashes, replay adjacency, episode boundaries, warm-up equality, paired starts and reset-role separation. '
        'Check `integrity-audit.json`, per-run `prefix-comparison.json`, `verification-tests.txt`, checkpoint manifests and traces.','',
        'This is a fixed-method duration experiment on one source topology and three training seeds. Monitoring resets are reused across checkpoints; '
        'no best-monitoring checkpoint was selected. Fifteen fresh episodes per method are still a small source test, not independent topology replications or a transfer result. '
        'The reward/value method remains TD-MPC2-inspired explicit-state learning, not the full published algorithm. '
        'No automatic method repair or multi-topology run was performed.','']
    (root/'report.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    report(parser.parse_args().output)
