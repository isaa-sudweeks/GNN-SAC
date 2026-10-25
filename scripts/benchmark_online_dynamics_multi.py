"""Full-architecture, disposable source benchmark; no campaign fitting data reused."""
from pathlib import Path
import sys
import time
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'sac')]
from omegaconf import OmegaConf
from scripts.run_online_dynamics_multi import worker
from scripts.collect_dynamics_training_data import write_json

if __name__=='__main__':
    root=ROOT/'outputs/dynamics-pilot/online-multi-benchmark-2026-10-24'
    root.mkdir(parents=True,exist_ok=False)
    cfg=OmegaConf.to_container(OmegaConf.load(ROOT/'config/dynamics_online_multi.yaml'))
    cfg.update(steps=320,evaluate_at=[320],evaluation_episodes=1,fresh_evaluation_episodes=1,checkpoint_every=320)
    begun=time.perf_counter()
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(worker,cfg,c,s,str(root),False) for c,s in [('shared',90),('separate',90),('shared',91)]]
        runs=[job.result() for job in jobs]
    if any(r['status']!='complete' for r in runs):raise RuntimeError(runs)
    # Last history interval excludes warm-up, initialization and evaluation at 320.
    intervals=[r['history'][2]['wall_seconds']-r['history'][1]['wall_seconds'] for r in runs]
    per_round=max(intervals)/80
    estimate=per_round*20000*6/3
    result=dict(wall_seconds=time.perf_counter()-begun,runs=[dict(condition=r['condition'],seed=r['seed'],wall_seconds=r['wall_seconds']) for r in runs],
        steady_round_seconds=per_round,estimated_campaign_elapsed_seconds=estimate*1.25,
        estimated_aggregate_group_seconds=estimate*1.25*3,
        method='Three concurrent workers, full pilot architecture; max steady interval 160->240 per-round cost extrapolated to six groups, plus 25 percent evaluation/checkpoint overhead.',
        declared_fitting_transitions=360000,stop='No method tuning or budget expansion')
    write_json(root/'benchmark.json',result)
    print(json.dumps(result,indent=2))
