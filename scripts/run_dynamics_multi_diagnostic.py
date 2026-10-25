"""Bounded frozen-checkpoint diagnostics on saved three-source real trajectories."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'sac')]
from dynamics.dataset import file_sha256
from dynamics.model import ObservableDynamicsGNN, physical_graph_loss
from dynamics.online import OnlineLearner
from dynamics.training import tensor_graph


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def valid_starts(arrays: dict, horizon: int) -> np.ndarray:
    """Require one episode throughout and no interior terminal/time-limit marker."""
    if horizon < 1:
        raise ValueError('Horizon must be positive')
    count = len(arrays['state']) - horizon + 1
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(count)
    good = np.ones(count, dtype=bool)
    for offset in range(horizon):
        good &= arrays['episode'][starts] == arrays['episode'][starts+offset]
        if offset < horizon-1:
            good &= ~(arrays['terminated'][starts+offset] | arrays['truncated'][starts+offset])
    return starts[good]


def cohorts(arrays: dict, horizon: int) -> dict[str, np.ndarray]:
    """Partition safe windows by the observed episode's future collapse distance."""
    starts = valid_starts(arrays, horizon)
    ends = starts+horizon-1
    terminal_by_episode = {int(arrays['episode'][i]): i for i in np.flatnonzero(arrays['terminated'])}
    distance = np.asarray([terminal_by_episode.get(int(arrays['episode'][i]), 10**12)-i for i in ends])
    return dict(terminal_ending=starts[distance == 0], precollapse=starts[(distance >= 1) & (distance <= 10)],
                ordinary=starts[distance > 10])


def draw_cohorts(arrays: dict, horizon: int, rng: np.random.Generator, limit: int = 32) -> dict:
    result = {}
    for name, available in cohorts(arrays, horizon).items():
        selected = np.sort(rng.choice(available, size=min(limit, len(available)), replace=False))
        result[name] = dict(available=len(available), starts=selected.tolist())
    return result


def model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def geometric_collapse(state: torch.Tensor, metadata: dict, graph: dict) -> torch.Tensor:
    """Use the unchanged campaign collapse classifier without a simulator instance."""
    context = SimpleNamespace(metadata=metadata, graph=graph,
        edges=torch.as_tensor(metadata['structural_edges'], dtype=torch.long))
    return OnlineLearner.collapsed(context, state)


def normalized_rigidity(state: torch.Tensor, metadata: dict, graph: dict) -> torch.Tensor:
    positions=state[:,graph['physical_order_index'],:3]
    edges=torch.as_tensor(metadata['structural_edges'],dtype=torch.long)
    axes=metadata['axis_indices'];delta=positions[:,edges[:,1]][:,:,axes]-positions[:,edges[:,0]][:,:,axes]
    length=delta.norm(dim=-1,keepdim=True)
    direction=torch.where(length>=1e-8,delta/length.clamp_min(1e-8),torch.zeros_like(delta))
    incidence=state.new_zeros((len(edges),positions.shape[1]))
    incidence[torch.arange(len(edges)),edges[:,0]]=-1;incidence[torch.arange(len(edges)),edges[:,1]]=1
    matrix=(direction[:,:,None,:]*incidence[None,:,:,None]).reshape(len(state),len(edges),-1)
    rank=positions.shape[1]*len(axes)-len(axes)-len(axes)*(len(axes)-1)//2
    eigen=torch.linalg.eigvalsh(matrix@matrix.transpose(1,2))[:,len(edges)-rank].clamp_min(0)
    if metadata['wcrm']:eigen=eigen/matrix.square().sum((1,2)).clamp_min(1e-12)
    return eigen/float(metadata['initial_rigidity_reference'])


@torch.no_grad()
def prediction_metrics(model: ObservableDynamicsGNN, arrays: dict, starts: np.ndarray,
                       horizon: int, metadata: dict, graph: dict) -> dict:
    """Recursive executed-action diagnostics on identical recorded windows."""
    if not len(starts):
        return dict(windows=0)
    legal=set(valid_starts(arrays,horizon).tolist())
    if any(int(i) not in legal for i in starts):raise ValueError('Unsafe diagnostic window')
    start=torch.as_tensor(starts,dtype=torch.long)
    predicted=torch.as_tensor(arrays['state'][starts],dtype=torch.float32)
    accumulated=predicted.new_zeros((len(starts),2));truth_progress=accumulated.clone()
    any_failure=torch.zeros(len(starts),dtype=torch.bool);nonfinite=any_failure.clone()
    for offset in range(horizon):
        action=torch.as_tensor(arrays['action'][starts+offset],dtype=torch.float32)
        predicted,move=model.step(predicted,action,graph)
        finite=torch.isfinite(predicted).all((1,2)) & torch.isfinite(move).all(1)
        nonfinite|=~finite
        safe=torch.nan_to_num(predicted,nan=0.,posinf=0.,neginf=0.)
        any_failure|=geometric_collapse(safe,metadata,graph)|~finite
        predicted=safe;accumulated+=torch.nan_to_num(move,nan=0.,posinf=0.,neginf=0.)
        truth_progress+=torch.as_tensor(arrays['progress'][starts+offset],dtype=torch.float32)
    truth=torch.as_tensor(arrays['next_state'][starts+horizon-1],dtype=torch.float32)
    physical=graph['physical_mask'].bool();diagonal=metadata['initial_diagonal']
    position=predicted[:,physical,:3]-truth[:,physical,:3]
    position[:,:,:2]+=(accumulated-truth_progress)[:,None]
    velocity=predicted[:,physical,3:]-truth[:,physical,3:]
    progress=(accumulated[:,0]-truth_progress[:,0]).abs()/diagonal
    position_mse=position.square().mean((1,2));velocity_mse=velocity.square().mean((1,2))
    progress=torch.where(nonfinite,100.,progress)
    position_mse=torch.where(nonfinite,(100*diagonal)**2,position_mse)
    velocity_mse=torch.where(nonfinite,10000.,velocity_mse)
    predicted_terminal=geometric_collapse(predicted,metadata,graph)|nonfinite
    true_terminal=torch.as_tensor(arrays['terminated'][starts+horizon-1],dtype=torch.bool)
    true_geometry=geometric_collapse(truth,metadata,graph)
    rig_error=(normalized_rigidity(predicted,metadata,graph)-normalized_rigidity(truth,metadata,graph)).abs()
    rig_error=torch.where(nonfinite,100.,rig_error)
    confusion=lambda values:dict(false_negatives=int((true_terminal&~values).sum()),
        false_positives=int((~true_terminal&values).sum()),predicted_collapses=int(values.sum()))
    return dict(windows=len(starts),episodes=len(set(arrays['episode'][starts].tolist())),
        progress_mae=float(progress.mean()),position_rmse=float(position_mse.mean().sqrt()),velocity_rmse=float(velocity_mse.mean().sqrt()),
        rigidity_mae=float(rig_error.mean()),true_collapses=int(true_terminal.sum()),
        true_geometry_label_disagreements=int((true_geometry!=true_terminal).sum()),
        nonfinite_windows=int(nonfinite.sum()),endpoint=confusion(predicted_terminal),any_imagined=confusion(any_failure))


def balanced_batch(replays: dict[str, dict], rng: np.random.Generator, horizon: int = 5,
                   half_batch: int = 16) -> tuple[dict, dict]:
    """Equal generator draws; independent one-step and episode-safe recursive inputs."""
    pieces=[];indices={}
    for generator,arrays in replays.items():
        available=valid_starts(arrays,horizon)
        if not len(available):raise ValueError('No valid recursive fitting windows')
        single=rng.integers(len(arrays['state']),size=half_batch)
        starts=rng.choice(available,size=half_batch,replace=True)
        indices[generator]=dict(one_step=single.tolist(),recursive=starts.tolist())
        pieces.append(dict(state=torch.tensor(arrays['state'][single]),next_state=torch.tensor(arrays['next_state'][single]),
            action=torch.tensor(arrays['action'][single]),progress=torch.tensor(arrays['progress'][single]),
            window_state=torch.tensor(arrays['state'][starts]),
            window_next=torch.tensor(np.stack([arrays['next_state'][starts+i] for i in range(horizon)],axis=1)),
            window_action=torch.tensor(np.stack([arrays['action'][starts+i] for i in range(horizon)],axis=1)),
            window_progress=torch.tensor(np.stack([arrays['progress'][starts+i] for i in range(horizon)],axis=1))))
    batch={key:torch.cat([piece[key] for piece in pieces]).float() for key in pieces[0]}
    return batch,indices


def diagnostic_loss(model: ObservableDynamicsGNN, batch: dict, graph: dict,
                    rollout_weight: float = 1.) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """Exact dynamics-only training objective, without update or stochastic sampling."""
    target=batch['next_state']-batch['state'];target=target.clone();target[:,:,:2]+=batch['progress'][:,None]
    one=physical_graph_loss(model(batch['state'],batch['action'],graph),target,graph['physical_mask'],model.delta_scale)
    imagined=batch['window_state'];progress=imagined.new_zeros((len(imagined),2));truth=progress.clone()
    horizon=batch['window_action'].shape[1];rollout=one.new_zeros(())
    for offset in range(horizon):
        imagined,move=model.step(imagined,batch['window_action'][:,offset],graph)
        progress=progress+move;truth=truth+batch['window_progress'][:,offset]
        rollout=rollout+physical_graph_loss(imagined,batch['window_next'][:,offset],graph['physical_mask'],model.state_scale)
        rollout=rollout+((progress-truth)/(model.delta_scale[:2]*(offset+1))).square().mean()
    rollout=rollout/horizon
    return one+rollout_weight*rollout,one,rollout


def frozen_gradients(model: ObservableDynamicsGNN, batch: dict, graph: dict,
                     rollout_weight: float = 1.) -> tuple[dict,dict[str,torch.Tensor]]:
    """Measure gradients without changing parameters, buffers, RNG or .grad fields."""
    before=model_digest(model);rng=torch.get_rng_state().clone()
    old_grads=[None if p.grad is None else p.grad.clone() for p in model.parameters()]
    total,one,rollout=diagnostic_loss(model,batch,graph,rollout_weight)
    if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite diagnostic loss')
    parameters=tuple(model.parameters())
    flatten=lambda values:torch.cat([value.detach().reshape(-1) for value in values])
    first=flatten(torch.autograd.grad(one,parameters,retain_graph=True))
    recursive=flatten(torch.autograd.grad(rollout,parameters))
    gradients=dict(one_step=first,rollout=recursive,total=first+rollout_weight*recursive)
    unchanged=before==model_digest(model) and torch.equal(rng,torch.get_rng_state())
    unchanged &= all((a is None and p.grad is None) or (a is not None and p.grad is not None and torch.equal(a,p.grad)) for a,p in zip(old_grads,parameters))
    if not unchanged:raise AssertionError('Frozen gradient diagnostic mutated state')
    return dict(loss=float(total.detach()),one_step_loss=float(one.detach()),rollout_loss=float(rollout.detach()),
        gradient_norms={key:float(value.norm()) for key,value in gradients.items()},no_mutation_verified=True),gradients


def cosine(first: torch.Tensor, second: torch.Tensor) -> float | None:
    denominator=first.norm()*second.norm()
    return float(torch.dot(first,second)/denominator) if float(denominator)>0 else None


def load_models(source: Path, seed: int, cfg: dict, hashes: dict) -> dict:
    models={}
    for condition in ['shared','separate']:
        path=source/f'{condition}-seed{seed}'/'checkpoint-020000.pt'
        hashes[str(path)]=file_sha256(path)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        records={}
        for topology,record in saved['topologies'].items():
            model=ObservableDynamicsGNN(cfg['width'],cfg['layers'])
            model.load_state_dict({key.removeprefix('dynamics.'):value for key,value in record['model'].items() if key.startswith('dynamics.')})
            model.eval();records[topology]=dict(model=model,metadata=record['metadata'],digest=model_digest(model))
        models[condition]=records
        del saved
    return models


def load_data(source: Path, seed: int, topology: str, hashes: dict) -> dict:
    result={'fitting':{},'fresh':{}}
    for condition in ['shared','separate']:
        folder=source/f'{condition}-seed{seed}'/topology
        path=folder/'replay.npz';hashes[str(path)]=file_sha256(path)
        with np.load(path) as arrays:result['fitting'][condition]=dict(arrays)
        run=json.loads((folder.parent/'report.json').read_text())
        episodes=[]
        for index,row in enumerate(run['fresh'][topology]['episodes']):
            path=folder/'fresh-final'/row['trace'];digest=file_sha256(path)
            if digest!=row['sha256']:raise ValueError('Fresh trace hash changed')
            hashes[str(path)]=digest
            with np.load(path) as arrays:
                episodes.append({**{key:arrays[key] for key in ('state','next_state','action','progress','terminated','truncated')},
                                 'episode':np.full(len(arrays['state']),index,dtype=np.int64)})
        result['fresh'][condition]={key:np.concatenate([episode[key] for episode in episodes]) for key in episodes[0]}
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/dynamics-pilot/online-multi-source-2026-10-24')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/dynamics-pilot/online-multi-diagnostic-2026-10-24')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cfg=json.loads((args.source/'protocol.json').read_text())['settings']
    source_protocol=json.loads((args.source/'protocol.json').read_text())
    for path,digest in source_protocol['sources'].items():
        if file_sha256(args.source/'source-snapshot'/path)!=digest:raise ValueError('Original source snapshot hash mismatch')
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=str(args.source),
        predeclared_plan_sha256=file_sha256(ROOT/'docs/plans/gnn_dynamics_multi_diagnostic.md'),
        diagnostic_script_sha256=file_sha256(Path(__file__)),settings=cfg,threads=1,elapsed_cap_seconds=1200,
        horizons=[1,5,10,20],prediction_cell_limit=32,gradient_replicates=4,gradient_batch_per_generator=16,
        diagnostic_seed=271828,no_fitting=True,no_simulator_execution=True,no_checkpoint_selection=True,
        cohorts='terminal-ending; endpoint 1-10 steps precollapse; ordinary farther than 10 or no observed true terminal',
        comparison='Same saved windows and equal-generator fitting batches for shared and corresponding separate model; gradient cosine only at common shared parameters')
    write_json(args.output/'protocol.json',protocol)
    started=time.perf_counter();hashes={};predictions=[];gradient_rows=[];cosines=[];samples=[]
    check_cap=lambda: time.perf_counter()-started<1200
    try:
        for seed in cfg['seeds']:
            models=load_models(args.source,seed,cfg,hashes)
            shared_gradients={}
            for topology_index,topology in enumerate(cfg['topologies']):
                if not check_cap():raise TimeoutError('Predeclared elapsed cap reached')
                data=load_data(args.source,seed,topology,hashes)
                metadata=models['shared'][topology]['metadata']
                graph=tensor_graph({**metadata,'state':data['fitting']['shared']['state'][:1]})
                for role_index,(role,generated) in enumerate(data.items()):
                    for generator_index,(generator,arrays) in enumerate(generated.items()):
                        for horizon in protocol['horizons']:
                            rng=np.random.default_rng(271828+seed*100000+topology_index*10000+role_index*1000+generator_index*100+horizon)
                            selected=draw_cohorts(arrays,horizon,rng)
                            samples.append(dict(seed=seed,topology=topology,role=role,generator=generator,horizon=horizon,cohorts=selected))
                            for stratum,record in selected.items():
                                starts=np.asarray(record['starts'],dtype=np.int64)
                                for condition in ['shared','separate']:
                                    metrics=prediction_metrics(models[condition][topology]['model'],arrays,starts,horizon,metadata,graph)
                                    predictions.append(dict(seed=seed,topology=topology,role=role,generator=generator,horizon=horizon,stratum=stratum,
                                        condition=condition,available=record['available'],**metrics))
                for replicate in range(4):
                    if not check_cap():raise TimeoutError('Predeclared elapsed cap reached')
                    rng=np.random.default_rng(314159+seed*10000+topology_index*100+replicate)
                    batch,index=balanced_batch(data['fitting'],rng)
                    samples.append(dict(seed=seed,topology=topology,role='gradient',replicate=replicate,indices=index))
                    for condition in ['shared','separate']:
                        metrics,gradients=frozen_gradients(models[condition][topology]['model'],batch,graph,cfg['rollout_weight'])
                        gradient_rows.append(dict(seed=seed,topology=topology,condition=condition,replicate=replicate,**metrics))
                        if condition=='shared':shared_gradients[topology,replicate]=gradients
                for condition in ['shared','separate']:
                    record=models[condition][topology]
                    if model_digest(record['model'])!=record['digest']:raise AssertionError('Frozen model changed')
                write_json(args.output/'partial-results.json',dict(predictions=predictions,gradients=gradient_rows,cosines=cosines))
                print(json.dumps(dict(seed=seed,topology=topology,wall_seconds=time.perf_counter()-started)),flush=True)
                del data
            for first,second in itertools.combinations(cfg['topologies'],2):
                for replicate in range(4):
                    cosines.append(dict(seed=seed,first=first,second=second,replicate=replicate,
                        **{component:cosine(shared_gradients[first,replicate][component],shared_gradients[second,replicate][component]) for component in ['one_step','rollout','total']}))
            del models,shared_gradients
        for path,digest in hashes.items():
            if file_sha256(Path(path))!=digest:raise AssertionError('Input artifact changed during diagnosis')
        result=dict(status='complete',wall_seconds=time.perf_counter()-started,protocol=protocol,predictions=predictions,
            gradients=gradient_rows,cosines=cosines,input_hashes=hashes,no_input_artifact_mutation=True,
            all_gradient_calls_no_mutation=True)
        write_json(args.output/'results.json',result);write_json(args.output/'samples.json',dict(samples=samples))
        from scripts.report_dynamics_multi_diagnostic import report
        report(args.output)
    except Exception:
        import traceback
        write_json(args.output/'failure.json',dict(traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-started))
        raise


if __name__=='__main__':main()
