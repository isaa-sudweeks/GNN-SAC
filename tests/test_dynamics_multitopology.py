"""Variable-size source graphs, balanced exposure, normalization and exact resume."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'sac')]
from dynamics.multitopology import normalize_warmups, share_dynamics, update_round
from dynamics.model import physical_graph_loss
from dynamics.online import OnlineLearner, OnlineReplay
from scripts.collect_dynamics_training_data import make_training_env, graph_metadata, observable_state
from scripts.run_online_dynamics_long import load_resume, nested_equal
from scripts.run_online_dynamics_multi import run_group


def settings():
    cfg=OmegaConf.to_container(OmegaConf.load(ROOT/'config/dynamics_online_multi.yaml'))
    cfg.update(steps=24,warmup=8,initial_updates=8,evaluate_at=[12,24],episode_steps=7,
        evaluation_episodes=1,fresh_evaluation_episodes=1,checkpoint_every=12,width=16,layers=1,
        batch_size=4,training_horizon=3)
    cfg['cem'].update(horizon=3,candidates=4,iterations=1,knots=2,elite_fraction=.5)
    return cfg


class TestMultiTopology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_variable_graphs_masks_routing_aliases_and_equal_initialization(self):
        cfg=settings(); learners={}; replays={}; sizes=[]
        for topology in cfg['topologies']:
            env=make_training_env(topology,OmegaConf.create(cfg))
            try:
                env.reset(); obs,_=env.unwrapped.reset(seed=123)
                metadata=graph_metadata(env.unwrapped,obs)
                state,_=observable_state(env.unwrapped,np.asarray(metadata['node_to_physical']))
                learner=learners[topology]=OnlineLearner(metadata,state,cfg,4,False)
                replay=replays[topology]=OnlineReplay(8,len(state))
                sizes.append((len(state),sum(metadata['physical_mask']),len(metadata['ctrlrange'])))
                for i in range(8):
                    replay.append(state=state,next_state=state,action=np.zeros((len(state),1)),progress=[0.,0.],
                        reward=0.,base_reward=0.,terminated=False,truncated=i%4==3,episode=i//4)
                graph=learner.graph
                s=torch.tensor(state,dtype=torch.float32)[None]
                a=torch.zeros((*s.shape[:2],1)); b=a.clone()
                b[:,~graph['action_mask'].bool()]=1000
                self.assertTrue(torch.equal(learner.model.dynamics(s,a,graph),learner.model.dynamics(s,b,graph)))
                nxt,_=learner.model.dynamics.step(s,a,graph)
                self.assertTrue(torch.equal(nxt,nxt[:,graph['representative_index']]))
                target=nxt.clone(); target[:,~graph['physical_mask']]+=1000
                self.assertEqual(float(physical_graph_loss(nxt,target,graph['physical_mask'],torch.ones(6)).detach()),0.)
                self.assertEqual(replay.starts(3).tolist(),[0,1,4,5])
            finally:
                env.close()
        self.assertEqual(len(set(sizes)),3)
        values=list(learners.values())
        for learner in values[1:]:
            self.assertTrue(nested_equal(values[0].model.dynamics.state_dict(),learner.model.dynamics.state_dict()))
        stats=normalize_warmups(learners,replays)
        mean=np.mean([r.state[:r.size,l.graph['physical_mask']].mean((0,1)).numpy() for l,r in zip(learners.values(),replays.values())],axis=0)
        np.testing.assert_allclose(stats['mean'],mean,atol=1e-7)
        frozen=values[0].model.dynamics.state_mean.clone()
        replays[cfg['topologies'][0]].next_state.fill_(100.)
        share_dynamics(learners)
        for _ in range(3): update_round(learners,replays)
        self.assertEqual([l.updates for l in learners.values()],[3,3,3])
        self.assertIs(values[0].optimizer,values[2].optimizer)
        steps={int(v['step']) for v in values[0].optimizer.state.values()}
        self.assertEqual(steps,{9})
        self.assertTrue(torch.equal(frozen,values[0].model.dynamics.state_mean))
        for learner in values:
            self.assertTrue(torch.equal(learner.model.dynamics.state_scale,values[0].model.dynamics.state_scale))

    def test_exact_mid_episode_resume_both_conditions(self):
        cfg=settings()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for condition in cfg['conditions']:
                full=root/(condition+'full'); split=root/(condition+'split')
                run_group(cfg,condition,2,full)
                run_group(cfg,condition,2,split,stop_after=12)
                first=load_resume(split,cfg)
                self.assertTrue(any(not r['runtime']['done'] for r in first['topologies'].values()))
                run_group(cfg,condition,2,split,resume=True)
                a,b=load_resume(full,cfg),load_resume(split,cfg)
                self.assertEqual(a['statistics'],b['statistics'])
                for topology in cfg['topologies']:
                    x,y=a['topologies'][topology],b['topologies'][topology]
                    for key in ('model','actor','target','target_actor','optimizer','actor_optimizer','updates','learner_rng'):
                        self.assertTrue(nested_equal(x[key],y[key]),condition+':'+topology+':'+key)
                    for key in x['replay']:
                        np.testing.assert_array_equal(x['replay'][key],y['replay'][key])
                    for trace in ['eval-24-0.npz','fresh-final/eval-24-0.npz']:
                        with np.load(full/topology/trace) as left,np.load(split/topology/trace) as right:
                            for key in left.files: np.testing.assert_array_equal(left[key],right[key])
                with self.assertRaises(ValueError):load_resume(split,dict(cfg,batch_size=8))


if __name__=='__main__':unittest.main()
