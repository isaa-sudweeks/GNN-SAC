"""Discount-only rerun: config isolation, planner weighting, and actual trace returns."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'sac')]
from dynamics.online import OnlineLearner
from scripts.run_online_dynamics_multi import run_group
from scripts.report_online_dynamics_discount_comparison import discounted_trace_return


class ProgressDynamics:
    def step(self,state,action,graph):
        progress=torch.cat((action[:,0,:],torch.zeros_like(action[:,0,:])),dim=1)
        return state,progress


class TestDiscountRerun(unittest.TestCase):
    def test_only_gamma_changes_and_planning_changes_action_ranking(self):
        old=OmegaConf.to_container(OmegaConf.load(ROOT/'config/dynamics_online_multi.yaml'))
        new=OmegaConf.to_container(OmegaConf.load(ROOT/'config/dynamics_online_multi_gamma095.yaml'))
        self.assertEqual([key for key in old if old[key]!=new[key]],['gamma'])
        self.assertEqual((old['gamma'],new['gamma']),(.995,.95))
        learner=OnlineLearner.__new__(OnlineLearner)
        learner.metadata=dict(initial_diagonal=1.,control_interval_seconds=1.)
        learner.use_value=False
        learner.cem=SimpleNamespace(horizon=10)
        learner.graph=dict(action_mask=torch.ones(1),routing_incidence=torch.zeros(1,1),
            speed=torch.tensor(1.),ctrlrange=torch.tensor([[-1.,1.]]))
        learner.model=SimpleNamespace(dynamics=ProgressDynamics())
        learner.collapsed=lambda state:torch.zeros(len(state),dtype=torch.bool)
        initial=np.zeros((1,6),dtype=np.float32)
        sequences=np.zeros((2,10,1),dtype=np.float32)
        sequences[0,0,0]=.2;sequences[1,9,0]=.21
        scores=[]
        for cfg in [old,new]:
            learner.settings=cfg
            value=learner.score(initial,sequences)
            np.testing.assert_allclose(value,[.2,.21*cfg['gamma']**9],rtol=1e-6)
            scores.append(value)
        self.assertEqual([int(np.argmax(score)) for score in scores],[1,0])

    def test_common_discount_rescoring_retains_late_collapse_and_rejects_invalid_data(self):
        reward=np.zeros(80);reward[0]=.2;reward[-1]=-1.
        self.assertAlmostEqual(discounted_trace_return(reward,.95),.2-.95**79)
        self.assertAlmostEqual(discounted_trace_return(reward,.995),.2-.995**79)
        self.assertAlmostEqual(discounted_trace_return(reward,1.),-.8)
        self.assertGreater(discounted_trace_return(reward,.95),discounted_trace_return(reward,.995))
        for data,gamma in [(np.array([np.nan]),.95),(np.zeros((2,2)),.95),(reward,0.),(reward,1.01)]:
            with self.assertRaises(ValueError):discounted_trace_return(data,gamma)

    def test_gamma095_native_collection_and_frozen_evaluation_discount(self):
        torch.set_num_threads(1)
        cfg=OmegaConf.to_container(OmegaConf.load(ROOT/'config/dynamics_online_multi_gamma095.yaml'))
        cfg.update(steps=24,warmup=8,initial_updates=8,evaluate_at=[24],episode_steps=7,
            evaluation_episodes=1,fresh_evaluation_episodes=1,checkpoint_every=24)
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'shared'
            result=run_group(cfg,'shared',0,folder)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(set(result['interactions'].values()),{24})
            self.assertEqual(set(result['updates'].values()),{24})
            for topology in cfg['topologies']:
                for evaluation,base in [(result['evaluations'][topology][0],folder/topology),
                                        (result['fresh'][topology],folder/topology/'fresh-final')]:
                    episode=evaluation['episodes'][0]
                    with np.load(base/episode['trace']) as trace:
                        reward=trace['reward']
                        expected=sum(float(r)*.95**t for t,r in enumerate(reward))
                        self.assertAlmostEqual(expected,episode['discounted_task_return'],places=12)


if __name__=='__main__':unittest.main()
