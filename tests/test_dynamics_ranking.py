"""Check ranking rollout score and termination bookkeeping independently of fitting."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'sac')]
from dynamics.rigidity import critical_eigenvalue
from scripts.evaluate_dynamics_ranking import model_scores


class FixedDynamics:
    def __init__(self, collapse=False):
        self.collapse = collapse

    def step(self, state, action, graph):
        result = state.clone()
        if self.collapse:
            result[:, :, :3] = 0
        return result, torch.full((len(state), 2), .01)


class DynamicsRankingTests(unittest.TestCase):
    def fixture(self):
        state = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        state[:, :2] -= state[:, :2].mean(0)
        edges = np.asarray([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
        episode = dict(state=np.concatenate((state,np.zeros_like(state)),axis=1)[None],
            edge_index=edges.T,node_to_physical=np.arange(4),physical_mask=np.ones(4,dtype=bool),
            action_mask=np.ones(4,dtype=bool),initial_diagonal=2.,initial_rigidity_reference=critical_eigenvalue(state,edges),
            structural_edges=edges,axis_indices=[0,1,2],wcrm=False,dt=[.2],speed=.05,
            routing_incidence=np.asarray([[-1,1,0,0]]),ctrlrange=np.asarray([[-1,1]]))
        cfg=OmegaConf.to_container(OmegaConf.load(Path(__file__).resolve().parents[1]/'config/dynamics_ranking.yaml'))
        return episode,cfg

    def test_routed_effort_and_progress_share_physical_time(self):
        episode,cfg=self.fixture()
        actions=np.zeros((2,10,4,1),dtype=np.float32);actions[1,:,1]=1
        scores,collapsed=model_scores(FixedDynamics(),episode['state'][0],actions,episode,cfg)
        np.testing.assert_allclose(scores,[.05,.05-.1*2*.05**2],atol=1e-8)
        self.assertFalse(collapsed.any())

    def test_collapse_stops_progress_and_pays_remaining_horizon(self):
        episode,cfg=self.fixture()
        scores,collapsed=model_scores(FixedDynamics(collapse=True),episode['state'][0],np.zeros((1,10,4,1)),episode,cfg)
        np.testing.assert_allclose(scores,[.01/2-1-1.8],atol=1e-8)
        self.assertTrue(collapsed[0])


if __name__ == '__main__':
    unittest.main()
