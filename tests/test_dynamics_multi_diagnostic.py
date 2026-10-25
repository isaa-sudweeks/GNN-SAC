"""Frozen gradients and diagnostic cohorts cannot mutate or cross real episodes."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'sac')]
from dynamics.model import ObservableDynamicsGNN
from dynamics.training import tensor_graph
from scripts.run_dynamics_multi_diagnostic import balanced_batch,cohorts,valid_starts,draw_cohorts,frozen_gradients,diagnostic_loss,model_digest,prediction_metrics


def toy_replay():
    state=np.zeros((12,4,6),np.float32)
    state[:,:,:3]=np.array([[0,0,1],[1,0,1],[0,1,1],[0,0,2]],np.float32)
    nxt=state.copy();nxt[:,:,2]+=.01
    return dict(state=state,next_state=nxt,action=np.zeros((12,4,1),np.float32),progress=np.zeros((12,2),np.float32),
                episode=np.repeat([0,1,2],4),terminated=np.array([0,0,0,1,0,0,0,0,0,0,0,1],bool),
                truncated=np.array([0,0,0,0,0,0,0,1,0,0,0,0],bool))


def metadata():
    edges=np.array([[a,b] for a in range(4) for b in range(4) if a!=b]).T
    return dict(node_to_physical=[0,1,2,3],physical_mask=[1,1,1,1],action_mask=[1,1,1,1],edge_index=edges,
        initial_diagonal=1.,structural_edges=[[a,b] for a in range(4) for b in range(a+1,4)],axis_indices=[0,1,2],
        wcrm=False,initial_rigidity_reference=1.,critical_eig_threshold=.01)


class TestMultiDiagnostic(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(27)

    def test_cohorts_partition_safe_windows_and_retain_terminal_endings(self):
        data=toy_replay()
        self.assertEqual(valid_starts(data,3).tolist(),[0,1,4,5,8,9])
        cells=cohorts(data,3)
        self.assertEqual(cells['terminal_ending'].tolist(),[1,9])
        self.assertEqual(cells['precollapse'].tolist(),[0,8])
        self.assertEqual(cells['ordinary'].tolist(),[4,5])
        drawn=draw_cohorts(data,3,np.random.default_rng(12),1)
        self.assertTrue(all(len(v['starts'])==1 and v['available']==2 for v in drawn.values()))
        self.assertEqual(drawn,draw_cohorts(data,3,np.random.default_rng(12),1))
        data['episode'][1]=9
        self.assertNotIn(0,valid_starts(data,3).tolist())
        data['truncated'][5]=True
        self.assertNotIn(4,valid_starts(data,3).tolist())
        with self.assertRaises(ValueError):valid_starts(data,0)

    def test_balanced_generators_and_exact_frozen_gradient(self):
        first=toy_replay();second=toy_replay();second['state']=second['state']+.1
        batch,indices=balanced_batch({'shared':first,'separate':second},np.random.default_rng(2),horizon=3,half_batch=3)
        self.assertEqual(batch['state'].shape[0],6)
        self.assertEqual([len(v['recursive']) for v in indices.values()],[3,3])
        for generator,data in [('shared',first),('separate',second)]:
            self.assertTrue(set(indices[generator]['recursive'])<=set(valid_starts(data,3)))
        graph=tensor_graph({**metadata(),'state':first['state'][:1]})
        model=ObservableDynamicsGNN(16,1);model.eval()
        for parameter in model.parameters():parameter.grad=torch.ones_like(parameter)*.25
        digest=model_digest(model);rng=torch.get_rng_state().clone()
        total,_,_=diagnostic_loss(model,batch,graph)
        expected=torch.cat([v.reshape(-1) for v in torch.autograd.grad(total,tuple(model.parameters()))])
        report,gradients=frozen_gradients(model,batch,graph)
        torch.testing.assert_close(gradients['total'],expected,rtol=1e-5,atol=1e-6)
        self.assertEqual(model_digest(model),digest);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(p.grad,torch.full_like(p,.25)) for p in model.parameters()))
        self.assertTrue(report['no_mutation_verified'])

    def test_nonfinite_predictions_are_penalized_and_unsafe_windows_rejected(self):
        class Nonfinite:
            def step(self,state,action,graph):return state*float('nan'),state.new_full((len(state),2),float('nan'))
        data=toy_replay();meta=metadata();graph=tensor_graph({**meta,'state':data['state'][:1]})
        measured=prediction_metrics(Nonfinite(),data,np.array([1]),3,meta,graph)
        self.assertEqual(measured['windows'],1);self.assertEqual(measured['nonfinite_windows'],1)
        self.assertEqual(measured['progress_mae'],100.)
        self.assertEqual(measured['position_rmse'],100.)
        self.assertEqual(measured['endpoint']['predicted_collapses'],1)
        with self.assertRaisesRegex(ValueError,'Unsafe'):
            prediction_metrics(Nonfinite(),data,np.array([3]),3,meta,graph)
        self.assertEqual(prediction_metrics(Nonfinite(),data,np.array([],dtype=int),3,meta,graph),{'windows':0})


if __name__=='__main__':unittest.main()
