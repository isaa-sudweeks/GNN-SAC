"""Regression checks for source-episode provenance and physical aliases."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sac'))
from dynamics.dataset import file_sha256, load_dataset, validate_episode
from omegaconf import OmegaConf
from scripts.collect_dynamics_training_data import collect


def episode():
    state = np.zeros((2, 3, 6))
    state[:, :, 2] = 1
    state[:, [0, 2], 0] = -1
    state[:, 1, 0] = 1
    return dict(state=state, next_state=state.copy(), action=np.zeros((2, 3, 1)),
                progress=np.ones((2,2)),global_xy=np.array([[0.,0.],[1.,1.]]),
                next_global_xy=np.array([[1.,1.],[2.,2.]]),dt=np.ones(2)*.2,
                node_to_physical=[0,1,0],physical_mask=[True,True,False],
                terminated=np.array([False,False]),truncated=np.array([False,False]))


class DynamicsDatasetTests(unittest.TestCase):
    def test_aliases_use_unique_physical_center(self):
        result=validate_episode(episode())
        self.assertEqual(result['physical_nodes'],2)
        self.assertEqual(result['transitions'],2)

    def test_alias_disagreement_and_duplicate_mask_rejected(self):
        value=episode();value['state'][0,2,0]=2
        with self.assertRaisesRegex(ValueError,'aliases'):
            validate_episode(value)
        value=episode();value['physical_mask']=[True,True,True]
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            validate_episode(value)

    def test_adjacency_and_terminal_boundary_rejected(self):
        value=episode();value['next_state'][0,:,2]=2
        with self.assertRaisesRegex(ValueError,'adjacency'):
            validate_episode(value)
        value=episode();value['terminated'][0]=True
        with self.assertRaisesRegex(ValueError,'after episode end'):
            validate_episode(value)

    def test_resume_rejects_changed_collection_settings_before_environment_creation(self):
        settings = OmegaConf.load(Path(__file__).resolve().parents[1] / 'config/dynamics_training_data.yaml')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            saved = OmegaConf.to_container(settings)
            saved['speed'] = 0.123
            (root / 'collection-protocol.json').write_text(json.dumps(dict(settings=saved)))
            with self.assertRaisesRegex(ValueError, 'changed settings'):
                collect(root, settings, ['tetrahedron'], resume=True)

    def test_progress_and_elapsed_time_checked(self):
        value=episode();value['progress'][0,0]=0
        with self.assertRaisesRegex(ValueError,'Progress'):
            validate_episode(value)
        value=episode();value['dt'][0]=0
        with self.assertRaisesRegex(ValueError,'positive'):
            validate_episode(value)

    def test_loader_filters_roles_and_rejects_holdouts_and_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);arrays=episode()
            metadata={k:arrays.pop(k) for k in ['node_to_physical','physical_mask']}
            metadata.update(edge_index=[[0,1],[1,0]],edge_static=[[1,0],[1,0]],
                            action_mask=[True,True,False],initial_bbox=[1,1,1],
                            routing_incidence=[[-1,1,0]],ctrlrange=[[-1,1]],structural_edges=[[0,1]])
            (root/'metadata.json').write_text(json.dumps(metadata))
            np.savez(root/'episode.npz',**arrays)
            record=dict(role='fitting',topology='tetrahedron',file='episode.npz',
                        sha256=file_sha256(root/'episode.npz'),metadata='metadata.json',
                        metadata_sha256=file_sha256(root/'metadata.json'),episode_id=0,seed=12000)
            manifest=dict(episodes=[record]);(root/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(len(load_dataset(root,['validation'])),0)
            self.assertEqual(len(load_dataset(root,['fitting'])),1)
            record['topology']='octahedron';(root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'non-source'):
                load_dataset(root)
            record['topology']='tetrahedron';(root/'manifest.json').write_text(json.dumps(manifest))
            (root/'metadata.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Metadata hash'):
                load_dataset(root)


if __name__=='__main__':
    unittest.main()
