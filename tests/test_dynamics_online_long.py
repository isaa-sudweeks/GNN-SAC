"""Long-run checkpoint integrity and exact mid-episode resume regression."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'sac')]
import numpy as np
from omegaconf import OmegaConf
import torch

from scripts.run_online_dynamics_long import run_long_one, load_resume, nested_equal


class TestOnlineLongResume(unittest.TestCase):
    def test_exact_resume_retains_models_optimizers_replay_and_actions(self):
        torch.set_num_threads(1)
        cfg=OmegaConf.to_container(OmegaConf.load('config/dynamics_online_long.yaml'))
        cfg.update(steps=48,warmup=16,initial_updates=16,evaluate_at=[16,32,48],
                   episode_steps=12,evaluation_episodes=1,fresh_evaluation_episodes=1,
                   checkpoint_every=16)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for condition in cfg['conditions']:
                uninterrupted=root/(condition+'-full')
                resumed=root/(condition+'-resumed')
                run_long_one(cfg,condition,4,uninterrupted,[])
                run_long_one(cfg,condition,4,resumed,[],stop_after=32)
                # Step 32 is eight steps into a 12-step episode: native/controller state matters.
                saved=load_resume(resumed,cfg)
                self.assertEqual(saved['episode_runtime'][1],8)
                run_long_one(cfg,condition,4,resumed,[],resume=True)
                left=torch.load(uninterrupted/'final-model.pt',weights_only=False)
                right=torch.load(resumed/'final-model.pt',weights_only=False)
                for key in ('model','actor','target','target_actor','optimizer','actor_optimizer'):
                    self.assertTrue(nested_equal(left[key],right[key]),key)
                with np.load(uninterrupted/'replay.npz') as a,np.load(resumed/'replay.npz') as b:
                    for key in a.files:
                        self.assertTrue(np.array_equal(a[key],b[key]),key)
                for name in ['eval-48-0.npz','fresh-final/eval-48-0.npz','fresh-random/eval-48-0.npz']:
                    with np.load(uninterrupted/name) as a,np.load(resumed/name) as b:
                        for key in a.files:
                            self.assertTrue(np.array_equal(a[key],b[key]),name+':'+key)
                changed=dict(cfg,learning_rate=cfg['learning_rate']*2)
                with self.assertRaises(ValueError):load_resume(resumed,changed)
                manifest=json.loads((resumed/'resume.json').read_text())
                manifest['sha256']='invalid'
                (resumed/'resume.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):load_resume(resumed,cfg)


if __name__=='__main__':unittest.main()
