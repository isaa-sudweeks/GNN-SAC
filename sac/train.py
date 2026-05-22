from pathlib import Path
import sys
import os
import platform
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

_default_mujoco_gl = 'glfw' if platform.system() == 'Darwin' else 'egl'
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", _default_mujoco_gl)
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch
import optuna

import hydra
from termcolor import colored
from omegaconf import OmegaConf

if sys.version_info >= (3, 14):
    if not getattr(argparse._ActionsContainer._check_help, "_hydra_py314_compat", False):
        _argparse_check_help = argparse._ActionsContainer._check_help

        def _check_help_compat(self, action):
            if action.help is not None and not isinstance(action.help, str):
                return
            return _argparse_check_help(self, action)

        _check_help_compat._hydra_py314_compat = True
        argparse._ActionsContainer._check_help = _check_help_compat

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from env import make_env
from sac import SAC
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True 
torch.set_float32_matmul_precision('high')

def run_training(cfg, trial=None):
    """
    Execute one training run and return the best objective value seen.
    """
    if getattr(cfg, 'device', 'cuda') == 'cuda':
        assert torch.cuda.is_available(), "CUDA not available, please run on a GPU"
    assert cfg.steps > 0, "Number of steps must be positive"
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

    env = make_env(cfg)
    trainer = OnlineTrainer(
        cfg = cfg,
        env = env, # I need to make this 
        agent=SAC(cfg),
        logger=Logger(cfg),
        buffer=Buffer(cfg),
        trial=trial,
    )

    try:
        trainer.train()
        objective_value, objective_metric = trainer.best_objective()
        if trial is not None:
            cfg.optuna_trial_state = "complete"
        print(
            colored("Optimization objective:", "cyan", attrs=["bold"]),
            f"{objective_metric}={objective_value:.6f}",
        )
        print(colored('Training completed!', 'green', attrs=['bold']))
        return objective_value
    except optuna.TrialPruned:
        cfg.optuna_trial_state = "pruned"
        trainer.logger.finish()
        raise
    except Exception:
        if trial is not None:
            cfg.optuna_trial_state = "failed"
        trainer.logger.finish()
        raise
    finally:
        env.close()


@hydra.main(config_name='config', config_path='../config')
def train(cfg):
    """
    Script for training SAC agents.

    Most relevant args:
		`env_name`: name of the environment 
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)
    
    Example usage:
    	python train.py env_name=ant-locomotion-v4 steps=10M seed=1
    
    """
    return run_training(cfg)


if __name__ == '__main__':
    train()
