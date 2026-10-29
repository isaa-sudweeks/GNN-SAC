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

from common.parser import capture_launch_command, parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from env import make_env
from sac import SAC
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

capture_launch_command()

torch.backends.cudnn.benchmark = True 
torch.set_float32_matmul_precision('high')

def validate_training_mode(cfg) -> None:
    """Reject ambiguous initialization and invalid offline/online budgets."""
    options = getattr(cfg, "distillation", {})
    get_option = options.get if hasattr(options, "get") else lambda key, default=None: getattr(options, key, default)
    offline_only = bool(get_option("offline_only", False))
    if offline_only:
        if not bool(get_option("enabled", False)) or int(get_option("pretrain_updates", 0)) <= 0:
            raise ValueError("Offline-only mode requires enabled distillation and positive pretrain_updates.")
        if int(cfg.steps) < 0:
            raise ValueError("Offline-only mode requires steps >= 0.")
    elif int(cfg.steps) <= 0:
        raise ValueError("Number of steps must be positive for online training.")
    if getattr(cfg, "init_from_agent", None) not in (None, "", "???") and getattr(
        cfg, "resume_from_checkpoint", None
    ) not in (None, "", "???", False):
        raise ValueError("init_from_agent and resume_from_checkpoint are mutually exclusive.")


def initialize_agent_from_export(agent, checkpoint_ref) -> None:
    """Load model/alpha while preserving freshly created optimizers and replay."""
    if checkpoint_ref in (None, "", "???"):
        return
    path = Path(checkpoint_ref).expanduser()
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "agent" in state:
        state = state["agent"]
    agent.load(state)
    print("Initialized agent from:", path)


def run_training(cfg, trial=None):
    """
    Execute one training run and return the best objective value seen.
    """
    sac_backend = str(getattr(cfg, "sac_backend", "mlp")).lower()
    if sac_backend not in {"mlp", "gnn", "padded_mlp"}:
        raise ValueError(
            f"Unsupported sac_backend={sac_backend!r}; expected 'mlp', 'gnn', or 'padded_mlp'."
        )

    if getattr(cfg, 'device', 'cuda') == 'cuda':
        assert torch.cuda.is_available(), "CUDA not available, please run on a GPU"
    cfg = parse_cfg(cfg)
    validate_training_mode(cfg)
    from common.training_validation import validate_training_config, validate_padded_mlp_environment

    validate_training_config(cfg)
    if sac_backend in {"gnn", "padded_mlp"} and str(getattr(cfg, "mujoco_backend", "mujoco")).lower() == "mjx":
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    set_seed(cfg.seed)

    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

    env = make_env(cfg)
    trainer = None
    try:
        validate_padded_mlp_environment(cfg, env)
        if sac_backend in {"gnn", "padded_mlp"}:
            from common.tensor_gnn_buffer import make_gnn_buffer
            if sac_backend == "gnn":
                from gnn_sac import GNNSAC

                agent = GNNSAC(cfg)
            else:
                from padded_mlp_sac import PaddedMLPSAC

                agent = PaddedMLPSAC(cfg)
            buffer = make_gnn_buffer(cfg)
        else:
            agent = SAC(cfg)
            buffer = Buffer(cfg)
        initialize_agent_from_export(agent, getattr(cfg, "init_from_agent", None))
        trainer = OnlineTrainer(
            cfg=cfg,
            env=env,
            agent=agent,
            logger=Logger(cfg),
            buffer=buffer,
            trial=trial,
        )
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
        if trainer is not None:
            trainer.logger.finish()
        raise
    except Exception:
        if trial is not None:
            cfg.optuna_trial_state = "failed"
        if trainer is not None:
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
