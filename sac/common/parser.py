import dataclasses
import hashlib
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf, open_dict


LAUNCH_COMMAND_ENV = "GNN_SAC_LAUNCH_COMMAND"


def capture_launch_command(argv: list[str] | None = None) -> str:
	"""Capture the user invocation before Hydra hands the job to Submitit."""
	if argv is None:
		argv = list(getattr(sys, "orig_argv", sys.argv))
	command = shlex.join(argv)
	if "submitit.core._submit" not in argv:
		# Slurm inherits the submission environment, so every Submitit worker in
		# a multirun receives the exact command that launched the whole sweep.
		os.environ[LAUNCH_COMMAND_ENV] = command
	return os.environ.get(LAUNCH_COMMAND_ENV, command)


def _hydra_multirun_id() -> str | None:
	"""Return a stable, collision-resistant identity for one Hydra sweep job."""
	from hydra.core.hydra_config import HydraConfig

	try:
		hydra_cfg = HydraConfig.get()
	except ValueError:
		return None

	job_num = OmegaConf.select(hydra_cfg, "job.num", default=None)
	if job_num is None:
		return None
	override_dirname = OmegaConf.select(
		hydra_cfg,
		"job.override_dirname",
		default="",
	)
	digest = hashlib.sha256(str(override_dirname).encode("utf-8")).hexdigest()[:12]
	return f"job_{int(job_num):04d}_{digest}"


def cfg_to_dataclass(cfg, frozen=False):
	"""
	Converts an OmegaConf config to a dataclass object.
	This prevents graph breaks when used with torch.compile.
	"""
	cfg_dict = OmegaConf.to_container(cfg, resolve=True)
	fields = []
	for key, value in cfg_dict.items():
		fields.append((key, Any, dataclasses.field(default_factory=lambda value_=value: value_)))
	dataclass_name = "Config"
	dataclass = dataclasses.make_dataclass(dataclass_name, fields, frozen=frozen)
	def get(self, val, default=None):
		return getattr(self, val, default)
	dataclass.get = get
	return dataclass()


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
	"""
	Parses a Hydra config. Mostly for convenience.
	"""

	# Logic
	for k in cfg.keys():
		try:
			v = cfg[k]
			if v == None:
				v = True
		except:
			pass

	# Algebraic expressions
	for k in cfg.keys():
		try:
			v = cfg[k]
			if isinstance(v, str):
				if v.replace('_', '').isdigit():
					cfg[k] = int(v.replace('_', ''))
					continue
				match = re.match(r"(\d+)([+\-*/])(\d+)", v)
				if match:
					cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
					if isinstance(cfg[k], float) and cfg[k].is_integer():
						cfg[k] = int(cfg[k])
		except:
			pass

	# Convenience
	with open_dict(cfg):
		project_root = Path(__file__).resolve().parents[2]
		if cfg.get("wandb_dir", None) in {None, "???"}:
			# wandb appends its own `wandb/` directory to this storage parent.
			cfg.wandb_dir = project_root
		cfg.launch_command = os.environ.get(
			LAUNCH_COMMAND_ENV,
			shlex.join(getattr(sys, "orig_argv", sys.argv)),
		)
		topologies = cfg.get("topologies", None)
		truss_topologies = cfg.get("truss_topologies", None)
		if topologies is not None:
			if truss_topologies is not None and list(topologies) != list(truss_topologies):
				raise ValueError("Use either topologies or truss_topologies, not both with different values.")
			cfg.truss_topologies = topologies
		if cfg.get("work_dir", None) not in {None, "???"}:
			cfg.work_dir = Path(cfg.work_dir)
		else:
			try:
				from hydra.core.hydra_config import HydraConfig
				cfg.work_dir = Path(HydraConfig.get().runtime.output_dir)
			except Exception:
				cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name
		if bool(cfg.get("isolate_multirun_runs", False)):
			multirun_id = _hydra_multirun_id()
			if multirun_id is not None:
				cfg.multirun_id = multirun_id
				cfg.work_dir = Path(cfg.work_dir) / multirun_id
		cfg.task_title = cfg.task.replace("-", " ").title()



	return cfg_to_dataclass(cfg)
