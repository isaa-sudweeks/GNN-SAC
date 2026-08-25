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

from common.cross_validation import resolve_cross_validation


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


def multirun_id(job_num: int, override_dirname: str) -> str:
	"""Return the stable, collision-resistant identity for one Hydra sweep job."""
	digest = hashlib.sha256(str(override_dirname).encode("utf-8")).hexdigest()[:12]
	return f"job_{int(job_num):04d}_{digest}"


def multirun_work_dir(
	work_dir: str | Path,
	*,
	isolate_multirun_runs: bool,
	job_num: int | None,
	override_dirname: str,
) -> Path:
	"""Resolve the run directory shared by Hydra workers and pre-submit checks."""
	work_dir = Path(work_dir)
	if not isolate_multirun_runs or job_num is None:
		return work_dir
	return work_dir / multirun_id(job_num, override_dirname)


def normalize_numeric_value(value: Any) -> Any:
	"""Normalize the integer and single-operation expressions accepted by configs."""
	if not isinstance(value, str):
		return value
	if value.replace('_', '').isdigit():
		return int(value.replace('_', ''))
	match = re.match(r"(\d+)([+\-*/])(\d+)", value)
	if match is None:
		return value
	left, operator, right = match.groups()
	result = eval(left + operator + right)
	if isinstance(result, float) and result.is_integer():
		return int(result)
	return result


def _hydra_multirun_identity() -> tuple[int, str] | None:
	"""Return Hydra's job number and override identity during a multirun."""
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
	return int(job_num), str(override_dirname)


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
			cfg[k] = normalize_numeric_value(cfg[k])
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
		resolve_cross_validation(cfg)
		if cfg.get("work_dir", None) not in {None, "???"}:
			cfg.work_dir = Path(cfg.work_dir)
		else:
			try:
				from hydra.core.hydra_config import HydraConfig
				cfg.work_dir = Path(HydraConfig.get().runtime.output_dir)
			except Exception:
				cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name
		if bool(cfg.get("isolate_multirun_runs", False)):
			identity = _hydra_multirun_identity()
			if identity is not None:
				job_num, override_dirname = identity
				cfg.multirun_id = multirun_id(job_num, override_dirname)
				cfg.work_dir = multirun_work_dir(
					cfg.work_dir,
					isolate_multirun_runs=True,
					job_num=job_num,
					override_dirname=override_dirname,
				)
		cfg.task_title = cfg.task.replace("-", " ").title()



	return cfg_to_dataclass(cfg)
