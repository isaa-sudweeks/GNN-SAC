"""Hydra-discoverable Submitit launcher extensions for GNN-SAC."""

from common import submitit_launcher as _submitit_launcher


class FilteringSlurmLauncher(_submitit_launcher.FilteringSlurmLauncher):
    """Expose the completed-job filtering launcher through Hydra's plugin namespace."""
