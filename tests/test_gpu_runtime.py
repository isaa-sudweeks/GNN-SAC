from importlib import metadata
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sac"))
from common.gpu_runtime import require_supported_mjx_gpu


class MjxGpuRuntimeTest(unittest.TestCase):
    def test_cuda13_rejects_v100_with_actionable_error(self):
        cfg = SimpleNamespace(device="cuda", mujoco_backend="mjx")
        with patch("common.gpu_runtime.torch.version.cuda", "13.0"), \
             patch("common.gpu_runtime.metadata.version", side_effect=metadata.PackageNotFoundError), \
             patch("common.gpu_runtime.torch.cuda.get_device_capability", return_value=(7, 0)), \
             patch("common.gpu_runtime.torch.cuda.get_device_name", return_value="Tesla V100"):
            with self.assertRaisesRegex(RuntimeError, "V100.*7.0"):
                require_supported_mjx_gpu(cfg)

    def test_cuda13_plugin_is_checked_even_if_torch_uses_cuda12(self):
        cfg = SimpleNamespace(device="cuda:0", mujoco_backend="mjx")
        with patch("common.gpu_runtime.torch.version.cuda", "12.9"), \
             patch("common.gpu_runtime.metadata.version", return_value="0.11.0"), \
             patch("common.gpu_runtime.torch.cuda.get_device_capability", return_value=(7, 0)), \
             patch("common.gpu_runtime.torch.cuda.get_device_name", return_value="Tesla V100"):
            with self.assertRaisesRegex(RuntimeError, "compute capability >= 7.5"):
                require_supported_mjx_gpu(cfg)

    def test_supported_devices_pass(self):
        cfg = SimpleNamespace(device="cuda", mujoco_backend="mjx")
        for capability in [(7, 5), (8, 0), (8, 9), (9, 0), (10, 0)]:
            with self.subTest(capability=capability), \
                 patch("common.gpu_runtime.torch.version.cuda", "13.0"), \
                 patch("common.gpu_runtime.metadata.version", return_value="0.11.0"), \
                 patch("common.gpu_runtime.torch.cuda.get_device_capability", return_value=capability):
                require_supported_mjx_gpu(cfg)

    def test_cpu_and_native_backends_do_not_probe_cuda(self):
        for cfg in [SimpleNamespace(device="cpu", mujoco_backend="mjx"),
                    SimpleNamespace(device="cuda", mujoco_backend="mujoco")]:
            with patch("common.gpu_runtime.torch.cuda.get_device_capability") as probe:
                require_supported_mjx_gpu(cfg)
                probe.assert_not_called()

    def test_cuda12_only_is_not_restricted_by_cuda13_rule(self):
        cfg = SimpleNamespace(device="cuda", mujoco_backend="mjx")
        with patch("common.gpu_runtime.torch.version.cuda", "12.9"), \
             patch("common.gpu_runtime.metadata.version", side_effect=metadata.PackageNotFoundError), \
             patch("common.gpu_runtime.torch.cuda.get_device_capability") as probe:
            require_supported_mjx_gpu(cfg)
            probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
