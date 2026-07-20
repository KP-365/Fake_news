"""Regression tests for ZeroGPU-safe startup device selection."""

from __future__ import annotations

import inspect
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import predict


class PredictDeviceTests(unittest.TestCase):
    @staticmethod
    def fake_torch(cuda_available: bool) -> SimpleNamespace:
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda_available),
            device=lambda name: name,
        )

    def test_zero_gpu_environment_forces_cpu(self) -> None:
        with patch.dict(os.environ, {"SPACES_ZERO_GPU": "true"}):
            device = predict._select_startup_device(self.fake_torch(True))

        self.assertEqual(device, "cpu")

    def test_normal_environment_uses_available_cuda(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            device = predict._select_startup_device(self.fake_torch(True))

        self.assertEqual(device, "cuda")

    def test_normal_environment_falls_back_to_cpu(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            device = predict._select_startup_device(self.fake_torch(False))

        self.assertEqual(device, "cpu")

    def test_predict_does_not_import_internal_spaces_config(self) -> None:
        self.assertNotIn("spaces.config", inspect.getsource(predict))


if __name__ == "__main__":
    unittest.main()
