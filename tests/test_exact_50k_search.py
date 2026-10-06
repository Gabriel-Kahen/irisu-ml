from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_exact_50k_search as search  # noqa: E402
import rl_exact_50k_gate_imitation as gate_imitation  # noqa: E402


class Exact50kSearchTests(unittest.TestCase):
    def test_config_is_exact_only(self) -> None:
        config = search.load_config()
        self.assertEqual(config["physics_backend"], "exact")
        self.assertEqual(config["state_producing_backends"], ["exact"])
        self.assertEqual(config["runtime"]["worker_sha256"], search.sha256_file(
            Path(config["runtime"]["worker_path"])
        ))

    def test_imitation_shape_is_identity_bound(self) -> None:
        model = search.ImitationModel(8, 12)
        output = model(torch.zeros((3, len(search.FEATURE_NAMES))))
        self.assertEqual(tuple(output.shape), (3, 12))

    def test_source_identity_declares_no_portable_state_producer(self) -> None:
        identity = search.source_identity()
        self.assertEqual(identity["physics_backend"], "exact")
        self.assertEqual(identity["state_producing_backends"], ["exact"])
        self.assertTrue(identity["promotion_eligible"])

    def test_gate_imitation_config_is_exact_only(self) -> None:
        config = gate_imitation.config()
        self.assertEqual(config["physics_backend"], "exact")
        self.assertEqual(config["state_producing_backends"], ["exact"])
        resolved = gate_imitation.gate_config(config)
        self.assertEqual(resolved.probe_ticks, 128)
        self.assertEqual(resolved.wait_ticks, 16)


if __name__ == "__main__":
    unittest.main()
