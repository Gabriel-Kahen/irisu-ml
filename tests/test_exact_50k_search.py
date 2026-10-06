from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
        self.assertRegex(config["runtime"]["worker_sha256"], r"^[0-9a-f]{64}$")

    def test_imitation_shape_is_identity_bound(self) -> None:
        model = search.ImitationModel(8, 12)
        output = model(torch.zeros((3, len(search.FEATURE_NAMES))))
        self.assertEqual(tuple(output.shape), (3, 12))

    def test_source_identity_declares_no_portable_state_producer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sources = [Path(directory) / name for name in (
                "core.py", "campaign.py", "worker", "checkpoint"
            )]
            for path in sources:
                path.write_text("# Frozen external proposal helpers.\n")
            modules = tuple(SimpleNamespace(__file__=str(path)) for path in sources[:2])
            config = search.load_config()
            config["runtime"].update(worker_path=str(sources[2]),
                worker_sha256=search.sha256_file(sources[2]))
            config["warm_start"].update(checkpoint=str(sources[3]),
                checkpoint_sha256=search.sha256_file(sources[3]))
            with (
                mock.patch.object(search.screen, "_load_external", return_value=modules),
                mock.patch.object(search, "load_config", return_value=config),
            ):
                identity = search.source_identity()
                for path in sources:
                    self.assertEqual(identity["files"][str(path)], search.sha256_file(path))
                sources[2].write_text("tampered worker\n")
                with self.assertRaisesRegex(RuntimeError, "exact worker bytes differ"):
                    search.source_identity()
                sources[2].write_text("# Frozen external proposal helpers.\n")
                sources[3].write_text("tampered checkpoint\n")
                with self.assertRaisesRegex(RuntimeError, "warm-start bytes differ"):
                    search.source_identity()
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
