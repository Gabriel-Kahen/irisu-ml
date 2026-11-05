from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import ExactProtocolError, ExactSimulator, ExactWorkerClient, NativeError  # noqa: E402
from irisu_metsu import MetsuSimulator, MetsuWorkerClient  # noqa: E402

METSU_WORKER = os.environ.get("IRISU_METSU_WORKER")
NORMAL_WORKER = os.environ.get("IRISU_EXACT_WORKER")
NORMAL_BACKEND = "exact-msvc9-r58-multiworld-forward"
METSU_BACKEND = "exact-msvc9-r58-metsu-forward"


class MetsuExplicitPathTests(unittest.TestCase):
    def test_metsu_never_resolves_the_normal_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit.*metsu"):
            MetsuSimulator(None)


@unittest.skipUnless(METSU_WORKER and NORMAL_WORKER, "requires both exact mode workers")
class MetsuClientIsolationTests(unittest.TestCase):
    def test_reciprocal_wrong_worker_rejection_and_unchanged_normal_default(self) -> None:
        for client_type, path in ((ExactWorkerClient, METSU_WORKER),
                                  (MetsuWorkerClient, NORMAL_WORKER)):
            with self.subTest(client_type=client_type.__name__):
                with self.assertRaisesRegex(ExactProtocolError, "required exact multiworld backend"):
                    client_type(path)
        with mock.patch.dict(os.environ, {"IRISU_EXACT_WORKER": NORMAL_WORKER}):
            with ExactSimulator() as normal, MetsuSimulator(METSU_WORKER) as metsu:
                self.assertEqual(normal.build_info()["worker_backend"], NORMAL_BACKEND)
                self.assertEqual(metsu.build_info()["worker_backend"], METSU_BACKEND)
                self.assertEqual(normal.config_hash(), 0xEC0E8463FEAF2670)
                self.assertNotEqual(metsu.config_hash(), normal.config_hash())
                self.assertEqual(normal.reset(41)["gauge"], 3_000)
                self.assertEqual(metsu.reset(41)["gauge"], 1_000)

    def test_reset_uses_fresh_metsu_worker_and_reproduces_seed(self) -> None:
        with MetsuSimulator(METSU_WORKER) as simulator:
            initial = simulator.reset(123)
            first_pid = simulator.build_info()["worker_pid"]
            simulator.step(2, 300, 360, 17)
            repeated = simulator.reset(123)
            self.assertEqual(repeated, initial)
            self.assertNotEqual(simulator.build_info()["worker_pid"], first_pid)
            self.assertEqual(simulator.build_info()["worker_backend"], METSU_BACKEND)
            self.assertEqual(len(repeated["bodies"]), 10)
            self.assertEqual(repeated["difficulty"]["active_colors"], 6)

    def test_durable_restore_reproduces_hidden_future_and_rejects_normal_snapshot(self) -> None:
        with MetsuSimulator(METSU_WORKER) as source, MetsuSimulator(METSU_WORKER) as restored, \
                ExactSimulator(NORMAL_WORKER) as normal:
            source.reset(41)
            source.step(0, 0, 0, 19)
            source.step(3, 300, 360, 1)
            source.step(0, 0, 0, 23)
            durable = source.clone_state()
            restored.restore_state(durable)
            self.assertEqual(restored.observation(), source.observation())
            self.assertEqual(restored.state_hash(), source.state_hash())
            self.assertEqual(restored.clone_state(), durable)
            for action in ((2, 280, 350, 1), (0, 0, 0, 70), (1, 330, 300, 1)):
                self.assertEqual(restored.step(*action), source.step(*action))
                self.assertEqual(restored.clone_state(), source.clone_state())
            normal.reset(41)
            normal_snapshot = normal.clone_state()
            metsu_stable = source.clone_state()
            with self.assertRaisesRegex(NativeError, "backend/config identity mismatch"):
                source.restore_state(normal_snapshot)
            with self.assertRaisesRegex(NativeError, "backend/config identity mismatch"):
                normal.restore_state(metsu_stable)
            self.assertEqual(source.clone_state(), metsu_stable)
            self.assertEqual(normal.clone_state(), normal_snapshot)

    @unittest.skipUnless(os.name == "posix" and sys.platform == "linux", "Linux fork checkpoints")
    def test_fast_branch_preserves_metsu_subclass_through_detached_reset(self) -> None:
        with MetsuSimulator(METSU_WORKER) as source:
            source.reset(73)
            source.step(0, 0, 0, 29)
            durable = source.clone_state()
            with source.fast_checkpoint() as checkpoint:
                with checkpoint.branch() as branch:
                    self.assertIsInstance(branch, MetsuSimulator)
                    self.assertEqual(branch.build_info()["worker_backend"], METSU_BACKEND)
                    self.assertEqual(branch.clone_state(), durable)
                    self.assertNotEqual(branch.build_info()["worker_pid"], source.build_info()["worker_pid"])
                    for action in ((2, 300, 350, 1), (0, 0, 0, 31)):
                        self.assertEqual(branch.step(*action), source.step(*action))
                    old_pid = branch.build_info()["worker_pid"]
                    branch.reset(74)
                    self.assertNotEqual(branch.build_info()["worker_pid"], old_pid)
                    self.assertEqual(branch.build_info()["worker_backend"], METSU_BACKEND)
                    self.assertEqual(branch.observation()["gauge"], 1_000)
                    branch.step(0, 0, 0, 1)
                    self.assertEqual(branch.observation()["tick"], 1)


if __name__ == "__main__":
    unittest.main()
