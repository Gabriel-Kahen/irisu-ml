from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from benchmarks import rl_r1, rl_r2b, rl_r3a  # noqa: E402
from irisu_rl.one_body import OneBodyTask  # noqa: E402


class ExactTrainingDefaultTests(unittest.TestCase):
    def test_one_body_fails_closed_without_explicit_exact_worker(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit worker_path"):
            OneBodyTask(1, 100.0)

    def test_one_body_rejects_implicit_portable_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            OneBodyTask(
                1,
                100.0,
                physics_backend="portable",
                library_path=Path("/tmp/portable.so"),
            )

    def test_one_body_portable_requires_explicit_library(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit library_path"):
            OneBodyTask(
                1,
                100.0,
                physics_backend="portable",
                diagnostic_portable=True,
            )

    def test_r1_default_requires_exact_worker(self) -> None:
        with mock.patch.object(sys, "argv", ["rl_r1.py"]):
            with self.assertRaises(SystemExit) as raised:
                rl_r1.main()
        self.assertEqual(raised.exception.code, 2)

    def test_r2b_default_requires_exact_worker(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["rl_r2b.py", "--checkpoint-root", "/tmp/unused-r2b-checkpoints"],
        ):
            with self.assertRaises(SystemExit) as raised:
                rl_r2b.main()
        self.assertEqual(raised.exception.code, 2)

    def test_r3a_portable_mode_is_explicit(self) -> None:
        exact = rl_r3a.parser().parse_args(["--runtime", "/tmp/exact-worker"])
        portable = rl_r3a.parser().parse_args(
            ["--runtime", "/tmp/portable.so", "--diagnostic-portable"]
        )
        self.assertFalse(exact.diagnostic_portable)
        self.assertTrue(portable.diagnostic_portable)

    def test_checked_training_configs_bind_exact_end_to_end(self) -> None:
        paths = (
            ROOT / "configs/rl/r0-r1.toml",
            ROOT / "configs/rl/experiments/r2a-smoke-v1.toml",
            ROOT / "configs/rl/experiments/r2b-one-body-v1.toml",
            ROOT / "configs/rl/experiments/r3a-multistep-v1.toml",
        )
        for path in paths:
            with self.subTest(path=path):
                simulator = tomllib.loads(path.read_text())["simulator"]
                self.assertEqual(simulator["training_backend"], "exact")
                self.assertEqual(
                    simulator["worker_path_policy"],
                    "explicit_absolute_attested_path_required",
                )
                self.assertEqual(
                    simulator["portable_mode"],
                    "explicit_diagnostic_only_non_promotable",
                )


if __name__ == "__main__":
    unittest.main()
