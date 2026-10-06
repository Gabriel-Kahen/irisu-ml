from __future__ import annotations

import unittest
from pathlib import Path

from benchmarks.rl_exact_controller_evolution import (
    DEFAULT_CONFIG,
    Candidate,
    choose_best,
    load_config,
)


class ExactControllerEvolutionTests(unittest.TestCase):
    def test_checked_grid_is_exact_genealogical_and_has_control(self) -> None:
        config, candidates, _payload = load_config(DEFAULT_CONFIG)
        self.assertEqual(config["physics_backend"], "exact")
        self.assertEqual(sum(not item.gate for item in candidates), 1)
        self.assertGreaterEqual(len(candidates), 4)
        incumbent = next(item for item in candidates if item.generation == 0 and item.gate)
        mutants = [item for item in candidates if item.generation == 1]
        self.assertTrue(mutants)
        self.assertTrue(all(item.parent == incumbent.id for item in mutants))

    def test_candidate_rejects_implicit_or_malformed_gate(self) -> None:
        with self.assertRaises(ValueError):
            Candidate.parse({"id": "bad", "generation": 0, "gate": True})
        with self.assertRaises(ValueError):
            Candidate.parse(
                {
                    "id": "bad-control",
                    "generation": 0,
                    "gate": False,
                    "probe_ticks": 1,
                }
            )

    def test_selection_requires_natural_termination_then_maximizes_score(self) -> None:
        rows = [
            {"terminal": True, "censored": False, "score": 50_001, "survival_ticks": 40},
            {"terminal": False, "censored": True, "score": 99_999, "survival_ticks": 100},
            {"terminal": True, "censored": False, "score": 50_001, "survival_ticks": 50},
        ]
        self.assertIs(choose_best(rows), rows[2])
        with self.assertRaises(RuntimeError):
            choose_best(rows[1:2])

    def test_default_artifact_name_is_scoped(self) -> None:
        from benchmarks.rl_exact_controller_evolution import DEFAULT_RUN_ROOT

        self.assertTrue(DEFAULT_RUN_ROOT.name.startswith("exact-50k-evo-"))
        self.assertIsInstance(DEFAULT_CONFIG, Path)

    def test_gate_ceiling_is_part_of_replay_contract(self) -> None:
        _config, candidates, _payload = load_config(DEFAULT_CONFIG)
        ceilings = {
            100_000 + (item.gate_config().probe_ticks if item.gate else 0)
            for item in candidates
        }
        self.assertGreater(len(ceilings), 1)


if __name__ == "__main__":
    unittest.main()
