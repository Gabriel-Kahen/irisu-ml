from __future__ import annotations

import unittest

from benchmarks.rl_exact_adaptive_controller import (
    DEFAULT_CONFIG,
    GaugeHysteresis,
    load_config,
)


class ExactAdaptiveControllerTests(unittest.TestCase):
    def test_checked_config_is_exact_and_uses_ordered_thresholds(self) -> None:
        config, _payload = load_config(DEFAULT_CONFIG)
        self.assertEqual(config["physics_backend"], "exact")
        self.assertLess(config["enter_survival_gauge"], config["exit_survival_gauge"])
        self.assertLess(
            config["score_mode"]["probe_ticks"],
            config["survival_mode"]["probe_ticks"],
        )

    def test_hysteresis_does_not_chatter_inside_band(self) -> None:
        gate = GaugeHysteresis(12_000, 20_000)
        self.assertEqual(gate.update(12_000), "score")
        self.assertEqual(gate.update(11_999), "survival")
        self.assertEqual(gate.update(15_000), "survival")
        self.assertEqual(gate.update(20_000), "survival")
        self.assertEqual(gate.update(20_001), "score")
        self.assertEqual(gate.update(15_000), "score")

    def test_invalid_thresholds_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            GaugeHysteresis(20_000, 12_000)


if __name__ == "__main__":
    unittest.main()
