from __future__ import annotations

import tomllib
from pathlib import Path

from irisu_env import Action


ROOT = Path(__file__).resolve().parents[1]


def test_expert_gate_config_is_exact_only():
    value = tomllib.loads(
        (ROOT / "configs/rl/experiments/exact-50k-expert-gate-v1.toml").read_text()
    )
    assert value["physics_backend"] == "exact"
    assert value["state_producing_backends"] == ["exact"]
    assert value["target_score"] == 50_000


def test_expert_gate_runner_has_no_legacy_checkpoint_dependency():
    source = (ROOT / "benchmarks/rl_exact_50k_expert_gate.py").read_text()
    for forbidden in ("BASE_CHECKPOINT", "POLICY_FACTORY", "rl_r3k", "rl_r3m"):
        assert forbidden not in source
    assert "ClosedLoopSteeringExpert" in source
    assert "ExactTrainingRuntime" in source


def test_expert_gate_trace_word_encoding():
    import sys

    sys.path.insert(0, str(ROOT / "benchmarks"))
    from rl_exact_50k_expert_gate import encode

    assert encode(Action.wait(1)) == 0
    assert encode(Action.strong(123, 45)) == (45 << 12) | (123 << 2) | 2
