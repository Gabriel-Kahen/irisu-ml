from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rl_exact_adaptive_checkpoint_eval import PolicyBundle
from rl_exact_proposal_blend_eval import parser, validate_blend_contract


SCHEMA_A = "a" * 64
SCHEMA_B = "b" * 64


def _bundle(training_seeds, schema=SCHEMA_A):
    policy = SimpleNamespace(
        model=SimpleNamespace(schema=SimpleNamespace(sha256=schema))
    )
    return PolicyBundle(
        checkpoint_path="checkpoint.pt",
        checkpoint_sha256="c" * 64,
        model_sha256="d" * 64,
        metadata={"training_seeds": list(training_seeds)},
        inference_config={},
        factory=lambda: policy,
    )


def test_blend_contract_unions_lineage_and_rejects_evaluation_overlap():
    base = _bundle([1, 2])
    residual = _bundle([1, 2, 3])

    training, evaluation = validate_blend_contract(
        base, residual, [1, 2], [1, 2, 3], [10, 11]
    )

    assert training == (1, 2, 3)
    assert evaluation == (10, 11)
    with pytest.raises(ValueError, match="overlap"):
        validate_blend_contract(base, residual, [1, 2], [1, 2, 3], [3])


def test_blend_contract_rejects_schema_or_declared_lineage_mismatch():
    base = _bundle([1])
    with pytest.raises(ValueError, match="schemas differ"):
        validate_blend_contract(base, _bundle([2], SCHEMA_B), [1], [2], [10])
    with pytest.raises(ValueError, match="declared training seeds differ"):
        validate_blend_contract(base, _bundle([2]), [9], [2], [10])


def test_development_cli_defaults_bind_fixed512_topk2_and_activation_band():
    options = parser().parse_args(
        [
            "--worker",
            "/worker",
            "--base-checkpoint",
            "base.pt",
            "--base-checkpoint-sha256",
            "a" * 64,
            "--base-training-seed-manifest",
            "base.json",
            "--residual-checkpoint",
            "residual.pt",
            "--residual-checkpoint-sha256",
            "b" * 64,
            "--residual-training-seed-manifest",
            "residual.json",
            "--seeds",
            "123",
            "--output",
            "report.json",
        ]
    )

    assert options.probe_ticks == 512
    assert options.top_k_pairs == 2
    assert options.activation_tick_exclusive == 50_000
    assert options.activation_gauge_exclusive == 20_000
