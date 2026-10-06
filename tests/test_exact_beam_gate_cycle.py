from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from benchmarks.rl_exact_beam_gate_cycle import (
    initialize,
    load_config,
    next_attempt,
    select,
    sha256,
)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    worker = tmp_path / "worker"
    worker.write_bytes(b"exact-worker")
    trace = tmp_path / "teacher.u32le"
    trace.write_bytes(struct.pack("<3I", 0, 1, 0))
    result = tmp_path / "teacher.json"
    dump(
        result,
        {
            "physics_backend": "exact",
            "state_producing_backends": ["exact"],
            "natural_terminal": True,
            "seed": 7,
            "trace_sha256": sha256(trace),
            "proposal_lineage": "mixed teacher",
            "final": {
                "tick": 3,
                "score": 10,
                "terminated": True,
                "truncated": False,
            },
        },
    )
    config = tmp_path / "cycle.toml"
    config.write_text(
        """
version = "exact-beam-gate-cycle-v1"
physics_backend = "exact"
target_score = 30
max_ticks = 100
parallelism = 1
[[arms]]
name = "p4"
cutoff = 2
probe = 4
wait = 2
gauge_advantage = 1
"""
    )
    root = tmp_path / "cycle"
    plan = initialize(root, config, result, trace, worker)
    return root, plan


def attempt_result(
    root: Path,
    plan: dict[str, object],
    name: str,
    score: int,
    *,
    natural: bool = True,
) -> None:
    directory = root / "arms" / name / "attempt-0001"
    trace = directory / "continuation.u32le"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_bytes(struct.pack("<4I", 0, 1, 0, 2))
    dump(
        directory / "result.json",
        {
            "physics_backend": "exact",
            "state_producing_backends": ["exact"],
            "natural_terminal": natural,
            "trace_sha256": sha256(trace),
            "proposal_lineage": "portable-trained frozen-v5; exact closed-loop continuation",
            "exact_runtime": {"identity": {"worker_sha256": plan["worker_sha256"]}},
            "final": {
                "tick": 4,
                "score": score,
                "terminated": natural,
                "truncated": not natural,
            },
        },
    )


def test_initialize_freezes_natural_exact_teacher(tmp_path: Path) -> None:
    root, plan = fixture(tmp_path)
    assert plan["teacher"]["score"] == 10
    assert plan["teacher"]["seed"] == 7
    assert plan["proposal_lineage"].startswith("portable-trained")
    assert initialize(
        root,
        Path(plan["config"]),
        Path(plan["teacher"]["result"]),
        Path(plan["teacher"]["trace"]),
        Path(plan["worker"]),
    ) == plan


def test_selection_rejects_nonnatural_and_is_append_only(tmp_path: Path) -> None:
    root, plan = fixture(tmp_path)
    attempt_result(root, plan, "p4", 20)
    attempt_result(root, plan, "bad", 999, natural=False)
    first = select(root)
    assert first["winner"]["score"] == 20
    assert len(first["rejected"]) == 1
    assert Path(first["next_parent_trace"]).read_bytes() == struct.pack(
        "<4I", 0, 1, 0, 2
    )
    assert select(root) == first
    attempt_result(root, plan, "better", 30)
    second = select(root)
    assert second["winner"]["score"] == 30
    assert second["target_reached"] is True
    assert Path(second["next_parent_trace"]).parent.name == "0002"


def test_interrupted_attempt_resumes_from_durable_progress(tmp_path: Path) -> None:
    root, plan = fixture(tmp_path)
    attempt = root / "arms/p4/attempt-0001"
    progress_trace = attempt / "progress.u32le"
    progress_trace.parent.mkdir(parents=True)
    progress_trace.write_bytes(struct.pack("<5I", 0, 1, 0, 2, 0))
    dump(
        attempt / "progress.json",
        {"action_count": 5, "trace_sha256": sha256(progress_trace)},
    )
    retry, source, cutoff, ordinal = next_attempt(root, plan["arms"][0], plan)
    assert retry.name == "attempt-0002"
    assert source == progress_trace
    assert cutoff == 4
    assert ordinal == 2


def test_config_rejects_duplicate_arm_names(tmp_path: Path) -> None:
    config = tmp_path / "bad.toml"
    config.write_text(
        """
version = "exact-beam-gate-cycle-v1"
physics_backend = "exact"
[[arms]]
name = "same"
cutoff = 1
probe = 1
[[arms]]
name = "same"
cutoff = 2
probe = 2
"""
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_config(config)
