from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from rl_exact_50k_ppo import load_exact_teacher_data, sha256_bytes  # noqa: E402


def _teacher(tmp_path, *, exact: bool):
    source = tmp_path / "source"
    (source / "dataset").mkdir(parents=True)
    (source / "teacher-traces").mkdir()
    dataset = source / "dataset/teacher-00.npz"
    np.savez_compressed(
        dataset,
        global_features=np.zeros((1, 2), dtype=np.float32),
        body_features=np.zeros((1, 3, 4), dtype=np.float16),
        body_mask=np.ones((1, 3), dtype=np.bool_),
        kind=np.zeros(1, dtype=np.int64),
        wait_index=np.zeros(1, dtype=np.int64),
        xy=np.zeros((1, 2), dtype=np.float32),
    )
    trace = source / "teacher-traces/00.u32le"
    replay = source / "teacher-traces/00.rpy"
    trace.write_bytes(b"\0\0\0\0")
    replay.write_bytes(b"replay")
    manifest = {
        "physics_backend": "exact" if exact else "portable",
        "targets_derived_exclusively_from_exact_rollouts": exact,
        "exact_runtime": {"physics_backend": "exact" if exact else "portable"},
        "episodes": [
            {
                "index": 0,
                "seed": 7,
                "terminal": True,
                "dataset": "dataset/teacher-00.npz",
                "dataset_sha256": sha256_bytes(dataset.read_bytes()),
                "trace": "teacher-traces/00.u32le",
                "trace_sha256": sha256_bytes(trace.read_bytes()),
                "replay": "teacher-traces/00.rpy",
                "replay_sha256": sha256_bytes(replay.read_bytes()),
            }
        ],
    }
    (source / "dataset-manifest.json").write_text(json.dumps(manifest))
    return source


def test_reused_teacher_requires_exact_provenance(tmp_path):
    source = _teacher(tmp_path, exact=False)
    with pytest.raises(RuntimeError, match="fail-closed exact provenance"):
        load_exact_teacher_data(source, tmp_path / "run")


def test_reused_exact_teacher_verifies_and_loads(tmp_path):
    source = _teacher(tmp_path, exact=True)
    run = tmp_path / "run"
    episodes, manifest = load_exact_teacher_data(source, run)
    assert episodes[0].source_seed == 7
    assert episodes[0].length == 1
    assert manifest["physics_backend"] == "exact"
    assert manifest["targets_derived_exclusively_from_exact_rollouts"] is True


def test_reused_teacher_rejects_mutated_trace(tmp_path):
    source = _teacher(tmp_path, exact=True)
    (source / "teacher-traces/00.u32le").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="trace hash mismatch"):
        load_exact_teacher_data(source, tmp_path / "run")
