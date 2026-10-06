from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_rl.actions import SemanticAction


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import rl_exact_pair_trace_distill as campaign  # noqa: E402


def shot(source: int, destination: int, x: float, y: float) -> SteeringDecision:
    return SteeringDecision(
        SemanticAction.strong(x, y),
        SteeringIntent.STEER_MATCH,
        source_body_id=source,
        destination_body_id=destination,
    )


def test_decode_action_word() -> None:
    word = (123 << 12) | (456 << 2) | 2
    action = campaign.decode_action(word)
    assert int(action.kind) == 2
    assert action.cursor_x == 456
    assert action.cursor_y == 123


def test_coordinate_binding_accepts_only_a_clear_nearest_pair() -> None:
    candidates = (
        shot(1, 2, 0.25, 0.5),
        shot(2, 1, 0.75, 0.5),
    )
    selected, evidence = campaign.bind_shot_to_pair(
        candidates,
        cursor_x=160,
        cursor_y=240,
        client_width=640,
        client_height=480,
        maximum_error_pixels=8.0,
        ambiguity_gap_pixels=4.0,
    )
    assert selected is candidates[0]
    assert evidence["best_error_pixels"] == 0.0
    assert evidence["rejection"] is None


def test_coordinate_binding_rejects_ambiguity_and_large_residual() -> None:
    candidates = (
        shot(1, 2, 100 / 640, 0.5),
        shot(2, 1, 102 / 640, 0.5),
    )
    selected, evidence = campaign.bind_shot_to_pair(
        candidates,
        cursor_x=101,
        cursor_y=240,
        client_width=640,
        client_height=480,
        maximum_error_pixels=8.0,
        ambiguity_gap_pixels=4.0,
    )
    assert selected is None
    assert evidence["rejection"] == "ambiguous-coordinate"
    selected, evidence = campaign.bind_shot_to_pair(
        candidates,
        cursor_x=300,
        cursor_y=240,
        client_width=640,
        client_height=480,
        maximum_error_pixels=8.0,
        ambiguity_gap_pixels=0.0,
    )
    assert selected is None
    assert evidence["rejection"] == "coordinate-error"


def test_trainer_has_no_embedded_evaluation_seeds() -> None:
    source = (ROOT / "benchmarks/rl_exact_pair_trace_distill.py").read_text()
    assert "3939967453" not in source
    assert "locked" + "_seeds" not in source.lower()


def test_trace_balancing_is_equal_and_deterministic() -> None:
    groups = (["a0", "a1", "a2"], ["b0", "b1"], ["c0", "c1", "c2", "c3"])
    first = campaign.trace_balanced(groups, seed=17)
    second = campaign.trace_balanced(groups, seed=17)
    assert first == second
    assert len(first) == 6
    assert [value[0] for value in first] == ["a", "b", "c", "a", "b", "c"]


def test_trace_manifest_binds_trace_source_and_expected_episode(tmp_path: Path) -> None:
    trace = tmp_path / "teacher.u32le"
    trace.write_bytes(struct.pack("<II", 0, 0))
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "schema": "test-source-v1",
                "episode": {"seed": 7, "score": 99, "tick": 2},
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": campaign.TRACE_MANIFEST_SCHEMA,
                "traces": [
                    {
                        "label": "teacher-7",
                        "seed": 7,
                        "trace": trace.name,
                        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                        "source_metadata": source.name,
                        "source_metadata_sha256": hashlib.sha256(
                            source.read_bytes()
                        ).hexdigest(),
                        "expected_score": 99,
                        "expected_ticks": 2,
                    }
                ],
            }
        )
    )
    entries = campaign.load_trace_manifest(manifest)
    assert entries[0]["seed"] == 7
    assert entries[0]["trace"] == trace.resolve()
    assert entries[0]["source_metadata_schema"] == "test-source-v1"
