#!/usr/bin/env python3
"""Independent exact-worker verifier for the ten R3N full episodes.

Exact fork/COW counterfactuals never mutate the live episode, so every compact
direct-action trace must reproduce without controller re-execution.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_r3m_shot_restraint as r3m
import rl_r3n_full_runs as r3n
from irisu_rl.exact_training_runtime import ExactTrainingRuntime


def direct_replay(row: dict[str, object]) -> tuple[bool, int | None]:
    core, campaign = r3m.screen._load_external()
    trace = (r3n.DEFAULT_RUN_ROOT / str(row["trace_file"])).read_bytes()
    if hashlib.sha256(trace).hexdigest() != row["trace_sha256"]:
        raise RuntimeError("R3N direct trace hash differs")
    expected = {int(item["tick"]): item for item in row["checkpoints"]}  # type: ignore[index]
    words = [word for (word,) in struct.iter_unpack("<I", trace)]
    mismatch = None
    with ExactTrainingRuntime(r3n.EXACT_WORKER).open_env(
        simulation_config={"max_episode_ticks": r3n.MAX_TICKS + r3m.GATE_CONFIG.probe_ticks},
    ) as exact_session:
        env = exact_session.environment
        observation, _ = env.reset(seed=int(row["seed"]))
        for word in words:
            observation, _reward, _terminated, _truncated, _info = env.step(
                r3m.decode_action(core, word)
            )
            tick = int(observation["tick"])
            if tick in expected and r3m.checkpoint(env, observation) != expected[tick]:
                mismatch = tick
                break
        if mismatch is None:
            exact = (
                r3m.checkpoint(env, observation) == row["final"]
                and hashlib.sha256(env.clone_state()).hexdigest()
                == row["final_snapshot_sha256"]
            )
        else:
            exact = False
    return exact, mismatch


def main() -> None:
    identity, plan = r3n.validate(r3n.DEFAULT_RUN_ROOT)
    summary = r3m.read_json(r3n.DEFAULT_RUN_ROOT / "summary.json")
    r3m.verify_self_hash(summary, "R3N summary")
    rows = r3n.load_units(r3n.DEFAULT_RUN_ROOT)
    direct_exact: list[int] = []
    mismatches: dict[str, int] = {}
    started = time.monotonic()
    for row in rows:
        exact, mismatch = direct_replay(row)
        index = int(row["index"])
        if exact:
            direct_exact.append(index)
        elif mismatch is not None:
            mismatches[str(index)] = mismatch
    expected_direct = list(range(r3n.EPISODE_COUNT))
    if direct_exact != expected_direct or mismatches:
        raise RuntimeError("R3N exact direct replay inventory differs")
    best = summary["best"]
    if int(best["index"]) not in direct_exact:
        raise RuntimeError("R3N winning unit lacks exact direct replay")
    result = r3m.with_sha(
        {
            "schema": "irisu-r3n-full-run-independent-verification-v2",
            "development_only": True,
            "sealed_test_allowed": False,
            "verifier_source_sha256": r3m.sha256_file(Path(__file__).resolve()),
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "summary_sha256": summary["sha256"],
            "unit_sha256s": [row["sha256"] for row in rows],
            "direct_replay_exact_indices": direct_exact,
            "direct_replay_mismatches": mismatches,
            "controller_replay_exact_indices": [],
            "winning_index": best["index"],
            "winning_score": best["score"],
            "winning_direct_replay_exact": True,
            "verified_episode_count": r3n.EPISODE_COUNT,
            "verified": True,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    output = r3n.DEFAULT_RUN_ROOT / "verification-v2.json"
    r3m.write_new(output, result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
