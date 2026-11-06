#!/usr/bin/env python3
"""Bounded, independently verified repair comparisons on real replay damage."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from irisu_search.engine import Engine  # noqa: E402
from irisu_search.proposals import generate, select_candidates  # noqa: E402
from irisu_search.verification import NativeVerifier, read_replay, replay_bytes, sha256_file  # noqa: E402


def words_hash(words):
    return hashlib.sha256(struct.pack(f"<{len(words)}I", *words)).hexdigest()


def terminal(summary):
    return bool(summary.get("terminated") and not summary.get("truncated"))


def damaged_reference(words, case):
    start = case["prefix_tick"]
    reference = list(words[start:])
    for edit in case["damage"]:
        offset = edit["tick"] - start
        if not 0 <= offset < len(reference) or reference[offset] != edit["original_word"]:
            raise ValueError("fixture damage does not match the replay")
        reference[offset] = edit["damaged_word"]
    if words_hash(words[:start]) != case["prefix_sha256"] or words_hash(reference) != case["damaged_suffix_sha256"]:
        raise ValueError("fixture recipe hash mismatch")
    return reference


def checkpoint(seed, prefix, row):
    return {"seed": seed, "words": prefix + row["words"], "summary": row["summary"]}


def damage_shot(reference, kind):
    """Apply a small spatial/timing error without introducing an extra edge."""
    if not reference or not reference[0] & 3:
        raise ValueError("fixture must begin at an existing shot")
    changed = list(reference)
    if kind in ("x", "y"):
        shift, mask, maximum = (2, 1023, 640) if kind == "x" else (12, 511, 480)
        coordinate = (reference[0] >> shift) & mask
        amount = 2 if coordinate <= maximum - 2 else -2
        changed[0] = (reference[0] & ~(mask << shift)) | ((coordinate + amount) << shift)
        return changed
    if kind == "timing":
        held = reference[0] & 3
        if len(reference) < 4 or reference[1] & held or reference[2] & 3 or reference[3] & held:
            return None
        changed[0], changed[2] = 0, reference[0]
        return changed
    raise ValueError("unknown fixture damage")


def discover_fixtures(engine, verifier, replay, cases, seconds=120):
    started = time.monotonic()
    seed, words, _ = read_replay(replay)
    anchor = min(96000, max(0, len(words) - 2048))
    shots = [tick for tick in range(anchor, len(words)) if words[tick] & 3]
    found, trials = [], []
    clean_receipt = None
    with Engine([str(engine)], timeout_seconds=600) as worker:
        for tick in shots:
            if time.monotonic() - started >= seconds:
                break
            mutation = ("x", "y", "timing")[len(found) % 3]
            reference = words[tick:]
            damaged = damage_shot(reference, mutation)
            if damaged is None:
                trials.append({"tick": tick, "mutation": mutation, "eligible": False,
                               "reason": "would interfere with neighboring held-button edges"})
                continue
            parent = worker.load(seed, words[:tick])
            horizon = min(100000, max(2048, len(reference) + 1024))
            candidates = [{"mode": 0, "search_seed": 0, "words": recipe} for recipe in (reference, damaged)]
            output = worker.batch(parent["handle"], horizon, candidates)
            clean, broken = output["results"]
            eligible = (terminal(clean["summary"]) and terminal(broken["summary"])
                        and clean["summary"]["canonical_score"] > broken["summary"]["canonical_score"])
            trials.append({"tick": tick, "mutation": mutation, "eligible": eligible,
                           "clean_terminal": terminal(clean["summary"]),
                           "damaged_terminal": terminal(broken["summary"]),
                           "clean_score": clean["summary"].get("canonical_score"),
                           "damaged_score": broken["summary"].get("canonical_score")})
            if eligible:
                if clean_receipt is None:
                    clean_receipt = verifier.verify(checkpoint(seed, words[:tick], clean))
                else:
                    clean_checkpoint = checkpoint(seed, words[:tick], clean)
                    data = replay_bytes(seed, clean_checkpoint["words"], clean_checkpoint["summary"])
                    if hashlib.sha256(data).hexdigest() != clean_receipt["replay_sha256"]:
                        raise ValueError("clean witness changed during fixture discovery")
                damage_receipt = verifier.verify(checkpoint(seed, words[:tick], broken))
                found.append({
                    "id": f"shot-{mutation}-{tick}", "mutation": mutation,
                    "seed": seed, "prefix_tick": tick,
                    "prefix_sha256": words_hash(words[:tick]), "horizon": horizon,
                    "damage": [{"tick": tick + offset, "original_word": before, "damaged_word": after}
                               for offset, (before, after) in enumerate(zip(reference, damaged)) if before != after],
                    "damaged_suffix_sha256": words_hash(damaged),
                    "damaged_score": broken["summary"]["canonical_score"],
                    "reachable_score": clean["summary"]["canonical_score"],
                    "damaged_verification": damage_receipt, "hidden_witness_verification": clean_receipt,
                    "contract": "Common prefix precedes damage; planner receives only the damaged suffix. Clean original is a hidden reachable witness.",
                })
            worker.release([parent["handle"], *(row["handle"] for row in output["results"])])
            if len(found) >= cases:
                break
    if len(found) != cases:
        raise ValueError(f"found only {len(found)} independently verified reachable fixtures")
    return {"schema": "irisu-combo-repair-fixtures-v1", "replay_sha256": sha256_file(replay),
            "generation_seconds": time.monotonic() - started,
            "selection": "First score-lowering terminal damage of each requested class; no search-policy outcomes used for selection.",
            "discovery_trials": trials, "cases": found}


def candidates_for(algorithm, summary, count, search_seed, reference, horizon, held):
    # This interface deliberately receives no healthy reference or witness edits.
    if algorithm == "current":
        catalog = generate(summary, count * 2, search_seed, "repair", reference,
                           horizon, reference, held)
        return select_candidates(summary, catalog, count, seed=search_seed)
    if algorithm == "combo":
        from irisu_search.combos import terminal_candidates
        return terminal_candidates(summary, count, search_seed, reference, horizon,
                                   operator="combo", held=held)
    raise ValueError("unknown search algorithm")


def run_arm(engine, verifier, seed, words, case, algorithm, candidate_budget, seconds,
            search_seed, output):
    started = time.monotonic()
    prefix = words[:case["prefix_tick"]]
    reference = damaged_reference(words, case)
    observed = set()
    attempted = ticks = censored = baseline_ties = novel_terminals = rejected = 0
    best = None
    with Engine([str(engine)], timeout_seconds=600) as worker:
        parent = worker.load(seed, prefix)
        identity = worker.identity
        wave = 0
        while attempted < candidate_budget and time.monotonic() - started < seconds:
            count = min(candidate_budget - attempted, identity.get("max_checkpoints", 64) - 1, 128)
            if count < 1:
                raise ValueError("worker cannot hold a parent and candidate")
            candidates = candidates_for(algorithm, parent["summary"], count, search_seed + wave,
                                        reference, case["horizon"], prefix[-1] & 3 if prefix else 0)
            if not candidates or len(candidates) > count:
                raise ValueError("candidate generator violated the qualification budget")
            result = worker.batch(parent["handle"], case["horizon"], candidates)
            if len(result["results"]) + len(result.get("rejected", [])) != len(candidates):
                raise RuntimeError("worker returned the wrong qualification candidate count")
            attempted += len(candidates)
            rejected += len(result.get("rejected", []))
            ticks += result["simulated_ticks"]
            for row in result["results"]:
                recipe = row["words"]
                observed.add(words_hash(recipe))
                baseline = recipe == reference[:len(recipe)]
                if not terminal(row["summary"]):
                    censored += 1
                elif baseline:
                    if row["summary"]["canonical_score"] != case["damaged_score"]:
                        raise RuntimeError("damaged reference disagrees with the independently verified fixture")
                    baseline_ties += 1
                else:
                    novel_terminals += 1
                    if best is None or row["summary"]["canonical_score"] > best["summary"]["canonical_score"]:
                        best = row
            worker.release([row["handle"] for row in result["results"]])
            wave += 1
    receipt = None
    gain = 0
    replay_path = None
    if best is not None and best["summary"]["canonical_score"] > case["damaged_score"]:
        winning = checkpoint(seed, prefix, best)
        receipt = verifier.verify(winning)
        gain = receipt["canonical_score"] - case["damaged_score"]
        replay_path = output / f"{case['id']}-{algorithm}-{search_seed}.rpy"
        with replay_path.open("xb") as stream:
            stream.write(replay_bytes(seed, winning["words"], winning["summary"]))
    elapsed = time.monotonic() - started
    return {
        "case": case["id"], "algorithm": algorithm, "search_seed": search_seed,
        "candidate_budget": candidate_budget, "wall_budget_seconds": seconds,
        "all_in_seconds": elapsed, "budget_overshoot_seconds": max(0., elapsed - seconds),
        "attempted_candidates": attempted, "simulated_ticks": ticks,
        "unique_executed_recipes": len(observed), "baseline_terminal_ties": baseline_ties,
        "novel_terminal_results": novel_terminals, "censored_results": censored,
        "rejected_candidates": rejected,
        "best_observed_novel_terminal_score": best["summary"]["canonical_score"] if best else None,
        "verified_novel_gain": gain,
        "recovery_fraction": gain / (case["reachable_score"] - case["damaged_score"]),
        "verification": receipt, "verified_replay": str(replay_path) if replay_path else None,
        "engine_identity": identity,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fixture", "eval"))
    parser.add_argument("--engine", type=Path, default=ROOT / "build-project-exact/irisu-search-engine")
    parser.add_argument("--oracle", type=Path, default=ROOT / "build-project-exact/irisu-game-oracle")
    parser.add_argument("--oracle-library", type=Path, default=ROOT / ".local/exact-runtime-20260824/exact-runtime/guest/libirisu_box2d_msvc_exact_multiworld.so")
    parser.add_argument("--replay", type=Path, default=ROOT / "irisu-high-score-4393101.rpy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests/fixtures/search/combo-repair.json")
    parser.add_argument("--cases", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args(argv)
    if not 1 <= args.cases <= 32 or not 1 <= args.candidates <= 1024 or not 0 < args.seconds <= 3600:
        parser.error("require cases1..32, candidates1..1024, and seconds0..3600")
    args.output.mkdir(parents=True, exist_ok=False)
    verifier = NativeVerifier(args.oracle, args.oracle_library, timeout=600)
    if args.mode == "fixture":
        data = discover_fixtures(args.engine, verifier, args.replay, args.cases, args.seconds)
        destination = args.output / "fixtures.json"
    else:
        import irisu_search.combos as combo_module
        import irisu_search.proposals as current_module
        fixtures = json.loads(args.fixture.read_text())
        if fixtures["schema"] != "irisu-combo-repair-fixtures-v1" or fixtures["replay_sha256"] != sha256_file(args.replay):
            raise ValueError("qualification fixture does not match replay")
        if args.cases > len(fixtures["cases"]):
            raise ValueError("requested more cases than the frozen fixture contains")
        seed, words, header = read_replay(args.replay)
        data = {"schema": "irisu-combo-qualification-v1", "fixture_sha256": sha256_file(args.fixture),
                "replay_sha256": sha256_file(args.replay), "reference_record_score": header["canonical_score"],
                "generator_sources": {name: {"path": module.__file__, "sha256": sha256_file(module.__file__)}
                                      for name, module in (("current", current_module), ("combo", combo_module))},
                "scope": "Finite damaged-reference repair comparison; unchanged baseline execution never counts as recovery.",
                "budget_policy": "Identical candidate limits and all-in wall caps; arms can finish below the wall cap. Batch and verification boundary overshoot is reported, not hidden.",
                "results": []}
        destination = args.output / "receipt.json"
        for index, case in enumerate(fixtures["cases"][:args.cases]):
            if case["seed"] != seed:
                raise ValueError("fixture seed mismatch")
            # Alternate arm order across cases to reduce systematic warm-up bias.
            algorithms = ("current", "combo") if index % 2 == 0 else ("combo", "current")
            for algorithm in algorithms:
                try:
                    row = run_arm(args.engine, verifier, seed, words, case, algorithm,
                                  args.candidates, args.seconds, args.seed, args.output)
                except Exception as error:
                    data["failure"] = {"case": case["id"], "algorithm": algorithm, "error": str(error)}
                    destination.write_text(json.dumps(data, indent=2) + "\n")
                    raise
                row["new_record"] = bool(row["verification"] and row["verification"]["canonical_score"] > header["canonical_score"])
                data["results"].append(row)
                destination.write_text(json.dumps(data, indent=2) + "\n")
    destination.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "mode": args.mode}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
