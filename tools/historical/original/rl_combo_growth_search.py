#!/usr/bin/env python3
"""Event-aware beam search that preserves unfinished combo opportunities.

Independent experiment runner: no live campaign configuration or planner is
modified. Saved frontier prefixes can restart this runner or ordinary MPC.
Heuristic potential is never published as a completed game score.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "benchmarks")]

import torch
from irisu_env.exact_ipc import _BODY, _OBSERVATION_HEADER, _TRANSITION
from irisu_rl.exact_training_runtime import ExactTrainingRuntime
from rl_combo_growth_proposals import propose_macros
from rl_combo_opportunity_value import ComboHistory, OpportunityConfig, evaluate_opportunity, _health
from rl_combo_mpc import (Artifacts, Budget, Recorder, StopRequested, continue_mpc,
                          jsonable, output_lock, read_source, stop_signals, write_json)
from rl_combo_replay_audit import unpack_events
from rl_combo_rollout import RolloutConfig, load_policy_bundle, run_feedback
from rl_exact_suffix_beam import atomic_write, snapshot, trace_bytes
from rl_record_rollout import ReplaySchedule, _snapshot, advance, evaluate
from rl_record_tail_search import save_best

VERSION = "combo-growth-beam-v1"


@dataclass
class Node:
    suffix: tuple[int, ...]
    observation: dict
    final: dict
    history: ComboHistory
    value: object
    family: str
    labels: tuple[str, ...]
    maximum_score_debt: int = 0

    @property
    def natural(self):
        return bool(self.final["terminated"] and not self.final["truncated"])

    @property
    def identifier(self):
        return hashlib.sha256(trace_bytes(self.suffix)).hexdigest()[:20]


def endpoint_key(node: Node):
    """Deduplicate exact observable endpoints and history, never score alone.

    Action prefix remains in the key because public observations omit hidden
    physics and random-generator state. Distinct histories must not be merged.
    """
    return node.identifier


def live_achievements(node: Node, root_history: ComboHistory):
    live = {int(b.get("chain_id", 0)) for b in node.observation.get("bodies", ())
            if b.get("kind") == "piece" and _health(b) > 0}
    existing, opened = [], []
    for gid in live - {0}:
        row = node.history.groups.get(gid, {})
        members = int(row.get("cumulative_members", 0))
        old = root_history.groups.get(gid)
        if old:
            existing.append(max(0, members - int(old["cumulative_members"])))
        else:
            opened.append(members)
    return max(existing, default=0), max(opened, default=0), sum(max(0, n-2) for n in opened)


def select_frontier(nodes, width, root_history, *, preserve_reference=True):
    """Round-robin protected growth/opening/value/score/gauge slots.

    There is deliberately no best-score-minus-X floor or wait-relative debt
    veto. A finite horizon and explicit potential decay bound patience.
    """
    if width < 1:
        raise ValueError("beam width must be positive")
    viable = [n for n in nodes if not (n.final["terminated"] or n.final["truncated"])]
    achievements = {id(n): live_achievements(n, root_history) for n in viable}
    keys = (
        lambda n: (n.value.objective, n.final["score"]),
        lambda n: (achievements[id(n)][0], n.value.extension_value, n.value.objective),
        lambda n: (achievements[id(n)][1:], n.value.opening_value, n.value.objective),
        lambda n: (n.value.extension_value, n.value.cashout_value, n.value.objective),
        lambda n: (n.final["score"], n.final["gauge"]),
        lambda n: (n.value.opening_value, n.value.objective),
        lambda n: (n.final["gauge"], n.value.objective),
    )
    pools = [sorted(viable, key=key, reverse=True) for key in keys]
    # Keep one exact inherited route as a measured control and reusable parent.
    reference = next((n for n in viable if preserve_reference and n.labels and
                      all(label == "reference" for label in n.labels)), None)
    selected = [reference] if reference is not None else []
    seen = {endpoint_key(reference)} if reference is not None else set()
    if len(selected) == width:
        return selected
    for rank in range(len(viable)):
        for pool in pools:
            node = pool[rank]
            key = endpoint_key(node)
            if key not in seen:
                seen.add(key)
                selected.append(node)
                if len(selected) == width:
                    return selected
    return selected


def replay_events(env, actions, start, stop, history, budget=None):
    """Replay batched neutral spans while retaining every native event."""
    native = env._native
    current = native.observation()
    final = snapshot(current)
    # Only decode this span; a short probe must not rescan 90,000 prefix words.
    for _, action in ReplaySchedule(actions[start:stop]).iter(0, stop-start):
        if budget:
            budget.check()
        native.send_step(int(action.kind), action.cursor_x, action.cursor_y, action.wait_ticks)
        payload = native.receive_step_raw()
        header = _OBSERVATION_HEADER.unpack_from(payload)
        trans = _TRANSITION.unpack_from(payload, _OBSERVATION_HEADER.size + header[15] * _BODY.size)
        if trans[15]:
            raise RuntimeError("invalid input during growth replay")
        history.update(unpack_events(payload))
        final = _snapshot(header, trans)
        if final["terminated"] or final["truncated"]:
            break
    current = native.observation()
    current.update({key: value for key, value in final.items() if key.startswith("canonical_")})
    return current, final


class HistoryRecorder(Recorder):
    def __init__(self, *args, history, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = history

    def step(self, action):
        result = super().step(action)
        self.history.update([{**e, "kind": e["kind_name"]} for e in result[-1]["events"]])
        return result


def node_report(node, root_history):
    new_sizes = [int(row["cumulative_members"]) for gid, row in node.history.groups.items()
                 if gid not in root_history.groups]
    return dict(id=node.identifier, final=node.final, family=node.family, labels=node.labels,
                maximum_score_debt=node.maximum_score_debt,
                achievements=live_achievements(node, root_history),
                new_combos=dict(largest=max(new_sizes, default=0),
                                at_least_three=sum(n >= 3 for n in new_sizes),
                                at_least_five=sum(n >= 5 for n in new_sizes)),
                value=asdict(node.value))


def beam_search(env, observation, history, actions, args, bundle, budget, output):
    root = int(observation["tick"])
    value_config = OpportunityConfig(**{
        name: getattr(args, name, getattr(OpportunityConfig(), name))
        for name in ("cashout_weight", "extension_weight", "opening_weight")})
    prefix = actions[:root]
    initial = Node((), observation, snapshot(observation), history,
                   evaluate_opportunity(observation, history, value_config), "root", ())
    frontier, archive, terminals = [initial], [], []
    rng = random.Random(args.random_seed)
    stats = Counter(branches=0, stages=0, policy_branches=0, deficit_survivors=0)
    baseline = {}
    schedule = ReplaySchedule(actions)
    stop_tick = min(args.max_ticks, root + args.horizon)
    search_deadline = time.monotonic() + args.search_seconds
    progress_time = 0.0
    source_hash = env.state_hash()
    with env.fast_checkpoint() as checkpoint, (output / "candidates.jsonl").open("w") as journal:
        # Fixed reference checkpoints quantify temporary deficits at equal time.
        with checkpoint.branch() as branch:
            cursor = root
            for end in range(root + args.segment, stop_tick + args.segment, args.segment):
                end = min(end, stop_tick)
                obs, final, natural = advance(branch, schedule, cursor, end, pipeline_size=16)
                baseline[end] = final
                cursor = final["tick"]
                if natural or final["truncated"] or end == stop_tick:
                    break
        children = []
        last_stage_seconds = 0.0
        try:
            while frontier and int(frontier[0].final["tick"]) < stop_tick:
                budget.check()
                if time.monotonic() + 1.2*last_stage_seconds >= search_deadline:
                    break
                stage_started = time.monotonic()
                children = []
                complete_stage = True
                for parent in frontier:
                    budget.check()
                    if time.monotonic() >= search_deadline:
                        complete_stage = False
                        break
                    start = int(parent.final["tick"])
                    end = min(start + args.segment, stop_tick)
                    duration = end - start
                    with checkpoint.branch() as branch:
                        if start > root:
                            restored, restored_final, _ = advance(branch,
                                ReplaySchedule(prefix + list(parent.suffix)), root, start, pipeline_size=16)
                            if restored_final != parent.final or restored != parent.observation:
                                raise RuntimeError("saved growth prefix failed exact reconstruction")
                        with branch.fast_checkpoint() as parent_checkpoint:
                            macros = propose_macros(parent.observation, parent.history,
                                                     duration=duration, cap=args.candidates, rng=rng)
                            plans = [(m.label, m.family, m) for m in macros]
                            plans.append(("reference", "reference", None))
                            if bundle is not None:
                                plans.append(("feedback", "feedback", None))
                            for label, family, macro in plans:
                                budget.check()
                                if time.monotonic() >= search_deadline:
                                    complete_stage = False
                                    break
                                child_history = parent.history.copy()
                                with parent_checkpoint.branch() as child:
                                    if family == "feedback":
                                        full_prefix = prefix + list(parent.suffix)
                                        recorder = HistoryRecorder(child, parent.observation, full_prefix,
                                            budget, history=child_history)
                                        result = run_feedback(recorder, parent.observation, args.seed, bundle,
                                            RolloutConfig(mode="policy", cooldown_ticks=args.cooldown,
                                                          max_ticks=args.max_ticks),
                                            prefix=full_prefix, stop_tick=end)
                                        actual = tuple(result["actions"][root:])
                                        obs, final = jsonable(result["observation"]), result["final"]
                                        obs.update({k: v for k, v in final.items() if k.startswith("canonical_")})
                                        stats["policy_branches"] += 1
                                    else:
                                        extension = list(actions[start:end]) if family == "reference" else [0]*duration
                                        extension += [0]*(duration-len(extension))
                                        if macro:
                                            for offset, word in macro.sparse:
                                                extension[offset] = word
                                        full = prefix + list(parent.suffix) + extension
                                        obs, final = replay_events(child, full, start, end, child_history, budget)
                                        actual = tuple(full[root:final["tick"]])
                                value = evaluate_opportunity(obs, child_history, value_config,
                                                             elapsed_ticks=int(final["tick"])-root)
                                reference_score = baseline.get(end, {}).get("score", 0)
                                debt = max(parent.maximum_score_debt, reference_score-int(final["score"]))
                                node = Node(actual, obs, final, child_history, value, family,
                                            parent.labels + (label,), debt)
                                stats["branches"] += 1
                                journal.write(json.dumps(jsonable(node_report(node, history))) + "\n")
                                if node.natural:
                                    terminals.append(node)
                                    stats["terminal_evaluations"] += 1
                                    # Keep useful complete replays without materializing
                                    # thousands of near-identical early game overs.
                                    terminals.sort(key=lambda n: n.final["canonical_score"], reverse=True)
                                    terminals = terminals[:8]
                                elif not final["truncated"]:
                                    children.append(node)
                if not children:
                    break
                if not complete_stage:
                    persist_frontier(output / "interrupted-stage",
                        select_frontier(children, args.beam_width, history), prefix, history, stats, baseline)
                    break
                frontier = select_frontier(children, args.beam_width, history)
                archive.extend(frontier)
                stats["stages"] += 1
                stats["deficit_survivors"] += sum(n.maximum_score_debt > 8000 for n in frontier)
                persist_frontier(output, frontier, prefix, history, stats, baseline)
                journal.flush()
                if time.monotonic()-progress_time > 20:
                    print(json.dumps(dict(progress=dict(stats), tick=frontier[0].final["tick"],
                        scores=[n.final["score"] for n in frontier],
                        retained_debt=max(n.maximum_score_debt for n in frontier))), flush=True)
                    progress_time = time.monotonic()
                last_stage_seconds = time.monotonic()-stage_started
        except StopRequested as error:
            if children:
                persist_frontier(output / "interrupted-stage",
                    select_frontier(children, args.beam_width, history), prefix, history, stats, baseline)
            stats["stop_reason"] = str(error)
            persist_frontier(output, frontier, prefix, history, stats, baseline)
    if env.state_hash() != source_hash:
        raise RuntimeError("growth search changed its source checkpoint")
    persist_frontier(output, frontier, prefix, history, stats, baseline)
    return frontier, archive, terminals, dict(stats), baseline


def persist_frontier(output, frontier, prefix, root_history, stats, baseline):
    rows = []
    for node in frontier:
        destination = output / "prefixes" / f"{node.identifier}.u32le"
        if not destination.exists():
            atomic_write(destination, trace_bytes(prefix + list(node.suffix)))
        rows.append({**node_report(node, root_history), "trace": str(destination)})
    write_json(output / "frontier.json", dict(nodes=rows, stats=dict(stats), baseline=baseline))


def finish_candidates(env, checkpoint, nodes, prefix, root_history, args, bundle, budget, output):
    results = []
    chosen = select_frontier(nodes, args.finish_candidates, root_history, preserve_reference=False)
    for index, node in enumerate(chosen):
        budget.check()
        directory = output / "continuations" / f"candidate-{index}-{node.identifier}"
        directory.mkdir(parents=True, exist_ok=True)
        full = prefix + list(node.suffix)
        with checkpoint.branch() as branch:
            if len(full) > len(prefix):
                obs, final, _ = advance(branch, ReplaySchedule(full), len(prefix), len(full), pipeline_size=16)
            else:
                obs = branch._native.observation()
                final = snapshot(obs)
            if final != node.final or obs != node.observation:
                raise RuntimeError("continuation prefix reconstruction failed")
            metadata = dict(version=VERSION, candidate=node.identifier,
                            source=str(args.source), seed=args.seed, parent=node_report(node, root_history))
            if args.finish_mode == "reference":
                source = list(args.source_actions)
                source[:len(full)] = full
                last, natural = evaluate(branch, ReplaySchedule(source), len(full), {}, args.max_ticks, pipeline_size=16)
                if natural:
                    save_best(directory, args.seed, source, {}, last, {**metadata, "natural": True})
                results.append(dict(candidate=node.identifier, final=last, natural=natural, mode="reference"))
            else:
                remaining = min(args.finish_seconds, budget.deadline-time.monotonic())
                if remaining <= 0:
                    raise StopRequested("wall_budget")
                local_budget = Budget(remaining)
                # Forward an already requested outer stop as well as its deadline.
                class CombinedBudget:
                    started = local_budget.started
                    def check(self):
                        budget.check()
                        local_budget.check()
                recorder = Recorder(branch, obs, full, CombinedBudget())
                writer = Artifacts(directory, args.seed, metadata, interval=30)
                recorder.progress = writer.checkpoint
                result = continue_mpc(recorder, args.seed, bundle,
                    RolloutConfig(mode="planner", cooldown_ticks=args.cooldown, probe_ticks=256,
                                  long_probe_ticks=512, top_k_pairs=2, max_ticks=args.max_ticks),
                    objective="score", packed=True)
                writer.finish(recorder, result)
                results.append({k: result[k] for k in ("final", "natural", "stop_reason", "stats")})
        write_json(output / "continuations.json", results)
        print(json.dumps(jsonable({"continuation": results[-1]})), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "worker", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--tick", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1298144938)
    parser.add_argument("--seconds", type=float, default=1200)
    parser.add_argument("--search-seconds", type=float, default=300)
    parser.add_argument("--horizon", type=int, default=1536)
    parser.add_argument("--segment", type=int, default=96)
    parser.add_argument("--beam-width", type=int, default=14)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--cooldown", type=int, default=16)
    parser.add_argument("--cashout-weight", type=float, default=1.0)
    parser.add_argument("--extension-weight", type=float, default=1.5)
    parser.add_argument("--opening-weight", type=float, default=1.25)
    parser.add_argument("--max-ticks", type=int, default=120000)
    parser.add_argument("--random-seed", type=int, default=290930)
    parser.add_argument("--finish-candidates", type=int, default=1)
    parser.add_argument("--finish-seconds", type=float, default=900)
    parser.add_argument("--finish-mode", choices=("mpc", "reference"), default="mpc")
    args = parser.parse_args()
    for key in ("seconds", "search_seconds", "finish_seconds"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be finite and positive")
    if min(args.horizon, args.segment, args.beam_width, args.candidates, args.cooldown) < 1:
        parser.error("search sizes must be positive")
    if not 0 <= args.seed < 2**32 or not 0 <= args.tick < args.max_ticks or args.finish_candidates < 0:
        parser.error("invalid seed, tick, or candidate count")
    if args.finish_candidates and args.finish_mode == "mpc" and not args.checkpoint:
        parser.error("MPC continuations require a checkpoint")
    torch.set_num_threads(1)
    args.source, args.worker, args.output = (p.resolve() for p in (args.source, args.worker, args.output))
    actions, source_sha = read_source(args.source, args.tick)
    plan = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    plan.update(version=VERSION, source_sha256=source_sha,
                controller_note="Every feedback branch starts a fresh policy; exact game prefixes are replayed.")
    plan["code_sha256"] = {name: hashlib.sha256((ROOT / "benchmarks" / name).read_bytes()).hexdigest()
        for name in (Path(__file__).name, "rl_combo_growth_proposals.py", "rl_combo_opportunity_value.py")}
    args.source_actions = actions
    bundle = load_policy_bundle(args.checkpoint, expected_sha256=args.checkpoint_sha256) if args.checkpoint else None
    budget = Budget(args.seconds)
    with output_lock(args.output), stop_signals(budget):
        if (args.output / "plan.json").exists():
            raise ValueError("output already used; resume a saved prefix into a new output directory")
        write_json(args.output / "plan.json", plan)
        for name in (Path(__file__).name, "rl_combo_growth_proposals.py", "rl_combo_opportunity_value.py"):
            atomic_write(args.output / "code" / name, (ROOT / "benchmarks" / name).read_bytes())
        with ExactTrainingRuntime(args.worker).open_env(simulation_config={"max_episode_ticks": args.max_ticks}) as session:
            env = session.environment
            env.reset(seed=args.seed)
            history = ComboHistory()
            observation, final = replay_events(env, actions, 0, args.tick, history, budget)
            if final["terminated"] or final["truncated"]:
                raise ValueError("source terminated before the requested root")
            write_json(args.output / "root.json", observation)
            write_json(args.output / "root-history.json", history.to_dict())
            result = dict(status="running", results=[])
            try:
                frontier, archive, terminals, stats, baseline = beam_search(
                    env, observation, history, actions, args, bundle, budget, args.output)
                for node in terminals:
                    directory = args.output / "terminals" / node.identifier
                    save_best(directory, args.seed, actions[:args.tick]+list(node.suffix), {}, node.final,
                              dict(version=VERSION, natural=True, candidate=node.identifier))
                result.update(stats=stats, baseline=baseline, terminal_count=len(terminals),
                              frontier=[node_report(n, history) for n in frontier])
                if args.finish_candidates and frontier:
                    with env.fast_checkpoint() as checkpoint:
                        result["results"] = finish_candidates(env, checkpoint, frontier,
                            actions[:args.tick], history, args, bundle, budget, args.output)
                result["status"] = "stopped" if stats.get("stop_reason") else "complete"
            except StopRequested as error:
                result.update(status="stopped", reason=str(error))
            result.update(seconds=time.monotonic()-budget.started, provenance=session.provenance_manifest)
            write_json(args.output / "result.json", result)
            print(json.dumps(jsonable({k: v for k, v in result.items() if k not in {"frontier", "provenance", "baseline"}})), flush=True)


if __name__ == "__main__":
    main()
