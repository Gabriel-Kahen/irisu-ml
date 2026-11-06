#!/usr/bin/env python3
"""Independent exact combo experiments: event beam or nested policy adaptation."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "benchmarks")]

import torch
import rl_combo_growth_search as growth
from irisu_rl.exact_training_runtime import ExactTrainingRuntime
from rl_combo_event_options import (EventOption, propose_options, propose_protection_options,
    propose_retirement_options, propose_preparation_options, execute_option, _step, _aim)
from rl_combo_growth_proposals import GrowthMacro
from rl_combo_diverse_value import evaluate_diverse
from rl_combo_mpc import Budget, StopRequested, jsonable, output_lock, read_source, stop_signals, write_json
from rl_combo_nrpa import Rollout, choose, search as nrpa_search
from rl_combo_opportunity_value import ComboHistory, OpportunityConfig, evaluate_opportunity, _health, _factor
from rl_combo_policy_screen import screen_candidates
from rl_combo_portfolio_finish import DeadlineBudget
from rl_combo_rollout import RolloutConfig, load_policy_bundle, run_feedback
from rl_combo_screened_growth import finish_candidates, CODE_FILES as FINISH_CODE_FILES
from rl_exact_suffix_beam import atomic_write, snapshot, trace_bytes
from rl_record_tail_search import save_best

VERSION = "combo-diverse-search-v1"
CODE_FILES = tuple(dict.fromkeys((Path(__file__).name, "rl_combo_nrpa.py", "rl_combo_event_options.py",
              "rl_combo_diverse_archive.py", "rl_combo_diverse_value.py", "rl_combo_diverse_objectives.py", "rl_combo_growth_search.py",
              "rl_combo_growth_proposals.py", "rl_combo_opportunity_value.py", *FINISH_CODE_FILES)))


def cpu_usage():
    return sum(resource.getrusage(w).ru_utime + resource.getrusage(w).ru_stime
               for w in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))


def value_config(args):
    return OpportunityConfig(extension_weight=args.extension_weight,
                             cashout_weight=getattr(args,"cashout_weight",1),
                             opening_weight=getattr(args,"opening_weight",1.25),
                             stale_half_life_ticks=args.stale_half_life,
                             elapsed_tick_cost=args.tick_cost)


def evaluate_value(observation,history,args,*,elapsed_ticks=0):
    return evaluate_diverse(observation,history,value_config(args),elapsed_ticks=elapsed_ticks)


def candidate_options(env,observation,history,args,duration,rng,budget):
    copay_cap=min(getattr(args,"co_clear_options",getattr(args,"cashout_options",0)),max(0,args.candidates-5))
    copay=propose_co_clear_options(observation,history,duration,copay_cap)
    prep_cap=min(getattr(args,"preparation_options",0),max(0,args.candidates-5-len(copay)))
    holds=propose_feeder_holds(observation,history,duration,prep_cap//3)
    prepared=propose_preparation_options(observation,history,duration=duration,cap=prep_cap-len(holds),rng=rng) if prep_cap else []
    prepared+=holds
    relocation_cap=min(getattr(args,'relocation_options',0),max(0,args.candidates-5-len(copay)-len(prepared)))
    relocation=propose_carrier_relocations(observation,history,duration,relocation_cap)
    retirement_cap=min(getattr(args,"retirement_options",0),max(0,args.candidates-5-len(copay)-len(prepared)-len(relocation)))
    eligible=any(b.get("kind")=="piece" and b.get("chain_id") and b.get("projectile_hits")==1
                 and float(b.get("size",0))<140 and _health(b)>0 for b in observation.get("bodies",()))
    retirement=propose_retirement_options(env,observation,history,duration=duration,
        cap=retirement_cap,budget=budget) if retirement_cap and eligible else []
    cap = min(getattr(args,"causal_options",8),max(0,args.candidates-5-len(retirement)-len(copay)-len(prepared)-len(relocation)))
    protection = propose_protection_options(env,observation,history,duration=duration,
        cap=cap,budget=budget) if cap else []
    ordinary = propose_options(observation,history,duration=duration,
        cap=args.candidates-len(protection)-len(retirement)-len(copay)-len(prepared)-len(relocation),rng=rng)
    return copay+prepared+relocation+retirement+protection+ordinary


def propose_carrier_relocations(observation,history,duration=32,cap=8):
    """Reserve first-hit relocation geometry for valuable unhit carriers."""
    if cap<=0:
        return []
    carriers=[b for b in observation.get('bodies',()) if b.get('kind')=='piece'
              and b.get('chain_id') and _health(b)>0]
    candidates=[]
    for body in carriers:
        if body.get('projectile_hits') or body.get('age_ticks',0)<=100 or _health(body)<.25:
            continue
        row=history.groups.get(int(body['chain_id']),{})
        product=row.get('cumulative_members',0)*(row.get('successful_clears',0)+1)
        if product>=12:
            candidates.append((-product,-float(body.get('vy',0)),body))
    candidates.sort(key=lambda p:(p[0],p[1],p[2]['id']))
    result,words=[],set()
    for angle in (-30,30,-60,60,0,-90,90,180):
        for radius in (.45,.15,.75,.6,.9):
            for strength in (1,2):
                for _,_,source in candidates[:3]:
                    label=f'relocate-s{source["id"]}-r{radius}-a{angle}-k{strength}'
                    option=EventOption(label,'reposition',duration,'reposition',source_id=int(source['id']),
                        group_id=int(source['chain_id']),radius=radius,angle=angle,strength=strength,max_shots=1)
                    target=dict(x=source['x'],y=560)
                    word=_aim(source,target,option,carriers,protect=True,direct=True)
                    if word is not None and word not in words:
                        result.append(option)
                        words.add(word)
                        if len(result)>=cap:
                            return result
                        # A one-tick delay can change contact order. Preserve
                        # the original aim as a separate measured hypothesis.
                        if strength==1 and duration>1:
                            delayed=label+'-delay1'
                            result.append(EventOption(delayed,'reposition',duration,
                                macro=GrowthMacro(delayed,'protect',((1,word),),duration),
                                source_id=int(source['id']),group_id=int(source['chain_id']),
                                radius=radius,angle=angle,strength=strength))
                            if len(result)>=cap:
                                return result
    return result


def propose_feeder_holds(observation,history,duration=32,cap=4):
    """Delay a fresh feeder's join so it can first become a shared payout source."""
    if cap<=0 or int(observation.get('qualifying_clear_count',0))<980:
        return []
    carriers=[b for b in observation.get('bodies',()) if b.get('kind')=='piece'
              and b.get('chain_id') and _health(b)>=.25]
    pairs=[]
    for source in observation.get('bodies',()):
        if (source.get('kind')!='piece' or source.get('chain_id') or source.get('rot_timer')
                or source.get('lifecycle')!='dynamic_fresh' or source.get('age_ticks',0)<=100):
            continue
        for target in carriers:
            distance=math.dist((source['x'],source['y']),(target['x'],target['y']))
            if source['color']!=target['color'] or distance>=200:
                continue
            row=history.groups.get(int(target['chain_id']),{})
            product=row.get('cumulative_members',0)*(row.get('successful_clears',0)+1)
            if product>=12:
                pairs.append((-product,distance,source,target))
    pairs.sort(key=lambda p:(p[0],p[1],p[2]['id'],p[3]['id']))
    result,words=[],set()
    for radius in (.6,1.15,.35):
        for angle in (0,-30,30,-60,60,90):
            for _,_,source,target in pairs[:4]:
                option=EventOption('hold-feeder','prepare',duration,'paired',source_id=int(source['id']),
                    target_id=int(target['id']),group_id=int(target['chain_id']),radius=radius,angle=angle)
                word=_aim(source,target,option,carriers,protect=True)
                offset=int(bool(observation.get('right_held')))
                if word is None or word in words or offset>=duration:
                    continue
                label=f'hold-feeder-s{source["id"]}-t{target["id"]}-r{radius}-a{angle}'
                result.append(EventOption(label,'prepare',duration,
                    macro=GrowthMacro(label,'protect',((offset,word),),duration)))
                words.add(word)
                if len(result)>=cap:
                    return result
    return result


def propose_co_clear_options(observation,history,duration=32,cap=8):
    """Reserve shots for visible, ungrouped, rot-running matching feeders."""
    if cap<=0:
        return []
    carriers=[b for b in observation.get("bodies",()) if b.get("kind")=="piece" and b.get("chain_id") and _health(b)>0]
    pairs=[]
    for feeder in observation.get("bodies",()):
        if feeder.get("kind")!='piece' or feeder.get("chain_id") or not feeder.get("rot_timer") or _health(feeder)<=0:
            continue
        for target in carriers:
            if feeder['color']==target['color']:
                past=history.groups.get(int(target['chain_id']),{})
                product=past.get('cumulative_members',0)*(past.get('successful_clears',0)+1)
                distance=math.dist((feeder['x'],feeder['y']),(target['x'],target['y']))
                pairs.append((product,distance,feeder,target))
    pairs.sort(key=lambda p:(-p[0],p[1],p[2]['id'],p[3]['id']))
    if pairs:
        pairs=[p for p in pairs if p[0]>=max(12,.25*pairs[0][0])]
    result,words=[],set()
    for radius in (.6,1.15,.35,.9):
        for angle in (0,-30,30,-60,60,-90,90,180):
            for _,_,source,target in pairs[:4]:
                for strength in (2,1):
                    template=EventOption('co-pay','co_clear',duration,'paired',source_id=int(source['id']),
                        target_id=int(target['id']),group_id=int(target['chain_id']),radius=radius,
                        angle=angle,strength=strength)
                    word=_aim(source,target,template,carriers)
                    offset=int(bool(observation.get('left_held' if strength==1 else 'right_held')))
                    if word is None or word in words or offset>=duration:
                        continue
                    label=f'co-pay-s{source["id"]}-t{target["id"]}-r{radius}-a{angle}-k{strength}'
                    macro=GrowthMacro(label,'cashout',((offset,word),),duration)
                    result.append(EventOption(label,'co_clear',duration,macro=macro))
                    words.add(word)
                    if len(result)>=cap:
                        return result
    return result


def execute_candidate(env,observation,history,option,args,budget,limit):
    outcome=execute_option(env,observation,history,option,budget,limit)
    # Optional fixed decision times compare equally old endpoints. Short waits
    # otherwise compete with full shot intervals while retaining more gauge.
    if (not getattr(args,"fixed_option_interval",False) or not outcome["actions"]
            or option.mode=='prepare_feeder'):
        return outcome
    end=min(int(observation["tick"])+option.duration,limit)
    tick=int(outcome["final"]["tick"])
    if tick < end and not (outcome["final"]["terminated"] or outcome["final"]["truncated"]):
        budget.check()
        obs,final,events=_step(env,0,end-tick)
        outcome["history"].update(events)
        padding=int(final["tick"])-tick
        outcome.update(actions=tuple(outcome["actions"])+(0,)*padding,observation=obs,final=final,
                       natural=bool(final["terminated"] and not final["truncated"]))
        outcome["diagnostics"].update(neutral_padding_ticks=padding,option_end_tick=tick,
                                      end_tick=int(final["tick"]))
        outcome["diagnostics"]["events"].extend(events)
    return outcome


def initial_node(observation, history, args):
    return growth.Node((), observation, snapshot(observation), history,
                       evaluate_value(observation, history, args), "root", ())


def retain_terminal(node, terminals, prefix, args, output):
    """Persist a measured best before another branch or shutdown can fail."""
    if not node.natural:
        raise ValueError("only canonical completed games are terminal results")
    if node.final["canonical_score"] > max((n.final["canonical_score"] for n in terminals), default=-1):
        save_best(output/"search-best", args.seed, prefix+list(node.suffix), {}, node.final,
                  dict(version=VERSION, natural=True, candidate=node.identifier))
    terminals.append(node)
    terminals.sort(key=lambda n:n.final["canonical_score"], reverse=True)
    del terminals[8:]


def finishing_nodes(nodes,args):
    if args.algorithm == "nrpa" and getattr(args,"nrpa_objective","score") == "asset":
        measured=[n for n in nodes if getattr(n,"asset_fitness",None) is not None]
        if measured:
            return [max(measured,key=lambda n:n.asset_fitness)]
    return nodes


def accept_route(delta,temperature,rng):
    return delta >= 0 or (temperature > 0 and rng.random() < math.exp(delta/temperature))


def source_revert(actions,current,tick,duration):
    original=actions[tick:tick+duration]
    original += [0]*max(0,duration-len(original))
    if original == current[tick:tick+duration]:
        return None
    shots=tuple((i,w) for i,w in enumerate(original) if w)
    try:
        macro=GrowthMacro('source-revert','protect' if shots else 'wait',shots,duration)
    except ValueError:
        # Historical held-button inputs are not legal new-shot macro grammars.
        return None
    return EventOption(macro.label,'reference',duration,macro=macro)


def persistent_asset_key(node):
    """A protected search niche, not a promised score from unseen feeders."""
    available = max(0, 989-int(node.observation.get("qualifying_clear_count", 0)))
    asset = 0.0
    for gid, row in node.history.groups.items():
        carriers = [b for b in node.observation.get("bodies", ()) if b.get("kind")=="piece"
                    and int(b.get("chain_id",0))==gid and _health(b)>0]
        if carriers:
            asset = max(asset, row["cumulative_members"]*(row["successful_clears"]+1)
                        *max(_health(c)*_factor(c) for c in carriers)*math.sqrt(available)
                        *(1+.15*min(2,len(carriers)-1)))
    return asset, int(node.observation["gauge"]), node.final["score"]


def select_diverse(nodes, width, history):
    """Protect reference, remaining clear budgets and distinct relay geometry."""
    from rl_combo_diverse_archive import describe_state
    viable = [n for n in nodes if not (n.final["terminated"] or n.final["truncated"])]
    chosen = growth.select_frontier(viable, max(1, width // 4), history)
    if viable and width > 1:
        persistent = max(viable, key=persistent_asset_key)
        if persistent_asset_key(persistent)[0] > 0 and all(persistent.identifier != n.identifier for n in chosen):
            chosen.append(persistent)
    def cell_for(node):
        if not hasattr(node, "behavior_cell"):
            description = describe_state(node.observation, node.history)
            node.behavior_cell = json.dumps([description["descriptor"],
                description["features"]["structure"], description["features"]["inventory"]], sort_keys=True)
        return node.behavior_cell
    seen, cells = {n.identifier for n in chosen}, {cell_for(n) for n in chosen}
    # A higher bank often represents already-spent qualifications. Compare
    # patient branches within their remaining budget before global value fills.
    phases = {}
    for node in viable:
        phases.setdefault(int(node.observation.get("qualifying_clear_count",0)),[]).append(node)
    phase_pools = [sorted(rows,key=lambda n:(persistent_asset_key(n),n.value.objective),reverse=True)
                   for _,rows in sorted(phases.items())]
    reserve_limit = min(width-1,max(len(chosen),3*width//4))
    for rank in range(max(map(len,phase_pools),default=0)):
        for pool in phase_pools:
            if len(chosen) >= reserve_limit:
                break
            if rank >= len(pool):
                continue
            node = pool[rank]
            if node.identifier not in seen and cell_for(node) not in cells:
                chosen.append(node)
                seen.add(node.identifier)
                cells.add(cell_for(node))
        if len(chosen) >= reserve_limit:
            break
    for node in sorted(viable, key=lambda n: n.value.objective, reverse=True):
        cell = cell_for(node)
        if node.identifier not in seen and cell not in cells:
            chosen.append(node)
            seen.add(node.identifier)
        cells.add(cell)
        if len(chosen) >= width:
            return chosen[:width]
    for node in growth.select_frontier(viable, width, history):
        if node.identifier not in seen:
            chosen.append(node)
            seen.add(node.identifier)
        if len(chosen) == width:
            break
    return chosen


def archive_frontier(nodes, width, history):
    # These states cannot earn another scored qualification. Completed games
    # are retained separately; scarce exploration slots need scoring capacity.
    viable = [n for n in nodes if int(n.observation.get("qualifying_clear_count",0)) < 989]
    selected = select_diverse(viable,width,history)
    protected = [n for n in viable if getattr(n,"asset_fitness",None) is not None]
    if protected:
        best = max(protected,key=lambda n:n.asset_fitness)
        if all(n.identifier != best.identifier for n in selected):
            selected = selected[:max(0,width-1)]+[best]
    return selected


def finish_exhausted(env, full, observation, final, history, max_ticks, budget):
    """Measure an exhausted ending cheaply instead of expanding useless shots."""
    if (int(observation.get("qualifying_clear_count",0)) < 989 or
            final["terminated"] or final["truncated"]):
        return full,observation,final,False
    full = list(full[:int(final["tick"])])+[0]*max(0,max_ticks-int(final["tick"]))
    observation,final = growth.replay_events(env,full,int(final["tick"]),max_ticks,history,budget)
    return full[:int(final["tick"])],observation,final,True


class AnchorHistory(ComboHistory):
    """Locate decisions around observed joins and impending carrier clears."""
    def __init__(self, history):
        self.groups, self.memberships = history.groups.copy(), history.memberships.copy()
        self.anchors = set()

    def update(self, events):
        for event in events:
            tick, kind = int(event["tick"]), event["kind"]
            gid = int(event["value"]) if kind == "chain_joined" else self.memberships.get(int(event.get("a",0)),0)
            if self.groups.get(gid, {}).get("cumulative_members", 0) >= 8:
                if kind == "chain_joined":
                    self.anchors.update((tick+1, tick+5))
                elif kind == "confirmed":
                    self.anchors.update((tick-48, tick-24))
            super().update((event,))


def sparse_search(env, observation, history, actions, args, bundle, budget, output):
    """Search one event option at a time, scoring the entire inherited ending.

    Successful complete routes become the next reference. Intermediate states
    survive independently, so a locally costly relay can seed a later search.
    Every comparison used for reference replacement is a canonical game score.
    """
    root, prefix = int(observation["tick"]), actions[:int(observation["tick"])]
    limit, root_hash = min(args.max_ticks, root+args.horizon), env.state_hash()
    rng, current = random.Random(args.random_seed), list(actions)
    archived, terminals, stats = [], [], Counter(branches=0, anchors=0, improvements=0)
    local = DeadlineBudget(budget, time.monotonic()+args.search_seconds, reason="search_budget")
    best_score = -1
    best_actions=list(actions)
    current_score=-1
    walk=getattr(args,"sparse_walk",False) or getattr(args,"sparse_temperature",0)>0
    restart_interval=getattr(args,"sparse_restart_anchors",64)
    next_restart=restart_interval
    def make_node(full, obs, final, ledger, label):
        return growth.Node(tuple(full[root:int(final["tick"])]), obs, dict(final), ledger.copy(),
            evaluate_value(obs, ledger, args, elapsed_ticks=int(final["tick"])-root),
            "sparse", (label,))
    with env.fast_checkpoint() as checkpoint, (output/"candidates.jsonl").open("w") as journal:
        try:
            with checkpoint.branch() as branch:
                full = current+[0]*max(0,args.max_ticks-len(current))
                ledger = history.copy()
                obs, final = growth.replay_events(branch, full, root, args.max_ticks, ledger, local)
                node = make_node(full,obs,final,ledger,"reference")
                if node.natural:
                    best_score = final["canonical_score"]
                    current_score=best_score
                    best_actions=full[:int(final["tick"])]
                    retain_terminal(node,terminals,prefix,args,output)
            while True:
                local.check()
                ledger = AnchorHistory(history)
                with checkpoint.branch() as scan:
                    growth.replay_events(scan,current+[0]*max(0,limit-len(current)),root,limit,ledger,local)
                anchors = sorted({root,*range(root,limit,max(16,args.segment)),
                                  *(t for t in ledger.anchors if root <= t < limit)})
                rng.shuffle(anchors)
                changed = False
                for tick in anchors:
                    local.check()
                    if walk and stats["anchors"] >= next_restart:
                        current=list(best_actions)
                        current_score=best_score
                        stats["route_restarts"]+=1
                        next_restart=stats["anchors"]+restart_interval
                    with checkpoint.branch() as branch:
                        parent_history = history.copy()
                        parent_obs,parent_final = growth.replay_events(branch,current+[0]*max(0,tick-len(current)),
                            root,tick,parent_history,local)
                        if parent_final["terminated"] or parent_final["truncated"]:
                            continue
                        if int(parent_obs.get("qualifying_clear_count",0)) >= 989:
                            continue
                        stats["anchors"] += 1
                        with branch.fast_checkpoint() as parent:
                            options = candidate_options(branch,parent_obs,parent_history,args,
                                min(args.segment,limit-tick),rng,local)
                            if walk:
                                duration=min(args.segment,limit-tick)
                                revert=source_revert(actions,current,tick,duration)
                                if revert is not None:
                                    options.append(revert)
                            rng.shuffle(options)
                            # Every option at this anchor sees the same parent.
                            candidate_reference = list(current)
                            for option in options:
                                local.check()
                                with parent.branch() as child:
                                    result = execute_candidate(child,parent_obs,parent_history,option,args,local,limit)
                                    if not result["actions"]:
                                        stats["rejected_options"] += 1
                                        continue
                                    full = candidate_reference[:tick]+list(result["actions"])
                                    obs,final,ledger = (result[k] for k in ("observation","final","history"))
                                    endpoint = make_node(full,obs,final,ledger,option.label)
                                    if not (final["terminated"] or final["truncated"]):
                                        archived.append(endpoint)
                                        full += candidate_reference[len(full):]
                                        full.extend([0]*max(0,args.max_ticks-len(full)))
                                        obs,final = growth.replay_events(child,full,int(final["tick"]),
                                            args.max_ticks,ledger,local)
                                    terminal = make_node(full,obs,final,ledger,option.label)
                                    stats["branches"] += 1
                                    if terminal.natural:
                                        score = int(final["canonical_score"])
                                        retain_terminal(terminal,terminals,prefix,args,output)
                                        if score > best_score:
                                            gain, best_score = score-best_score,score
                                            current = full[:int(final["tick"])]
                                            current_score=score
                                            best_actions=list(current)
                                            changed = True
                                            stats["improvements"] += 1
                                            save_best(output/"sparse-best",args.seed,current,{},final,
                                                dict(version=VERSION,natural=True,anchor=tick,option=option.label))
                                            print(json.dumps(dict(sparse_gain=gain,score=score,anchor=tick,
                                                option=option.label)),flush=True)
                                        elif walk:
                                            fraction=max(.05,min(1,(local.deadline-time.monotonic())/args.search_seconds))
                                            temperature=getattr(args,"sparse_temperature",0)*fraction
                                            if accept_route(score-current_score,temperature,rng):
                                                current=full[:int(final["tick"])]
                                                stats["downhill_acceptances" if score<current_score else "plateau_acceptances"]+=1
                                                current_score=score
                                                changed=True
                                    journal.write(json.dumps(dict(tick=tick,option=option.label,
                                        natural=terminal.natural,score=final.get("canonical_score"),
                                        endpoint=endpoint.final,current_score=current_score))+"\n")
                                    if stats["branches"] % 32 == 0:
                                        archived[:] = archive_frontier(archived,max(32,args.beam_width*4),history)
                                        terminals.sort(key=lambda n:n.final["canonical_score"],reverse=True)
                                        terminals[:] = terminals[:8]
                                        growth.persist_frontier(output,archived,prefix,history,stats,{})
                                        journal.flush()
                stats["sweeps"] += 1
                if not changed:
                    # Another pass jitters aim geometry and option ordering.
                    stats["flat_sweeps"] += 1
        except StopRequested as stopped:
            stats["stop_reason"] = str(stopped)
    if env.state_hash() != root_hash:
        raise RuntimeError("sparse search mutated root")
    archived = archive_frontier(archived,max(32,args.beam_width*4),history)
    frontier = select_diverse(archived,args.beam_width,history)
    terminals.sort(key=lambda n:n.final["canonical_score"],reverse=True)
    growth.persist_frontier(output,frontier,prefix,history,stats,{})
    return frontier,archived,terminals[:8],dict(stats),{}


def event_search(env, observation, history, actions, args, bundle, budget, output):
    root, rng = int(observation["tick"]), random.Random(args.random_seed)
    prefix, initial = actions[:int(observation["tick"])], initial_node(observation, history, args)
    limit, source_hash = min(args.max_ticks, root + args.horizon), env.state_hash()
    frontier, archived, terminals, finished = [initial], [], [], []
    stats = Counter(branches=0, stages=0, event_branches=0, deficit_survivors=0)
    local = DeadlineBudget(budget, time.monotonic() + args.search_seconds, reason="search_budget")
    with env.fast_checkpoint() as checkpoint, (output / "candidates.jsonl").open("w") as journal:
        try:
            while frontier:
                local.check()
                children = []
                for parent in frontier:
                    local.check()
                    start = int(parent.final["tick"])
                    if start >= limit:
                        finished.append(parent)
                        continue
                    with checkpoint.branch() as branch:
                        ledger = history.copy()
                        restored, final = growth.replay_events(branch, prefix + list(parent.suffix),
                            root, start, ledger, local)
                        if final != parent.final or ledger.to_dict() != parent.history.to_dict():
                            raise RuntimeError("event beam prefix failed exact reconstruction")
                        with branch.fast_checkpoint() as parent_checkpoint:
                            options = candidate_options(branch,parent.observation,parent.history,args,
                                min(args.segment,limit-start),rng,local)
                            for option in [None, *options]:
                                local.check()
                                with parent_checkpoint.branch() as child:
                                    if option is None:
                                        end = min(start+args.segment, limit)
                                        extension = actions[start:end] + [0]*max(0, end-len(actions))
                                        full = prefix + list(parent.suffix) + extension
                                        child_history = parent.history.copy()
                                        obs, last = growth.replay_events(child, full, start, end,
                                                                        child_history, local)
                                        suffix = tuple(full[root:int(last["tick"])])
                                        family, label = "reference", "reference"
                                    else:
                                        outcome = execute_candidate(child,parent.observation,parent.history,
                                                                    option,args,local,limit)
                                        obs, last, child_history = (outcome[k] for k in ("observation", "final", "history"))
                                        if not outcome["actions"]:
                                            stats["rejected_options"] += 1
                                            continue
                                        suffix = parent.suffix + tuple(outcome["actions"])
                                        family, label = option.family, option.label
                                        stats["event_branches"] += 1
                                    if int(last["tick"]) <= start:
                                        raise RuntimeError("option made no progress")
                                    full,obs,last,exhausted = finish_exhausted(child,prefix+list(suffix),
                                        obs,last,child_history,args.max_ticks,local)
                                    if exhausted:
                                        suffix = tuple(full[root:])
                                        stats["exhausted_finishes"] += 1
                                    value = evaluate_value(obs, child_history, args,
                                                                 elapsed_ticks=int(last["tick"])-root)
                                    node = growth.Node(suffix, obs, last, child_history, value, family,
                                                       parent.labels+(label,), parent.maximum_score_debt)
                                    stats["branches"] += 1
                                    journal.write(json.dumps(jsonable(growth.node_report(node, history)))+"\n")
                                    if node.natural:
                                        retain_terminal(node,terminals,prefix,args,output)
                                    elif not last["truncated"]:
                                        children.append(node)
                if not children:
                    break
                frontier = select_diverse(children, args.beam_width, history)
                archived.extend(frontier)
                stats["stages"] += 1
                growth.persist_frontier(output, frontier, prefix, history, stats, {})
                journal.flush()
                print(json.dumps(dict(progress=dict(stats), ticks=[n.final["tick"] for n in frontier],
                                      best=max(n.final["score"] for n in frontier))), flush=True)
        except StopRequested as stopped:
            stats["stop_reason"] = str(stopped)
            if "children" in locals() and children:
                archived.extend(select_diverse(children, args.beam_width, history))
    if env.state_hash() != source_hash:
        raise RuntimeError("event search mutated root")
    candidates = select_diverse(finished + frontier + archived[-4*args.beam_width:],
                                args.beam_width, history)
    growth.persist_frontier(output, candidates, prefix, history, stats, {})
    return candidates, archived, terminals, dict(stats), {}


def nrpa(env, observation, history, actions, args, bundle, budget, output):
    root, prefix = int(observation["tick"]), actions[:int(observation["tick"])]
    limit, source_hash = min(args.max_ticks, root+args.horizon), env.state_hash()
    local = DeadlineBudget(budget, time.monotonic()+args.search_seconds, reason="search_budget")
    archived, terminals, stats = [], [], Counter(branches=0, policy_branches=0)
    best_endpoint = [None]
    objective = getattr(args,"nrpa_objective","score")
    def endpoint_fitness(node):
        if objective == "asset":
            from rl_combo_diverse_objectives import AssetConfig, asset_fitness
            config=AssetConfig(wasted_qualification_penalty=getattr(args,"wasted_qualification_penalty",0),
                               size_weight=getattr(args,"asset_size_weight",0),
                               gauge_window=getattr(args,"asset_gauge_window",2000))
            return asset_fitness(node.observation,node.history,observation,history,config=config)
        if objective == "opportunity":
            return node.value.objective
        return node.final["canonical_score"] if node.natural else -1.0
    with env.fast_checkpoint() as checkpoint:
        def rollout(policy, rng, temperature, force_reference=False):
            ledger, suffix, decisions, labels = history.copy(), [], [], []
            archived_tick = root
            with checkpoint.branch() as branch:
                obs, final = observation, snapshot(observation)
                while int(final["tick"]) < limit and not (final["terminated"] or final["truncated"]):
                    local.check()
                    start = int(final["tick"])
                    duration = min(args.segment, limit-start)
                    options = candidate_options(branch,obs,ledger,args,duration,rng,local)
                    # Every decision has its own learned action weights. Short
                    # event waits must not alias several choices in one segment.
                    context = f"{len(decisions)}:{int(obs.get('qualifying_clear_count', 0))//10}"
                    legal = [(context+":reference", args.reference_bias)]
                    legal.extend((context+":"+option.label,
                        {"extend": 1.0, "relay": 1.2, "co_clear":1.5,"wait": .2, "protect": .4}.get(option.family, .1))
                        for option in options)
                    index, decision = choose(policy, legal, rng, temperature)
                    if force_reference:
                        from rl_combo_nrpa import Decision
                        index, decision = 0, Decision(legal[0][0], tuple(legal))
                    decisions.append(decision)
                    if index == 0:
                        end = start+duration
                        extension = actions[start:end] + [0]*max(0, end-len(actions))
                        full = prefix+suffix+extension
                        obs, final = growth.replay_events(branch, full, start, end, ledger, local)
                        suffix = full[root:int(final["tick"])]
                        labels.append("reference")
                    else:
                        option = options[index-1]
                        outcome = execute_candidate(branch,obs,ledger,option,args,local,limit)
                        if not outcome["actions"]:
                            # A rejected geometry is an uncommitted intention.
                            # Make one neutral observation step and record it.
                            full = prefix+suffix+[0]*min(4, limit-start)
                            obs, final = growth.replay_events(branch, full, start, len(full), ledger, local)
                            suffix = full[root:int(final["tick"])]
                            labels.append("rejected:"+option.label)
                            stats["rejected_options"] += 1
                            continue
                        suffix.extend(outcome["actions"])
                        obs, final, ledger = (outcome[k] for k in ("observation", "final", "history"))
                        labels.append(option.label)
                    if int(final["tick"]) <= start:
                        raise RuntimeError("NRPA option made no progress")
                    if (int(final["tick"])-archived_tick >= 256
                            and not (final["terminated"] or final["truncated"])):
                        intermediate_value = evaluate_value(obs, ledger, args,
                            elapsed_ticks=int(final["tick"])-root)
                        archived.append(growth.Node(tuple(suffix), obs, dict(final), ledger.copy(),
                            intermediate_value, "nrpa-intermediate", tuple(labels)))
                        archived_tick = int(final["tick"])
                    if len(decisions) >= args.max_decisions:
                        break
                if int(final["tick"]) < limit and not (final["terminated"] or final["truncated"]):
                    start = int(final["tick"])
                    full = prefix+suffix+actions[start:limit]+[0]*max(0,limit-max(start,len(actions)))
                    obs,final = growth.replay_events(branch,full,start,limit,ledger,local)
                    suffix = full[root:int(final["tick"])]
                    labels.append("decision-cap-reference")
                    stats["decision_cap_continuations"] += 1
                value = evaluate_value(obs, ledger, args,
                                             elapsed_ticks=int(final["tick"])-root)
                node = growth.Node(tuple(suffix), obs, final, ledger, value, "nrpa", tuple(labels))
                # Keep live endpoints before a cheap terminal assessment changes them.
                if not (final["terminated"] or final["truncated"]):
                    archived.append(node)
                fitness = endpoint_fitness(node)
                if objective == "asset" and not (final["terminated"] or final["truncated"]):
                    node.asset_fitness = fitness
                    if best_endpoint[0] is None or fitness > best_endpoint[0].asset_fitness:
                        best_endpoint[0] = node
                if node.natural:
                    retain_terminal(node,terminals,prefix,args,output)
                elif args.nrpa_finish != "none":
                    with branch.fast_checkpoint() as endpoint, endpoint.branch() as tail:
                        full = prefix+suffix
                        tail_history = ledger.copy()
                        tail_budget = DeadlineBudget(local, time.monotonic()+args.assessment_seconds,
                                                     reason="assessment_budget")
                        if args.nrpa_finish == "reference":
                            continuation = full+actions[len(full):]
                            continuation.extend([0]*max(0, args.max_ticks-len(continuation)))
                            try:
                                tail_obs, tail_final = growth.replay_events(tail, continuation, len(full),
                                    args.max_ticks, tail_history, tail_budget)
                            except StopRequested:
                                local.check()
                                stats["assessment_timeouts"] += 1
                                return Rollout(float(fitness), tuple(decisions), node.identifier, node)
                            tail_actions = continuation[:int(tail_final["tick"])]
                        else:
                            recorder = growth.HistoryRecorder(tail, obs, full, tail_budget, history=tail_history)
                            try:
                                outcome = run_feedback(recorder, obs, args.seed, bundle,
                                    RolloutConfig(mode="policy", cooldown_ticks=args.cooldown, max_ticks=args.max_ticks), prefix=full)
                            except StopRequested:
                                local.check()
                                outcome = dict(actions=recorder.actions, observation=recorder.observation, final=recorder.final)
                            tail_obs, tail_final, tail_actions = (outcome[k] for k in ("observation", "final", "actions"))
                            stats["policy_branches"] += 1
                        if tail_final["terminated"] and not tail_final["truncated"]:
                            terminal = growth.Node(tuple(tail_actions[root:]), tail_obs, tail_final,
                                tail_history, evaluate_value(tail_obs, tail_history, args),
                                "nrpa-assessment", tuple(labels))
                            retain_terminal(terminal,terminals,prefix,args,output)
                            if objective == "score":
                                fitness = tail_final["canonical_score"]
                stats["branches"] += 1
                archived[:] = archive_frontier(archived, max(32, args.beam_width*4), history)
                if best_endpoint[0] is not None and all(n.identifier != best_endpoint[0].identifier for n in archived):
                    archived.append(best_endpoint[0])
                terminals.sort(key=lambda n: n.final["canonical_score"], reverse=True)
                terminals[:] = terminals[:8]
                if stats["branches"] % 8 == 0:
                    growth.persist_frontier(output, archived, prefix, history, stats, {})
                    print(json.dumps(dict(progress=dict(stats), best_terminal=max(
                        (n.final["canonical_score"] for n in terminals), default=0))), flush=True)
                return Rollout(float(fitness), tuple(decisions), node.identifier, node)
        try:
            warm_start = rollout({}, random.Random(args.random_seed), args.temperature, True)
        except StopRequested:
            warm_start = None
        # Limited repetitions ends one concentrated search, then another seed
        # explores under the same remaining allowance. All outcomes survive.
        for restart in range(args.nrpa_restarts):
            result = nrpa_search(rollout, seed=args.random_seed+104729*restart, levels=args.nrpa_levels,
                iterations=args.nrpa_iterations, repetitions=args.nrpa_repetitions,
                learning_rate=args.nrpa_rate, temperature=args.temperature*(1+.25*(restart%3)), budget=local,
                stop_exception=StopRequested, initial_best=warm_start)
            for key in ("rollouts", "adaptations", "repetition_stops", "distinct_sequences"):
                stats[key] += getattr(result, key)
            stats["nrpa_restarts"] += 1
            if result.best is not None and (warm_start is None or result.best.score > warm_start.score):
                warm_start = result.best
            if result.stop_reason:
                stats["stop_reason"] = result.stop_reason
                break
    if env.state_hash() != source_hash:
        raise RuntimeError("NRPA changed original root")
    if best_endpoint[0] is not None:
        archived.append(best_endpoint[0])
        stats["asset_best_fitness"] = best_endpoint[0].asset_fitness
    frontier = archive_frontier(archived, args.beam_width, history)
    growth.persist_frontier(output, frontier, prefix, history, stats, {})
    return frontier, archived, terminals, dict(stats), {}


def run(args):
    torch.set_num_threads(1)
    os.nice(5)
    actions, source_sha = read_source(args.source, args.tick)
    args.source_actions = actions
    bundle = load_policy_bundle(args.checkpoint, expected_sha256=args.checkpoint_sha256) if args.checkpoint else None
    output, budget, before = args.output, Budget(args.seconds), cpu_usage()
    result = dict(version=VERSION, status="running", results=[], algorithm=args.algorithm)
    with output_lock(output), stop_signals(budget):
        if (output / "plan.json").exists():
            raise ValueError("output already used")
        write_json(output / "plan.json", {**{k: v for k,v in vars(args).items() if k != "source_actions"},
            "source_sha256": source_sha, "version": VERSION,
            "code_sha256": {name: hashlib.sha256((ROOT/"benchmarks"/name).read_bytes()).hexdigest() for name in CODE_FILES}})
        for name in CODE_FILES:
            atomic_write(output / "code" / name, (ROOT/"benchmarks"/name).read_bytes())
        try:
            with ExactTrainingRuntime(args.worker).open_env(simulation_config={"max_episode_ticks": args.max_ticks}) as session:
                env = session.environment
                env.reset(seed=args.seed)
                history = ComboHistory()
                observation, final = growth.replay_events(env, actions, 0, args.tick, history, budget)
                if final["tick"] != args.tick or final["terminated"] or final["truncated"]:
                    raise ValueError("root is not live at requested tick")
                write_json(output / "root.json", observation)
                write_json(output / "root-history.json", history.to_dict())
                engine = {"baseline": growth.beam_search, "event": event_search,
                          "nrpa": nrpa, "sparse": sparse_search}[args.algorithm]
                frontier, archive, terminals, stats, baseline = engine(env, observation, history,
                    actions, args, bundle, budget, output)
                result.update(stats=stats, terminal_count=len(terminals))
                for node in terminals:
                    save_best(output/"terminals"/node.identifier, args.seed, actions[:args.tick]+list(node.suffix),
                              {}, node.final, dict(version=VERSION, natural=True, candidate=node.identifier))
                # Durable proposals are independent of their finishing scores.
                archive_rows = []
                for node in archive_frontier(archive+frontier, args.archive_candidates, history):
                    trace = output/"archive-prefixes"/f"{node.identifier}.u32le"
                    atomic_write(trace, trace_bytes(actions[:args.tick]+list(node.suffix)))
                    archive_rows.append(dict(source=str(trace), observation=node.observation,
                        history=node.history.to_dict(), parent_tick=args.tick, value=asdict(node.value)))
                write_json(output / "archive-proposals.json", archive_rows)
                if args.finish_candidates and frontier and not budget.reason:
                    finish_nodes=finishing_nodes(frontier,args)
                    result["finish_input_candidates"]=[n.identifier for n in finish_nodes]
                    with env.fast_checkpoint() as checkpoint:
                        result["results"] = finish_candidates(env, checkpoint, finish_nodes, actions[:args.tick],
                            history, args, bundle, budget, output, policy_screen_seconds=30)
                result["provenance"] = session.provenance_manifest
                result["status"] = "stopped" if budget.reason else "complete"
                if budget.reason:
                    result["stop_reason"] = budget.reason
        except StopRequested as stopped:
            result.update(status="stopped", stop_reason=str(stopped))
        except Exception as error:
            result.update(status="error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            result.update(cpu_seconds=cpu_usage()-before, seconds=time.monotonic()-budget.started)
            write_json(output / "result.json", result)
    print(json.dumps(jsonable({k:v for k,v in result.items() if k != "provenance"})), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "worker", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--algorithm", choices=("baseline", "event", "nrpa", "sparse"), default="event")
    parser.add_argument("--fixed-option-interval",action="store_true")
    parser.add_argument("--sparse-walk",action="store_true")
    parser.add_argument("--sparse-temperature",type=float,default=0)
    for name, default in dict(tick=95680, seed=1298144938, max_ticks=120000, horizon=3072,
        segment=96, beam_width=14, candidates=24, random_seed=261001, cooldown=16,
        finish_candidates=1, archive_candidates=32, max_decisions=256, nrpa_levels=1,
        nrpa_iterations=256, nrpa_repetitions=8, nrpa_restarts=64, causal_options=8, retirement_options=0,
        co_clear_options=0,preparation_options=0,relocation_options=0).items():
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=default)
    parser.add_argument("--sparse-restart-anchors",type=int,default=64)
    for name, default in dict(seconds=1200, search_seconds=300, finish_seconds=900,
        extension_weight=3, cashout_weight=1, opening_weight=1.25, stale_half_life=1536,
        tick_cost=.5, nrpa_rate=.8, temperature=1, reference_bias=4.5, assessment_seconds=3).items():
        parser.add_argument("--"+name.replace("_", "-"), type=float, default=default)
    parser.add_argument("--wasted-qualification-penalty", type=float, default=0)
    parser.add_argument("--asset-size-weight", type=float, default=0)
    parser.add_argument("--asset-gauge-window",type=float,default=2000)
    parser.add_argument("--finish-mode", choices=("mpc", "reference"), default="mpc")
    parser.add_argument("--nrpa-finish", choices=("none", "reference", "policy"), default="reference")
    parser.add_argument("--nrpa-objective", choices=("score","opportunity","asset"), default="score")
    args = parser.parse_args(argv)
    for key in ("seconds", "search_seconds", "finish_seconds", "stale_half_life", "nrpa_rate",
                "temperature", "assessment_seconds","asset_gauge_window"):
        if not math.isfinite(getattr(args,key)) or getattr(args,key) <= 0:
            parser.error(f"{key} must be finite and positive")
    for key in ("horizon", "segment", "beam_width", "candidates", "archive_candidates", "max_decisions",
                "nrpa_iterations", "nrpa_repetitions", "nrpa_restarts", "cooldown", "max_ticks"):
        if getattr(args,key) < 1:
            parser.error(f"{key} must be positive")
    if args.sparse_restart_anchors < 1:
        parser.error("sparse_restart_anchors must be positive")
    if (not 0 <= args.seed < 2**32 or not 0 <= args.tick < args.max_ticks
            or not 0 <= args.finish_candidates <= 1 or not 0 <= args.nrpa_levels <= 3
            or min(args.causal_options,args.retirement_options,args.co_clear_options,args.preparation_options,args.relocation_options)<0):
        parser.error("invalid seed, tick, finishing count or NRPA depth")
    for key in ("extension_weight", "cashout_weight", "opening_weight", "tick_cost", "reference_bias",
                "wasted_qualification_penalty","sparse_temperature","asset_size_weight"):
        if not math.isfinite(getattr(args,key)) or getattr(args,key) < 0:
            parser.error(f"{key} must be finite and nonnegative")
    if args.finish_candidates and args.finish_mode == "mpc" and not args.checkpoint:
        parser.error("MPC finishing requires checkpoint")
    if args.nrpa_finish == "policy" and not args.checkpoint:
        parser.error("policy assessment requires checkpoint")
    if args.checkpoint and not args.checkpoint_sha256:
        parser.error("checkpoint requires its expected SHA256")
    for key in ("source", "worker", "output", "checkpoint"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    run(args)


if __name__ == "__main__":
    main()
