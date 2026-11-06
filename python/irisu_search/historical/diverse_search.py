"""Recovered October proposal allocation and diverse beam selection."""
from __future__ import annotations
import json
import math
from types import SimpleNamespace
from . import growth_search as growth
from .growth_proposals import GrowthMacro
from .event_options import (EventOption, propose_options, propose_protection_options,
    propose_retirement_options, propose_preparation_options, execute_option, _step, _aim)
from .opportunity_value import OpportunityConfig, _health, _factor
from .diverse_value import evaluate_diverse
from .common import snapshot

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



def finishing_nodes(nodes,args):
    if args.algorithm == "nrpa" and getattr(args,"nrpa_objective","score") == "asset":
        measured=[n for n in nodes if getattr(n,"asset_fitness",None) is not None]
        if measured:
            return [max(measured,key=lambda n:n.asset_fitness)]
    return nodes



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
    from .diverse_archive import describe_state
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



def defaults(**overrides):
    """Historical record-winning event parameters and original value defaults."""
    values = dict(seed=1298144938, tick=96000, max_ticks=120000, horizon=3072,
        segment=32, beam_width=24, candidates=48, random_seed=661003, cooldown=16,
        fixed_option_interval=True, causal_options=4, retirement_options=6,
        co_clear_options=8, preparation_options=6, relocation_options=0,
        extension_weight=3, cashout_weight=1, opening_weight=1.25,
        stale_half_life=1536, tick_cost=.5, finish_candidates=0)
    values.update(overrides)
    return SimpleNamespace(**values)
