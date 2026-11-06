
import pytest

from irisu_search.historical.diverse_value import evaluate_diverse
from irisu_search.historical.opportunity_value import ComboHistory, OpportunityConfig, evaluate_opportunity


def state(count=988):
    def body(bid,gid,x):
        return dict(id=bid,kind="piece",chain_id=gid,color=0,x=x,y=150,size=54,
            lifecycle="confirmed" if gid else "scripted_falling",rot_timer=0,
            projectile_hits=0,remaining_lifetime=2000)
    obs=dict(tick=100,level=99,score=3000000,gauge=20000,qualifying_clear_count=count,
        bodies=[body(1,7,300),body(2,0,350),body(3,0,400)])
    history=ComboHistory.from_groups([dict(chain_id=7,cumulative_members=64,
        successful_clears=53,first_tick=1,member_ids=[1])])
    return obs,history


def test_visible_feeders_have_value_with_one_scoring_qualification_left():
    obs,history=state()
    config=OpportunityConfig(extension_weight=3)
    old=evaluate_opportunity(obs,history,config)
    new=evaluate_diverse(obs,history,config)
    assert old.extension_value==0
    assert new.extension_value>0 and new.objective>old.objective
    assert new.group_details[0]["co_clear_feeder_ids"]==[2,3]
    assert new.guaranteed_future_value==0
    assert obs["qualifying_clear_count"]==988 and history.groups[7]["cumulative_members"]==64


@pytest.mark.parametrize("count",[980,989,990])
def test_extra_credit_is_disabled_outside_last_scored_qualifications(count):
    obs,history=state(count)
    config=OpportunityConfig()
    assert evaluate_diverse(obs,history,config)==evaluate_opportunity(obs,history,config)


def test_no_credit_for_rotten_distant_wrong_color_or_invisible_feeders():
    obs,history=state()
    obs["bodies"][1].update(rot_timer=1,lifecycle='rotten')
    obs["bodies"][2].update(color=1)
    config=OpportunityConfig()
    assert evaluate_diverse(obs,history,config).extension_value==0
    obs["bodies"][1].update(rot_timer=0,lifecycle='dynamic_fresh',x=1000)
    assert evaluate_diverse(obs,history,config).extension_value==0
    obs["bodies"]=obs["bodies"][:1]
    assert evaluate_diverse(obs,history,config).extension_value==0


def test_zero_confidence_restores_original_and_invalid_confidence_rejected():
    obs,history=state()
    config=OpportunityConfig()
    assert evaluate_diverse(obs,history,config,co_clear_confidence=0)==evaluate_opportunity(obs,history,config)
    with pytest.raises(ValueError):
        evaluate_diverse(obs,history,config,co_clear_confidence=float("nan"))


def test_unpaid_live_carriers_are_not_all_cashable_under_one_qualification():
    obs,history=state()
    obs['bodies']=obs['bodies'][:1]
    config=OpportunityConfig()
    one=evaluate_diverse(obs,history,config)
    obs['bodies'].append(dict(obs['bodies'][0],id=4,x=320))
    crowded=evaluate_diverse(obs,history,config)
    assert crowded.cashout_value==one.cashout_value
    assert len(crowded.group_details[0]['cashout_carrier_ids'])==1
    obs['qualifying_clear_count']=987
    two=evaluate_diverse(obs,history,config)
    assert two.cashout_value==2*one.cashout_value


def test_prepared_fresh_feeder_is_preserved_before_the_last_shared_payment():
    obs,history=state()
    obs['bodies']=obs['bodies'][:2]
    config=OpportunityConfig()
    fresh=evaluate_diverse(obs,history,config)
    obs['bodies'][1].update(lifecycle='dynamic_fresh',rot_timer=4)
    prepared=evaluate_diverse(obs,history,config)
    assert prepared.extension_value>fresh.extension_value
    assert prepared.group_details[0]['prepared_feeder_ids']==[2]
    obs['bodies'][0].update(y=524)
    assert evaluate_diverse(obs,history,config).extension_value==0
