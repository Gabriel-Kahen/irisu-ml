#include "irisu/metsu/simulator.hpp"
#include "irisu/simulator.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace irisu::metsu {
// Isolate the same contact/timer components as the original-byte oracle. No
// physics step runs here, and each comparison states the measured projection.
struct SimulatorTestAccess {
  static void clear(Simulator& sim) {
    sim.bodies_.clear();
    // Full dispatcher reserves space for reentrant spawns; keep fixture references
    // stable across this contact-only component suite (at most eight actors).
    sim.bodies_.reserve(5*MechanicsConfig::actor_pool_capacity);
    sim.groups_.clear();
    sim.ex_timers_.fill(-1); sim.actor_pool_colors_.fill(-1);
    sim.gauge_=0; sim.qualifying_clear_count_=0; sim.level_=1;
    sim.scene_frame_=1; sim.next_chain_id_=1; sim.active_contact_pairs_.clear();
  }
  static Body& actor(Simulator& sim,bool ex) {
    Body body;
    body.id=static_cast<BodyId>(sim.bodies_.size()+1);
    body.actor_slot=body.id+4; body.rule_guard_f0=ex;
    body.shape=ex?Shape::Circle:Shape::Box; body.position={300,200};
    body.age_ticks=150; body.freshness_state=2; body.color=0;
    body.remaining_lifetime=-1;
    sim.bodies_.push_back(body);
    sim.actor_pool_colors_[body.actor_slot]=body.color;
    return sim.bodies_.back();
  }
  static void group(Simulator& sim,Body& body,std::uint32_t chain,std::uint32_t shadow=0) {
    GroupState group{sim.next_chain_id_++,chain,shadow,0};
    sim.groups_.push_back(group); body.chain_id=group.id;
  }
  static void timer(Simulator& sim,Body& body,int value) { sim.ex_timers_[body.actor_slot]=value; }
  static void contact(Simulator& sim,BodyId a,BodyId b,BoundaryKind boundary=BoundaryKind::None) {
    std::vector<Event> events; sim.process_contacts({{a,b,boundary}},false,false,0,0,events);
  }
  static void update(Simulator& sim,unsigned count=1) {
    std::vector<Event> events;
    for(unsigned i=0;i<count;++i) sim.process_actor_updates({},events);
  }
  static void gauge(Simulator& sim,std::int64_t value) { sim.gauge_=value; }
  static void pool_color(Simulator& sim,std::uint32_t slot,int color) { sim.actor_pool_colors_[slot]=color; }
  static std::int64_t get(const Simulator& sim,const Body& body,const std::string& key) {
    if(key=="dynamic") return body.physics_owned;
    if(key=="grouped") return body.grouped;
    if(key=="clear_pending") return body.successful_clear_pending;
    if(key=="rot_timer") return body.rot_timer;
    if(key=="ex_timer") return sim.ex_timers_[body.actor_slot];
    if(key=="dynamic_updates") return body.physics_update_count;
    if(key=="gate_seen") return body.top_contact_pending;
    if(key=="delete") return body.delete_marked||body.pending_delete;
    if(key=="hits") return body.projectile_hits;
    if(key=="gauge") return sim.gauge_;
    if(key=="qualifying_clears") return sim.qualifying_clear_count_;
    const auto found=std::find_if(sim.groups_.begin(),sim.groups_.end(),[&](const auto& g){return g.id==body.chain_id;});
    if(found==sim.groups_.end()) return 0;
    if(key=="chain") return found->chain;
    if(key=="shadow") return found->secondary_count;
    if(key=="clears") return found->num;
    throw std::runtime_error("unknown measured field: "+key);
  }
};
}

namespace {
using Access=irisu::metsu::SimulatorTestAccess;
using Sim=irisu::metsu::Simulator;
using Values=std::map<std::string,std::int64_t>;
std::map<std::string,Values> oracle;
unsigned comparisons=0,cases=0;
std::set<std::string> visited;

void require(bool valid,const std::string& message) { if(!valid) throw std::runtime_error(message); }
void load_oracle(const std::filesystem::path& path) {
  std::ifstream file(path); require(file.good(),"cannot read original component fixture");
  std::string line,name; Values row;
  while(std::getline(file,line)) {
    const auto first=line.find('"'); if(first==std::string::npos) {
      if(!name.empty()&&line.find('}')!=std::string::npos) { oracle.emplace(name,row); name.clear(); row.clear(); }
      continue;
    }
    const auto end=line.find('"',first+1),colon=line.find(':',end);
    const auto key=line.substr(first+1,end-first-1);
    if(key=="case") {
      const auto begin=line.find('"',colon+1),finish=line.find('"',begin+1);
      name=line.substr(begin+1,finish-begin-1);
    } else if(!name.empty()) row[key]=std::stoll(line.substr(colon+1));
  }
  require(oracle.size()==21,"original component fixture case count changed");
}
void compare(const std::string& name,const Sim& sim,const irisu::Body& body,std::initializer_list<const char*> fields) {
  require(visited.insert(name).second,"duplicate clone case: "+name);
  const auto& expected=oracle.at(name);
  for(const auto* field:fields) {
    const auto actual=Access::get(sim,body,field);
    require(actual==expected.at(field),name+" / "+field+": clone="+std::to_string(actual)+" original="+std::to_string(expected.at(field)));
    ++comparisons;
  }
  ++cases;
}
irisu::MechanicsConfig config() {
  auto config=irisu::metsu::default_config();
  config.initial_falling_count=0; config.initial_rotten_count=0;
  config.gravity_y=0; config.passive_gauge_decay_per_tick=0;
  config.spawn_interval_ticks=100000;
  return config;
}
void timers_and_groups() {
  {
    Sim sim(config()); Access::clear(sim);
    auto& a=Access::actor(sim,true); a.physics_owned=true;
    Access::group(sim,a,2); Access::timer(sim,a,120);
    Access::update(sim,8);
    compare("countdown_8",sim,a,{"dynamic","grouped","ex_timer","dynamic_updates","chain"});
    auto& b=Access::actor(sim,true);
    Access::contact(sim,b.id,a.id);
    compare("propagate_112",sim,b,{"dynamic","grouped","ex_timer","chain","shadow"});
    // The source shares the now incremented chain from the propagation case.
    Access::timer(sim,a,1); Access::update(sim);
    compare("countdown_expired",sim,a,{"grouped","ex_timer","dynamic_updates","chain","shadow"});
  }
  {
    Sim sim(config()); Access::clear(sim);
    auto& a=Access::actor(sim,true); a.physics_owned=true;
    Access::group(sim,a,3,1); Access::timer(sim,a,113);
    auto& b=Access::actor(sim,true);
    Access::contact(sim,b.id,a.id);
    compare("propagate_113_rejected",sim,b,{"dynamic","grouped","ex_timer","chain"});
  }
  {
    Sim sim(config()); Access::clear(sim);
    auto& source=Access::actor(sim,false); source.grouped=true; source.physics_owned=true;
    Access::group(sim,source,4,4);
    auto& target=Access::actor(sim,true); Access::timer(sim,target,57);
    Access::contact(sim,target.id,source.id);
    compare("mixed_group_resets_countdown",sim,target,{"dynamic","grouped","ex_timer","chain","shadow"});
  }
}
void gates_and_activation() {
  {
    Sim sim(config()); Access::clear(sim); auto& a=Access::actor(sim,true);
    Access::contact(sim,a.id,0,irisu::BoundaryKind::Floor);
    compare("ex_bottom_gate",sim,a,{"dynamic","gate_seen","ex_timer"});
  }
  {
    Sim sim(config()); Access::clear(sim); auto& a=Access::actor(sim,true);
    Access::contact(sim,a.id,0,irisu::BoundaryKind::Top);
    compare("ex_top_open",sim,a,{"dynamic","gate_seen"});
  }
  for(bool dynamic: {false,true}) {
    Sim sim(config()); Access::clear(sim); auto& target=Access::actor(sim,true);
    auto& source=Access::actor(sim,false); source.color=1; source.physics_owned=dynamic;
    Access::contact(sim,target.id,source.id);
    compare(dynamic?"ex_dynamic_ordinary_rejected":"ex_activated_by_scripted_ordinary",
            sim,target,{"dynamic","grouped","ex_timer"});
  }
  {
    Sim sim(config()); Access::clear(sim); auto& target=Access::actor(sim,true);
    auto& shot=Access::actor(sim,false); shot.kind=irisu::BodyKind::Projectile;
    shot.color=-1; shot.physics_owned=true;
    Access::contact(sim,target.id,shot.id);
    compare("ex_activated_by_projectile",sim,target,{"dynamic","grouped","ex_timer"});
    require(shot.delete_marked,"activation must consume projectile");
  }
}
void rot_and_scripted() {
  for(unsigned updates: {4U,5U}) {
    Sim sim(config()); Access::clear(sim); auto& target=Access::actor(sim,true);
    // Dynamic state bypasses the separate bottom startup gate; the measured
    // rot-start projection is otherwise identical to the direct original call.
    target.physics_owned=true; target.physics_update_count=updates;
    Access::contact(sim,target.id,0,irisu::BoundaryKind::Floor);
    compare(updates==4?"ex_rot_before_5_rejected":"ex_rot_at_5",sim,target,{"rot_timer","dynamic_updates"});
    if(updates==5) {
      Access::update(sim,130);
      compare("ex_rot_does_not_advance",sim,target,{"rot_timer","grouped","ex_timer"});
      require(target.freshness_state==2,"EX actor became rotten");
    }
  }
  {
    Sim sim(config()); Access::clear(sim); auto& a=Access::actor(sim,true);
    auto& b=Access::actor(sim,false); b.color=1;
    Access::contact(sim,a.id,b.id);
    // An Activated event proves the mixed scripted dispatcher did not return
    // early. Its measured grouping/countdown state stays unchanged.
    compare("mixed_scripted_contact_allowed",sim,a,{"grouped","ex_timer","delete"});
    require(a.physics_owned,"mixed scripted contact did not reach activation");
  }
  {
    Sim sim(config()); Access::clear(sim); auto& a=Access::actor(sim,true);
    auto& b=Access::actor(sim,true); Access::contact(sim,a.id,b.id);
    compare("same_ex_scripted_short_circuit",sim,a,{"dynamic","grouped","ex_timer","chain"});
  }
}
void bursts_hits_and_special() {
  {
    Sim sim(config()); Access::clear(sim); auto& a=Access::actor(sim,true);
    a.physics_owned=true; a.grouped=true; Access::group(sim,a,7);
    auto& source=Access::actor(sim,true); source.color=1;
    Access::gauge(sim,1000); Access::contact(sim,a.id,source.id);
    compare("ex_source_bursts_mature_ex",sim,a,{"clear_pending","chain","clears","gauge","qualifying_clears"});
  }
  for(bool ex: {true,false}) {
    Sim sim(config()); Access::clear(sim); auto& target=Access::actor(sim,ex);
    target.physics_owned=true;
    auto& shot=Access::actor(sim,false); shot.kind=irisu::BodyKind::Projectile;
    shot.color=-1; shot.physics_owned=true; shot.non_wall_contacts=1;
    Access::contact(sim,target.id,shot.id);
    compare(ex?"ex_direct_hit_retains_projectile":"ordinary_direct_hit_deletes_projectile",sim,shot,{"delete","hits"});
  }
  {
    Sim sim(config()); Access::clear(sim); auto& a=Access::actor(sim,true); a.color=3;
    auto& orb=Access::actor(sim,false); orb.kind=irisu::BodyKind::Bonus;
    orb.special=true; orb.top_contact_enabled=true; orb.physics_owned=true; orb.color=-2;
    Access::pool_color(sim,a.actor_slot,3); Access::pool_color(sim,42,3);
    Access::gauge(sim,1000); Access::contact(sim,a.id,orb.id);
    compare("metsu_orb_clears_color_without_gauge",sim,a,{"delete","gauge"});
    require(orb.delete_marked,"orb was not consumed");
  }
  {
    Sim sim(config()); Access::clear(sim);
    auto& shot=Access::actor(sim,false); shot.kind=irisu::BodyKind::Projectile; shot.physics_owned=true;
    auto& target=Access::actor(sim,false);
    // A same-color projectile cannot itself join the chain. This isolates the
    // original fresh-target grouping call within the complete dispatcher.
    Access::contact(sim,target.id,shot.id);
    compare("fresh_ordinary_chain_control",sim,target,{"grouped","chain","shadow"});
  }
}
void normal_positive_control() {
  auto cfg=config(); cfg.gauge_clear_unit=120;
  irisu::Simulator sim(cfg);
  const auto target=sim.spawn_piece(irisu::Shape::Box,3,30,{300,200});
  const auto orb=sim.spawn_bonus({300,200});
  auto state=sim.clone_state(); state.scene_frame=1; state.gauge=1000;
  state.actor_pool_colors.fill(-1); state.actor_pool_colors[42]=3;
  for(auto& body:state.bodies) {
    body.age_ticks=150; body.freshness_state=2;
    body.top_contact_enabled=true; body.physics_owned=true;
    body.lifecycle=irisu::Lifecycle::DynamicFresh;
    state.actor_pool_colors[body.actor_slot]=body.color;
  }
  sim.restore_state(state);
  sim.step({irisu::ActionKind::Wait,0,0,1});
  require(sim.observation().gauge==oracle.at("normal_color_clear_positive_control").at("gauge"),
          "normal color-clear positive control differs from original");
  require(target&&orb,"normal control fixture allocation failed");
  visited.insert("normal_color_clear_positive_control");
  ++comparisons; ++cases;
}
}

int main(int argc,char** argv) {
  try {
    const auto fixture=argc==2?std::filesystem::path(argv[1]):
        std::filesystem::path(__FILE__).parent_path().parent_path().parent_path()/"reference/metsu-contacts-original-components.json";
    load_oracle(fixture); timers_and_groups(); gates_and_activation(); rot_and_scripted();
    bursts_hits_and_special(); normal_positive_control();
    require(cases==oracle.size()&&visited.size()==oracle.size(),"clone/original case inventory mismatch");
    std::cout<<cases<<" original component cases matched ("<<comparisons<<" measured fields)\n";
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
