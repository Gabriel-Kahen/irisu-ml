// Deterministic simulator-generated coverage input; this is not a browser or
// original-recorded replay. Every cursor coordinate is packed before stepping.
#include "irisu/floating_point.hpp"
#include "irisu/metsu/simulator.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

int main(int argc,char** argv) {
  try {
    if(argc!=3) throw std::invalid_argument("usage: generate_controlled_replay UINT32_SEED OUTPUT.rpy");
    const auto parsed=std::stoull(argv[1]);
    if(parsed>std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("seed range");
    const auto seed=static_cast<std::uint32_t>(parsed);
    irisu::detail::install_canonical_floating_point_environment();
    irisu::metsu::Simulator sim;
    sim.reset(seed);
    std::mt19937 random(seed+12345U);
    std::vector<std::uint32_t> words;
    std::uint32_t held=0,x=320,y=400,fires=0;
    bool ex_chain_seen=false,special_seen=false,finished=false;
    irisu::StepDiagnostics final;
    std::uint32_t highest_level=1;
    std::uint64_t clears=0;
    for(std::uint32_t frame=0;frame<100000;++frame) {
      std::uint32_t buttons=0;
      if(frame>2&&frame%24<3) {
        if(frame%24==0) {
          held=(fires++%2)?2U:1U;
          x=124U+random()%380U;y=380;
          const irisu::Body* target=nullptr;
          for(const auto& body:sim.bodies()) {
            if(body.pending_delete||body.lifecycle==irisu::Lifecycle::Deleted||
               body.kind!=irisu::BodyKind::Piece||body.special||body.grouped||!body.rule_guard_f0) continue;
            if(!target||(!body.physics_owned&&target->physics_owned)||body.age_ticks<target->age_ticks) target=&body;
          }
          if(target) {
            x=static_cast<std::uint32_t>(std::clamp(target->position.x,0.0,639.0));
            y=static_cast<std::uint32_t>(std::clamp(target->position.y+target->size/2+18,0.0,479.0));
          }
        }
        buttons=held;
      }
      words.push_back((y<<12U)|(x<<2U)|buttons);
      irisu::Action action{static_cast<irisu::ActionKind>(buttons),double(x),double(y),1};
      action.suppress_fresh_edges=frame<2;
      const auto result=sim.step(action);
      for(const auto& body:sim.bodies()) {
        ex_chain_seen|=body.rule_guard_f0&&body.chain_id;
        special_seen|=body.special;
      }
      const auto state=sim.observation();
      highest_level=std::max(highest_level,state.level);clears=state.qualifying_clear_count;
      if(result.terminated||result.truncated) { final=result.diagnostics;finished=result.terminated;break; }
    }
    if(!finished||!final.terminal_metadata_recorded) throw std::runtime_error("no natural canonical finish within100000inputs");
    std::ofstream output(argv[2],std::ios::binary);
    if(!output) throw std::runtime_error("cannot create replay");
    const auto write=[&](std::uint32_t value) {
      for(unsigned shift=0;shift<32;shift+=8) output.put(static_cast<char>(value>>shift));
    };
    write(seed);write(final.recorded_final_level);write(static_cast<std::uint32_t>(final.recorded_final_score));
    write(final.recorded_final_highest_chain);write(1);
    for(unsigned index=0;index<32;++index) output.put(0);
    for(const auto value:words) write(value);
    output.close();if(!output) throw std::runtime_error("replay write failed");
    std::cout<<"{\"source\":\"deterministic-native-policy\",\"seed\":"<<seed
      <<",\"inputs\":"<<words.size()<<",\"canonical_score\":"<<final.recorded_final_score
      <<",\"canonical_level\":"<<final.recorded_final_level<<",\"canonical_chain\":"<<final.recorded_final_highest_chain
      <<",\"highest_live_level\":"<<highest_level<<",\"qualifying_clears\":"<<clears
      <<",\"shot_edges\":"<<fires<<",\"held_frames_per_edge\":3,\"ex_chain_seen\":"<<(ex_chain_seen?"true":"false")
      <<",\"special_seen\":"<<(special_seen?"true":"false")<<",\"config_hash\":\""<<sim.config_hash()<<"\"}\n";
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
