// Exact replay genealogy. Group identities come from native membership events;
// score attribution uses the native score -> scored actor teardown sequence.
#include "irisu/simulator.hpp"
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef IRISU_ORCHESTRATOR_BACKEND
#define IRISU_ORCHESTRATOR_BACKEND "unknown"
#endif
#ifndef IRISU_TRACE_SOURCE_SHA256
#define IRISU_TRACE_SOURCE_SHA256 "unknown"
#endif
#ifndef IRISU_TRACE_EXACT_LIBRARY_SHA256
#define IRISU_TRACE_EXACT_LIBRARY_SHA256 "unknown"
#endif
namespace {
struct Group {
 std::uint32_t id{},chain{},num{};
 std::uint64_t first_tick{},last_join_tick{},last_score_tick{},last_tick{},members{},scored_pieces{};
 std::optional<std::uint64_t> extinct_tick;
 std::int64_t score_live{},score_canonical{};
};
struct PendingScore {std::uint64_t tick{},sequence{};std::int64_t delta{};bool canonical_eligible{};};
std::uint32_t word(const std::vector<unsigned char>& bytes,std::size_t at){return std::uint32_t(bytes[at])|(std::uint32_t(bytes[at+1])<<8U)|(std::uint32_t(bytes[at+2])<<16U)|(std::uint32_t(bytes[at+3])<<24U);}
void fields(std::uint64_t tick,const char* kind,const Group& group,std::uint32_t body=0){std::cout<<"{\"tick\":"<<tick<<",\"kind\":"<<std::quoted(kind)<<",\"group_id\":"<<group.id<<",\"body_id\":"<<body<<",\"chain\":"<<group.chain<<",\"num\":"<<group.num;}
void maybe_tick(std::optional<std::uint64_t> tick){if(tick)std::cout<<*tick;else std::cout<<"null";}
}
int main(int argc,char** argv){
 try{
  if(argc!=2)throw std::runtime_error("usage: irisu-search-trace padded-normal-mode-replay.rpy");
  std::ifstream file(argv[1],std::ios::binary|std::ios::ate);if(!file)throw std::runtime_error("cannot open replay");
  const auto length=std::streamoff(file.tellg());constexpr std::streamoff limit=52+4*1000000;
  if(length<56||length>limit||(length-52)%4)throw std::runtime_error("expected padded replay with 1..1000000 input records");
  std::vector<unsigned char> bytes(static_cast<std::size_t>(length));file.seekg(0);file.read(reinterpret_cast<char*>(bytes.data()),length);if(!file)throw std::runtime_error("cannot read complete replay");
  if(word(bytes,16)!=0)throw std::runtime_error("only normal-mode replay is supported");
  const auto seed=word(bytes,0),header_level=word(bytes,4),header_score=word(bytes,8),header_chain=word(bytes,12);const auto count=(bytes.size()-52)/4;
  const std::uint16_t control=0x027f;asm volatile("fldcw %0"::"m"(control));
  const auto start=std::chrono::steady_clock::now();irisu::Simulator simulator;simulator.reset(seed);irisu::StepResult step;
  std::map<unsigned,Group> groups;std::map<unsigned,unsigned> body_groups;std::set<unsigned> live_members;
  std::optional<PendingScore> pending;std::optional<std::uint64_t> first_finish_tick;
  bool finished=false;std::uint64_t score_events=0,total_joins=0,total_qualifications=0,total_removals=0;
  std::int64_t attributed_live=0,attributed_canonical=0;
  std::cout<<"{\"kind\":\"identity\",\"schema\":\"irisu-search-genealogy-v1\",\"backend\":"<<std::quoted(IRISU_ORCHESTRATOR_BACKEND)<<",\"source_sha256\":"<<std::quoted(IRISU_TRACE_SOURCE_SHA256)<<",\"exact_library_sha256\":"<<std::quoted(IRISU_TRACE_EXACT_LIBRARY_SHA256)<<",\"config_hash\":\""<<simulator.config_hash()<<"\",\"seed\":"<<seed<<",\"input_words\":"<<count<<",\"header_score\":"<<header_score<<",\"header_level\":"<<header_level<<",\"header_chain\":"<<header_chain<<"}\n";
  for(std::size_t index=0;index<count;++index){
   if(step.terminated||step.truncated)throw std::runtime_error("replay contains post-terminal input records");
   const auto input=word(bytes,52+4*index);if(input>>21U)throw std::runtime_error("replay contains invalid input bits");
   step=simulator.step({static_cast<irisu::ActionKind>(input&3U),double((input>>2U)&1023U),double((input>>12U)&511U),1,index<2});
   const auto tick=simulator.observation().tick;
   auto possibly_removed=live_members;std::set<unsigned> touched;
   for(const auto& event:step.events){
    if(event.kind==irisu::EventKind::ChainJoined){
     if(event.value<=0||event.value>UINT32_MAX)throw std::runtime_error("invalid native group identifier");const auto id=unsigned(event.value);auto& group=groups[id];if(!group.id){group.id=id;group.first_tick=event.tick;}
     if(body_groups.count(event.a))throw std::runtime_error("body was recruited more than once");body_groups[event.a]=id;++group.chain;group.last_join_tick=group.last_tick=event.tick;++total_joins;touched.insert(id);possibly_removed.insert(event.a);
     fields(event.tick,"join",group,event.a);std::cout<<",\"source_body_id\":"<<event.b<<",\"sequence\":"<<event.sequence<<"}\n";
    }else if(event.kind==irisu::EventKind::Confirmed){
     const auto member=body_groups.find(event.a);if(member==body_groups.end())throw std::runtime_error("qualification has no known group");auto& group=groups.at(member->second);if(event.value!=std::int64_t(group.num)+1)throw std::runtime_error("nonconsecutive group qualification counter");group.num=unsigned(event.value);group.last_tick=event.tick;++total_qualifications;touched.insert(group.id);
     fields(event.tick,"qualify",group,event.a);std::cout<<",\"source_body_id\":"<<event.b<<",\"sequence\":"<<event.sequence<<"}\n";
    }else if(event.kind==irisu::EventKind::GameOver||event.kind==irisu::EventKind::LevelCompleted){
     if(!finished){finished=true;first_finish_tick=event.tick;std::cout<<"{\"tick\":"<<event.tick<<",\"kind\":\"first_finish\",\"sequence\":"<<event.sequence<<",\"recorded_score\":"<<step.diagnostics.recorded_final_score<<"}\n";}
    }else if(event.kind==irisu::EventKind::ScoreChanged){
     if(pending)throw std::runtime_error("unassigned prior score event");if(event.value<=0||event.detail!="normal burst block")throw std::runtime_error("unsupported score attribution event");pending=PendingScore{event.tick,event.sequence,event.value,!finished};++score_events;
    }else if(event.kind==irisu::EventKind::Cleared&&event.detail=="normal burst actor teardown"){
     if(!pending)throw std::runtime_error("scored actor teardown has no score event");const auto member=body_groups.find(event.a);if(member==body_groups.end())throw std::runtime_error("scored actor has no known group");auto& group=groups.at(member->second);if(event.value!=group.num)throw std::runtime_error("scored actor group counter mismatch");group.score_live+=pending->delta;attributed_live+=pending->delta;if(pending->canonical_eligible){group.score_canonical+=pending->delta;attributed_canonical+=pending->delta;}++group.scored_pieces;group.last_score_tick=group.last_tick=event.tick;touched.insert(group.id);
     fields(event.tick,"score",group,event.a);std::cout<<",\"delta\":"<<pending->delta<<",\"canonical_eligible\":"<<(pending->canonical_eligible?"true":"false")<<",\"score_sequence\":"<<pending->sequence<<",\"sequence\":"<<event.sequence<<"}\n";pending.reset();
    }else if(event.kind==irisu::EventKind::Ejected||event.kind==irisu::EventKind::Destroyed||event.kind==irisu::EventKind::ProjectileHit||event.kind==irisu::EventKind::Rotten){
     const auto member=body_groups.find(event.a);if(member!=body_groups.end()){auto& group=groups.at(member->second);fields(event.tick,event.kind==irisu::EventKind::Ejected?"ejected":event.kind==irisu::EventKind::Rotten?"rotten":event.kind==irisu::EventKind::Destroyed?"deletion_mark":"projectile_hit",group,event.a);std::cout<<",\"detail\":"<<std::quoted(event.detail)<<",\"sequence\":"<<event.sequence<<"}\n";}
     // ProjectileHit stores projectile in a, grouped target in b.
     if(event.kind==irisu::EventKind::ProjectileHit){const auto target=body_groups.find(event.b);if(target!=body_groups.end()){fields(event.tick,"projectile_hit",groups.at(target->second),event.b);std::cout<<",\"projectile_id\":"<<event.a<<",\"previous_hits\":"<<event.value<<",\"sequence\":"<<event.sequence<<"}\n";}}
    }
   }
   if(pending)throw std::runtime_error("score event was not attributed within its update");
   std::set<unsigned> now;std::map<unsigned,unsigned> member_counts;
   for(const auto& body:simulator.bodies())if(body.lifecycle!=irisu::Lifecycle::Deleted&&body.chain_id){const auto found=body_groups.find(body.id);if(found==body_groups.end()||found->second!=body.chain_id)throw std::runtime_error("live body group disagrees with event genealogy");now.insert(body.id);++member_counts[body.chain_id];}
   for(const auto body:possibly_removed)if(!now.count(body)){const auto id=body_groups.at(body);auto& group=groups.at(id);group.last_tick=tick;fields(tick,"removed",group,body);std::cout<<"}\n";touched.insert(id);++total_removals;}
   for(auto& [id,group]:groups){const auto after=member_counts[id];if(after!=group.members||touched.count(id)){
     fields(tick,"carriers",group);std::cout<<",\"before\":"<<group.members<<",\"after\":"<<after<<",\"members\":[";bool comma=false;for(auto body:now)if(body_groups.at(body)==id){if(comma)std::cout<<',';comma=true;std::cout<<body;}std::cout<<"]}\n";
     if(!after&&!group.extinct_tick){group.extinct_tick=tick;fields(tick,"extinct",group);std::cout<<"}\n";}else if(after&&group.extinct_tick){fields(tick,"revived",group);std::cout<<"}\n";group.extinct_tick.reset();}
    }group.members=after;
   }
   live_members=std::move(now);
  }
  const auto observation=simulator.observation();const auto& diagnostics=step.diagnostics;
  if(!step.terminated||step.truncated||!diagnostics.terminal_metadata_recorded)throw std::runtime_error("replay did not naturally terminate");
  if(diagnostics.recorded_final_score!=header_score||diagnostics.recorded_final_level!=header_level||diagnostics.recorded_final_highest_chain!=header_chain)throw std::runtime_error("replayed canonical result disagrees with header");
  if(attributed_live!=observation.score||attributed_canonical!=diagnostics.recorded_final_score)throw std::runtime_error("group score attribution does not reconcile with live/canonical totals");
  std::cout<<"{\"kind\":\"summary\",\"schema\":\"irisu-search-genealogy-v1\",\"verified\":true,\"seed\":"<<seed<<",\"tick\":"<<observation.tick<<",\"canonical_score\":"<<diagnostics.recorded_final_score<<",\"live_score\":"<<observation.score<<",\"level\":"<<diagnostics.recorded_final_level<<",\"highest_chain\":"<<diagnostics.recorded_final_highest_chain<<",\"canonical_attributed_score\":"<<attributed_canonical<<",\"live_attributed_score\":"<<attributed_live<<",\"unassigned_score_events\":0,\"score_events\":"<<score_events<<",\"joins\":"<<total_joins<<",\"qualifications\":"<<total_qualifications<<",\"removals\":"<<total_removals<<",\"first_finish_tick\":";maybe_tick(first_finish_tick);std::cout<<",\"elapsed_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"groups\":[";
  bool comma=false;for(const auto& [id,group]:groups){if(comma)std::cout<<',';comma=true;std::cout<<"{\"id\":"<<id<<",\"chain\":"<<group.chain<<",\"num\":"<<group.num<<",\"score_canonical\":"<<group.score_canonical<<",\"score_live\":"<<group.score_live<<",\"scored_pieces\":"<<group.scored_pieces<<",\"first_tick\":"<<group.first_tick<<",\"last_join_tick\":"<<group.last_join_tick<<",\"last_score_tick\":"<<group.last_score_tick<<",\"last_tick\":"<<group.last_tick<<",\"members\":"<<group.members<<",\"extinct_tick\":";maybe_tick(group.extinct_tick);std::cout<<'}';}
  std::cout<<"]}\n";if(!std::cout.flush())throw std::runtime_error("failed writing genealogy output");return 0;
 }catch(const std::exception& error){std::cerr<<"irisu-search-trace: "<<error.what()<<'\n';return 1;}
}
