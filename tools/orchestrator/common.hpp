#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __CUDACC__
#define ORCH_HD __host__ __device__
#else
#define ORCH_HD
#endif
namespace orch {
constexpr unsigned max_bodies=512, max_groups=512, max_horizon=100000;
constexpr unsigned max_event_horizon=512, max_captured_events=32768;
struct Body {
 unsigned id{},kind{},chain_id{},hits{},rot_timer{};int color{};
 double x{},y{},vx{},vy{},size{};long long lifetime{};
 bool pending_clear{},pending_delete{},special{},physics_owned{};
 unsigned shape{},lifecycle{};double angle{},angular_velocity{};std::uint64_t age_ticks{};
};
struct Group {unsigned id{},chain{},num{};};
struct GroupDelta {unsigned id{},chain_before{},num_before{},chain_after{},num_after{},members_after{};};
struct Metrics {std::uint64_t recruitment_events{},new_groups{},max_new_chain{},qualifications{};};
struct Summary {
 unsigned seed{},level{},highest_chain{},canonical_level{},canonical_chain{},body_count{},group_count{};
 unsigned long long tick{},qualifying{},config_hash{};
 long long score{},gauge{},canonical_score{};
 bool terminated{},truncated{},failed{},left_held{},right_held{};
 Body bodies[max_bodies]{};Group groups[max_groups]{};
 unsigned group_delta_count{};GroupDelta group_deltas[max_groups]{};Metrics metrics{};
};
struct Candidate {unsigned mode{},search_seed{};std::vector<unsigned> words;};
inline unsigned number(std::istringstream& in,unsigned long long maximum,const char* name){
 std::string token;if(!(in>>token)||token.empty()||token[0]=='-')throw std::runtime_error(std::string("missing/negative ")+name);
 std::size_t end{};const auto value=std::stoull(token,&end);if(end!=token.size()||value>maximum)throw std::runtime_error(std::string("invalid ")+name);return unsigned(value);
}
inline void end(std::istringstream& in){std::string extra;if(in>>extra)throw std::runtime_error("unexpected trailing input");}
inline std::vector<unsigned> words(std::istringstream& in,unsigned count){std::vector<unsigned> out;out.reserve(count);for(unsigned i=0;i<count;++i)out.push_back(number(in,0x1fffffU,"input word"));end(in);return out;}
inline Candidate candidate(const std::string& line,unsigned horizon){std::istringstream in(line);Candidate c;c.mode=number(in,3,"continuation mode");c.search_seed=number(in,UINT32_MAX,"search seed");const auto n=number(in,horizon,"prefix length");c.words=words(in,n);return c;}
struct MixedCandidate {unsigned parent{},horizon{};Candidate continuation;};
inline MixedCandidate mixed_candidate(const std::string& line){
 std::istringstream in(line);MixedCandidate result;result.parent=number(in,UINT32_MAX,"parent");result.horizon=number(in,max_event_horizon,"horizon");if(!result.parent||!result.horizon)throw std::runtime_error("mixed futures require positive parent and horizon");const auto n=number(in,result.horizon,"word count");result.continuation.words=words(in,n);return result;
}
ORCH_HD inline unsigned mix(unsigned x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16);}
ORCH_HD inline double absolute(double x){return x<0?-x:x;}
// A deterministic, inexpensive proposal controller. Exact search evaluates its
// suggestions; this controller does not claim an optimal or safe strategy.
ORCH_HD inline unsigned policy(const Summary& s,unsigned mode,unsigned random){
 if(!mode||s.tick<2||s.left_held||s.right_held)return 0;
 const unsigned interval=mode==2?16U+(random%17U):6U+(random%13U);
 if((s.tick+random)%interval)return 0;
 double best=-1e30;int chosen=-1,target=-1;
 for(unsigned i=0;i<s.body_count;++i){const auto& a=s.bodies[i];
  if(a.kind!=0||a.special||a.pending_clear||a.pending_delete||a.color<0||a.y<0||a.y>440||a.x<10||a.x>630)continue;
  // Preserve the last life of grouped pieces: shoot fresh recruits instead.
  if(a.chain_id&&a.hits)continue;
  if(mode==2&&a.chain_id)continue;
  for(unsigned j=0;j<s.body_count;++j){if(i==j)continue;const auto& b=s.bodies[j];
   if(b.kind!=0||b.special||b.pending_clear||b.pending_delete||a.color!=b.color||(a.chain_id&&b.chain_id))continue;
   double value=100-absolute(a.x-b.x)*.13-absolute(a.y-b.y)*.07;
   if(!a.chain_id&&b.chain_id){for(unsigned k=0;k<s.group_count;++k)if(s.groups[k].id==b.chain_id)value+=12*s.groups[k].chain+8*s.groups[k].num;}
   if(mode==3)value+=b.rot_timer?100:0;
   if(mode==2)value+=a.y>260?50:0;
   value+=double(mix(random^a.id^(b.id*31U)^unsigned(s.tick/interval))%1000)*.03;
   if(value>best){best=value;chosen=int(i);target=int(j);}
  }
 }
 if(chosen<0)return 0;const auto& a=s.bodies[chosen];const auto& b=s.bodies[target];
 const double side=(b.x>=a.x?1.0:-1.0),offset=.25+double((random>>4)%4)*.1;
 const double x=a.x-side*a.size*offset,y=a.y+a.size*(.65+double((random>>7)%3)*.1);
 if(x<0||x>640||y<0||y>480)return 0;
 const unsigned kind=mode==2?1U:((random>>11)&1U?1U:2U);
 return kind|(unsigned(x+.5)<<2)|(unsigned(y+.5)<<12);
}
inline const char* event_kind(unsigned kind){
 static constexpr const char* names[]={"invalid_action","spawned","shot_fired","activated","contact","confirmed","chain_joined","cleared","rotten","ejected","destroyed","gauge_changed","score_changed","level_changed","game_over","projectile_hit","projectile_contact","held_input_ignored","level_completed"};
 if(kind>=std::size(names))throw std::runtime_error("unknown event kind");return names[kind];
}
template<class E>inline void emit_events(const E* events,std::size_t count){
 std::cout<<'[';for(std::size_t i=0;i<count;++i){if(i)std::cout<<',';const auto& e=events[i];std::cout<<"{\"tick\":"<<e.tick<<",\"kind\":"<<std::quoted(event_kind(unsigned(e.kind)))<<",\"a\":"<<e.a<<",\"b\":"<<e.b<<",\"value\":"<<e.value<<",\"sequence\":"<<e.sequence<<'}';}std::cout<<']';
}
inline void emit_words(const std::vector<unsigned>& values){std::cout<<'[';for(std::size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<values[i];}std::cout<<']';}
inline void emit_summary(const Summary& s){
 if(s.failed)throw std::runtime_error("simulation capacity or numerical failure");
 std::cout<<std::setprecision(17)<<"{\"seed\":"<<s.seed<<",\"tick\":"<<s.tick<<",\"score\":"<<s.score<<",\"gauge\":"<<s.gauge<<",\"level\":"<<s.level<<",\"qualifying\":"<<s.qualifying<<",\"qualifying_clear_count\":"<<s.qualifying<<",\"left_held\":"<<(s.left_held?"true":"false")<<",\"right_held\":"<<(s.right_held?"true":"false")<<",\"highest_chain\":"<<s.highest_chain<<",\"terminated\":"<<(s.terminated?"true":"false")<<",\"truncated\":"<<(s.truncated?"true":"false")<<",\"canonical_score\":"<<s.canonical_score<<",\"canonical_level\":"<<s.canonical_level<<",\"canonical_chain\":"<<s.canonical_chain<<",\"config_hash\":\""<<s.config_hash<<"\",\"groups\":[";
 for(unsigned i=0;i<s.group_count;++i){if(i)std::cout<<',';const auto& g=s.groups[i];std::cout<<"{\"id\":"<<g.id<<",\"chain\":"<<g.chain<<",\"num\":"<<g.num<<",\"members\":[";bool comma=false;for(unsigned j=0;j<s.body_count;++j)if(s.bodies[j].chain_id==g.id){if(comma)std::cout<<',';comma=true;std::cout<<s.bodies[j].id;}std::cout<<"]}";}
 std::cout<<"],\"bodies\":[";
 for(unsigned i=0;i<s.body_count;++i){if(i)std::cout<<',';const auto& b=s.bodies[i];std::cout<<"{\"id\":"<<b.id<<",\"kind\":"<<b.kind<<",\"color\":"<<b.color<<",\"x\":"<<b.x<<",\"y\":"<<b.y<<",\"vx\":"<<b.vx<<",\"vy\":"<<b.vy<<",\"size\":"<<b.size<<",\"chain_id\":"<<b.chain_id<<",\"hits\":"<<b.hits<<",\"rot_timer\":"<<b.rot_timer<<",\"lifetime\":"<<b.lifetime<<",\"remaining_lifetime\":"<<b.lifetime<<",\"projectile_hits\":"<<b.hits<<",\"shape\":"<<b.shape<<",\"lifecycle\":"<<b.lifecycle<<",\"angle\":"<<b.angle<<",\"angular_velocity\":"<<b.angular_velocity<<",\"age_ticks\":"<<b.age_ticks<<",\"pending_clear\":"<<(b.pending_clear?"true":"false")<<",\"pending_delete\":"<<(b.pending_delete?"true":"false")<<",\"special\":"<<(b.special?"true":"false")<<",\"physics_owned\":"<<(b.physics_owned?"true":"false")<<'}';}
 std::cout<<"]}";
}
inline void emit_metrics(const Metrics& m){
 std::cout<<"{\"recruitment_events\":"<<m.recruitment_events<<",\"new_groups\":"<<m.new_groups<<",\"max_new_chain\":"<<m.max_new_chain<<",\"qualifications\":"<<m.qualifications<<'}';
}
inline void emit_group_deltas(const Summary& s){
 std::cout<<'[';for(unsigned i=0;i<s.group_delta_count;++i){if(i)std::cout<<',';const auto& g=s.group_deltas[i];std::cout<<"{\"id\":"<<g.id<<",\"chain_before\":"<<g.chain_before<<",\"num_before\":"<<g.num_before<<",\"chain_after\":"<<g.chain_after<<",\"num_after\":"<<g.num_after<<",\"members_after\":"<<g.members_after<<'}';}std::cout<<']';
}
inline void error(const std::exception& e){std::cout<<"{\"ok\":false,\"error\":"<<std::quoted(e.what())<<"}\n"<<std::flush;}
}
