#include "shared_future_plan.hpp"
#include <cassert>
#include <iostream>
#include <random>
int main(int argc,char** argv){
 if(argc==5){const auto input=irisu::cuda_search::load_inputs(argv[1],argv[2],argv[3],argv[4]);const auto plan=irisu::shared_future::build_adaptive(input);std::cout<<"split="<<plan.split<<" groups="<<plan.representatives.size()<<" independent="<<plan.independent_ticks<<" planned="<<plan.planned_ticks<<"\n";return 0;}
 using namespace irisu;cuda_search::Inputs input;input.options.horizon=4;input.replay={0,0,0,0};
 cuda_search::Candidate a{},b{},c{},d{};b.count=1;b.mutations[0]={2,1};c.count=1;c.mutations[0]={3,2};d.count=1;d.mutations[0]={0,3};input.candidates={a,b,c,d};
 auto plan=shared_future::build(input);assert(plan.split==2&&plan.representatives.size()==2&&plan.planned_ticks==12&&plan.independent_ticks==16);assert(plan.selection[0]==plan.selection[1]&&plan.selection[0]==plan.selection[2]&&plan.selection[0]!=plan.selection[3]);
 // Mutation storage order and edits equal to the incumbent do not split keys.
 a.count=2;a.mutations[0]={1,2};a.mutations[1]={0,1};b=a;std::swap(b.mutations[0],b.mutations[1]);c=b;c.count=3;c.mutations[2]={3,0};input.candidates={a,b,c};plan=shared_future::build(input);assert(plan.split==4&&plan.representatives.size()==2&&plan.planned_ticks==4);
 input.options.horizon=100000;input.candidates={cuda_search::Candidate{},cuda_search::Candidate{}};plan=shared_future::build(input);assert(plan.split==100000&&plan.planned_ticks==100000);
 input.options.horizon=4;input.candidates={a};plan=shared_future::build(input);assert(!plan.split&&plan.selection.empty());
 bool rejected=false;try{shared_future::build(input,5);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 std::mt19937 random(71931);
 for(unsigned trial=0;trial<1000;++trial){
  input.options.horizon=1+random()%20;input.options.prefix=0;input.replay.resize(input.options.horizon);for(auto& word:input.replay)word=random()%4;
  input.candidates.assign(1+random()%30,{});
  for(auto& candidate:input.candidates){std::set<unsigned> used;const auto mutations=random()%5;for(unsigned m=0;m<mutations;++m){const unsigned offset=random()%input.options.horizon;if(used.insert(offset).second)candidate.mutations[candidate.count++]={offset,unsigned(random()%5)};}}
  const auto automatic=shared_future::build(input);auto minimum=std::uint64_t(input.candidates.size())*input.options.horizon;
  for(unsigned split=1;split<=input.options.horizon;++split){const auto explicit_plan=shared_future::build(input,split);minimum=std::min(minimum,explicit_plan.planned_ticks);}
  assert(automatic.planned_ticks==minimum);
  const auto adaptive=shared_future::build_adaptive(input);assert(adaptive.planned_ticks<=adaptive.independent_ticks&&adaptive.representatives.size()<=128);
  if(adaptive.split){std::uint64_t actual=0;for(auto depth:adaptive.depths)actual+=depth;for(unsigned lane=0;lane<input.candidates.size();++lane){const auto group=adaptive.selection[lane],representative=adaptive.representatives[group],depth=adaptive.depths[group];actual+=input.options.horizon-depth;for(unsigned t=0;t<depth;++t)assert(cuda_search::input_at(input.candidates[lane],t,input.replay[t])==cuda_search::input_at(input.candidates[representative],t,input.replay[t]));}assert(actual==adaptive.planned_ticks);}

 }
 std::cout<<"shared prefix planner: fixed cases and1000 randomized brute-force minima passed\n";
}
