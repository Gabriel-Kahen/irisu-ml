#pragma once
#include "../cuda-exact/search_io.hpp"
#include "shared_future_plan.hpp"
#include <numeric>

namespace irisu::gpu_search {
inline constexpr unsigned maximum_frontier=131072,maximum_roots=32;
struct Root {std::uint32_t seed{},prefix{},horizon{};};
struct Job {std::uint32_t root{},candidate{},checkpoint{},begin{};};
struct Checkpoint {std::uint32_t root{},candidate{},depth{};};
struct Inputs {
 std::vector<std::uint32_t> replay;
 std::vector<cuda_search::Candidate> candidates;
 Root original;
};
inline Inputs load(const char* replay_path,const char* specs_path,const char* prefix,const char* horizon){
 const auto bytes=cuda_search::read_bytes(replay_path),specs=cuda_search::read_bytes(specs_path);
 if(bytes.size()<=52||(bytes.size()-52)%4||cuda_search::read_word(bytes,16))throw std::runtime_error("expected padded normal-mode replay");
 if(specs.empty()||specs.size()%sizeof(cuda_search::Candidate)||specs.size()/sizeof(cuda_search::Candidate)>maximum_frontier)throw std::runtime_error("expected 1..131072 fixed candidates");
 const auto p=std::stoull(prefix),h=std::stoull(horizon);
 if(p>(bytes.size()-52)/4||!h||h>100000||p+h>UINT32_MAX)throw std::runtime_error("invalid prefix/horizon");
 Inputs input;input.original={cuda_search::read_word(bytes,0),unsigned(p),unsigned(h)};
 for(std::size_t i=52;i<bytes.size();i+=4)input.replay.push_back(cuda_search::read_word(bytes,i));
 input.candidates.resize(specs.size()/sizeof(cuda_search::Candidate));std::memcpy(input.candidates.data(),specs.data(),specs.size());
 for(const auto& candidate:input.candidates)if(!cuda_search::valid(candidate,unsigned(h)))throw std::runtime_error("invalid sparse candidate");
 return input;
}
struct Plan {
 std::vector<Root> roots;
 std::vector<Job> jobs;
 std::vector<Checkpoint> checkpoints;
 std::uint64_t logical_ticks{},planned_ticks{};
};
inline Plan plan(const Inputs& input,unsigned roots,unsigned seed_step,unsigned prefix_step,bool shared){
 if(!roots||roots>maximum_roots||roots>input.candidates.size())throw std::runtime_error("root count outside bounded frontier");
 Plan output;output.jobs.resize(input.candidates.size());
 for(unsigned root=0;root<roots;++root){
  const auto seed=std::uint64_t(input.original.seed)+std::uint64_t(root)*seed_step;
  const auto prefix=std::uint64_t(input.original.prefix)+std::uint64_t(root)*prefix_step;
  if(seed>UINT32_MAX||prefix>input.replay.size()||prefix+input.original.horizon>UINT32_MAX)throw std::runtime_error("root seed/prefix overflow");
  output.roots.push_back({unsigned(seed),unsigned(prefix),input.original.horizon});
  output.checkpoints.push_back({root,0,0});
 }
 for(unsigned root=0;root<roots;++root){
  cuda_search::Inputs subset;subset.replay=input.replay;subset.options.seed=output.roots[root].seed;subset.options.prefix=output.roots[root].prefix;subset.options.horizon=input.original.horizon;
  std::vector<unsigned> indices;
  for(unsigned i=root;i<input.candidates.size();i+=roots){indices.push_back(i);subset.candidates.push_back(input.candidates[i]);output.jobs[i]={root,i,root,0};}
  if(!shared)continue;
  const auto cache=shared_future::build_adaptive(subset);
  if(cache.representatives.empty())continue;
  std::vector<unsigned> slots(cache.representatives.size());slots[0]=root;
  for(unsigned i=1;i<slots.size();++i){slots[i]=output.checkpoints.size();output.checkpoints.push_back({root,indices[cache.representatives[i]],cache.depths[i]});}
  for(unsigned i=0;i<indices.size();++i){const auto group=cache.selection[i];output.jobs[indices[i]].checkpoint=slots[group];output.jobs[indices[i]].begin=cache.depths[group];}
 }
 for(const auto& checkpoint:output.checkpoints)output.planned_ticks+=checkpoint.depth;
 for(const auto& job:output.jobs){output.logical_ticks+=output.roots[job.root].horizon;output.planned_ticks+=output.roots[job.root].horizon-job.begin;}
 return output;
}
inline std::vector<unsigned> schedule(const Plan& plan){
 std::vector<unsigned> order(plan.jobs.size());std::iota(order.begin(),order.end(),0U);
 std::sort(order.begin(),order.end(),[&](unsigned a,unsigned b){
  const auto x=plan.jobs[a],y=plan.jobs[b];const auto xr=plan.roots[x.root].horizon-x.begin,yr=plan.roots[y.root].horizon-y.begin;
  if(xr!=yr)return xr>yr;if(x.root!=y.root)return x.root<y.root;if(x.checkpoint!=y.checkpoint)return x.checkpoint<y.checkpoint;return a<b;
 });return order;
}
} // namespace irisu::gpu_search
