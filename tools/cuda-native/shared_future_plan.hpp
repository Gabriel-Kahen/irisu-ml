#pragma once
#include "../cuda-exact/search_io.hpp"
#include <algorithm>
#include <array>
#include <map>
#include <set>

namespace irisu::shared_future {
// A key records only raw input edits before the split. Equal keys therefore
// imply exactly equal input words, without approximating game semantics.
using Key=std::array<std::uint32_t,1+2*cuda_search::maximum_mutations>;
struct Plan {
 unsigned split{};
 std::vector<unsigned> representatives,selection,depths;
 std::uint64_t independent_ticks{},planned_ticks{};
};
inline Plan build(const cuda_search::Inputs& input,unsigned requested=0){
 const unsigned count=input.candidates.size(),horizon=input.options.horizon;
 Plan best;best.independent_ticks=best.planned_ticks=std::uint64_t(count)*horizon;
 std::set<unsigned> splits;
 if(requested){if(requested>horizon)throw std::runtime_error("shared split exceeds horizon");splits.insert(requested);}
 else {
  // Lexicographic action-word order makes every common prefix contiguous.
  // Adjacent divergence depths give all group-count changes in O(N log N).
  std::vector<std::vector<cuda_search::Mutation>> edits(count);
  auto original=[&](unsigned offset){const auto tick=input.options.prefix+offset;return tick<input.replay.size()?input.replay[tick]:0U;};
  for(unsigned lane=0;lane<count;++lane){const auto& c=input.candidates[lane];for(unsigned i=0;i<c.count;++i)if(c.mutations[i].input!=original(c.mutations[i].offset))edits[lane].push_back(c.mutations[i]);std::sort(edits[lane].begin(),edits[lane].end(),[](auto a,auto b){return a.offset<b.offset;});}
  auto difference=[&](unsigned a,unsigned b){
   unsigned i=0,j=0;const auto& left=edits[a];const auto& right=edits[b];
   while(i<left.size()||j<right.size()){
    const unsigned lo=i<left.size()?left[i].offset:horizon,ro=j<right.size()?right[j].offset:horizon,offset=std::min(lo,ro);
    const unsigned av=lo==offset?left[i++].input:original(offset),bv=ro==offset?right[j++].input:original(offset);
    if(av!=bv)return std::pair<unsigned,int>{offset,av<bv?-1:1};
   }return std::pair<unsigned,int>{horizon,0};
  };
  std::vector<unsigned> order(count);for(unsigned i=0;i<count;++i)order[i]=i;
  std::sort(order.begin(),order.end(),[&](unsigned a,unsigned b){const auto diff=difference(a,b);return diff.second?diff.second<0:a<b;});
  std::map<unsigned,unsigned> changes;for(unsigned i=1;i<count;++i){const auto diff=difference(order[i-1],order[i]);if(diff.second)++changes[diff.first];}
  unsigned groups=1,best_split=0;auto cost=best.planned_ticks;
  for(const auto [offset,added]:changes){const auto candidate=std::uint64_t(groups)*offset+std::uint64_t(count)*(horizon-offset);if(offset&&candidate<cost){cost=candidate;best_split=offset;}groups+=added;}
  const auto candidate=std::uint64_t(groups)*horizon;if(candidate<cost)best_split=horizon;
  if(best_split)splits.insert(best_split);
 }
 for(unsigned split:splits){
  std::map<Key,unsigned> groups;Plan plan;plan.split=split;plan.selection.reserve(count);
  for(unsigned lane=0;lane<count;++lane){
   const auto& c=input.candidates[lane];std::vector<cuda_search::Mutation> edits;
   for(unsigned i=0;i<c.count;++i){const auto m=c.mutations[i];const auto tick=input.options.prefix+m.offset;
    const auto original=tick<input.replay.size()?input.replay[tick]:0;
    if(m.offset<split&&m.input!=original)edits.push_back(m);
   }
   std::sort(edits.begin(),edits.end(),[](auto a,auto b){return a.offset<b.offset;});Key key{};key[0]=edits.size();
   for(unsigned i=0;i<edits.size();++i){key[1+2*i]=edits[i].offset;key[2+2*i]=edits[i].input;}
   const auto [it,inserted]=groups.emplace(key,groups.size());if(inserted)plan.representatives.push_back(lane);plan.selection.push_back(it->second);
  }
  plan.independent_ticks=best.independent_ticks;plan.planned_ticks=std::uint64_t(plan.representatives.size())*split+std::uint64_t(count)*(horizon-split);
  if(requested||plan.planned_ticks<best.planned_ticks)best=std::move(plan);
 }
 // No sharing means no additional allocation, launch or clone overhead.
 if(best.planned_ticks>=best.independent_ticks){best.split=0;best.representatives.clear();best.selection.clear();}
 if(best.split){
  // Singleton prefixes save no work. Map them to root slot zero and run their
  // uninterrupted future; only groups used by multiple leaves own a cache.
  std::vector<unsigned> uses(best.representatives.size()),remap(uses.size());for(auto group:best.selection)++uses[group];
  std::vector<unsigned> reused{0};for(unsigned group=0;group<uses.size();++group)if(uses[group]>1){remap[group]=reused.size();reused.push_back(best.representatives[group]);}
  for(auto& group:best.selection)group=remap[group];best.representatives=std::move(reused);best.depths.assign(best.representatives.size(),best.split);best.depths[0]=0;
 }
 return best;
}
// Bounded cache forest. Each checkpoint is prepared independently, and a
// leaf resumes from its deepest selected exact input prefix. Preparation cost
// is included in the greedy gain and in planned/executed physics ticks.
inline Plan build_adaptive(const cuda_search::Inputs& input,unsigned max_checkpoints=128){
 const unsigned count=input.candidates.size(),horizon=input.options.horizon;
 Plan plan;plan.independent_ticks=plan.planned_ticks=std::uint64_t(count)*horizon;
 struct Group {unsigned depth,representative;std::vector<unsigned> leaves;};
 std::set<unsigned> split_set{horizon};for(const auto& c:input.candidates)for(unsigned i=0;i<c.count;++i)if(c.mutations[i].offset)split_set.insert(c.mutations[i].offset);
 std::vector<unsigned> splits(split_set.begin(),split_set.end());
 if(splits.size()>64){std::vector<unsigned> sampled;for(unsigned i=0;i<64;++i)sampled.push_back(splits[std::size_t(i)*(splits.size()-1)/63]);splits=std::move(sampled);}
 std::vector<Group> choices;
 for(unsigned depth:splits){
  std::map<Key,unsigned> groups;std::vector<Group> level;
  for(unsigned lane=0;lane<count;++lane){const auto& c=input.candidates[lane];std::vector<cuda_search::Mutation> edits;
   for(unsigned i=0;i<c.count;++i){const auto m=c.mutations[i];const auto tick=input.options.prefix+m.offset;const auto original=tick<input.replay.size()?input.replay[tick]:0;if(m.offset<depth&&m.input!=original)edits.push_back(m);}
   std::sort(edits.begin(),edits.end(),[](auto a,auto b){return a.offset<b.offset;});Key key{};key[0]=edits.size();for(unsigned i=0;i<edits.size();++i){key[1+2*i]=edits[i].offset;key[2+2*i]=edits[i].input;}
   const auto [it,inserted]=groups.emplace(key,groups.size());if(inserted)level.push_back({depth,lane,{}});level[it->second].leaves.push_back(lane);
  }
  for(auto& group:level)if(group.leaves.size()>1)choices.push_back(std::move(group));
 }
 auto gross=[](const Group& group){return std::uint64_t(group.depth)*(group.leaves.size()-1);};
 std::sort(choices.begin(),choices.end(),[&](const Group& a,const Group& b){return gross(a)>gross(b);});if(choices.size()>4096)choices.resize(4096);
 plan.representatives={0};plan.depths={0};plan.selection.assign(count,0);std::vector<unsigned> covered(count,0);
 while(plan.representatives.size()<max_checkpoints){
  std::int64_t best_gain=0;unsigned best=choices.size();
  for(unsigned i=0;i<choices.size();++i){const auto& group=choices[i];std::int64_t gain=-std::int64_t(group.depth);for(auto lane:group.leaves)if(group.depth>covered[lane])gain+=group.depth-covered[lane];if(gain>best_gain){best_gain=gain;best=i;}}
  if(best==choices.size())break;const auto& group=choices[best];const unsigned slot=plan.representatives.size();plan.representatives.push_back(group.representative);plan.depths.push_back(group.depth);plan.split=std::max(plan.split,group.depth);plan.planned_ticks-=best_gain;
  for(auto lane:group.leaves)if(group.depth>covered[lane]){covered[lane]=group.depth;plan.selection[lane]=slot;}
 }
 std::vector<unsigned> uses(plan.representatives.size()),remap(uses.size());for(auto slot:plan.selection)++uses[slot];
 std::vector<unsigned> reps{0},depths{0};for(unsigned slot=1;slot<uses.size();++slot)if(uses[slot]){remap[slot]=reps.size();reps.push_back(plan.representatives[slot]);depths.push_back(plan.depths[slot]);}
 for(auto& slot:plan.selection)slot=remap[slot];plan.representatives=std::move(reps);plan.depths=std::move(depths);plan.planned_ticks=0;plan.split=0;
 for(auto depth:plan.depths){plan.planned_ticks+=depth;plan.split=std::max(plan.split,depth);}for(auto slot:plan.selection)plan.planned_ticks+=horizon-plan.depths[slot];
 if(plan.representatives.size()==1){plan.representatives.clear();plan.depths.clear();plan.selection.clear();plan.split=0;}
 return plan;
}
} // namespace irisu::shared_future
