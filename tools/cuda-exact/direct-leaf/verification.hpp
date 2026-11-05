#pragma once
#include "../../../clone/cuda/guest.hpp"
#ifndef __CUDA_ARCH__
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>
namespace irisu::cuda_direct {
struct LeafStats {std::uint64_t calls{},verified{},blocks{},original_ns{},typed_ns{},declined{};};
inline std::map<unsigned,LeafStats> leaf_stats;
struct LeafSummary {~LeafSummary(){for(const auto& [entry,s]:leaf_stats)std::fprintf(stderr,"scalar_leaf entry=%u calls=%llu verified=%llu blocks=%llu original_ns=%llu typed_ns=%llu declined=%llu\n",entry,(unsigned long long)s.calls,(unsigned long long)s.verified,(unsigned long long)s.blocks,(unsigned long long)s.original_ns,(unsigned long long)s.typed_ns,(unsigned long long)s.declined);}};
inline LeafSummary leaf_summary;
template<unsigned Entry,class Original,class Typed>
inline void verified_leaf(cuda_guest::Guest& g,std::uint64_t& budget,unsigned depth,Original original,Typed typed){
 if(g.pc!=Entry||g.depth||g.error||g.word_stride!=1){original(g,budget,depth);return;}
 auto& s=leaf_stats[Entry];++s.calls;const auto before=g.instructions;
#ifdef IRISU_DIRECT_LEAF_PROFILE_ONLY
 const auto begin=std::chrono::steady_clock::now();original(g,budget,depth);
 s.original_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();s.blocks+=g.instructions-before;
#else
 auto expected=g;auto expected_budget=budget;std::vector<unsigned char> arena(g.memory,g.memory+g.capacity);expected.memory=arena.data();
 auto begin=std::chrono::steady_clock::now();original(expected,expected_budget,depth);s.original_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
 begin=std::chrono::steady_clock::now();if(!typed(g,budget)){++s.declined;original(g,budget,depth);}s.typed_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();s.blocks+=g.instructions-before;
 expected.memory=g.memory;
 if(expected_budget!=budget||std::memcmp(&expected,&g,sizeof(g))||std::memcmp(arena.data(),g.memory,g.capacity)){
  std::fprintf(stderr,"scalar_leaf MISMATCH entry=%u call=%llu pc=%u/%u depth=%u/%u flags=%x/%x status=%x/%x blocks=%llu/%llu budget=%llu/%llu\n",Entry,(unsigned long long)s.calls,g.pc,expected.pc,g.depth,expected.depth,g.flags,expected.flags,g.status,expected.status,(unsigned long long)g.instructions,(unsigned long long)expected.instructions,(unsigned long long)budget,(unsigned long long)expected_budget);
  for(unsigned i=0;i<8;++i)if(g.r[i]!=expected.r[i])std::fprintf(stderr,"r%u %x/%x\n",i,g.r[i],expected.r[i]);
  for(unsigned i=0;i<8;++i)if(std::memcmp(g.fp+i,expected.fp+i,8))std::fprintf(stderr,"fp%u %.17g/%.17g\n",i,g.fp[i],expected.fp[i]);
  for(unsigned i=0;i<g.capacity;++i)if(arena[i]!=g.memory[i]){std::fprintf(stderr,"arena first difference=%x actual=%x expected=%x\n",i,g.memory[i],arena[i]);break;}
  throw std::runtime_error("scalar leaf Guest/budget/arena mismatch");
 }
 ++s.verified;
#endif
}
}
#endif
