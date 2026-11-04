#pragma once
#include "velocity.cuh"
#include <cstdio>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace irisu::cuda_guest { IRISU_GUEST_HD inline void execute(Guest&,std::uint64_t); }
namespace irisu::cuda_direct {
#ifndef __CUDA_ARCH__
inline thread_local bool bypass{};
inline std::uint64_t calls{},verified{},baseline_ns{},typed_ns{},solver_blocks{},nonempty_calls{};
struct Summary {~Summary(){std::fprintf(stderr,"direct_velocity_calls=%llu verified=%llu baseline_solver_ns=%llu typed_solver_ns=%llu solver_blocks=%llu nonempty_calls=%llu\n",(unsigned long long)calls,(unsigned long long)verified,(unsigned long long)baseline_ns,(unsigned long long)typed_ns,(unsigned long long)solver_blocks,(unsigned long long)nonempty_calls);}};
inline Summary summary;
#endif
IRISU_GUEST_HD inline bool hook(cuda_guest::Guest& g) {
  if(g.pc!=0x7610 || g.depth || g.error)return false;
#ifndef __CUDA_ARCH__
  if(bypass)return false;
  ++calls;if(g.r[1]<=g.capacity-12 && g.load<std::int32_t>(g.r[1]+8)>0)++nonempty_calls;
#ifdef IRISU_DIRECT_LEAF_VERIFY
  auto expected=g;std::vector<unsigned char> memory(g.memory,g.memory+g.capacity);
  expected.memory=memory.data();const auto return_pc=expected.load<std::uint32_t>(expected.r[4]);
  expected.store<std::uint32_t>(expected.r[4],0);bypass=true;
  const auto baseline_begin=std::chrono::steady_clock::now();
  cuda_guest::execute(expected,100000000);baseline_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-baseline_begin).count();bypass=false;solver_blocks+=expected.instructions-g.instructions;
  expected.store<std::uint32_t>(g.r[4],return_pc);expected.pc=return_pc;
#endif
#endif
#if !defined(__CUDA_ARCH__) && defined(IRISU_DIRECT_LEAF_VERIFY)
  const auto typed_begin=std::chrono::steady_clock::now();
#endif
  if(!velocity(g))return false;
#if !defined(__CUDA_ARCH__) && defined(IRISU_DIRECT_LEAF_VERIFY)
  typed_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-typed_begin).count();
#endif
#ifndef __CUDA_ARCH__
#ifdef IRISU_DIRECT_LEAF_VERIFY
  expected.memory=g.memory;
  // A bytewise comparison also checks every raw FP/register/status field and
  // the entire arena, including the native function's scratch stack writes.
  if(std::memcmp(&expected,&g,sizeof(g)) || std::memcmp(memory.data(),g.memory,g.capacity)) {
    std::fprintf(stderr,"leaf mismatch call=%llu pc=%x/%x esp=%x/%x depth=%u/%u flags=%x/%x status=%x/%x instructions=%llu/%llu\n",
      (unsigned long long)calls,g.pc,expected.pc,g.r[4],expected.r[4],g.depth,expected.depth,g.flags,expected.flags,g.status,expected.status,
      (unsigned long long)g.instructions,(unsigned long long)expected.instructions);
    for(unsigned i=0;i<8;++i)if(g.r[i]!=expected.r[i])std::fprintf(stderr,"r%u %x/%x\n",i,g.r[i],expected.r[i]);
    for(unsigned i=0;i<8;++i)if(std::memcmp(g.fp+i,expected.fp+i,8))std::fprintf(stderr,"fp%u %.17g/%.17g\n",i,g.fp[i],expected.fp[i]);
    for(unsigned i=0;i<g.capacity;++i)if(memory[i]!=g.memory[i]){std::fprintf(stderr,"arena first difference=%x actual=%x expected=%x\n",i,g.memory[i],memory[i]);break;}
    throw std::runtime_error("typed leaf post-call state differs from interpreter");
  }
  ++verified;
#endif
#endif
  // execute() charges one dispatch for an intercepted callback. Internal leaf
  // block accounting above already charged the original blocks.
  --g.instructions;return true;
}
}
