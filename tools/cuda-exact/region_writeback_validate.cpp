// Host differential gate for generated region caching. Compile against both
// cache modes and compare stdout; normal physics is checked by physics_probe.
#include "../../clone/cuda/guest_program.cuh"
#include "image_identity.hpp"
#include <bit>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>
using namespace irisu::cuda_guest;

std::uint64_t hash_memory(const std::vector<unsigned char>& memory) {
  std::uint64_t hash=14695981039346656037ULL;
  for(auto byte:memory)hash=(hash^byte)*1099511628211ULL;
  return hash;
}
void report(const char* name,const Guest& g,const std::vector<unsigned char>& memory) {
  std::printf("%s",name);
  for(auto value:g.r)std::printf(" %08x",value);
  for(auto value:g.fp)std::printf(" %016llx",static_cast<unsigned long long>(std::bit_cast<std::uint64_t>(value)));
  const unsigned values[]={g.top,g.depth,g.status,g.control,g.flags,g.pc,g.error,g.fault_pc,
    g.heap,g.heap_limit,g.capacity,g.free_head,g.allocation_reuses};
  for(auto value:values)std::printf(" %08x",value);
  std::printf(" %016llx %llu %016llx\n",static_cast<unsigned long long>(std::bit_cast<std::uint64_t>(g.fault_value)),
    static_cast<unsigned long long>(g.instructions),static_cast<unsigned long long>(hash_memory(memory)));
}
IRISU_GUEST_HD void exercise(Guest& g,unsigned test) {
  if(test<5) {
      const unsigned args[]={float_word(-100),float_word(-100),float_word(100),float_word(100),float_word(10),float_word(1)};
      g.r[4]=g.capacity-16;for(unsigned i=6;i;--i)g.push(args[i-1]);g.push(0);g.pc=msvc_b2d_world_create;
      const unsigned budgets[]={0,1,2,8,8};
      if(test==4)g.r[4]=1;
      execute(g,budgets[test]);

    } else {
      auto f=[] IRISU_GUEST_HD (float value){return float_word(value);};
      const auto world=call(g,msvc_b2d_world_create,f(-100),f(-100),f(100),f(100),f(10),f(1));
      const auto body=call(g,msvc_b2d_world_create_circle,world,f(.5f),f(0),f(0),f(1),f(.3f),f(.1f));
      call(g,msvc_b2d_world_set_v,world,body,0x7fc00000u,f(0));
      call(g,msvc_b2d_world_step,world,f(1.0f/60),10);

    }
}

#ifndef IRISU_REGION_WRITEBACK_NO_MAIN
int main(int argc,char** argv) {
 try {
  if(argc!=2)return 2;
  std::ifstream stream(argv[1],std::ios::binary);
  const std::vector<unsigned char> image((std::istreambuf_iterator<char>(stream)),{});
  irisu::cuda_exact::verify_image(image.data(),image.size());
  for(unsigned test=0;test<6;++test) {
    std::vector<unsigned char> memory(524288);std::copy(image.begin(),image.end(),memory.begin());
    Guest g{};g.memory=memory.data();g.capacity=memory.size();g.heap=0x20000;g.heap_limit=g.capacity-0x10000;
    exercise(g,test);
    const auto expected=test==4?bad_address:test==5?unsupported_fp:budget_exhausted;
    if(g.error!=expected)throw std::runtime_error("fault/budget gate did not exercise expected failure");
    report(test==4?"bad-stack":test==5?"unsupported-fp":"budget",g,memory);
  }
  return 0;
 }catch(const std::exception& error){std::fprintf(stderr,"region-writeback: %s\n",error.what());return 1;}
}

#endif
