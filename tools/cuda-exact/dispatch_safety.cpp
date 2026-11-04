// Host-only safety gate for generated separately compiled guest regions.
#include "../../clone/cuda/guest_program.cuh"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
using namespace irisu::cuda_guest;
namespace {
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
struct Fixture {
  std::array<unsigned char,4096> arena{};Guest guest{};
  Fixture(){guest.memory=arena.data();guest.capacity=arena.size();guest.r[4]=1024;}
};
}
int main(int argc,char** argv){
  try {
    if(argc<5)throw std::runtime_error("usage: dispatch-safety text-low text-high abort-callback memset-callback [empty-owner-pc ...]");
    const auto low=std::stoul(argv[1]),high=std::stoul(argv[2]);
    unsigned invalid=0;
    for(auto pc:{1U,static_cast<unsigned>(low-1),static_cast<unsigned>(high),UINT32_MAX}){
      Fixture f;f.guest.pc=pc;execute(f.guest,1);require(f.guest.error==bad_pc,"invalid PC did not fail bad_pc");++invalid;
    }
    for(int i=5;i<argc;++i){Fixture f;f.guest.pc=std::stoul(argv[i]);execute(f.guest,1);
      require(f.guest.error==bad_pc,"empty owner did not fail bad_pc");++invalid;}
    {Fixture f;f.guest.pc=std::stoul(argv[3]);execute(f.guest,1);
      require(f.guest.error==aborted&&f.guest.instructions==1,"below-text abort callback was blocked");}
    {Fixture f;f.guest.pc=std::stoul(argv[4]);f.guest.store<std::uint32_t>(1028,256);
      f.guest.store<std::uint32_t>(1032,37);f.guest.store<std::uint32_t>(1036,16);execute(f.guest,1);
      require(f.guest.error==ok&&f.guest.pc==0&&f.guest.instructions==1,"high callback failed");
      for(unsigned i=256;i<272;++i)require(f.arena[i]==37,"callback did not execute memset");}
    {Fixture f;f.guest.pc=low;execute(f.guest,0);require(f.guest.error==budget_exhausted,"zero budget did not fail closed");}
    // The pinned first stub calls a second function after its initial push.
    // Budget 2 therefore exhausts inside a native callee and must propagate.
    for(unsigned budget:{1U,2U}){Fixture f;f.guest.pc=low;execute(f.guest,budget);
      require(f.guest.error==budget_exhausted&&f.guest.instructions==budget,"nested budget exhaustion did not propagate");}
    {Fixture f;f.guest.pc=0;execute(f.guest,1);require(f.guest.error==ok,"normal return sentinel rejected");}
    std::printf("{\"invalid_pc_cases\":%u,\"callback_cases\":2,\"zero_budget_case\":true,\"positive_budget_cases\":2,\"normal_return_sentinel\":true,\"pass\":true}\n",invalid);
    return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"dispatch-safety: %s\n",e.what());return 1;}
}
