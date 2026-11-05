// Helper-only raw guard qualification, not a physics or timing result.
#include "prefix_guard.cuh"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
namespace guard=irisu::cuda_native::prefix_guard;
struct alignas(8) World {std::uint64_t words[33*1024+1];};
struct alignas(8) Scene {std::uint64_t words[65*1024+1];};
struct alignas(8) Result {std::uint64_t words[37];};
// Field order deliberately differs from the runner's Validation layout.
struct Validation {unsigned errors{},prefix_errors{},first_section{},first_offset{~0U};};
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
struct Device {
  std::vector<void*> pointers;
  ~Device(){for(auto* p:pointers)cudaFree(p);}
  template<class T>T* allocate(){T* p{};check(cudaMalloc(&p,sizeof(T)));pointers.push_back(p);return p;}
};
int main(){try{
  Device device;auto* w=device.allocate<World>();auto* s=device.allocate<Scene>();auto* r=device.allocate<Result>();
  auto* sw=device.allocate<World>();auto* ss=device.allocate<Scene>();auto* sr=device.allocate<Result>();auto* dv=device.allocate<Validation>();
  const std::size_t sizes[]={sizeof(World),sizeof(Scene),sizeof(Result)};
  void* current[]={w,s,r};void* saved[]={sw,ss,sr};
  std::vector<unsigned char> baseline[3],changed[3];
  for(unsigned section=0;section<3;++section){baseline[section].resize(sizes[section]);
    for(std::size_t i=0;i<sizes[section];++i)baseline[section][i]=static_cast<unsigned char>((i*31U+section*73U)^((i>>8U)&255U));}
  unsigned cases=0,negative_cases=0;
  auto run=[&](Validation initial,bool mismatch,unsigned section,unsigned offset,const char* label){
    for(unsigned i=0;i<3;++i){check(cudaMemcpy(current[i],changed[i].data(),sizes[i],cudaMemcpyHostToDevice));
      check(cudaMemcpy(saved[i],baseline[i].data(),sizes[i],cudaMemcpyHostToDevice));}
    check(cudaMemcpy(dv,&initial,sizeof(initial),cudaMemcpyHostToDevice));
    check(guard::check_prefix_async(w,s,r,sw,ss,sr,dv));Validation actual;
    check(cudaMemcpy(&actual,dv,sizeof(actual),cudaMemcpyDeviceToHost));
    if(mismatch){
      if(actual.errors<=initial.errors||actual.prefix_errors<=initial.prefix_errors||
         (section&&actual.first_section!=section)||actual.first_offset!=offset)
        throw std::runtime_error(std::string("mismatch diagnostics differ: ")+label);
      ++negative_cases;
    }else if(actual.errors!=initial.errors||actual.prefix_errors!=initial.prefix_errors||
             actual.first_section!=initial.first_section||actual.first_offset!=initial.first_offset)
      throw std::runtime_error(std::string("equal guard changed validation: ")+label);
    for(unsigned i=0;i<3;++i){std::vector<unsigned char> copy(sizes[i]);
      check(cudaMemcpy(copy.data(),current[i],sizes[i],cudaMemcpyDeviceToHost));
      if(copy!=changed[i])throw std::runtime_error("guard modified current checkpoint");
      check(cudaMemcpy(copy.data(),saved[i],sizes[i],cudaMemcpyDeviceToHost));
      if(copy!=baseline[i])throw std::runtime_error("guard modified saved checkpoint");
    }
    ++cases;return actual;
  };
  auto reset=[&](){for(unsigned i=0;i<3;++i)changed[i]=baseline[i];};
  reset();run({},false,0,~0U,"equal");
  for(unsigned section=0;section<3;++section){
    std::vector<std::size_t> offsets={0,1,7,8,9,2047,2048,2049,262143,262144,262145,sizes[section]-2,sizes[section]-1};
    std::sort(offsets.begin(),offsets.end());offsets.erase(std::unique(offsets.begin(),offsets.end()),offsets.end());
    for(auto offset:offsets)if(offset<sizes[section]){
      reset();changed[section][offset]^=static_cast<unsigned char>(1U<<(offset%8));
      run({},true,section+3,unsigned(offset),"single-byte-word-or-grid-boundary");
    }
    reset();changed[section][7]^=0x80;changed[section][3]^=0x20;
    run({},true,section+3,3,"two-bytes-one-word");
  }
  reset();changed[0].back()^=1;changed[1][0]^=0x80;changed[2][8]^=1;
  const auto multiple=run({},true,0,0,"multiple-regions-independent-diagnostics");
  if(multiple.first_section<3||multiple.first_section>5)throw std::runtime_error("invalid multi-region section");
  Validation prior{7,11,1,2};reset();run(prior,false,1,2,"preserve-prior-validation");
  changed[0][100]^=1;run(prior,true,1,2,"accumulate-existing-error");
  if(guard::check_prefix_async(static_cast<const World*>(nullptr),s,r,sw,ss,sr,dv)!=cudaErrorInvalidValue)
    throw std::runtime_error("null input was not rejected");
  std::printf("{\"schema\":\"irisu-parallel-prefix-guard-qualifier-v1\",\"passed\":true,\"cases\":%u,\"mismatch_cases\":%u,\"null_input_rejected\":true,\"all_current_saved_bytes_unchanged\":true,\"prior_validation_preserved\":true,\"multi_region_diagnostics\":\"section CAS winner and independent minimum relative offset, matching existing guard\",\"scope\":\"Raw guard helper only; no game/search or performance admission\"}\n",cases,negative_cases);
  return 0;
}catch(const std::exception& error){std::fprintf(stderr,"prefix-guard-validate: %s\n",error.what());return 1;}}
