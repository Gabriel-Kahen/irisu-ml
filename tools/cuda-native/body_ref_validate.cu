// Standalone qualifier always selects the wide DEVICE path. The HOST helper
// still uses its original memcpy path because __CUDA_ARCH__ is not defined.
#if !defined(IRISU_CUDA_NATIVE_WIDE_REBIND) || !IRISU_CUDA_NATIVE_WIDE_REBIND
#undef IRISU_CUDA_NATIVE_WIDE_REBIND
#define IRISU_CUDA_NATIVE_WIDE_REBIND 1
#endif
#include "../../clone/cuda_native/game_adapter.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace game=irisu::cuda_game;
using Refs=game::vector<game::Body*>;
struct Case {Refs refs;std::uintptr_t old_base{},new_base{};unsigned live_only{};};
static_assert(sizeof(void*)==8&&sizeof(std::uintptr_t)==8&&alignof(Refs)>=8&&sizeof(Refs)%8==0);
static_assert(std::is_trivially_copyable_v<Refs>&&std::is_trivially_copyable_v<Case>);
__global__ void qualify(const Case* cases,Refs* output,unsigned* returned,unsigned count){
  const unsigned i=blockIdx.x;if(i>=count)return;
  const auto* source=reinterpret_cast<const unsigned char*>(&cases[i].refs);
  auto* destination=reinterpret_cast<unsigned char*>(output+i);
  // Preserve padding/header bytes as well as pointer storage before relocation.
  for(std::size_t word=threadIdx.x;word<sizeof(Refs)/8;word+=blockDim.x){
    std::uint64_t value;
    asm volatile("ld.global.u64 %0, [%1];":"=l"(value):"l"(__cvta_generic_to_global(source+8*word)):"memory");
    asm volatile("st.global.u64 [%0], %1;"::"l"(__cvta_generic_to_global(destination+8*word)),"l"(value):"memory");
  }
  __syncthreads();
  if(!threadIdx.x)returned[i]=game::relocate_native_body_refs(output[i],cases[i].old_base,cases[i].new_base,cases[i].live_only!=0);
}
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
void put(Refs& refs,unsigned i,std::uintptr_t word){std::memcpy(refs.storage+i*8,&word,8);}
std::uintptr_t get(const Refs& refs,unsigned i){std::uintptr_t word;std::memcpy(&word,refs.storage+i*8,8);return word;}
struct Device {
  std::vector<void*> pointers;
  ~Device(){for(auto* p:pointers)cudaFree(p);}
  template<class T>T* allocate(std::size_t count){T* p{};check(cudaMalloc(&p,count*sizeof(T)));pointers.push_back(p);return p;}
};
int main(){try{
  constexpr std::uintptr_t old_base=0x100000000ULL,new_base=0x900000000ULL;
  constexpr std::size_t stride=sizeof(game::Body),capacity=Refs::capacity,span=stride*capacity;
  const unsigned counts[]={0,1,511,512,513},slots[]={0,1,255,510,511};
  std::vector<Case> cases;std::vector<std::string> labels;std::vector<unsigned> partial_stop;
  cases.reserve(4096);labels.reserve(4096);partial_stop.reserve(4096);
  auto add=[&](const std::string& label,unsigned count,bool live,std::uintptr_t old=0x100000000ULL,std::uintptr_t next=0x900000000ULL)->Case&{
    cases.emplace_back();auto& c=cases.back();
    std::memset(static_cast<void*>(&c),int((cases.size()*37U+0x42U)&255U),sizeof(c));
    c.refs.count=count;c.refs.overflow=(cases.size()&1U)!=0;c.old_base=old;c.new_base=next;c.live_only=live;
    for(unsigned i=0;i<capacity;++i)put(c.refs,i,old+i*stride);
    labels.push_back(label);partial_stop.push_back(~0U);return c;
  };
  const std::uintptr_t edges[]={0,old_base,old_base+stride,old_base+(capacity-1)*stride,
    old_base-1,old_base+span,old_base+span+1,old_base+1,old_base+stride-1,old_base+span-1,
    new_base,std::numeric_limits<std::uintptr_t>::max(),0x7ff8000000000001ULL,old_base+8,1};
  for(unsigned count:counts)for(bool live:{false,true}){
    add("all-aligned",count,live);
    auto& zero=add("all-null",count,live);for(unsigned i=0;i<capacity;++i)put(zero.refs,i,0);
    for(unsigned slot:slots)for(auto word:edges){auto& c=add("edge-word",count,live);put(c.refs,slot,word);}
  }
  // Every possible early-return position, with valid words both before and
  // after it. Earlier words MUST relocate, later words MUST remain original.
  for(bool live:{false,true})for(unsigned bad=0;bad<capacity;++bad){
    auto& c=add("exhaustive-partial-mutation",512,live);put(c.refs,bad,old_base+bad*stride+1);partial_stop.back()=bad;
  }
  for(bool live:{false,true}){
    auto& inactive=add("inactive-garbage",0,live);
    for(unsigned i=0;i<capacity;++i)put(inactive.refs,i,i%3==0?0:i%3==1?old_base+i*stride:0xffff800000000001ULL);
    auto& multiple=add("two-invalid-preserve-first-stop",512,live);put(multiple.refs,7,old_base+7*stride+1);put(multiple.refs,511,1);partial_stop.back()=7;
    add("same-base",512,live,old_base,old_base);
    add("zero-old-base",512,live,0,new_base);
    add("high-old-base",512,live,std::numeric_limits<std::uintptr_t>::max()-span+1,new_base);
    add("wrapped-new-base",512,live,old_base,std::numeric_limits<std::uintptr_t>::max()-2*stride+1);
  }
  const auto bytes=cases.size()*sizeof(Case);std::vector<unsigned char> original(bytes);
  std::memcpy(original.data(),cases.data(),bytes);
  std::vector<Refs> expected(cases.size()),actual(cases.size());std::vector<unsigned> returns(cases.size()),device_returns(cases.size());
  unsigned rejected=0,partial_cases=0;
  for(unsigned i=0;i<cases.size();++i){const auto& c=cases[i];std::memcpy(&expected[i],&c.refs,sizeof(Refs));
    returns[i]=game::relocate_native_body_refs(expected[i],c.old_base,c.new_base,c.live_only!=0);rejected+=!returns[i];
    if(partial_stop[i]!=~0U){++partial_cases;if(returns[i])throw std::runtime_error("partial mutation fixture did not reject");
      for(unsigned slot=0;slot<capacity;++slot){const auto wanted=slot<partial_stop[i]?c.new_base+slot*stride:get(c.refs,slot);
        if(get(expected[i],slot)!=wanted)throw std::runtime_error("host reference violated sequential partial mutation order");}
    }
    if(c.live_only&&c.refs.count>capacity&&(returns[i]||std::memcmp(&expected[i],&c.refs,sizeof(Refs))))
      throw std::runtime_error("invalid live count must reject before mutation");
    if(!c.live_only&&c.refs.count==513&&labels[i]=="all-null"&&!returns[i])
      throw std::runtime_error("full-capacity mode unexpectedly changed count513 behavior");
  }
  if(std::memcmp(original.data(),cases.data(),bytes))throw std::runtime_error("host reference modified source cases");
  Device device;auto* input=device.allocate<Case>(cases.size());auto* output=device.allocate<Refs>(cases.size());auto* result=device.allocate<unsigned>(cases.size());
  check(cudaMemcpy(input,cases.data(),bytes,cudaMemcpyHostToDevice));check(cudaMemset(output,0xcc,actual.size()*sizeof(Refs)));
  qualify<<<unsigned(cases.size()),256>>>(input,output,result,unsigned(cases.size()));check(cudaGetLastError());
  check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(Refs),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(device_returns.data(),result,device_returns.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
  std::vector<unsigned char> source_after(bytes);check(cudaMemcpy(source_after.data(),input,bytes,cudaMemcpyDeviceToHost));
  if(source_after!=original)throw std::runtime_error("GPU qualifier modified source cases");
  for(unsigned i=0;i<cases.size();++i)if(returns[i]!=device_returns[i]||std::memcmp(&expected[i],&actual[i],sizeof(Refs))){
    const auto* a=reinterpret_cast<const unsigned char*>(&actual[i]);const auto* e=reinterpret_cast<const unsigned char*>(&expected[i]);
    std::size_t byte=0;while(byte<sizeof(Refs)&&a[byte]==e[byte])++byte;
    std::fprintf(stderr,"case %u (%s), count%u live%u: bool host/device%u/%u first_vector_byte%zu\n",
      i,labels[i].c_str(),cases[i].refs.count,cases[i].live_only,returns[i],device_returns[i],byte);return 1;
  }
  std::printf("{\"schema\":\"irisu-wide-body-reference-qualifier-v1\",\"passed\":true,\"wide_device_path_forced\":true,\"host_reference\":\"original memcpy helper\",\"cases\":%zu,\"rejected_cases\":%u,\"partial_mutation_cases\":%u,\"capacity\":%zu,\"body_bytes\":%zu,\"vector_bytes\":%zu,\"all_raw_vector_header_padding_bytes_equal\":true,\"source_case_bytes_unchanged\":true,\"scope\":\"Synthetic integer pointer words never dereferenced; relocation transport and existing validation behavior only, not live-world admission or performance\"}\n",
    cases.size(),rejected,partial_cases,capacity,stride,sizeof(Refs));
  return 0;
}catch(const std::exception& error){std::fprintf(stderr,"body-ref-validate: %s\n",error.what());return 1;}}
