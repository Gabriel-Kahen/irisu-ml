#pragma once
// Optional read-only raw checkpoint guard. No normalization, hashing or scratch
// omission: every byte of all three object representations participates.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
namespace irisu::cuda_native::prefix_guard {
__device__ __forceinline__ std::uint64_t load_global_word64(const unsigned char* source){
  std::uint64_t word;
  asm volatile("ld.global.u64 %0, [%1];":"=l"(word):"l"(__cvta_generic_to_global(source)):"memory");
  return word;
}
template<class World,class Scene,class Result,class Validation> __global__ void compare_raw(
    const World* world,const Scene* scene,const Result* result,
    const World* saved_world,const Scene* saved_scene,const Result* saved_result,Validation* validation){
  constexpr std::size_t world_words=sizeof(World)/8,scene_words=sizeof(Scene)/8;
  constexpr std::size_t total_words=world_words+scene_words+sizeof(Result)/8;
  unsigned first[3]={~0U,~0U,~0U};
  const auto stride=std::size_t(gridDim.x)*blockDim.x;
  for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total_words;i+=stride){
    const unsigned char *current,*saved;std::size_t word;unsigned section;
    if(i<world_words){section=0;word=i;
      current=reinterpret_cast<const unsigned char*>(world);saved=reinterpret_cast<const unsigned char*>(saved_world);
    }else if(i<world_words+scene_words){section=1;word=i-world_words;
      current=reinterpret_cast<const unsigned char*>(scene);saved=reinterpret_cast<const unsigned char*>(saved_scene);
    }else{section=2;word=i-world_words-scene_words;
      current=reinterpret_cast<const unsigned char*>(result);saved=reinterpret_cast<const unsigned char*>(saved_result);
    }
    const auto different=load_global_word64(current+8*word)^load_global_word64(saved+8*word);
    if(different){
      // CUDA global u64 loads are little-endian. ffs identifies the first
      // differing BYTE, including changes confined to the high bit of a byte.
      const unsigned byte=unsigned(8*word)+unsigned((__ffsll(static_cast<long long>(different))-1)/8);
      first[section]=first[section]<byte?first[section]:byte;
    }
  }
  for(unsigned section=0;section<3;++section)if(first[section]!=~0U){
    atomicAdd(&validation->errors,1U);atomicAdd(&validation->prefix_errors,1U);
    atomicCAS(&validation->first_section,0U,section+3U);
    atomicMin(&validation->first_offset,first[section]);
  }
}

// Caller owns CUDA-global, eight-byte-aligned complete objects, including any
// padding/inactive/scratch bytes. Validation fields are unsigned 32-bit atomics.
// As in the original guard: mismatch counts count detecting threads/regions,
// not differing bytes; only zero/nonzero is invariant across launch geometries.
// first_section is the first CAS winner (IDs3/4/5); first_offset is independently
// the minimum REGION-RELATIVE byte offset. Simultaneous multi-region faults need
// not produce a paired section/offset. Existing validation values are retained.
// Initialize errors/prefix_errors/first_section=0, first_offset=~0U for a fresh
// check. The launch is stream ordered and asynchronous; it neither resets state
// nor allocates workspace nor transfers anything to the host.
template<class World,class Scene,class Result,class Validation> inline cudaError_t check_prefix_async(
    const World* world,const Scene* scene,const Result* result,
    const World* saved_world,const Scene* saved_scene,const Result* saved_result,
    Validation* validation,cudaStream_t stream=0){
  static_assert(alignof(World)>=8&&alignof(Scene)>=8&&alignof(Result)>=8);
  static_assert(sizeof(World)%8==0&&sizeof(Scene)%8==0&&sizeof(Result)%8==0);
  static_assert(sizeof(World)<=0xffffffffULL&&sizeof(Scene)<=0xffffffffULL&&sizeof(Result)<=0xffffffffULL);
  if(!world||!scene||!result||!saved_world||!saved_scene||!saved_result||!validation)return cudaErrorInvalidValue;
  constexpr std::size_t words=(sizeof(World)+sizeof(Scene)+sizeof(Result))/8;
  constexpr unsigned blocks=words<128U*256U?unsigned((words+255U)/256U):128U;
  compare_raw<<<blocks,256,0,stream>>>(world,scene,result,saved_world,saved_scene,saved_result,validation);
  return cudaGetLastError();
}
} // namespace irisu::cuda_native::prefix_guard
#endif
