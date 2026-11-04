#pragma once
// Optional transport/scheduler helpers only. Result bits and physics are untouched.
#include "../../clone/cuda/search.hpp"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cub/device/device_merge_sort.cuh>
#define IRISU_ACCEL_HD __host__ __device__
#else
#define IRISU_ACCEL_HD
#endif

namespace irisu::cuda_native::search_accel {
// Index makes this a strict total order even for identical Result keys. CUB's
// non-stable merge sort therefore produces the same canonical rank permutation.
struct IndexLess {
  const cuda_search::Result* results{};
  IRISU_ACCEL_HD bool operator()(unsigned a,unsigned b) const {
    return cuda_search::better(results[a],a,results[b],b);
  }
};

#ifdef __CUDACC__
template<class Unused=void> __global__ void initialize_indices(unsigned* order,unsigned count){
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)order[i]=i;
}
// Query once after device selection. Caller owns/reuses workspace and order;
// neither may overlap Results. Null pointers are allowed for a zero-count no-op.
inline cudaError_t rank_workspace_bytes(const cuda_search::Result* results,unsigned* order,
                                       unsigned count,std::size_t& bytes,cudaStream_t stream=0){
  bytes=0;if(count<2)return cudaSuccess;
  if(!results||!order||count>0x7fffffffU)return cudaErrorInvalidValue;
  return cub::DeviceMergeSort::SortKeys(nullptr,bytes,order,count,IndexLess{results},stream);
}
// Stream ordered, asynchronous, no allocation or synchronization. Initialize on
// every invocation: the caller must not rely on the preceding batch's ordering.
inline cudaError_t rank_async(const cuda_search::Result* results,unsigned* order,unsigned count,
                             void* workspace,std::size_t workspace_bytes,cudaStream_t stream=0){
  if(!count)return cudaSuccess;
  if(!results||!order||count>0x7fffffffU||(count>1&&(!workspace||!workspace_bytes)))return cudaErrorInvalidValue;
  initialize_indices<><<<(count+127U)/128U,128,0,stream>>>(order,count);
  auto status=cudaGetLastError();if(status!=cudaSuccess||count==1)return status;
  return cub::DeviceMergeSort::SortKeys(workspace,workspace_bytes,order,count,IndexLess{results},stream);
}

__device__ __forceinline__ void copy_global_word64(unsigned char* destination,const unsigned char* source){
  std::uint64_t word;
  asm volatile("ld.global.u64 %0, [%1];":"=l"(word):"l"(__cvta_generic_to_global(source)):"memory");
  asm volatile("st.global.u64 [%0], %1;"::"l"(__cvta_generic_to_global(destination)),"l"(word):"memory");
}
template<class World,class Scene,unsigned tile_bytes> __global__ void copy_tiles(
    World* worlds,Scene* scenes,const World* source_worlds,const Scene* source_scenes,
    unsigned count,const unsigned* selection,std::size_t scene_bytes,std::size_t tiles_per_clone){
  const std::size_t lane=blockIdx.x/tiles_per_clone,tile=blockIdx.x%tiles_per_clone;
  if(lane>=count)return;
  const unsigned source=selection?selection[lane]:0;
  auto* w=reinterpret_cast<unsigned char*>(worlds+lane);
  auto* s=reinterpret_cast<unsigned char*>(scenes+lane);
  const auto* sw=reinterpret_cast<const unsigned char*>(source_worlds+source);
  const auto* ss=reinterpret_cast<const unsigned char*>(source_scenes+source);
  constexpr std::size_t world_words=sizeof(World)/8,tile_words=tile_bytes/8;
  const std::size_t first=tile*tile_words,total=world_words+scene_bytes/8;
  const std::size_t end=first+tile_words<total?first+tile_words:total;
  for(std::size_t i=first+threadIdx.x;i<end;i+=blockDim.x){
    if(i<world_words)copy_global_word64(w+8*i,sw+8*i);
    else copy_global_word64(s+8*(i-world_words),ss+8*(i-world_words));
  }
}
// Copy only; caller MUST launch the existing pointer-rebind kernel afterward
// on this stream, before any destination is read or simulated. Kernel completion
// is the global copy/rebind boundary; no in-kernel cross-block barrier is used.
// Source/destination arrays must be disjoint, CUDA-global allocations. Selection
// indices must be valid, and scene_bytes must be an independently admitted prefix
// (full Scene or qualified physical prefix). Uncopied scratch stays untouched.
template<class World,class Scene,unsigned tile_bytes=32768> inline cudaError_t copy_tiled_async(
    World* worlds,Scene* scenes,const World* source_worlds,const Scene* source_scenes,
    unsigned count,const unsigned* selection,std::size_t scene_bytes,cudaStream_t stream=0){
  static_assert(alignof(World)>=8&&alignof(Scene)>=8&&sizeof(World)%8==0&&sizeof(Scene)%8==0);
  static_assert(tile_bytes>=8&&tile_bytes%8==0);
  if(!count)return cudaSuccess;
  if(!worlds||!scenes||!source_worlds||!source_scenes||!scene_bytes||scene_bytes>sizeof(Scene)||scene_bytes%8)
    return cudaErrorInvalidValue;
  const auto tiles=(sizeof(World)+scene_bytes+tile_bytes-1)/tile_bytes;
  const std::uint64_t blocks=std::uint64_t(count)*tiles;
  if(blocks>0x7fffffffULL)return cudaErrorInvalidConfiguration;
  copy_tiles<World,Scene,tile_bytes><<<unsigned(blocks),256,0,stream>>>(
    worlds,scenes,source_worlds,source_scenes,count,selection,scene_bytes,tiles);
  return cudaGetLastError();
}
#endif
} // namespace irisu::cuda_native::search_accel
#undef IRISU_ACCEL_HD
