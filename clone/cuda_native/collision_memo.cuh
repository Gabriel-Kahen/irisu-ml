#pragma once
#include "collision.hpp"
#include <cstddef>
#include <cstring>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda/atomic>
#endif

namespace irisu::cuda_native::collision_memo {
// Only consumed geometry fields enter the key. No ABI padding, inactive
// vertices, pointers, or allocation metadata is inspected.
inline constexpr unsigned key_words=96,value_words=(sizeof(Manifold)+sizeof(StageResult))/4;
struct Key {std::uint32_t words[key_words];unsigned count{};};
struct Entry {unsigned state{};unsigned count{};std::uint64_t hash{};std::uint32_t key[key_words],value[value_words];};
struct Counters {unsigned long long calls{},hits{},misses{},inserts{},busy{},full{},bypass{};};
struct Descriptor {Entry* entries{};unsigned capacity{};Counters* counters{};unsigned counter_count{};};
NATIVE_HD inline void put(Key& key,unsigned value){key.words[key.count++]=value;}
NATIVE_HD inline void put(Key& key,float value){put(key,irisu::cuda_exact::bits_float(value));}
NATIVE_HD inline void put(Key& key,Vec2 value){put(key,value.x);put(key,value.y);}
NATIVE_HD inline bool put_shape(Key& key,const ShapeState& shape){
 put(key,unsigned(shape.kind));
 if(shape.kind==ShapeKind::circle){put(key,shape.position);put(key,shape.radius);return true;}
 if(shape.kind!=ShapeKind::polygon||!shape.vertex_count||shape.vertex_count>8)return false;
 put(key,shape.vertex_count);put(key,shape.position);put(key,shape.R.col1);put(key,shape.R.col2);
 for(unsigned i=0;i<shape.vertex_count;++i){put(key,shape.vertices[i]);put(key,shape.normals[i]);}return true;
}
NATIVE_HD inline bool make_key(Key& key,const Manifold& input,const ShapeState& a,const ShapeState& b,bool conservative){
 key.count=0;
 if(a.kind==ShapeKind::circle&&b.kind!=ShapeKind::circle)return false;
 if(a.kind!=ShapeKind::circle&&a.kind!=ShapeKind::polygon)return false;
 if(b.kind!=ShapeKind::circle&&b.kind!=ShapeKind::polygon)return false;
 if(!put_shape(key,a)||!put_shape(key,b))return false;
 for(const auto& point:input.points){put(key,point.position);put(key,point.separation);put(key,point.normal_impulse);put(key,point.tangent_impulse);put(key,point.id);}
 put(key,input.normal);put(key,std::uint32_t(input.point_count));put(key,unsigned(conservative));return key.count<=key_words;
}
NATIVE_HD inline std::uint64_t hash_key(const Key& key){
 std::uint64_t hash=0xcbf29ce484222325ULL;
 for(unsigned i=0;i<key.count;++i){hash^=key.words[i];hash*=0x100000001b3ULL;}return hash;
}
NATIVE_HD inline bool equal_key(const Key& a,const Key& b){if(a.count!=b.count)return false;for(unsigned i=0;i<a.count;++i)if(a.words[i]!=b.words[i])return false;return true;}
#ifdef __CUDACC__
static __device__ __constant__ Descriptor device_cache{};
inline cudaError_t configure(Descriptor descriptor){
 if(descriptor.entries&&(descriptor.capacity<4||(descriptor.capacity&(descriptor.capacity-1))))return cudaErrorInvalidValue;
 return cudaMemcpyToSymbol(device_cache,&descriptor,sizeof(descriptor));
}
__device__ inline Counters* counters(const Descriptor& descriptor){
 const auto thread=blockIdx.x*blockDim.x+threadIdx.x;
 return descriptor.counters&&descriptor.counter_count?descriptor.counters+thread%descriptor.counter_count:nullptr;
}
__device__ inline void count(unsigned long long* value){if(value)atomicAdd(value,1ULL);}
__device__ inline bool matches(const Entry& entry,const Key& key,std::uint64_t hash){
 const volatile Entry* stored=&entry;if(stored->hash!=hash||stored->count!=key.count)return false;
 for(unsigned i=0;i<key.count;++i)if(stored->key[i]!=key.words[i])return false;return true;
}
__device__ inline void load_value(const Entry& entry,Manifold& manifold,StageResult& status){
 const volatile auto* source=entry.value;unsigned char bytes[sizeof(Manifold)+sizeof(StageResult)];
 for(unsigned i=0;i<value_words;++i){const auto value=source[i];std::memcpy(bytes+4*i,&value,4);}
 std::memcpy(&manifold,bytes,sizeof(manifold));std::memcpy(&status,bytes+sizeof(manifold),sizeof(status));
}
__device__ inline void publish(Entry& entry,const Key& key,std::uint64_t hash,const Manifold& manifold,StageResult status){
 volatile Entry* destination=&entry;destination->hash=hash;destination->count=key.count;
 for(unsigned i=0;i<key.count;++i)destination->key[i]=key.words[i];
 unsigned char bytes[sizeof(Manifold)+sizeof(StageResult)];std::memcpy(bytes,&manifold,sizeof(manifold));std::memcpy(bytes+sizeof(manifold),&status,sizeof(status));
 for(unsigned i=0;i<value_words;++i){unsigned value;std::memcpy(&value,bytes+4*i,4);destination->value[i]=value;}
 cuda::atomic_ref<unsigned,cuda::thread_scope_device>(entry.state).store(2U,cuda::memory_order_release);
}
#endif
// Entries are immutable after publication. Collisions/busy/full tables merely
// execute the original stage; no hash-only hit and no eviction race exists.
NATIVE_HD inline StageResult collide_exact(Manifold& manifold,const ShapeState& a,const ShapeState& b,bool conservative=false){
#ifdef __CUDA_ARCH__
 const auto descriptor=device_cache;if(!descriptor.entries)return ::irisu::cuda_native::collide(manifold,a,b,conservative);
 auto* stats=counters(descriptor);if(stats)count(&stats->calls);Key key;
 if(!make_key(key,manifold,a,b,conservative)){if(stats)count(&stats->bypass);return ::irisu::cuda_native::collide(manifold,a,b,conservative);}
 const auto hash=hash_key(key);unsigned vacant=~0U;
 for(unsigned probe=0;probe<4;++probe){const unsigned index=(unsigned(hash)+probe)&(descriptor.capacity-1);auto& entry=descriptor.entries[index];const auto state=cuda::atomic_ref<unsigned,cuda::thread_scope_device>(entry.state).load(cuda::memory_order_acquire);
  if(state==2&&matches(entry,key,hash)){StageResult result;load_value(entry,manifold,result);if(stats)count(&stats->hits);return result;}
  if(!state&&vacant==~0U)vacant=index;else if(state==1&&stats)count(&stats->busy);
 }
 if(stats)count(&stats->misses);const auto result=::irisu::cuda_native::collide(manifold,a,b,conservative);
 unsigned expected=0;
 if(vacant!=~0U&&cuda::atomic_ref<unsigned,cuda::thread_scope_device>(descriptor.entries[vacant].state).compare_exchange_strong(expected,1U,cuda::memory_order_relaxed)){publish(descriptor.entries[vacant],key,hash,manifold,result);if(stats)count(&stats->inserts);}
 else if(vacant==~0U&&stats)count(&stats->full);return result;
#else
 return ::irisu::cuda_native::collide(manifold,a,b,conservative);
#endif
}
static_assert(sizeof(Manifold)%4==0&&sizeof(StageResult)%4==0);
} // namespace irisu::cuda_native::collision_memo
