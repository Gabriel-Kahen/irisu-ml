#pragma once
#include <cstddef>
#include <cstdint>
namespace irisu::cuda_native::wire {
inline constexpr std::uint32_t broadphase_magic=0x3153424eU;
inline constexpr std::uint32_t native_broadphase_bytes=0x1743cU;
inline constexpr std::uint32_t native_broadphase_words=native_broadphase_bytes/4;
struct BroadHeader {std::uint32_t magic,version,count,op_words;float world[4];};
struct BroadOperation {
  std::uint32_t kind,slot,user_or_limit;
  float aabb[4];
  std::uint32_t reserved;
};
struct BroadResultHeader {std::uint32_t operation,result,query_count,event_count,snapshot_words;};
struct BroadEvent {std::uint32_t kind,first,second,pair_token;};
static_assert(sizeof(BroadHeader)==32&&sizeof(BroadOperation)==32);
static_assert(offsetof(BroadOperation,aabb)==12&&sizeof(BroadResultHeader)==20&&sizeof(BroadEvent)==16);
} // namespace irisu::cuda_native::wire
