#pragma once
// Explicit geometry wire; independent of evolving world/shape metadata ABI.
#include <cstddef>
#include <cstdint>
#include <type_traits>
namespace irisu::cuda_native::wire {
inline constexpr std::uint32_t collision_magic_v2=0x32534e49U;
inline constexpr std::uint32_t collision_version_v2=2;
struct Header {std::uint32_t magic,version,count,case_words;};
struct ShapePacket {
  std::uint32_t kind{},vertex_count{}; // circle=0, polygon=1
  float position[2]{},R[4]{1,0,0,1},radius{};
  float vertices[16]{},normals[16]{};
};
struct CollisionCaseV2 {
  ShapePacket first{},second{};
  std::uint32_t manifold[15]{};
  std::uint32_t conservative{};
};
inline constexpr std::uint32_t collision_case_words_v2=98;
static_assert(sizeof(Header)==16);
static_assert(sizeof(ShapePacket)==164&&alignof(ShapePacket)==4);
static_assert(sizeof(CollisionCaseV2)==392&&offsetof(CollisionCaseV2,manifold)==328);
static_assert(offsetof(ShapePacket,position)==8&&offsetof(ShapePacket,R)==16);
static_assert(offsetof(ShapePacket,radius)==32&&offsetof(ShapePacket,vertices)==36&&offsetof(ShapePacket,normals)==100);
static_assert(std::is_trivially_copyable_v<CollisionCaseV2>);
} // namespace irisu::cuda_native::wire
