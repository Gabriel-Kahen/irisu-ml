#pragma once
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#define NATIVE_HD __host__ __device__
#else
#define NATIVE_HD
#endif

namespace irisu::cuda_native {
inline constexpr std::uint32_t null_index=0xffffffffU;
inline constexpr std::uint32_t max_bodies=512,max_shapes=512,max_contacts=4096;
struct Vec2 {float x{},y{};};
struct Mat22 {Vec2 col1{1,0},col2{0,1};};
struct AABB {Vec2 min{},max{};};
enum BodyFlags : std::uint32_t {
  static_body=1,frozen_body=2,island_body=4,sleeping_body=8,allow_sleep=16,destroy_body=32
};
// All arithmetic fields represent actual native binary32 stores. Stable IDs
// and array links replace pointers; they do not define traversal order.
struct BodyState {
  Vec2 position{},velocity{};
  float rotation{},angular_velocity{},inv_mass{},inv_i{};
  Mat22 R{};
  Vec2 force{},position0{},center{};
  float torque{},rotation0{},mass{},inertia{},linear_damping{1},angular_damping{1},sleep_time{};
  std::uint32_t flags{allow_sleep},user_data{},id{};
  std::uint32_t prev{null_index},next{null_index},shape_head{null_index},contact_head{null_index};
  bool occupied{},pending_destroy{};
};
struct ConstraintPoint {
  Vec2 local_anchor1{},local_anchor2{};
  float normal_impulse{},tangent_impulse{},position_impulse{};
  float normal_mass{},tangent_mass{},separation{},velocity_bias{};
};
struct ContactConstraint {
  ConstraintPoint points[2]{};
  Vec2 normal{};
  std::uint32_t body1{null_index},body2{null_index},manifold{null_index};
  float friction{},restitution{};
  std::int32_t point_count{};
};
enum class ShapeKind : std::uint32_t {circle,polygon};
struct ShapeState {
  ShapeKind kind{};
  std::uint32_t id{},body{null_index},next{null_index},vertex_count{},user_data{};
  Vec2 local_position{},local_centroid{},vertices[8]{},normals[8]{};
  Vec2 position{};
  Mat22 R{};
  float radius{},max_radius{},friction{},restitution{};
  std::uint16_t category_bits{1},mask_bits{0xffff};
  std::uint16_t proxy_id{0xffff};
  std::int16_t group_index{};
  bool occupied{};
};
struct ManifoldPoint {
  Vec2 position{};
  float separation{},normal_impulse{},tangent_impulse{};
  std::uint32_t id{};
};
struct Manifold {ManifoldPoint points[2]{};Vec2 normal{};std::int32_t point_count{};};
// Edge references are2*contact_slot+side; this preserves each body's native
// intrusive contact-node order without pointers or a separate allocation.
struct ContactEdge {std::uint32_t other{null_index},prev{null_index},next{null_index};};
struct ContactState {
  std::uint32_t id{},shape1{null_index},shape2{null_index},flags{};
  std::uint32_t prev{null_index},next{null_index};
  ContactEdge edges[2]{};
  float friction{},restitution{};
  std::int32_t manifold_count{};
  Manifold manifold{};
  bool occupied{};
};
struct TimeStep {float dt{},inv_dt{};std::int32_t iterations{1};};
enum class Status : std::uint32_t {
  ok,capacity,invalid_body,invalid_shape,unsupported_numeric,unsupported_collision,
  unsupported_joint,bridge_required,invalid_snapshot
};
struct StageResult {
  Status status{Status::ok};
  std::uint32_t body{null_index};
  NATIVE_HD explicit operator bool() const {return status==Status::ok;}
};
static_assert(std::is_trivially_copyable_v<BodyState>);
static_assert(std::is_trivially_copyable_v<ContactConstraint>);
} // namespace irisu::cuda_native
