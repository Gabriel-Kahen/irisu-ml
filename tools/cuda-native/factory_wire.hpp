#pragma once
#include "scene_wire.hpp"
namespace irisu::cuda_native {
inline constexpr std::uint32_t factory_magic=0x464e5349,factory_version=1,factory_case_words=12;
struct FactoryInput {std::uint32_t kind;float width,height,x,y,angle,density,friction,restitution,scale;};
static_assert(sizeof(FactoryInput)==40);
template<class B,class S,class V>NATIVE_HD void visit_factory(B& b,S& s,V& v){
#define B(x) v(b.x)
  B(flags);B(position.x);B(position.y);B(rotation);B(R.col1.x);B(R.col1.y);B(R.col2.x);B(R.col2.y);
  B(position0.x);B(position0.y);B(rotation0);B(velocity.x);B(velocity.y);B(angular_velocity);B(force.x);B(force.y);
  B(torque);B(center.x);B(center.y);B(mass);B(inv_mass);B(inertia);B(inv_i);B(linear_damping);B(angular_damping);B(sleep_time);B(user_data);
#undef B
#define S(x) v(s.x)
  S(kind);S(R.col1.x);S(R.col1.y);S(R.col2.x);S(R.col2.y);S(position.x);S(position.y);
  S(friction);S(restitution);S(max_radius);S(category_bits);S(mask_bits);S(group_index);S(user_data);
  if(s.kind==ShapeKind::circle){S(local_position.x);S(local_position.y);S(radius);}
  else {S(local_centroid.x);S(local_centroid.y);S(vertex_count);S(local_obb_R.col1.x);S(local_obb_R.col1.y);S(local_obb_R.col2.x);S(local_obb_R.col2.y);
    S(local_obb_center.x);S(local_obb_center.y);S(local_obb_extents.x);S(local_obb_extents.y);
    for(unsigned i=0;i<s.vertex_count;++i){v(s.vertices[i].x);v(s.vertices[i].y);v(s.normals[i].x);v(s.normals[i].y);v(s.core_vertices[i].x);v(s.core_vertices[i].y);}}
#undef S
}
NATIVE_HD inline std::uint64_t factory_hash(const BodyState& b,const ShapeState& s){SceneHash h;visit_factory(b,s,h);return h.value;}
} // namespace irisu::cuda_native
