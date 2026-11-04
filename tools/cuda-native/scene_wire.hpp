#pragma once
#include "../../clone/cuda_native/world.hpp"

namespace irisu::cuda_native {
// Versioned semantic words, never sizeof(BodyState)/sizeof(ShapeState). This
// visitor also defines the pointer-free host/device comparison digest.
template<class W,class V> NATIVE_HD void visit_scene(W& w,V& v){
  v(w.gravity.x);v(w.gravity.y);v(w.bounds.min.x);v(w.bounds.min.y);v(w.bounds.max.x);v(w.bounds.max.y);
  v(w.body_head);v(w.destroy_head);v(w.contact_head);v(w.body_count);v(w.shape_count);v(w.contact_count);
  v(w.next_body_id);v(w.next_shape_id);v(w.tick);v(w.allow_sleeping);v(w.status);
  for(auto& b:w.bodies){v(b.occupied);if(!b.occupied)continue;
#define F(x) v(b.x)
    F(position.x);F(position.y);F(velocity.x);F(velocity.y);F(rotation);F(angular_velocity);F(inv_mass);F(inv_i);
    F(R.col1.x);F(R.col1.y);F(R.col2.x);F(R.col2.y);F(force.x);F(force.y);F(position0.x);F(position0.y);
    F(center.x);F(center.y);F(torque);F(rotation0);F(mass);F(inertia);F(linear_damping);F(angular_damping);F(sleep_time);
    F(flags);F(user_data);F(id);F(prev);F(next);F(shape_head);F(contact_head);F(pending_destroy);
#undef F
  }
  for(auto& s:w.shapes){v(s.occupied);if(!s.occupied)continue;
#define F(x) v(s.x)
    F(kind);F(id);F(body);F(next);F(vertex_count);F(user_data);F(local_position.x);F(local_position.y);F(local_centroid.x);F(local_centroid.y);
    F(position.x);F(position.y);F(R.col1.x);F(R.col1.y);F(R.col2.x);F(R.col2.y);
    for(auto& p:s.vertices){v(p.x);v(p.y);}for(auto& p:s.normals){v(p.x);v(p.y);}
    F(radius);F(max_radius);F(friction);F(restitution);F(category_bits);F(mask_bits);F(proxy_id);F(group_index);
#undef F
  }
  for(auto& c:w.contacts){v(c.occupied);if(!c.occupied)continue;
#define F(x) v(c.x)
    F(id);F(shape1);F(shape2);F(flags);F(prev);F(next);F(friction);F(restitution);F(manifold_count);
    for(auto& e:c.edges){v(e.other);v(e.prev);v(e.next);}
    for(auto& p:c.manifold.points){v(p.position.x);v(p.position.y);v(p.separation);v(p.normal_impulse);v(p.tangent_impulse);v(p.id);}
    F(manifold.normal.x);F(manifold.normal.y);F(manifold.point_count);
#undef F
  }
}
struct SceneHash {
  std::uint64_t value{14695981039346656037ULL};
  NATIVE_HD void word(std::uint32_t w){for(unsigned i=0;i<4;++i){value^=(w>>(8*i))&255U;value*=1099511628211ULL;}}
  template<class T>NATIVE_HD void operator()(const T& v){
    if constexpr(std::is_same_v<T,float>){union{float f;std::uint32_t u;}b{v};word(b.u);}
    else if constexpr(sizeof(T)==8){word(std::uint32_t(v));word(std::uint32_t(std::uint64_t(v)>>32));}
    else word(std::uint32_t(v));
  }
};
NATIVE_HD inline std::uint64_t scene_hash(const World& world){SceneHash hash;visit_scene(world,hash);return hash.value;}
} // namespace irisu::cuda_native

#include <vector>
#include <stdexcept>
namespace irisu::cuda_native {
inline constexpr std::uint32_t scene_wire_magic=0x4e435349,scene_wire_version=1;
struct SceneEncode {
  std::vector<std::uint32_t>& words;
  template<class T>void operator()(const T& v){
    if constexpr(std::is_same_v<T,float>){union{float f;std::uint32_t u;}b{v};words.push_back(b.u);}
    else if constexpr(sizeof(T)==8){words.push_back(std::uint32_t(v));words.push_back(std::uint32_t(std::uint64_t(v)>>32));}
    else words.push_back(std::uint32_t(v));
  }
};
struct SceneDecode {
  const std::vector<std::uint32_t>& words;std::size_t offset{};
  std::uint32_t take(){if(offset==words.size())throw std::runtime_error("truncated scene words");return words[offset++];}
  template<class T>void operator()(T& v){
    const auto a=take();
    if constexpr(std::is_same_v<T,float>){union{std::uint32_t u;float f;}b{a};v=b.f;}
    else if constexpr(sizeof(T)==8)v=T(std::uint64_t(a)|(std::uint64_t(take())<<32));
    else v=T(a);
  }
};
inline std::vector<std::uint32_t> encode_scene(const World& world){std::vector<std::uint32_t> words;SceneEncode v{words};visit_scene(world,v);return words;}
inline void decode_scene(World& world,const std::vector<std::uint32_t>& words){
  world.reset();SceneDecode v{words};visit_scene(world,v);
  if(v.offset!=words.size()||!world.validate_snapshot())throw std::runtime_error("invalid scene snapshot words");
}
} // namespace irisu::cuda_native
