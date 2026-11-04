#pragma once
#include "oracle_native.h"
#include "../../clone/cuda_native/world.hpp"
#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace scene_oracle {
using namespace irisu::cuda_native;
struct Body {
  std::uint32_t flags;Vec2 position;float rotation;Mat22 R;Vec2 position0;float rotation0;
  Vec2 velocity;float angular_velocity;Vec2 force;float torque;Vec2 center;
  std::uint32_t world,prev,next,shape_list;std::int32_t shape_count;
  std::uint32_t joint_list,contact_list;float mass,inv_mass,inertia,inv_i,linear_damping,angular_damping,sleep_time;
  std::uint32_t user_data;
};
struct Circle {
  std::uint32_t vptr,next;Mat22 R;Vec2 position;
  std::uint32_t type,user_data,body;float friction,restitution,max_radius;
  std::uint16_t proxy_id,category,mask;std::int16_t group;Vec2 local_position;float radius;
};
struct Polygon {
  std::uint32_t vptr,next;Mat22 R;Vec2 position;
  std::uint32_t type,user_data,body;float friction,restitution,max_radius;
  std::uint16_t proxy_id,category,mask;std::int16_t group;Vec2 local_centroid;
  Mat22 local_obb_R;Vec2 local_obb_center,local_obb_extents,vertices[8],core_vertices[8];
  std::int32_t vertex_count;Vec2 normals[8];
};
struct Edge {std::uint32_t other,contact,prev,next;};
struct Contact {
  std::uint32_t vptr,flags,prev,next;Edge edges[2];std::uint32_t shape1,shape2;
  std::int32_t manifold_count;float friction,restitution;Manifold manifold;
};
static_assert(sizeof(Body)==136&&sizeof(Circle)==76&&sizeof(Contact)==128);
static_assert(sizeof(Polygon)==300&&offsetof(Polygon,vertex_count)==232);
static_assert(offsetof(Contact,manifold)==68);
template<class T>const T& at(const void* p,std::size_t offset){return *reinterpret_cast<const T*>(static_cast<const unsigned char*>(p)+offset);}
template<class T>T* pointer(std::uint32_t p){return reinterpret_cast<T*>(std::uintptr_t(p));}
inline void copy_body(BodyState& b,const Body& n){
#define COPY(x) b.x=n.x
  COPY(flags);COPY(position);COPY(rotation);COPY(R);COPY(position0);COPY(rotation0);COPY(velocity);COPY(angular_velocity);
  COPY(force);COPY(torque);COPY(center);COPY(mass);COPY(inv_mass);COPY(inertia);COPY(inv_i);COPY(linear_damping);COPY(angular_damping);COPY(sleep_time);COPY(user_data);
#undef COPY
}
struct Api {
  NativeOracle native{};
  void* (*create)(float,float,float,float,float,float){};
  void (*destroy)(void*){};void (*step)(void*,float,int){};
  void* (*circle)(void*,float,float,float,float,float,float){};
  void* (*box)(void*,float,float,float,float,float,float,float,float){};
  void* (*triangle)(void*,float,float,float,float,float,float,float,float){};
  void (*destroy_body)(void*,void*){};
  void (*set_velocity)(void*,void*,float,float){};
  void (*set_origin)(void*,void*,float,float,float){};
  void (*set_tag)(void*,void*,void*){};
  Api(const char* path):native(oracle_open(path)){
    create=reinterpret_cast<decltype(create)>(dlsym(native.library,"b2d_world_create"));
    destroy=reinterpret_cast<decltype(destroy)>(dlsym(native.library,"b2d_world_destroy"));
    step=reinterpret_cast<decltype(step)>(dlsym(native.library,"b2d_world_step"));
    circle=reinterpret_cast<decltype(circle)>(dlsym(native.library,"b2d_world_create_circle"));
    box=reinterpret_cast<decltype(box)>(dlsym(native.library,"b2d_world_create_box"));
    triangle=reinterpret_cast<decltype(triangle)>(dlsym(native.library,"b2d_world_create_triangle"));
    destroy_body=reinterpret_cast<decltype(destroy_body)>(dlsym(native.library,"b2d_world_destroy_body"));
    set_velocity=reinterpret_cast<decltype(set_velocity)>(dlsym(native.library,"b2d_world_set_v"));
    set_origin=reinterpret_cast<decltype(set_origin)>(dlsym(native.library,"b2d_world_set_position"));
    set_tag=reinterpret_cast<decltype(set_tag)>(dlsym(native.library,"b2d_world_set_user_data"));
    if(!create||!destroy||!step||!circle||!box||!triangle||!destroy_body||!set_velocity||!set_origin||!set_tag)throw std::runtime_error("native scene API missing");
  }
  ~Api(){oracle_close(&native);}
};
struct Mapping {
  std::vector<Body*> bodies;std::vector<Circle*> shapes;std::vector<Contact*> contacts;
  template<class T>static std::uint32_t index(const std::vector<T*>& list,std::uint32_t raw){
    if(!raw)return null_index;
    for(unsigned i=0;i<list.size();++i)if(reinterpret_cast<std::uintptr_t>(list[i])==raw)return i;
    throw std::runtime_error("native pointer outside imported scene");
  }
  std::uint32_t edge(std::uint32_t raw)const{
    if(!raw)return null_index;
    for(unsigned i=0;i<contacts.size();++i)for(unsigned side=0;side<2;++side)
      if(reinterpret_cast<std::uintptr_t>(&contacts[i]->edges[side])==raw)return 2*i+side;
    throw std::runtime_error("native contact edge outside imported scene");
  }
};
inline Mapping import_scene(World& world,void* handle,std::uint64_t tick=0){
  void* native_world=*static_cast<void**>(handle);Mapping map;world.reset();world.tick=tick;
  const auto* broadphase=pointer<unsigned char>(at<std::uint32_t>(native_world,102868));
  world.bounds=at<AABB>(broadphase,95260);
  world.gravity=at<Vec2>(native_world,102980);world.allow_sleeping=at<std::uint8_t>(native_world,102988)!=0;
  if(at<std::uint32_t>(native_world,102960)||at<std::uint32_t>(native_world,102976))throw std::runtime_error("scene importer requires no joints or deferred bodies");
  for(auto* b=pointer<Body>(at<std::uint32_t>(native_world,102952));b;b=pointer<Body>(b->next)){
    if(map.bodies.size()==max_bodies)throw std::runtime_error("native scene body capacity exceeded");map.bodies.push_back(b);
    for(auto* s=pointer<Circle>(b->shape_list);s;s=pointer<Circle>(s->next)){
      if((s->type!=0&&s->type!=2)||map.shapes.size()==max_shapes)throw std::runtime_error("scene importer requires circles or polygons");map.shapes.push_back(s);
    }
  }
  for(auto* c=pointer<Contact>(at<std::uint32_t>(native_world,102956));c;c=pointer<Contact>(c->next)){
    if(map.contacts.size()==max_contacts)throw std::runtime_error("native scene contact capacity exceeded");map.contacts.push_back(c);
  }
  world.body_count=map.bodies.size();world.shape_count=map.shapes.size();world.contact_count=map.contacts.size();
  world.next_body_id=world.body_count+1;world.next_shape_id=world.shape_count+1;
  world.body_head=world.body_count?0:null_index;world.contact_head=world.contact_count?0:null_index;
  for(unsigned i=0;i<map.bodies.size();++i){
    const auto& n=*map.bodies[i];auto& b=world.bodies[i];copy_body(b,n);b.occupied=true;b.id=i+1;
    b.prev=Mapping::index(map.bodies,n.prev);b.next=Mapping::index(map.bodies,n.next);
    b.shape_head=Mapping::index(map.shapes,n.shape_list);b.contact_head=map.edge(n.contact_list);
    if(n.joint_list)throw std::runtime_error("native scene has joints");
  }
  for(unsigned i=0;i<map.shapes.size();++i){
    const auto& n=*map.shapes[i];auto& s=world.shapes[i];s.occupied=true;s.kind=n.type==0?ShapeKind::circle:ShapeKind::polygon;s.id=i+1;
    s.next=Mapping::index(map.shapes,n.next);s.body=Mapping::index(map.bodies,n.body);s.R=n.R;s.position=n.position;
    s.user_data=n.user_data;s.friction=n.friction;s.restitution=n.restitution;s.max_radius=n.max_radius;
    s.proxy_id=n.proxy_id;s.category_bits=n.category;s.mask_bits=n.mask;s.group_index=n.group;
    if(s.kind==ShapeKind::circle){s.local_position=n.local_position;s.radius=n.radius;}
    else {
      const auto& poly=*reinterpret_cast<const Polygon*>(&n);s.local_centroid=poly.local_centroid;
      s.local_obb_R=poly.local_obb_R;s.local_obb_center=poly.local_obb_center;s.local_obb_extents=poly.local_obb_extents;
      if(poly.vertex_count<0||poly.vertex_count>8)throw std::runtime_error("native polygon vertex capacity exceeded");
      s.vertex_count=poly.vertex_count;
      for(unsigned j=0;j<s.vertex_count;++j){s.vertices[j]=poly.vertices[j];s.normals[j]=poly.normals[j];s.core_vertices[j]=poly.core_vertices[j];}
    }
  }
  for(unsigned i=0;i<map.contacts.size();++i){
    const auto& n=*map.contacts[i];auto& c=world.contacts[i];c.occupied=true;c.id=i+1;c.flags=n.flags;
    c.prev=Mapping::index(map.contacts,n.prev);c.next=Mapping::index(map.contacts,n.next);
    c.shape1=Mapping::index(map.shapes,n.shape1);c.shape2=Mapping::index(map.shapes,n.shape2);
    c.friction=n.friction;c.restitution=n.restitution;c.manifold_count=n.manifold_count;c.manifold=n.manifold;
    for(unsigned side=0;side<2;++side){c.edges[side].other=Mapping::index(map.bodies,n.edges[side].other);c.edges[side].prev=map.edge(n.edges[side].prev);c.edges[side].next=map.edge(n.edges[side].next);}
  }
  if(!world.validate_snapshot())throw std::runtime_error("invalid imported native scene");
  return map;
}
} // namespace scene_oracle
