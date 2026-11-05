#pragma once
#include "world.hpp"
#include "world_shapes.hpp"
#include "world_contacts.hpp"
#include "broadphase.hpp"
#include "collision_memo.cuh"
#include <cstddef>
#include <cstring>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda/atomic>
#endif

namespace irisu::cuda_native {
NATIVE_HD inline StageResult freeze_body(World&,BroadPhase&,std::uint32_t);
}
namespace irisu::cuda_native::parallel_physics {
// One controller warp plus one dedicated worker warp per world. The controller
// alone mutates canonical state; workers calculate disjoint scratch outcomes.
// No warp barrier is entered through divergent game/worker call stacks.
inline constexpr unsigned worker_count=32,block_threads=64;
enum Command : unsigned {idle,collision,velocity,position,bounds,stop};
struct Stats {unsigned long long collision_calls{},velocity_calls{},position_calls{},bounds_calls{},collision_jobs{},body_jobs{},shape_jobs{};};
struct CollisionOutput {Manifold manifold{};StageResult status{};bool evaluated{},skip{};};
struct BodyOutput {BodyState body{};StageResult status{};};
struct ShapeOutput {Vec2 position{};Mat22 R{};AABB bounds{};StageResult quick{},swept{};bool bounds_written{};};
struct BodyBounds {StageResult status{};unsigned generation{};};
struct Channel {
 Stats stats{};unsigned epoch{},completed{},command{},count{},active{},prepared_epoch{};
 World* world{};const unsigned* order{};float dt{};bool canonical{};
 unsigned contacts[max_contacts]{},shape_owner[max_shapes]{},mask[(max_shapes+31)/32]{};
 CollisionOutput collision_output[max_contacts]{};BodyOutput body_output[max_bodies]{};
 ShapeOutput shape_output[max_shapes]{};BodyBounds body_bounds[max_bodies]{};
};
struct Descriptor {
 void* scene_base{};std::size_t scene_stride{};unsigned slots{};Channel* channels{};
 bool collision_enabled{true},integration_enabled{true},bounds_enabled{true};unsigned minimum_items{2};
};
inline constexpr std::size_t bytes_required(unsigned slots){return std::size_t(slots)*sizeof(Channel);}

NATIVE_HD inline CollisionOutput calculate_collision(const World& world,unsigned index){
 CollisionOutput out;
 if(index>=max_contacts){out.status={Status::invalid_snapshot};return out;}
 const auto& c=world.contacts[index];
 if(!c.occupied||c.shape1>=max_shapes||c.shape2>=max_shapes){out.status={Status::invalid_snapshot};return out;}
 if(c.flags&2){out.status={Status::bridge_required};return out;}
 const auto& a=world.shapes[c.shape1];const auto& b=world.shapes[c.shape2];
 if(a.body>=max_bodies||b.body>=max_bodies){out.status={Status::invalid_body};return out;}
 if((world.bodies[a.body].flags&sleeping_body)&&(world.bodies[b.body].flags&sleeping_body)){out.skip=true;return out;}
 std::memcpy(&out.manifold,&c.manifold,sizeof(Manifold));out.evaluated=true;
#if defined(IRISU_CUDA_NATIVE_COLLISION_CACHE) && IRISU_CUDA_NATIVE_COLLISION_CACHE
 out.status=collision_memo::collide_exact(out.manifold,a,b);
#else
 out.status=::irisu::cuda_native::collide(out.manifold,a,b);
#endif
 if(!out.status)return out;
 if(a.kind==ShapeKind::polygon&&b.kind==ShapeKind::polygon){
  bool matched[2]{};
  for(int i=0;i<out.manifold.point_count;++i){auto& point=out.manifold.points[i];point.normal_impulse=point.tangent_impulse=0;
   for(int j=0;j<c.manifold.point_count;++j)if(!matched[j]&&point.id==c.manifold.points[j].id){matched[j]=true;point.normal_impulse=c.manifold.points[j].normal_impulse;point.tangent_impulse=c.manifold.points[j].tangent_impulse;break;}
  }
 }
 return out;
}
NATIVE_HD inline void commit_collision(World& world,unsigned index,const CollisionOutput& output){
 if(!output.evaluated)return;auto& c=world.contacts[index];const auto previous=c.manifold_count;
 std::memcpy(&c.manifold,&output.manifold,sizeof(Manifold));if(!output.status)return;
 c.manifold_count=c.manifold.point_count?1:0;
 if(!previous&&c.manifold_count)connect_contact(world,index);else if(previous&&!c.manifold_count)disconnect_contact(world,index);
}
NATIVE_HD inline bool previous_rotation(const BodyState& body,bool canonical,Mat22& result,StageResult& status){
#if IRISU_CUDA_NATIVE_CACHED_ROTATION
 if(canonical&&math::sincos_input_supported(body.rotation)&&irisu::cuda_exact::bits_float(body.rotation0)==irisu::cuda_exact::bits_float(body.rotation)){result=body.R;return true;}
#endif
 const auto trig=math::sincos(body.rotation0);if(!trig.supported){status={Status::unsupported_numeric,body.id};return false;}
 result={{trig.cosine,trig.sine},{-trig.sine,trig.cosine}};return true;
}
NATIVE_HD inline ShapeOutput calculate_shape(const ShapeState& source,const BodyState& body,const Mat22& previous_R){
 ShapeOutput out;ShapeState shape{};shape.kind=source.kind;shape.radius=source.kind==ShapeKind::circle?source.radius:0;
 shape.local_position=source.kind==ShapeKind::circle?source.local_position:Vec2{};
 shape.local_centroid=source.kind==ShapeKind::circle?Vec2{}:source.local_centroid;
 if(source.kind==ShapeKind::polygon){shape.local_obb_R=source.local_obb_R;shape.local_obb_center=source.local_obb_center;shape.local_obb_extents=source.local_obb_extents;}
 out.quick=shape_quick_sync(shape,body.position,body.R);if(!out.quick)return out;out.position=shape.position;out.R=shape.R;
 if(shape.kind==ShapeKind::circle){
  // An unsupported old QuickSync leaves the destination bounds untouched.
  ShapeState previous{};previous.kind=ShapeKind::circle;previous.local_position=shape.local_position;
  const auto old=shape_quick_sync(previous,body.position0,previous_R);
  if(!old){out.swept=old;return out;}
  out.swept=circle_swept_bounds(out.bounds,shape,body.position0,previous_R);out.bounds_written=true;
 }else if(shape.kind==ShapeKind::polygon){out.swept=polygon_swept_bounds(out.bounds,shape,body.position0,previous_R,body.position,body.R);out.bounds_written=bool(out.swept);}
 else out.swept={Status::unsupported_collision};return out;
}

#ifdef __CUDACC__
static __device__ __constant__ Descriptor device_workspace{};
inline cudaError_t configure(Descriptor descriptor){
 if(descriptor.channels&&(!descriptor.scene_base||descriptor.scene_stride<sizeof(World)||!descriptor.slots))return cudaErrorInvalidValue;
 return cudaMemcpyToSymbol(device_workspace,&descriptor,sizeof(descriptor));
}
__device__ inline cuda::atomic_ref<unsigned,cuda::thread_scope_block> control(unsigned& value){return cuda::atomic_ref<unsigned,cuda::thread_scope_block>(value);}
__device__ inline Channel* channel_for(World& world){
 const auto d=device_workspace;if(!d.channels)return nullptr;
 const auto address=reinterpret_cast<std::uintptr_t>(&world),base=reinterpret_cast<std::uintptr_t>(d.scene_base);
 if(address<base||(address-base)%d.scene_stride)return nullptr;const auto slot=(address-base)/d.scene_stride;
 if(slot>=d.slots)return nullptr;auto* c=d.channels+slot;return c->active&&c->world==&world?c:nullptr;
}
// Controller thread0 calls this, then ALL64 threads execute __syncthreads()
// before branching into controller and worker paths. Reset every launch.
__device__ inline void begin_slot(unsigned slot,World* world){
 auto& c=device_workspace.channels[slot];c.stats={};c.world=world;c.order=nullptr;c.count=0;c.prepared_epoch=0;c.active=1;
 c.command=idle;control(c.completed).store(0,cuda::memory_order_relaxed);control(c.epoch).store(0,cuda::memory_order_relaxed);
}
__device__ inline void stop_slot(unsigned slot){auto& c=device_workspace.channels[slot];c.active=0;c.command=stop;control(c.epoch).fetch_add(1,cuda::memory_order_release);}
__device__ inline void request(Channel& channel,Command command){
 channel.command=command;control(channel.completed).store(0,cuda::memory_order_relaxed);
 control(channel.epoch).fetch_add(1,cuda::memory_order_release);
 while(control(channel.completed).load(cuda::memory_order_acquire)!=worker_count)__nanosleep(32);
}
__device__ inline void run_worker(unsigned slot){
 auto& c=device_workspace.channels[slot];unsigned seen=0;const unsigned worker=threadIdx.x-32;
 for(;;){unsigned epoch;while((epoch=control(c.epoch).load(cuda::memory_order_acquire))==seen)__nanosleep(32);seen=epoch;
  const auto command=Command(c.command);if(command==stop)return;auto& world=*c.world;
  for(unsigned i=worker;i<c.count;i+=worker_count){
   if(command==collision)c.collision_output[i]=calculate_collision(world,c.contacts[i]);
   else if(command==velocity||command==position){
    auto& out=c.body_output[i];const auto body=c.order[i];std::memcpy(&out.body,world.bodies+body,sizeof(BodyState));
    out.status=command==velocity?integrate_velocity(out.body,c.dt,world.gravity):integrate_position(out.body,c.dt,c.canonical);
   }else if(command==bounds){
    const auto body=c.order[i];auto copy=world.bodies[body];StageResult status{};
    if(copy.inv_mass!=0){
#if IRISU_CUDA_NATIVE_CACHED_ROTATION
     if(!c.canonical||!math::sincos_input_supported(copy.rotation))status=rotation_matrix(copy);
#else
     status=rotation_matrix(copy);
#endif
     Mat22 previous_R;if(status&&previous_rotation(copy,c.canonical,previous_R,status))
      for(auto shape=copy.shape_head;shape!=null_index;shape=world.shapes[shape].next)c.shape_output[shape]=calculate_shape(world.shapes[shape],copy,previous_R);
    }
    c.body_bounds[body]={status,epoch};
   }
  }
  control(c.completed).fetch_add(1,cuda::memory_order_acq_rel);
 }
}
__device__ inline bool valid_order(Channel& channel,const unsigned* order,unsigned count){
 if(count>max_bodies)return false;for(auto& word:channel.mask)word=0;
 for(unsigned i=0;i<count;++i){const auto body=order[i];if(body>=max_bodies||(channel.mask[body/32]>>(body%32)&1U))return false;channel.mask[body/32]|=1U<<(body%32);}return true;
}
#endif
NATIVE_HD inline void try_collide(World& world,StageResult& result,bool& handled){
 handled=false;
#ifdef __CUDA_ARCH__
 auto* c=channel_for(world);if(!c)return;c->prepared_epoch=0;if(!device_workspace.collision_enabled)return;
 unsigned count=0;StageResult tail{};
 for(auto index=world.contact_head;index!=null_index;index=world.contacts[index].next){if(index>=max_contacts||count==max_contacts){tail={Status::invalid_snapshot};break;}c->contacts[count++]=index;}
 if(count<device_workspace.minimum_items&&tail)return;
 handled=true;c->count=count;if(count){request(*c,collision);++c->stats.collision_calls;c->stats.collision_jobs+=count;}
 for(unsigned i=0;i<count;++i){const auto& out=c->collision_output[i];if(out.skip)continue;commit_collision(world,c->contacts[i],out);if(!out.status){result=out.status;return;}}
 result=tail;
#endif
}
NATIVE_HD inline void try_integrate_velocity(World& world,const unsigned* order,unsigned count,float dt,StageResult& result,bool& handled){
 handled=false;
#ifdef __CUDA_ARCH__
 auto* c=channel_for(world);if(!c)return;c->prepared_epoch=0;
 if(!device_workspace.integration_enabled||count<device_workspace.minimum_items||!valid_order(*c,order,count))return;
 c->order=order;c->count=count;c->dt=dt;request(*c,velocity);handled=true;++c->stats.velocity_calls;c->stats.body_jobs+=count;
 for(unsigned i=0;i<count;++i){const auto& out=c->body_output[i];std::memcpy(world.bodies+order[i],&out.body,sizeof(BodyState));if(!out.status){result=out.status;return;}}result={};
#endif
}
NATIVE_HD inline void try_integrate_position(World& world,const unsigned* order,unsigned count,float dt,bool canonical,StageResult& result,bool& handled){
 handled=false;
#ifdef __CUDA_ARCH__
 auto* c=channel_for(world);if(!c||!device_workspace.integration_enabled||count<device_workspace.minimum_items||!valid_order(*c,order,count))return;
 c->order=order;c->count=count;c->dt=dt;c->canonical=canonical;request(*c,position);handled=true;++c->stats.position_calls;c->stats.body_jobs+=count;
 for(unsigned i=0;i<count;++i){const auto& out=c->body_output[i];std::memcpy(world.bodies+order[i],&out.body,sizeof(BodyState));if(!out.status){result=out.status;return;}}result={};
#endif
}
NATIVE_HD inline void prepare_shape_bounds(World& world,const unsigned* order,unsigned count,bool canonical){
#ifdef __CUDA_ARCH__
 auto* c=channel_for(world);if(!c)return;c->prepared_epoch=0;
 if(!device_workspace.bounds_enabled||count<device_workspace.minimum_items||!valid_order(*c,order,count))return;
 for(auto& word:c->mask)word=0;unsigned shapes=0;
 for(unsigned i=0;i<count;++i){const auto body=order[i];if(world.bodies[body].inv_mass==0)continue;
  for(auto shape=world.bodies[body].shape_head;shape!=null_index;shape=world.shapes[shape].next){
   if(shape>=max_shapes||!world.shapes[shape].occupied||(c->mask[shape/32]>>(shape%32)&1U))return;
   c->mask[shape/32]|=1U<<(shape%32);++shapes;
  }
 }
 c->order=order;c->count=count;c->canonical=canonical;request(*c,bounds);c->prepared_epoch=control(c->epoch).load(cuda::memory_order_relaxed);++c->stats.bounds_calls;c->stats.shape_jobs+=shapes;
#endif
}
NATIVE_HD inline void try_sync_body_shapes(World& world,AABB* swept,unsigned body,BroadPhase* broadphase,bool force_static,bool canonical,StageResult& result,bool& handled){
 handled=false;
#ifdef __CUDA_ARCH__
 auto* c=channel_for(world);if(!c||force_static||!c->prepared_epoch||c->body_bounds[body].generation!=c->prepared_epoch)return;
 handled=true;c->body_bounds[body].generation=0;result=c->body_bounds[body].status;if(!result)return;
 for(auto shape=world.bodies[body].shape_head;shape!=null_index;shape=world.shapes[shape].next){auto& target=world.shapes[shape];const auto& out=c->shape_output[shape];
  result=out.quick;if(!result)return;target.R=out.R;target.position=out.position;if(broadphase&&target.proxy_id==0xffff)continue;
  if(out.bounds_written)swept[shape]=out.bounds;result=out.swept;if(!result)return;const auto& box=swept[shape];
  if(broadphase){if(broadphase->InRange(box))broadphase->MoveProxy(target.proxy_id,box);else {result=::irisu::cuda_native::freeze_body(world,*broadphase,body);if(!result)return;}
   if(broadphase->error!=Status::ok){result={broadphase->error};return;}
  }else if(box.min.x<=world.bounds.min.x||box.min.y<=world.bounds.min.y||box.max.x>=world.bounds.max.x||box.max.y>=world.bounds.max.y){result={Status::bridge_required};return;}
 }
 result={};
#endif
}
} // namespace irisu::cuda_native::parallel_physics
