#pragma once
#include "../cuda/game_types.cuh"
#include "world_scene.hpp"
#if defined(IRISU_CUDA_NATIVE_ENTRY_INDEX) && IRISU_CUDA_NATIVE_ENTRY_INDEX
#include "entry_index.cuh"
#endif
#include <cstring>
#ifdef __CUDACC__
#define NATIVE_GAME_CALL __noinline__
#else
#define NATIVE_GAME_CALL
#endif

namespace irisu::cuda_game {
GAME_HD inline bool relocate_native_body_refs(vector<Body*>& refs,std::uintptr_t old_base,std::uintptr_t new_base,bool live_only=false){
  if(live_only&&refs.count>refs.capacity)return false;
  const auto count=live_only?refs.count:refs.capacity;
  for(unsigned i=0;i<count;++i){
    std::uintptr_t word{};std::memcpy(&word,refs.storage+i*sizeof(Body*),sizeof(word));
    if(word>=old_base&&word-old_base<sizeof(Body)*vector<Body>::capacity){
      if((word-old_base)%sizeof(Body))return false;
      word=new_base+(word-old_base);std::memcpy(refs.storage+i*sizeof(Body*),&word,sizeof(word));
    }else if(i<refs.count&&word)return false;
  }
  return true;
}
// Independent typed physics adapter for the existing generated game rules.
// Binding is external so snapshots can relocate without copying a host pointer.
class NativePhysicsWorld {
  using Scene=irisu::cuda_native::ScenePipelineState;
  using NativeVec=irisu::cuda_native::Vec2;
 public:
  struct Entry {
    BodyId id{};std::uint32_t slot{};Shape shape{};
    double size{},density{},friction{},restitution{};
    Vec2 position{},velocity{};double angle{};bool pending{};
  };
 private:
  MechanicsConfig config_{};Scene* scene_{};
  vector<Entry> entries_{};vector<Contact> contacts_{};vector<Body*> ordered_{};
#if defined(IRISU_CUDA_NATIVE_ENTRY_INDEX) && IRISU_CUDA_NATIVE_ENTRY_INDEX
  EntryIndex entry_index_{};
  GAME_HD void rebuild_entry_index(){
    entry_index_.rebuild();
    for(unsigned i=0;i<entries_.count;++i)if(!entry_index_.insert(entries_[i].id,i)){fail(irisu::cuda_native::Status::capacity);return;}
  }
#endif
  GAME_HD void fail(irisu::cuda_native::Status status=irisu::cuda_native::Status::invalid_snapshot) const {
    if(scene_&&scene_->world.status==irisu::cuda_native::Status::ok)scene_->world.status=status;
  }
  GAME_HD bool accept(irisu::cuda_native::StageResult result) const {
    if(!result)fail(result.status);return bool(result);
  }
  GAME_HD Entry* find(BodyId id){
#if defined(IRISU_CUDA_NATIVE_ENTRY_INDEX) && IRISU_CUDA_NATIVE_ENTRY_INDEX
    const auto index=entry_index_.find(id);if(index==EntryIndex::missing)return nullptr;
    if(index>=entries_.count||entries_[index].id!=id){fail();return nullptr;}return &entries_[index];
#else
    for(auto& e:entries_)if(e.id==id)return &e;return nullptr;
#endif
  }
  GAME_HD const Entry* find(BodyId id)const{
#if defined(IRISU_CUDA_NATIVE_ENTRY_INDEX) && IRISU_CUDA_NATIVE_ENTRY_INDEX
    const auto index=entry_index_.find(id);if(index==EntryIndex::missing)return nullptr;
    if(index>=entries_.count||entries_[index].id!=id){fail();return nullptr;}return &entries_[index];
#else
    for(const auto& e:entries_)if(e.id==id)return &e;return nullptr;
#endif
  }
  GAME_HD static bool same(Vec2 a,Vec2 b){return a.x==b.x&&a.y==b.y;}
  GAME_HD static void mirror(Entry& e,const Body& b){e.position=b.position;e.angle=b.angle;e.velocity=b.native_velocity;}
  GAME_HD float scale() const {return static_cast<float>(config_.world_magnification);}
  GAME_HD Vec2 raw(std::uint32_t slot) const {
    if(!scene_||slot>=irisu::cuda_native::max_bodies||!scene_->world.bodies[slot].occupied){fail();return {};}
    const auto v=scene_->world.bodies[slot].velocity;return {v.x,v.y};
  }
  GAME_HD void set_wrapped_velocity(std::uint32_t slot,Vec2 velocity){
    irisu::cuda_native::math::Context m;
    // Preserve both the caller's scale multiplication and wrapper's division.
    const auto x=m.store(m.div(m.mul_store(static_cast<float>(velocity.x),scale()),scale()));
    const auto y=m.store(m.div(m.mul_store(static_cast<float>(velocity.y),scale()),scale()));
    if(!m.ok()){fail(m.status);return;}accept(scene_->set_velocity(slot,{x,y}));
  }
  GAME_HD NATIVE_GAME_CALL std::uint32_t construct(unsigned kind,float width,float height,float x,float y,float angle,
                                  float density,float friction,float restitution,std::uint32_t tag){
    using namespace irisu::cuda_native;
    BodyDefinition bd{};ShapeDefinition sd{};std::uint32_t slot=null_index;
    if(!accept(wrapper_definition(bd,sd,kind,width,height,x,y,angle,density,friction,restitution,scale())))return slot;
    bd.user_data=tag;accept(scene_->create_body_shape(bd,sd,slot));return slot;
  }
  GAME_HD void boundary(BoundaryKind kind,double width,double height,double x,double y,double friction,double restitution){
    construct(1,float(width),float(height),float(x),float(y),0,0,float(friction),float(restitution),unsigned(kind));
  }
  GAME_HD NATIVE_GAME_CALL void create(Body& b){
    const unsigned kind=b.shape==Shape::Circle?0:b.shape==Shape::Box?1:2;
    const auto slot=construct(kind,b.shape==Shape::Circle?float(b.size)/2.0f:float(b.size),float(b.size),
      float(b.position.x),float(b.position.y),float(b.angle),float(b.density),float(b.friction),float(b.restitution),b.id+16);
    if(failed())return;set_wrapped_velocity(slot,b.native_velocity);
    Entry e{};e.id=b.id;e.slot=slot;e.shape=b.shape;e.size=b.size;e.density=b.density;
    e.friction=b.friction;e.restitution=b.restitution;mirror(e,b);entries_.push_back(e);
    if(entries_.overflow)fail(irisu::cuda_native::Status::capacity);
#if defined(IRISU_CUDA_NATIVE_ENTRY_INDEX) && IRISU_CUDA_NATIVE_ENTRY_INDEX
    else if(!entry_index_.insert(e.id,entries_.count-1))fail(irisu::cuda_native::Status::capacity);
#endif
  }
  GAME_HD NATIVE_GAME_CALL void external(Body& b,Entry& e){
    const auto requested=b.native_velocity;
    if(!same(b.position,e.position)||b.angle!=e.angle){
      irisu::cuda_native::math::Context m;
      const NativeVec origin{m.store(m.div(float(b.position.x),scale())),m.store(m.div(float(b.position.y),scale()))};
      if(!m.ok()){fail(m.status);return;}
      if(!accept(scene_->set_origin(e.slot,origin,float(b.angle))))return;
      b.native_position={static_cast<double>(float(b.position.x))/scale(),static_cast<double>(float(b.position.y))/scale()};
      b.native_angle=float(b.angle);b.native_velocity={};
    }
    if(!same(requested,e.velocity)){set_wrapped_velocity(e.slot,requested);b.native_velocity=raw(e.slot);}
    mirror(e,b);
  }
 public:
  GAME_HD explicit NativePhysicsWorld(MechanicsConfig config={}):config_(config){}
  template<class Visitor> GAME_HD void snapshot_vectors(Visitor& visit){visit(entries_);visit(contacts_);visit(ordered_);}
  template<class Visitor> GAME_HD void snapshot_vectors(Visitor& visit)const{visit(entries_);visit(contacts_);visit(ordered_);}
  GAME_HD void bind(Scene* scene){scene_=scene;}
  GAME_HD void initialize_pointer_storage(){for(auto& byte:ordered_.storage)byte=0;}
  GAME_HD bool relocate_body_pointers(std::uintptr_t old_base,std::uintptr_t new_base,bool live_only=false){
    const bool valid=relocate_native_body_refs(ordered_,old_base,new_base,live_only);if(!valid)fail();return valid;
  }
  GAME_HD bool failed() const {return !scene_||scene_->world.status!=irisu::cuda_native::Status::ok||entries_.overflow||contacts_.overflow;}
  GAME_HD NATIVE_GAME_CALL void reset(){
    if(!scene_)return;using namespace irisu::cuda_native;math::Context m;
    const AABB bounds{{m.store(m.div(float(config_.world_min_x),scale())),m.store(m.div(float(config_.world_min_y),scale()))},
      {m.store(m.div(float(config_.world_max_x),scale())),m.store(m.div(float(config_.world_max_y),scale()))}};
    const NativeVec gravity{0,m.store(m.div(float(config_.gravity_y),scale()))};
    entries_.clear();contacts_.clear();ordered_.clear();
#if defined(IRISU_CUDA_NATIVE_ENTRY_INDEX) && IRISU_CUDA_NATIVE_ENTRY_INDEX
    entry_index_.reset();
#endif
    if(!accept(scene_->initialize(bounds,gravity)))return;
    if(!m.ok()){fail(m.status);return;}
    BodyState ground{};ground.flags=static_body|allow_sleep;std::uint32_t slot{};
    if(!accept(rotation_matrix(ground))||!accept(scene_->world.create_body(ground,slot)))return;
    const double ht=trunc(config_.field_thickness/2),hh=trunc(config_.field_height/2),hw=trunc(config_.field_width/2);
    const auto center=config_.field_x+hw+config_.field_thickness;
    boundary(BoundaryKind::LeftWall,config_.field_thickness,config_.field_height,config_.field_x+ht,config_.field_y+hh,1,1);
    boundary(BoundaryKind::RightWall,config_.field_thickness,config_.field_height,config_.field_x+config_.field_width+config_.field_thickness,config_.field_y+hh,1,1);
    boundary(BoundaryKind::Floor,config_.field_width+2*config_.field_thickness,config_.field_bottom_height,center,config_.field_y+config_.field_height+config_.field_blank+trunc(config_.field_bottom_height/2),1,0);
    boundary(BoundaryKind::Top,config_.field_top_width,config_.field_top_height,center,config_.field_top,1,.5);
  }
  GAME_HD void initialize_mass(Body& b) const {
    b.native_position={static_cast<double>(float(b.position.x))/scale(),static_cast<double>(float(b.position.y))/scale()};
    b.native_velocity={float(b.velocity.x),float(b.velocity.y)};b.native_angle=float(b.angle);b.native_angular_velocity=float(b.angular_velocity);
    b.native_state_valid=true;b.native_center_valid=false;b.sleeping=false;b.sleep_time=0;
    if(b.density<=0||b.lifecycle==Lifecycle::Deleted){b.inverse_mass=0;b.inverse_inertia=0;return;}
    const double width=b.size/config_.world_magnification;double area=width*width;
    if(b.shape==Shape::Triangle)area*=.5;
    if(b.shape==Shape::Circle){const double r=width*.5;area=3.14159265358979323846*r*r;}
    const auto mass=b.density*area;b.inverse_mass=mass>0?1/mass:0;b.inverse_inertia=0;
  }
  GAME_HD void queue_destroy(BodyId id){
    auto* e=find(id);if(!e){fail();return;}
    if(!e->pending&&accept(scene_->queue_destroy(e->slot)))e->pending=true;
  }
  GAME_HD NATIVE_GAME_CALL void synchronize(vector<Body>& bodies){
    if(failed())return;ordered_.clear();for(auto& b:bodies)ordered_.push_back(&b);
    sort(ordered_.begin(),ordered_.end(),[] GAME_HD(const Body* a,const Body* b){
      if(a->actor_slot!=b->actor_slot)return a->actor_slot<b->actor_slot;
      if((a->lifecycle==Lifecycle::Deleted)!=(b->lifecycle==Lifecycle::Deleted))return a->lifecycle==Lifecycle::Deleted;
      return a->id<b->id;
    });
    if(ordered_.overflow){fail(irisu::cuda_native::Status::capacity);return;}
    for(auto* candidate:ordered_){auto& b=*candidate;auto* e=find(b.id);
      if(b.lifecycle==Lifecycle::Deleted){if(e){if(!e->pending)external(b,*e);queue_destroy(b.id);}}
      else if(!e)create(b);
      else if(e->pending||e->shape!=b.shape||e->size!=b.size||e->density!=b.density||e->friction!=b.friction||e->restitution!=b.restitution){fail();return;}
      else external(b,*e);
      if(failed())return;
    }
  }
  GAME_HD Vec2 raw_velocity(BodyId id) const {const auto* e=find(id);if(!e){fail();return {};}return raw(e->slot);}
  GAME_HD NATIVE_GAME_CALL const vector<Contact>& step(vector<Body>& bodies){
    synchronize(bodies);contacts_.clear();if(failed()||!accept(scene_->step(float(config_.tick_seconds),config_.solver_iterations)))return contacts_;
    for(auto& e:entries_)if(e.pending)for(auto& b:bodies)if(b.id==e.id&&b.lifecycle==Lifecycle::Deleted)b.pending_delete=false;
#if defined(IRISU_CUDA_NATIVE_ENTRY_INDEX) && IRISU_CUDA_NATIVE_ENTRY_INDEX
    const auto old_entry_count=entries_.count;
#endif
    entries_.erase(remove_if(entries_.begin(),entries_.end(),[] GAME_HD(const Entry& e){return e.pending;}),entries_.end());
#if defined(IRISU_CUDA_NATIVE_ENTRY_INDEX) && IRISU_CUDA_NATIVE_ENTRY_INDEX
    if(entries_.count!=old_entry_count)rebuild_entry_index();
#endif
    const auto& w=scene_->world;unsigned count=0;
    for(auto c=w.contact_head;c!=irisu::cuda_native::null_index;c=w.contacts[c].next){
      if(c>=irisu::cuda_native::max_contacts||++count>irisu::cuda_native::max_contacts){fail();break;}
      const auto& contact=w.contacts[c];if(!contact.manifold_count)continue;
      const auto first=w.bodies[w.shapes[contact.shape1].body].user_data,second=w.bodies[w.shapes[contact.shape2].body].user_data;
      const auto a=first>=16?first-16:0,b=second>=16?second-16:0;
      const auto ba=first>=1&&first<=4?static_cast<BoundaryKind>(first):BoundaryKind::None;
      const auto bb=second>=1&&second<=4?static_cast<BoundaryKind>(second):BoundaryKind::None;
      if(a||b)contacts_.push_back({a,b,a==0?ba:b==0?bb:BoundaryKind::None});
      if(contacts_.overflow){fail(irisu::cuda_native::Status::capacity);break;}
    }
    for(auto& b:bodies)if(b.lifecycle!=Lifecycle::Deleted){
      auto* e=find(b.id);if(!e)continue;const auto& native=w.bodies[e->slot];irisu::cuda_native::math::Context m;
      const auto origin=irisu::cuda_native::collision_detail::subtract(m,native.position,
        irisu::cuda_native::collision_detail::rotate(m,native.R,native.center));
      const float x=m.mul_store(origin.x,scale()),y=m.mul_store(origin.y,scale());
      if(!m.ok()){fail(m.status);break;}const auto v=raw(e->slot);
      b.position={x,y};b.velocity=v;b.angle=native.rotation;
      b.native_position={static_cast<double>(x)/scale(),static_cast<double>(y)/scale()};b.native_center=b.native_position;
      b.native_velocity=v;b.native_angle=native.rotation;b.native_state_valid=true;b.native_center_valid=false;
      b.sleeping=false;b.sleep_time=0;mirror(*e,b);
    }
    return contacts_;
  }
};
} // namespace irisu::cuda_game
#undef NATIVE_GAME_CALL
