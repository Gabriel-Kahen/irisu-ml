#pragma once

#include "game_types.cuh"
#include "guest_program.cuh"

namespace irisu::cuda_game {

// Source-equivalent forward adapter for physics_wrapper_forward.cpp. Every
// native pointer is a guest offset; the two host/device bindings are rebound
// after a byte-arena branch before any method is called.
class PhysicsWorld {
 public:
  struct Entry {
    BodyId id{};
    std::uint32_t native{};
    Shape shape{};
    double size{},density{},friction{},restitution{};
    Vec2 position{},velocity{};
    double angle{};
    bool pending{};
  };
 private:
  MechanicsConfig config_{};
  cuda_guest::Guest* guest_{};
  std::uint32_t world_{};
  vector<Entry> entries_{};
  vector<Contact> contacts_{};

  GAME_HD std::uint32_t f(double x) const {return cuda_guest::float_word(static_cast<float>(x));}
  template<class... T> GAME_HD std::uint32_t call(std::uint32_t fn,T... args) const {
    if(!guest_)return 0;
    return cuda_guest::call(*guest_,fn,args...);
  }
  GAME_HD void fail() const {if(guest_)guest_->fail(cuda_guest::bad_address);}
  GAME_HD Entry* find(BodyId id) {
    for(auto& e:entries_)if(e.id==id)return &e;
    return nullptr;
  }
  GAME_HD const Entry* find(BodyId id) const {
    for(const auto& e:entries_)if(e.id==id)return &e;
    return nullptr;
  }
  GAME_HD void mirror(Entry& e,const Body& b) {
    e.position=b.position;e.angle=b.angle;e.velocity=b.native_velocity;
  }
  GAME_HD bool same(Vec2 a,Vec2 b) const {return a.x==b.x&&a.y==b.y;}
  GAME_HD Vec2 raw(std::uint32_t native) const {
    call(cuda_guest::msvc_b2d_world_get_v,world_,native,0x19000u,0x19004u);
    return {guest_->load<float>(0x19000),guest_->load<float>(0x19004)};
  }
  GAME_HD float scalar(std::uint32_t fn,std::uint32_t native) {
    call(fn,world_,native);
    const float v=static_cast<float>(guest_->st());guest_->fpop();return v;
  }
  GAME_HD void boundary(BoundaryKind kind,double width,double height,double x,double y,double friction,double restitution) {
    auto p=call(cuda_guest::msvc_b2d_world_create_box,world_,f(width),f(height),f(x),f(y),f(0),f(0),f(friction),f(restitution));
    call(cuda_guest::msvc_b2d_world_set_v,world_,p,f(0),f(0));
    call(cuda_guest::msvc_b2d_world_set_user_data,world_,p,static_cast<unsigned>(kind));
    if(!p)fail();
  }
  GAME_HD void create(Body& b) {
    std::uint32_t p{};
    if(b.shape==Shape::Circle)
      p=call(cuda_guest::msvc_b2d_world_create_circle,world_,f(static_cast<float>(b.size)/2.0f),f(b.position.x),f(b.position.y),f(b.density),f(b.friction),f(b.restitution));
    else
      p=call(b.shape==Shape::Box?cuda_guest::msvc_b2d_world_create_box:cuda_guest::msvc_b2d_world_create_triangle,
        world_,f(b.size),f(b.size),f(b.position.x),f(b.position.y),f(b.angle),f(b.density),f(b.friction),f(b.restitution));
    const float scale=static_cast<float>(config_.world_magnification);
    call(cuda_guest::msvc_b2d_world_set_v,world_,p,f(static_cast<float>(b.native_velocity.x)*scale),f(static_cast<float>(b.native_velocity.y)*scale));
    call(cuda_guest::msvc_b2d_world_set_user_data,world_,p,b.id+16);
    Entry e{};e.id=b.id;e.native=p;e.shape=b.shape;e.size=b.size;e.density=b.density;e.friction=b.friction;e.restitution=b.restitution;mirror(e,b);
    entries_.push_back(e);if(entries_.overflow||!p)fail();
  }
  GAME_HD void external(Body& b,Entry& e) {
    const auto requested=b.native_velocity;
    if(!same(b.position,e.position)||b.angle!=e.angle) {
      call(cuda_guest::msvc_b2d_world_set_position,world_,e.native,f(b.position.x),f(b.position.y),f(b.angle));
      const float scale=static_cast<float>(config_.world_magnification);
      b.native_position={static_cast<double>(static_cast<float>(b.position.x))/scale,static_cast<double>(static_cast<float>(b.position.y))/scale};
      b.native_angle=static_cast<float>(b.angle);b.native_velocity={};
    }
    if(!same(requested,e.velocity)) {
      const float scale=static_cast<float>(config_.world_magnification);
      call(cuda_guest::msvc_b2d_world_set_v,world_,e.native,f(static_cast<float>(requested.x)*scale),f(static_cast<float>(requested.y)*scale));
      b.native_velocity=raw(e.native);
    }
    mirror(e,b);
  }

 public:
  GAME_HD explicit PhysicsWorld(MechanicsConfig config={}):config_(config){}
  GAME_HD void bind(cuda_guest::Guest* g) {guest_=g;}
  GAME_HD bool failed() const {return !guest_||guest_->error||entries_.overflow;}
  GAME_HD std::uint64_t state_hash(std::uint64_t hash=14695981039346656037ULL) const {
    hash=fnv_scalar(hash,world_);hash=fnv_scalar(hash,entries_.size());
    for(const auto& e:entries_) {
#define HASH(name) hash=fnv_scalar(hash,e.name)
      HASH(id);HASH(native);HASH(shape);HASH(size);HASH(density);HASH(friction);HASH(restitution);
      HASH(position.x);HASH(position.y);HASH(velocity.x);HASH(velocity.y);HASH(angle);HASH(pending);
#undef HASH
    }
    hash=fnv_scalar(hash,contacts_.size());
    for(const auto& c:contacts_) {
#define HASH(name) hash=fnv_scalar(hash,c.name)
      HASH(a);HASH(b);HASH(boundary);HASH(point.x);HASH(point.y);HASH(normal.x);HASH(normal.y);HASH(penetration);
#undef HASH
    }
    hash=fnv_scalar(hash,entries_.overflow);hash=fnv_scalar(hash,contacts_.overflow);
    return hash;
  }

  GAME_HD void reset() {
    if(!guest_)return;
    if(world_)call(cuda_guest::msvc_b2d_world_destroy,world_);
    entries_.clear();
    world_=call(cuda_guest::msvc_b2d_world_create,f(config_.world_min_x),f(config_.world_min_y),f(config_.world_max_x),f(config_.world_max_y),f(config_.gravity_y),f(config_.world_magnification));
    const double ht=trunc(config_.field_thickness/2),hh=trunc(config_.field_height/2),hw=trunc(config_.field_width/2);
    const auto center=config_.field_x+hw+config_.field_thickness;
    boundary(BoundaryKind::LeftWall,config_.field_thickness,config_.field_height,config_.field_x+ht,config_.field_y+hh,1,1);
    boundary(BoundaryKind::RightWall,config_.field_thickness,config_.field_height,config_.field_x+config_.field_width+config_.field_thickness,config_.field_y+hh,1,1);
    boundary(BoundaryKind::Floor,config_.field_width+2*config_.field_thickness,config_.field_bottom_height,center,config_.field_y+config_.field_height+config_.field_blank+trunc(config_.field_bottom_height/2),1,0);
    boundary(BoundaryKind::Top,config_.field_top_width,config_.field_top_height,center,config_.field_top,1,.5);
    if(!world_)fail();
  }
  GAME_HD void initialize_mass(Body& b) const {
    const float scale=static_cast<float>(config_.world_magnification);
    b.native_position={static_cast<double>(static_cast<float>(b.position.x))/scale,static_cast<double>(static_cast<float>(b.position.y))/scale};
    b.native_velocity={static_cast<float>(b.velocity.x),static_cast<float>(b.velocity.y)};
    b.native_angle=static_cast<float>(b.angle);b.native_angular_velocity=static_cast<float>(b.angular_velocity);
    b.native_state_valid=true;b.native_center_valid=false;b.sleeping=false;b.sleep_time=0;
    if(b.density<=0||b.lifecycle==Lifecycle::Deleted){b.inverse_mass=0;b.inverse_inertia=0;return;}
    const double width=b.size/config_.world_magnification;
    double area=width*width;
    if(b.shape==Shape::Triangle)area*=.5;
    if(b.shape==Shape::Circle){const double r=width*.5;area=3.14159265358979323846*r*r;}
    const auto mass=b.density*area;b.inverse_mass=mass>0?1/mass:0;b.inverse_inertia=0;
  }
  GAME_HD void queue_destroy(BodyId id) {
    auto* e=find(id);if(!e){fail();return;}
    if(!e->pending){call(cuda_guest::msvc_b2d_world_destroy_body,world_,e->native);e->pending=true;}
  }
  GAME_HD void synchronize(vector<Body>& bodies) {
    vector<Body*> ordered;
    for(auto& b:bodies)ordered.push_back(&b);
    sort(ordered.begin(),ordered.end(),[] GAME_HD (const Body* a,const Body* b){
      if(a->actor_slot!=b->actor_slot)return a->actor_slot<b->actor_slot;
      if((a->lifecycle==Lifecycle::Deleted)!=(b->lifecycle==Lifecycle::Deleted))return a->lifecycle==Lifecycle::Deleted;
      return a->id<b->id;
    });
    for(auto* candidate:ordered) {
      auto& b=*candidate;auto* e=find(b.id);
      if(b.lifecycle==Lifecycle::Deleted){if(e){if(!e->pending)external(b,*e);queue_destroy(b.id);}}
      else if(!e)create(b);
      else {
        if(e->pending||e->shape!=b.shape||e->size!=b.size||e->density!=b.density||e->friction!=b.friction||e->restitution!=b.restitution){fail();return;}
        external(b,*e);
      }
    }
    if(ordered.overflow)fail();
  }
  GAME_HD Vec2 raw_velocity(BodyId id) const {
    const auto* e=find(id);if(!e){fail();return {};}
    return raw(e->native);
  }
  GAME_HD const vector<Contact>& step(vector<Body>& bodies) {
    synchronize(bodies);
    call(cuda_guest::msvc_b2d_world_step,world_,f(config_.tick_seconds),config_.solver_iterations);
    for(auto& e:entries_)if(e.pending)for(auto& b:bodies)if(b.id==e.id&&b.lifecycle==Lifecycle::Deleted)b.pending_delete=false;
    entries_.erase(remove_if(entries_.begin(),entries_.end(),[] GAME_HD (const Entry& e){return e.pending;}),entries_.end());
    auto& contacts=contacts_;contacts.clear();
    unsigned count=0;
    while(!guest_->error&&call(cuda_guest::msvc_b2d_world_get_contact,world_,0x19000u,0x19004u)) {
      const auto first=guest_->load<std::uint32_t>(0x19000),second=guest_->load<std::uint32_t>(0x19004);
      const auto a=first>=16?first-16:0,b=second>=16?second-16:0;
      const auto ba=first>=1&&first<=4?static_cast<BoundaryKind>(first):BoundaryKind::None;
      const auto bb=second>=1&&second<=4?static_cast<BoundaryKind>(second):BoundaryKind::None;
      if(a||b)contacts.push_back({a,b,a==0?ba:b==0?bb:BoundaryKind::None});
      if(++count>4096||contacts.overflow){fail();break;}
    }
    const float scale=static_cast<float>(config_.world_magnification);
    for(auto& b:bodies)if(b.lifecycle!=Lifecycle::Deleted) {
      auto* e=find(b.id);if(!e)continue;
      const float x=scalar(cuda_guest::msvc_b2d_world_get_x,e->native),y=scalar(cuda_guest::msvc_b2d_world_get_y,e->native),angle=scalar(cuda_guest::msvc_b2d_world_get_r,e->native);
      const auto v=raw(e->native);
      b.position={x,y};b.velocity=v;b.angle=angle;
      b.native_position={static_cast<double>(x)/scale,static_cast<double>(y)/scale};b.native_center=b.native_position;b.native_velocity=v;b.native_angle=angle;
      b.native_state_valid=true;b.native_center_valid=false;b.sleeping=false;b.sleep_time=0;mirror(*e,b);
    }
    return contacts;
  }
};
} // namespace irisu::cuda_game
