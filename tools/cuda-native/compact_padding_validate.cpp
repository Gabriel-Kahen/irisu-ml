// Host-only exhaustive one-byte perturbation check of the LIVE element mask.
// Body coverage comes from the independently maintained game-digest visitor.
#include "../../clone/cuda_native/compact_game_snapshot.cuh"
#include <cstdio>
#include <cstring>
#include <stdexcept>
namespace g=irisu::cuda_game;namespace c=irisu::cuda_native::compact_snapshot;
using Hash=irisu::game_digest::Hash;
std::uint64_t hash(const g::Body& body){Hash h;irisu::game_digest::body(h,body);return h.value;}
std::uint64_t hash(const g::Contact& contact){Hash h;
  h.scalar(contact.a);h.scalar(contact.b);h.scalar(contact.boundary);
  h.scalar(contact.point.x);h.scalar(contact.point.y);h.scalar(contact.normal.x);h.scalar(contact.normal.y);h.scalar(contact.penetration);return h.value;}
std::uint64_t hash(const g::ScriptedOrigin& origin){Hash h;
  h.scalar(origin.id);h.scalar(origin.position.x);h.scalar(origin.position.y);h.scalar(origin.angle);return h.value;}
std::uint64_t hash(const g::NativePhysicsWorld::Entry& entry){Hash h;
  h.scalar(entry.id);h.scalar(entry.slot);h.scalar(entry.shape);h.scalar(entry.size);h.scalar(entry.density);
  h.scalar(entry.friction);h.scalar(entry.restitution);h.scalar(entry.position.x);h.scalar(entry.position.y);
  h.scalar(entry.velocity.x);h.scalar(entry.velocity.y);h.scalar(entry.angle);h.scalar(entry.pending);return h.value;}
template<class T>unsigned validate(const char* name){
  alignas(T) unsigned char initial[sizeof(T)]{};::new(static_cast<void*>(initial))T{};
  const auto expected=hash(*reinterpret_cast<const T*>(initial));unsigned padding=0;
  for(unsigned byte=0;byte<sizeof(T);++byte){
    alignas(T) unsigned char changed[sizeof(T)];std::memcpy(changed,initial,sizeof(T));changed[byte]^=1;
    const bool value=hash(*reinterpret_cast<const T*>(changed))!=expected;
    if(value!=c::element_value_byte<T>(byte)){
      std::fprintf(stderr,"type%s byte%u visitor_value%d mask_value%d\n",name,byte,value,c::element_value_byte<T>(byte));
      throw std::runtime_error("compact live-element mask omits a value or admits padding");
    }
    padding+=!value;
  }
  std::printf("{\"type\":\"%s\",\"bytes\":%zu,\"value_bytes\":%zu,\"padding_bytes\":%u}\n",name,sizeof(T),sizeof(T)-padding,padding);
  return sizeof(T);
}
int main(){try{
  unsigned bytes=validate<g::Body>("Body");bytes+=validate<g::Contact>("Contact");
  bytes+=validate<g::ScriptedOrigin>("ScriptedOrigin");bytes+=validate<g::NativePhysicsWorld::Entry>("Entry");
  std::printf("{\"pass\":true,\"host_only\":true,\"one_byte_perturbations\":%u,\"body_field_enumeration\":\"game_digest::visit_body\"}\n",bytes);return 0;
}catch(const std::exception& error){std::fprintf(stderr,"compact-padding-validate: %s\n",error.what());return 1;}}
