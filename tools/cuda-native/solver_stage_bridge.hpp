#pragma once
// Test adapter only: conversion and original translated reference stay outside
// the independent typed solver. This is not a complete native backend.
#include "../../clone/cuda_native/solver.hpp"
#include "../../clone/cuda/guest.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace irisu::cuda_native::solver_test {
inline std::uint64_t calls[4]{},constraints_seen[4]{},points_seen[4]{};
struct Summary {~Summary(){for(unsigned i=0;i<4;++i)std::fprintf(stderr,"typed_solver stage=%u calls=%llu constraints=%llu points=%llu\n",i,(unsigned long long)calls[i],(unsigned long long)constraints_seen[i],(unsigned long long)points_seen[i]);}};
inline Summary summary;
inline BodyState body(cuda_guest::Guest& g,std::uint32_t a){
 BodyState b{};b.position={g.load<float>(a+4),g.load<float>(a+8)};b.rotation=g.load<float>(a+12);b.R={{g.load<float>(a+16),g.load<float>(a+20)},{g.load<float>(a+24),g.load<float>(a+28)}};
 b.velocity={g.load<float>(a+44),g.load<float>(a+48)};b.angular_velocity=g.load<float>(a+52);b.inv_mass=g.load<float>(a+108);b.inv_i=g.load<float>(a+116);return b;
}
inline bool same(float a,float b){return irisu::cuda_exact::bits_float(a)==irisu::cuda_exact::bits_float(b);}
inline void append(std::vector<std::uint32_t>& v,const BodyState& b){for(float f:{b.position.x,b.position.y,b.rotation,b.R.col1.x,b.R.col1.y,b.R.col2.x,b.R.col2.y,b.velocity.x,b.velocity.y,b.angular_velocity,b.inv_mass,b.inv_i})v.push_back(irisu::cuda_exact::bits_float(f));}
inline void append(std::vector<std::uint32_t>& v,const ContactConstraint& c){
 for(const auto& p:c.points)for(float f:{p.local_anchor1.x,p.local_anchor1.y,p.local_anchor2.x,p.local_anchor2.y,p.normal_impulse,p.tangent_impulse,p.position_impulse,p.normal_mass,p.tangent_mass,p.separation,p.velocity_bias})v.push_back(irisu::cuda_exact::bits_float(f));
 for(float f:{c.normal.x,c.normal.y,c.friction,c.restitution})v.push_back(irisu::cuda_exact::bits_float(f));v.push_back(c.body1);v.push_back(c.body2);v.push_back(c.manifold);v.push_back(c.point_count);
}
inline Manifold manifold(cuda_guest::Guest& g,std::uint32_t a){Manifold m{};for(unsigned j=0;j<2;++j){auto& p=m.points[j];const auto b=a+j*24;p.position={g.load<float>(b),g.load<float>(b+4)};p.separation=g.load<float>(b+8);p.normal_impulse=g.load<float>(b+12);p.tangent_impulse=g.load<float>(b+16);p.id=g.load<std::uint32_t>(b+20);}m.normal={g.load<float>(a+48),g.load<float>(a+52)};m.point_count=g.load<std::int32_t>(a+56);return m;}
inline void append(std::vector<std::uint32_t>& v,const Manifold& m){for(const auto& p:m.points){for(float f:{p.position.x,p.position.y,p.separation,p.normal_impulse,p.tangent_impulse})v.push_back(irisu::cuda_exact::bits_float(f));v.push_back(p.id);}v.push_back(irisu::cuda_exact::bits_float(m.normal.x));v.push_back(irisu::cuda_exact::bits_float(m.normal.y));v.push_back(m.point_count);}
struct Packet {
 std::vector<std::uint32_t> addresses;std::vector<BodyState> bodies;std::vector<ContactConstraint> constraints;std::vector<std::uint32_t> manifold_addresses;std::vector<Manifold> manifolds;
 std::uint32_t id(cuda_guest::Guest& g,std::uint32_t address){for(std::uint32_t i=0;i<addresses.size();++i)if(addresses[i]==address)return i;addresses.push_back(address);bodies.push_back(body(g,address));return addresses.size()-1;}
 std::uint32_t mid(cuda_guest::Guest& g,std::uint32_t address){for(std::uint32_t i=0;i<manifold_addresses.size();++i)if(manifold_addresses[i]==address)return i;manifold_addresses.push_back(address);manifolds.push_back(manifold(g,address));return manifold_addresses.size()-1;}
 void read(cuda_guest::Guest& g,std::uint32_t first,std::uint32_t count){
  constraints.clear();for(std::uint32_t i=0;i<count;++i){const auto a=first+i*120;ContactConstraint c{};
   for(unsigned j=0;j<2;++j){const auto p=a+j*44;auto& q=c.points[j];q.local_anchor1={g.load<float>(p),g.load<float>(p+4)};q.local_anchor2={g.load<float>(p+8),g.load<float>(p+12)};
    q.normal_impulse=g.load<float>(p+16);q.tangent_impulse=g.load<float>(p+20);q.position_impulse=g.load<float>(p+24);q.normal_mass=g.load<float>(p+28);q.tangent_mass=g.load<float>(p+32);q.separation=g.load<float>(p+36);q.velocity_bias=g.load<float>(p+40);}
   c.normal={g.load<float>(a+88),g.load<float>(a+92)};c.manifold=mid(g,g.load<std::uint32_t>(a+96));c.body1=id(g,g.load<std::uint32_t>(a+100));c.body2=id(g,g.load<std::uint32_t>(a+104));c.friction=g.load<float>(a+108);c.restitution=g.load<float>(a+112);c.point_count=g.load<std::int32_t>(a+116);constraints.push_back(c);
  }
 }
};
template<unsigned Entry,class Original>inline void stage(cuda_guest::Guest& g,std::uint64_t& budget,unsigned depth,Original original){
 if(g.pc!=Entry||g.error){original(g,budget,depth);return;}
 constexpr unsigned index=Entry==29424?0:Entry==30224?1:Entry==31584?2:3;
 const auto n=g.load<std::int32_t>(g.r[1]+8);const auto first=g.load<std::uint32_t>(g.r[1]+4);
 const float beta=Entry==31584?g.load<float>(g.r[4]+4):0.0f;
 const bool warm=g.load<std::uint32_t>(99272)!=0;
 if(n<0||n>4096)throw std::runtime_error("invalid reference solver count");Packet p;p.read(g,first,n);auto before=p;
 SolverResult result;
 if constexpr(Entry==29424)result=solver_presolve(p.bodies.data(),p.bodies.size(),p.constraints.data(),p.constraints.size(),warm);
 if constexpr(Entry==30224)result=solver_velocity(p.bodies.data(),p.bodies.size(),p.constraints.data(),p.constraints.size());
 if constexpr(Entry==31584)result=solver_position(p.bodies.data(),p.bodies.size(),p.constraints.data(),p.constraints.size(),beta);
 if constexpr(Entry==29968)result=solver_postsolve(p.constraints.data(),p.constraints.size(),p.manifolds.data(),p.manifolds.size());
 if(!result){std::fprintf(stderr,"typed solver rejected stage=%u constraint=%u point=%u error=%u\n",Entry,result.constraint,result.point,unsigned(result.status));throw std::runtime_error("typed solver rejected reference stage");}
 original(g,budget,depth);if(g.error)throw std::runtime_error("reference solver failed");
 if constexpr(Entry==31584)if(result.converged!=bool(g.r[0]))throw std::runtime_error("position convergence differs");
 Packet expected;expected.addresses=p.addresses;for(auto a:p.addresses)expected.bodies.push_back(body(g,a));expected.manifold_addresses=p.manifold_addresses;for(auto a:p.manifold_addresses)expected.manifolds.push_back(manifold(g,a));expected.read(g,first,n);
 std::vector<std::uint32_t> actual_words,expected_words;
 for(const auto& b:p.bodies)append(actual_words,b);for(const auto& b:expected.bodies)append(expected_words,b);
 for(const auto& c:p.constraints)append(actual_words,c);for(const auto& c:expected.constraints)append(expected_words,c);
 for(const auto& m:p.manifolds)append(actual_words,m);for(const auto& m:expected.manifolds)append(expected_words,m);
 if(actual_words!=expected_words){
  unsigned k=0;while(k<actual_words.size()&&actual_words[k]==expected_words[k])++k;
  std::fprintf(stderr,"typed solver MISMATCH stage=%u call=%llu constraints=%d fieldword=%u got=%08x expected=%08x\n",Entry,(unsigned long long)calls[index],n,k,actual_words[k],expected_words[k]);
  for(unsigned i=0;i<p.bodies.size();++i)std::fprintf(stderr,"body%u v=(%a,%a,%a) expected=(%a,%a,%a) before=(%a,%a,%a)\n",i,p.bodies[i].velocity.x,p.bodies[i].velocity.y,p.bodies[i].angular_velocity,expected.bodies[i].velocity.x,expected.bodies[i].velocity.y,expected.bodies[i].angular_velocity,before.bodies[i].velocity.x,before.bodies[i].velocity.y,before.bodies[i].angular_velocity);
  throw std::runtime_error("independent typed solver physical state mismatch");
 }
 ++calls[index];constraints_seen[index]+=n;for(const auto& c:p.constraints)points_seen[index]+=c.point_count;
}
}
