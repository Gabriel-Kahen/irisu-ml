#pragma once
/*
* Copyright (c) 2006-2007 Erin Catto http://www.gphysics.com
*
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
*/
// Altered version: fixed pointer-free state, typed callbacks, CUDA/host methods.
#include "solver_math.hpp"
#include <cstddef>
#include <cstdint>
#include <climits>
namespace irisu::cuda_native::broadphase_detail {
using uint16=std::uint16_t;using uint32=std::uint32_t;using int32=std::int32_t;
inline constexpr int32 b2_maxProxies=512,b2_maxPairs=4096,b2_tableCapacity=4096,b2_tableMask=4095;
inline constexpr uint16 b2_nullProxy=65535,b2_nullPair=65535,b2_invalid=65535;
template<class T>NATIVE_HD inline void swap_values(T& a,T& b){T copy=a;a=b;b=copy;}
NATIVE_HD inline bool valid_aabb(const AABB& a){return math::finite(a.min.x)&&math::finite(a.min.y)&&math::finite(a.max.x)&&math::finite(a.max.y)&&a.min.x<=a.max.x&&a.min.y<=a.max.y;}
struct Bound {
 uint16 value{},proxyId{},stabbingCount{};
 NATIVE_HD bool IsLower() const{return !(value&1);}
 NATIVE_HD bool IsUpper() const{return value&1;}
};
struct BoundValues {uint16 lowerValues[2],upperValues[2];};
struct Proxy {
 uint16 lowerBounds[2]{},upperBounds[2]{},overlapCount{},timeStamp{};uint32 userData{};
 NATIVE_HD uint16 GetNext() const{return lowerBounds[0];}
 NATIVE_HD void SetNext(uint16 value){lowerBounds[0]=value;}
 NATIVE_HD bool IsValid() const{return overlapCount!=b2_invalid;}
};
struct Pair {
 uint32 userData{};uint16 proxyId1{},proxyId2{},next{},status{};
 NATIVE_HD bool IsBuffered() const{return status&1;} NATIVE_HD void SetBuffered(){status|=1;} NATIVE_HD void ClearBuffered(){status&=~1;}
 NATIVE_HD bool IsRemoved() const{return status&2;} NATIVE_HD void SetRemoved(){status|=2;} NATIVE_HD void ClearRemoved(){status&=~2;}
 NATIVE_HD bool IsFinal() const{return status&4;} NATIVE_HD void SetFinal(){status|=4;}
};
struct BufferedPair {uint16 proxyId1{},proxyId2{};};
NATIVE_HD inline uint32 Hash(uint32 a,uint32 b){uint32 key=(b<<16)|a;key=~key+(key<<15);key^=key>>12;key+=key<<2;key^=key>>4;key*=2057;return key^(key>>16);}
struct PairManager {
 // Native-layout reserved words; no persistent pointers or callbacks.
 uint32 m_broadPhase{},m_callback{};Pair m_pairs[b2_maxPairs]{};
 uint16 m_freePair{},padding{};int32 m_pairCount{};BufferedPair m_pairBuffer[b2_maxPairs]{};
 int32 m_pairBufferCount{};uint16 m_hashTable[b2_tableCapacity]{};
 NATIVE_HD void Initialize(){
  m_broadPhase=m_callback=0;m_pairBufferCount=m_pairCount=0;m_freePair=0;
  for(int32 i=0;i<b2_tableCapacity;++i)m_hashTable[i]=b2_nullPair;
  for(int32 i=0;i<b2_maxPairs;++i){auto& p=m_pairs[i];p.proxyId1=p.proxyId2=b2_nullProxy;p.userData=0;p.status=0;p.next=i+1;}
  m_pairs[b2_maxPairs-1].next=b2_nullPair;
 }
 NATIVE_HD Pair* Find(int32 a,int32 b,uint32 hash){uint16 index=m_hashTable[hash];while(index!=b2_nullPair&&(m_pairs[index].proxyId1!=a||m_pairs[index].proxyId2!=b))index=m_pairs[index].next;return index==b2_nullPair?nullptr:m_pairs+index;}
 NATIVE_HD Pair* Find(int32 a,int32 b){if(a>b)swap_values(a,b);return Find(a,b,Hash(a,b)&b2_tableMask);}
 NATIVE_HD Pair* AddPair(int32 a,int32 b,Status& error){
  if(a>b)swap_values(a,b);const auto hash=Hash(a,b)&b2_tableMask;auto* p=Find(a,b,hash);if(p)return p;
  if(m_pairCount>=b2_maxPairs||m_freePair==b2_nullPair){error=Status::capacity;return nullptr;}
  const uint16 index=m_freePair;p=m_pairs+index;m_freePair=p->next;p->proxyId1=a;p->proxyId2=b;p->status=0;p->userData=0;p->next=m_hashTable[hash];m_hashTable[hash]=index;++m_pairCount;return p;
 }
 NATIVE_HD uint32 RemovePair(int32 a,int32 b,Status& error){
  if(a>b)swap_values(a,b);uint16* node=m_hashTable+(Hash(a,b)&b2_tableMask);
  while(*node!=b2_nullPair){auto& p=m_pairs[*node];if(p.proxyId1==a&&p.proxyId2==b){const uint16 index=*node;*node=p.next;const auto token=p.userData;p.next=m_freePair;p.proxyId1=p.proxyId2=b2_nullProxy;p.userData=0;p.status=0;m_freePair=index;--m_pairCount;return token;}node=&p.next;}
  error=Status::invalid_snapshot;return 0;
 }
 NATIVE_HD void AddBufferedPair(int32 a,int32 b,Status& error){
  if(error!=Status::ok)return;auto* p=AddPair(a,b,error);if(!p)return;
  if(!p->IsBuffered()){if(m_pairBufferCount>=b2_maxPairs){error=Status::capacity;return;}p->SetBuffered();m_pairBuffer[m_pairBufferCount++]={p->proxyId1,p->proxyId2};}p->ClearRemoved();
 }
 NATIVE_HD void RemoveBufferedPair(int32 a,int32 b,Status& error){
  if(error!=Status::ok)return;auto* p=Find(a,b);if(!p)return;
  if(!p->IsBuffered()){if(m_pairBufferCount>=b2_maxPairs){error=Status::capacity;return;}p->SetBuffered();m_pairBuffer[m_pairBufferCount++]={p->proxyId1,p->proxyId2};}p->SetRemoved();
 }
 template<class Callback>NATIVE_HD void Commit(Proxy* proxies,Callback& callback,Status& error){
  if(error!=Status::ok)return;int32 removeCount=0;
  for(int32 i=0;i<m_pairBufferCount;++i){auto* p=Find(m_pairBuffer[i].proxyId1,m_pairBuffer[i].proxyId2);if(!p){error=Status::invalid_snapshot;return;}p->ClearBuffered();auto& a=proxies[p->proxyId1];auto& b=proxies[p->proxyId2];
   if(p->IsRemoved()){if(p->IsFinal())callback.PairRemoved(a.userData,b.userData,p->userData);m_pairBuffer[removeCount++]={p->proxyId1,p->proxyId2};}
   else if(!p->IsFinal()){p->userData=callback.PairAdded(a.userData,b.userData);p->SetFinal();}
  }
  for(int32 i=0;i<removeCount;++i)RemovePair(m_pairBuffer[i].proxyId1,m_pairBuffer[i].proxyId2,error);
  m_pairBufferCount=0;
 }
};
NATIVE_HD inline int32 BinarySearch(Bound* bounds,int32 count,uint16 value){int32 low=0,high=count-1;while(low<=high){const int32 mid=(low+high)>>1;if(bounds[mid].value>value)high=mid-1;else if(bounds[mid].value<value)low=mid+1;else return mid;}return low;}
NATIVE_HD inline void move_bounds(Bound* destination,const Bound* source,int32 count){if(destination<source){for(int32 i=0;i<count;++i)destination[i]=source[i];}else if(destination>source){for(int32 i=count;i>0;--i)destination[i-1]=source[i-1];}}
struct BroadPhase {
 PairManager m_pairManager{};Proxy m_proxyPool[b2_maxProxies]{};uint16 m_freeProxy{};
 Bound m_bounds[2][2*b2_maxProxies]{};uint16 m_queryResults[b2_maxProxies]{},queryPadding{};int32 m_queryResultCount{};
 AABB m_worldAABB{};Vec2 m_quantizationFactor{};int32 m_proxyCount{};uint16 m_timeStamp{},endPadding{};
 Status error{Status::ok};
 NATIVE_HD Status Initialize(const AABB& world){
  error=Status::ok;if(!valid_aabb(world)||world.min.x>=world.max.x||world.min.y>=world.max.y){error=Status::unsupported_numeric;return error;}
  m_pairManager.Initialize();m_worldAABB=world;m_proxyCount=m_queryResultCount=0;m_freeProxy=0;m_timeStamp=1;
  math::Context m;const float dx=m.sub_store(world.max.x,world.min.x),dy=m.sub_store(world.max.y,world.min.y);
  m_quantizationFactor={m.div_store(65535,dx),m.div_store(65535,dy)};if(!m.ok()){error=m.status;return error;}
  for(uint16 i=0;i<b2_maxProxies;++i){auto& p=m_proxyPool[i];p.SetNext(i+1);p.timeStamp=0;p.overlapCount=b2_invalid;p.userData=0;}m_proxyPool[b2_maxProxies-1].SetNext(b2_nullProxy);return error;
 }
 NATIVE_HD bool InRange(const AABB& a){math::Context m;const float ax=m.sub_store(a.min.x,m_worldAABB.max.x),ay=m.sub_store(a.min.y,m_worldAABB.max.y),bx=m.sub_store(m_worldAABB.min.x,a.max.x),by=m.sub_store(m_worldAABB.min.y,a.max.y);const float x=ax>bx?ax:bx,y=ay>by?ay:by;if(!m.ok())error=m.status;return (x>y?x:y)<0;}
 NATIVE_HD Proxy* GetProxy(int32 id){return id<0||id>=b2_maxProxies||!m_proxyPool[id].IsValid()?nullptr:m_proxyPool+id;}
 NATIVE_HD static float Clamp(float value,float low,float high){const float top=value<high?value:high;return low>top?low:top;}
 NATIVE_HD void ComputeBounds(uint16* lower,uint16* upper,const AABB& a){
  if(!valid_aabb(a)){error=Status::unsupported_numeric;return;}math::Context m;
  const float low[2]={m_worldAABB.min.x,m_worldAABB.min.y},high[2]={m_worldAABB.max.x,m_worldAABB.max.y},amin[2]={a.min.x,a.min.y},amax[2]={a.max.x,a.max.y},factor[2]={m_quantizationFactor.x,m_quantizationFactor.y};
  for(unsigned axis=0;axis<2;++axis){const float first=Clamp(amin[axis],low[axis],high[axis]),last=Clamp(amax[axis],low[axis],high[axis]);const double l=m.mul(m.sub(first,low[axis]),factor[axis]),u=m.mul(m.sub(last,low[axis]),factor[axis]);
   if(!m.ok()||!(l>=0&&l<2147483648.0&&u>=0&&u<2147483648.0)){error=Status::unsupported_numeric;return;}
   lower[axis]=static_cast<uint16>(static_cast<int32>(l))&65534u;upper[axis]=static_cast<uint16>(static_cast<int32>(u))|1u;
  }
 }
 NATIVE_HD void IncrementTimeStamp();NATIVE_HD void IncrementOverlapCount(int32 id);
 NATIVE_HD void Query(int32*,int32*,uint16,uint16,Bound*,int32,int32);
 template<class Callback>NATIVE_HD uint16 CreateProxy(const AABB&,uint32,Callback&);
 template<class Callback>NATIVE_HD void DestroyProxy(int32,Callback&);
 NATIVE_HD void MoveProxy(int32,const AABB&);NATIVE_HD int32 Query(const AABB&,uint32*,int32);
 NATIVE_HD bool TestOverlap(const BoundValues&,Proxy*);
 template<class Callback>NATIVE_HD Status Commit(Callback& callback){m_pairManager.Commit(m_proxyPool,callback,error);return error;}
 NATIVE_HD bool Validate() const{
  for(unsigned axis=0;axis<2;++axis){uint16 stabbing=0;for(int32 i=0;i<2*m_proxyCount;++i){const auto& b=m_bounds[axis][i];if((i&&m_bounds[axis][i-1].value>b.value)||b.proxyId>=b2_maxProxies||!m_proxyPool[b.proxyId].IsValid())return false;const auto& p=m_proxyPool[b.proxyId];if(b.IsLower()){if(p.lowerBounds[axis]!=i)return false;++stabbing;}else{if(p.upperBounds[axis]!=i)return false;--stabbing;}if(b.stabbingCount!=stabbing)return false;}}
  return true;
 }
};
static_assert(sizeof(Bound)==6&&sizeof(Proxy)==16&&sizeof(Pair)==12&&sizeof(PairManager)==0x12014);
static_assert(offsetof(BroadPhase,m_proxyPool)==0x12014&&offsetof(BroadPhase,m_bounds)==0x14016&&offsetof(BroadPhase,m_queryResults)==0x17016&&offsetof(BroadPhase,m_worldAABB)==0x1741c&&offsetof(BroadPhase,error)==0x1743c);
#include "broadphase_algorithm.inc"
}
namespace irisu::cuda_native {using BroadPhase=broadphase_detail::BroadPhase;}
