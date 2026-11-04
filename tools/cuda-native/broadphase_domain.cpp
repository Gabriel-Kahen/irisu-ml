#include "../../clone/cuda_native/broadphase.hpp"
#include <cassert>
#include <cstring>
#include <limits>
namespace n=irisu::cuda_native;
struct Callback {unsigned adds{},removes{};unsigned PairAdded(unsigned,unsigned){return ++adds;}void PairRemoved(unsigned,unsigned,unsigned){++removes;}};
int main(){
 n::BroadPhase bp{};Callback callback;assert(bp.Initialize({{-100,-100},{100,100}})==n::Status::ok);
 for(unsigned i=0;i<512;++i){const float x=-90.0f+float(i%32)*5.0f,y=-90.0f+float(i/32)*5.0f;assert(bp.CreateProxy({{x,y},{x+.25f,y+.25f}},i+1,callback)==i);assert(bp.Validate());}
 assert(bp.m_proxyCount==512&&bp.m_pairManager.m_pairCount==0);unsigned char saved[0x1743c];std::memcpy(saved,&bp,sizeof(saved));assert(bp.CreateProxy({{0,0},{1,1}},999,callback)==65535&&bp.error==n::Status::capacity);assert(!std::memcmp(saved,&bp,sizeof(saved)));
 n::broadphase_detail::PairManager pairs{};pairs.Initialize();n::Status status=n::Status::ok;unsigned a_ids[4096],b_ids[4096],count=0;
 for(unsigned a=0;a<512&&count<4096;++a)for(unsigned b=a+1;b<512&&count<4096;++b){a_ids[count]=a;b_ids[count]=b;pairs.AddBufferedPair(a,b,status);++count;}
 assert(status==n::Status::ok&&pairs.m_pairCount==4096&&pairs.m_pairBufferCount==4096&&pairs.m_freePair==65535);
 pairs.RemoveBufferedPair(0,1,status);pairs.AddBufferedPair(0,1,status);assert(status==n::Status::ok&&pairs.m_pairBufferCount==4096);
 n::broadphase_detail::Proxy proxies[512]{};for(unsigned i=0;i<512;++i)proxies[i].userData=i+1;Callback full;pairs.Commit(proxies,full,status);assert(full.adds==4096&&pairs.m_pairBufferCount==0&&status==n::Status::ok);
 for(unsigned i=0;i<4096;++i)pairs.RemoveBufferedPair(a_ids[i],b_ids[i],status);pairs.Commit(proxies,full,status);assert(full.removes==4096&&pairs.m_pairCount==0&&pairs.m_freePair==4095&&status==n::Status::ok);
 pairs.Initialize();for(unsigned i=0;i<4096;++i)pairs.AddBufferedPair(a_ids[i],b_ids[i],status);pairs.AddBufferedPair(510,511,status);assert(status==n::Status::capacity&&pairs.m_pairCount==4096&&pairs.m_pairBufferCount==4096);
 n::BroadPhase invalid{};Callback unused;assert(invalid.Initialize({{0,0},{0,1}})==n::Status::unsupported_numeric);
 invalid={};assert(invalid.Initialize({{-1,-1},{1,1}})==n::Status::ok);assert(invalid.CreateProxy({{std::numeric_limits<float>::quiet_NaN(),0},{1,1}},1,unused)==65535&&invalid.error==n::Status::unsupported_numeric);
 invalid={};invalid.Initialize({{-1,-1},{1,1}});invalid.MoveProxy(65535,{{0,0},{1,1}});assert(invalid.error==n::Status::invalid_snapshot);
}
