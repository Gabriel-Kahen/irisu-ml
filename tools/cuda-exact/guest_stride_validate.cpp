#include "../../clone/cuda/guest.hpp"
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using irisu::cuda_guest::Guest;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
 constexpr unsigned capacity=8192,lanes=33,padded=64,guard=64;
 std::vector<unsigned char> packed(std::size_t(capacity)*padded+2*guard,0xa5);
 std::vector<unsigned char> flat(std::size_t(capacity)*lanes);
 std::array<Guest,lanes> guests{};
 for(unsigned lane=0;lane<lanes;++lane){auto& g=guests[lane];g.capacity=capacity;g.word_stride=32;
  g.memory=packed.data()+guard+std::size_t(lane/32)*capacity*32+(lane%32)*4;g.fill(0,0,capacity);}
 unsigned rng=12345;
 for(unsigned tick=0;tick<30000;++tick){
  rng=rng*1664525U+1013904223U;const auto lane=rng%lanes;auto& g=guests[lane];auto* expected=flat.data()+lane*capacity;
  rng=rng*1664525U+1013904223U;const auto address=rng%(capacity-512);
  const auto value=(std::uint64_t(tick)<<32)|rng;
  switch((rng>>20)%6){
   case 0:g.store<std::uint64_t>(address,value);std::memcpy(expected+address,&value,8);break;
   case 1:{auto v=std::uint32_t(value);g.store(address,v);std::memcpy(expected+address,&v,4);break;}
   case 2:{auto v=std::uint16_t(value);g.store(address,v);std::memcpy(expected+address,&v,2);break;}
   case 3:{const auto length=1+((rng>>8)%257);g.fill(address,rng,length);std::memset(expected+address,rng&255,length);break;}
   default:{const unsigned other=address+((rng>>8)&31),length=1+((rng>>13)&255);
    const auto a=(rng&1)?address:other,b=(rng&1)?other:address;
    g.copy(a,b,length);std::memmove(expected+a,expected+b,length);break;}
  }
  std::uint64_t a;std::uint32_t b;std::uint16_t c;
  std::memcpy(&a,expected+address,8);std::memcpy(&b,expected+address,4);std::memcpy(&c,expected+address,2);
  require(g.load<std::uint64_t>(address)==a&&g.load<std::uint32_t>(address)==b&&g.load<std::uint16_t>(address)==c,"typed/cross-word load mismatch");
  require(!g.error,"valid packed operation failed");
 }
 for(unsigned lane=0;lane<lanes;++lane)for(unsigned a=0;a<capacity;++a)
  require(guests[lane].byte_at(a)==flat[lane*capacity+a],"logical bytes or neighboring lane corrupted");
 for(unsigned i=0;i<guard;++i)require(packed[i]==0xa5&&packed[packed.size()-1-i]==0xa5,"outer canary corrupted");
 for(unsigned lane=lanes;lane<padded;++lane)for(unsigned a=0;a<capacity;++a)
  require(packed[guard+std::size_t(lane/32)*capacity*32+(lane%32)*4+std::size_t(a&~3U)*32+(a&3)]==0xa5,"inactive padded lane corrupted");
 auto bad=guests[0];bad.error=0;bad.load<std::uint64_t>(capacity-7);
 require(bad.error==irisu::cuda_guest::bad_address,"logical bounds not enforced");
 bad=guests[0];bad.word_stride=2;bad.load<unsigned>(0);
 require(bad.error==irisu::cuda_guest::bad_address,"unsupported stride accepted");
 require(guests[0].byte_offset(0x10000004U)==0x200000080ULL,"physical byte offset truncated to32bits");
 // Allocator metadata, copyable freelists and partial fill paths use the same mapping.
 std::vector<unsigned char> dense(capacity);Guest reference;reference.memory=dense.data();reference.capacity=capacity;reference.heap=1024;reference.heap_limit=capacity-256;
 auto allocator=guests[0];allocator.fill(0,0,capacity);allocator.heap=1024;allocator.heap_limit=capacity-256;
 std::array<unsigned,16> live{},normal{};
 for(unsigned tick=0;tick<10000;++tick){rng=rng*1664525U+1013904223U;auto i=rng%live.size();
  if(live[i]){allocator.release(live[i]);reference.release(normal[i]);live[i]=normal[i]=0;}
  else{auto n=1+((rng>>8)%128);live[i]=allocator.alloc(n);normal[i]=reference.alloc(n);require(live[i]==normal[i],"allocator choices diverged");}
  require(!allocator.error&&!reference.error,"allocator error");}
 for(unsigned a=0;a<capacity;++a)require(allocator.byte_at(a)==dense[a],"allocator arena mismatch");
 std::puts("{\"status\":\"passed\",\"lanes\":33,\"random_memory_transitions\":30000,\"allocator_transitions\":10000,\"cross_word_64bit_and_unaligned\":true,\"overlap_copy\":true,\"neighbor_and_padding_canaries\":true,\"logical_bounds_and_stride_rejection\":true}");return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
