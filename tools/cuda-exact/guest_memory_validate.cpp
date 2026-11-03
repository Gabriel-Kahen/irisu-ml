#include "../../clone/cuda/guest.hpp"
#include <array>
#include <cstdio>
#include <vector>
using irisu::cuda_guest::Guest;
struct Item {unsigned address{},size{},pattern{};};
int main(){
 std::vector<unsigned char> arena(65536);Guest g;g.memory=arena.data();g.capacity=arena.size();g.heap=1024;g.heap_limit=60000;
 std::array<Item,32> items{};unsigned rng=123;
 for(unsigned tick=0;tick<50000;++tick){
  rng=rng*1664525+1013904223;auto& x=items[rng%items.size()];
  if(x.address){for(unsigned j=0;j<x.size;++j)if(arena[x.address+j]!=(x.pattern&255))return 2;g.release(x.address);x={};}
  else{rng=rng*1664525+1013904223;x.size=1+(rng%256);x.address=g.alloc(x.size);x.pattern=tick;g.fill(x.address,x.pattern,x.size);}
  if(g.error)return 3;
 }
 // A byte checkpoint must preserve all allocator links and deterministic choices.
 auto copied=arena;auto clone=g;clone.memory=copied.data();
 for(auto x:items)if(x.address){g.release(x.address);clone.release(x.address);}
 for(unsigned n=1;n<1000;n+=13){auto a=g.alloc(n),b=clone.alloc(n);if(a!=b||g.error!=clone.error)return 4;if(g.error)break;}
 if(arena!=copied||!g.allocation_reuses)return 5;
 Guest bad=g;bad.error=0;bad.release(1);if(!bad.error)return 6;
 std::printf("50000 recycled allocation transitions and cloned allocator parity passed; heap=%u reuses=%u\n",g.heap,g.allocation_reuses);
}
