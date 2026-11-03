#include "../../clone/cuda/search.hpp"
#include <cstdio>
#include <stdexcept>
using namespace irisu::cuda_search;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){
  try {
    Candidate c{};require(valid(c,1),"incumbent rejected");
    c.count=5;require(!valid(c,1),"oversize mutation list accepted");
    c={};c.count=2;c.mutations[0]={0,1};c.mutations[1]={0,2};
    require(!valid(c,1),"duplicate mutation tick accepted");
    c.count=1;c.mutations[0].offset=1;require(!valid(c,1),"out-of-horizon accepted");
    c.mutations[0]={0,1U<<31};require(!valid(c,1),"unencoded action accepted");
    c.mutations[0]={0,7};require(input_at(c,0,3)==7&&input_at(c,1,3)==3,"sparse mutation mismatch");
    Result complete{},death{};complete.final.score=999;complete.final.canonical_score=888;complete.canonical_score=100;
    complete.final.tick=99;complete.canonical_level=100;complete.natural=1;
    death.final.score=10000;death.canonical_score=10000;death.final.tick=100;
    death.canonical_level=99;death.natural=1;
    rank_keys(complete,100,101);rank_keys(death,100,101);
    require(better(complete,1,death,0),"incomplete death outranked completion");
    auto worse=complete;worse.canonical_score=99;worse.final.canonical_score=100000;worse.final.score=100000;
    rank_keys(worse,100,101);require(better(complete,1,worse,0),"live score replaced canonical score");
    require(better(complete,0,complete,1)&&!better(complete,1,complete,0),"unstable ties");
    worse=complete;worse.failed=1;require(better(complete,1,worse,0),"failed lane selected");
    Result r;const auto initial=r.trace_hash;fold(r,{});require(r.trace_hash!=initial,"frame digest omitted");
    std::puts("search sparse mutation, canonical objective, stable rank and failure gates passed");return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
