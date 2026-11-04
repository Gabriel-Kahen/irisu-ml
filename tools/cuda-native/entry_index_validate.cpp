#include "../../clone/cuda_native/entry_index.cuh"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>
using Index=irisu::cuda_game::EntryIndex;
namespace {
unsigned comparisons{};
void require(bool value){if(!value)throw std::runtime_error("entry index differs from first-match linear reference");}
void rebuild(Index& index,const std::vector<unsigned>& ids){index.rebuild();for(unsigned i=0;i<ids.size();++i)require(index.insert(ids[i],i));}
void compare(const Index& index,const std::vector<unsigned>& ids,unsigned id){
  unsigned expected=Index::missing;for(unsigned i=0;i<ids.size();++i)if(ids[i]==id){expected=i;break;}
  require(index.find(id)==expected);++comparisons;
}
}
int main(){try{
  Index index;index.reset();std::vector<unsigned> ids;
  for(unsigned i=0;i<512;++i){const unsigned id=i%7?i*1024:0x80000000U+i;
    ids.push_back(id);require(index.insert(id,i));for(unsigned j=0;j<=i;++j)compare(index,ids,ids[j]);compare(index,ids,0xffffffffU);}
  // Duplicate keys deliberately retain the first vector entry.
  require(index.insert(ids[17],511));compare(index,ids,ids[17]);
  unsigned random=1;
  for(unsigned cycle=0;cycle<1000;++cycle){
    random=random*1664525U+1013904223U;const unsigned begin=random%ids.size();
    ids.erase(ids.begin()+begin,ids.begin()+std::min<std::size_t>(ids.size(),begin+1+cycle%11));rebuild(index,ids);
    for(unsigned j=0;j<17&&ids.size()<512;++j){const unsigned id=0x40000000U+cycle*17+j;ids.push_back(id);require(index.insert(id,ids.size()-1));}
    for(unsigned id:ids)compare(index,ids,id);compare(index,ids,0xffffffffU);
  }
  index.generation=0xffffffffU;rebuild(index,ids);for(unsigned id:ids)compare(index,ids,id);
  index.reset();compare(index,{},0);compare(index,{},0xffffffffU);
  for(unsigned i=0;i<Index::capacity;++i)require(index.insert(i,i));require(!index.insert(0xffffffffU,0));
  require(index.insert(0,999)&&index.find(0)==0);
  std::printf("{\"pass\":true,\"host_only\":true,\"lookup_comparisons\":%u,\"delete_rebuild_cycles\":1000,\"collision_chains\":true,\"duplicate_first_match\":true,\"epoch_wrap\":true,\"full_table_rejection\":true}\n",comparisons);return 0;
}catch(const std::exception& error){std::fprintf(stderr,"entry-index-validate: %s\n",error.what());return 1;}}
