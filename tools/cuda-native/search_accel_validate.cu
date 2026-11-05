// Helper-only qualification; real Game/Scene continuation is a separate gate.
#include "search_accel.cuh"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace accel=irisu::cuda_native::search_accel;
namespace wire=irisu::cuda_search;
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
struct Device {
  std::vector<void*> pointers;
  ~Device(){for(auto* p:pointers)cudaFree(p);}
  template<class T>T* allocate(std::size_t count){T* p{};check(cudaMalloc(&p,std::max<std::size_t>(1,count*sizeof(T))));pointers.push_back(p);return p;}
};
struct alignas(8) MockWorld {std::uint64_t words[9217];};
struct alignas(8) MockScene {std::uint64_t words[12289];};
int main(){try{
  const unsigned counts[]={0,1,2,3,31,32,33,127,128,129,257,511,512,513,1023,1024,12288,16384};
  unsigned rank_cases=0;
  for(unsigned count:counts)for(unsigned pattern=0;pattern<3;++pattern){
    std::vector<wire::Result> values(count);std::uint64_t state=661003;
    for(unsigned i=0;i<count;++i){auto& value=values[i];value.failed=i%7==0?7U:0U;
      for(unsigned key=0;key<11;++key){
        state=state*6364136223846793005ULL+1442695040888963407ULL;
        std::int64_t number{};std::memcpy(&number,&state,8);
        value.objective[key]=pattern==0?0:pattern==2&&key<10?11:number;
      }
      if(pattern&&i%9==0)for(auto& key:value.objective)key=std::numeric_limits<std::int64_t>::min();
      if(pattern&&i%13==0)for(auto& key:value.objective)key=std::numeric_limits<std::int64_t>::max();
      if(i%17==0&&i)value=values[i-1]; // Exact ties must break by ORIGINAL index.
    }
    std::vector<unsigned> expected(count),actual(count);std::iota(expected.begin(),expected.end(),0);
    std::sort(expected.begin(),expected.end(),[&](unsigned a,unsigned b){return wire::better(values[a],a,values[b],b);});
    Device device;auto* dr=device.allocate<wire::Result>(count);auto* order=device.allocate<unsigned>(count);
    if(count)check(cudaMemcpy(dr,values.data(),count*sizeof(wire::Result),cudaMemcpyHostToDevice));
    std::size_t bytes{};check(accel::rank_workspace_bytes(dr,order,count,bytes));auto* workspace=device.allocate<unsigned char>(bytes);
    // Reverse/nontrivial initial contents prove initialization is not skipped.
    check(cudaMemset(order,0xa5,std::max<std::size_t>(1,count*sizeof(unsigned))));
    check(accel::rank_async(dr,order,count,workspace,bytes));
    if(count)check(cudaMemcpy(actual.data(),order,count*sizeof(unsigned),cudaMemcpyDeviceToHost));
    if(actual!=expected)throw std::runtime_error("rank permutation differs from canonical scalar comparator");
    std::vector<wire::Result> unchanged(count);
    if(count)check(cudaMemcpy(unchanged.data(),dr,count*sizeof(wire::Result),cudaMemcpyDeviceToHost));
    if(count&&std::memcmp(values.data(),unchanged.data(),count*sizeof(wire::Result)))throw std::runtime_error("rank helper changed Results");
    ++rank_cases;
  }
  constexpr unsigned source_count=13,count=5;const unsigned selection[count]={12,0,7,7,3};
  std::vector<MockWorld> worlds(source_count),actual_worlds(count);
  std::vector<MockScene> scenes(source_count),actual_scenes(count);
  for(unsigned i=0;i<source_count;++i){
    for(unsigned j=0;j<9217;++j)worlds[i].words[j]=0x1122334400000000ULL^(std::uint64_t(i)<<32)^j;
    for(unsigned j=0;j<12289;++j)scenes[i].words[j]=0x8877665500000000ULL^(std::uint64_t(i)<<32)^j;
  }
  unsigned copy_cases=0;
  for(std::size_t scene_bytes:{sizeof(MockScene),std::size_t(81928)})for(bool selected:{false,true}){
    Device device;auto* sw=device.allocate<MockWorld>(source_count);auto* ss=device.allocate<MockScene>(source_count);
    auto* dw=device.allocate<MockWorld>(count);auto* ds=device.allocate<MockScene>(count);auto* order=device.allocate<unsigned>(count);
    check(cudaMemcpy(sw,worlds.data(),worlds.size()*sizeof(MockWorld),cudaMemcpyHostToDevice));
    check(cudaMemcpy(ss,scenes.data(),scenes.size()*sizeof(MockScene),cudaMemcpyHostToDevice));
    check(cudaMemcpy(order,selection,sizeof(selection),cudaMemcpyHostToDevice));
    check(cudaMemset(dw,0xa5,count*sizeof(MockWorld)));check(cudaMemset(ds,0xa5,count*sizeof(MockScene)));
    check(accel::copy_tiled_async(dw,ds,sw,ss,count,selected?order:nullptr,scene_bytes));
    check(cudaMemcpy(actual_worlds.data(),dw,count*sizeof(MockWorld),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual_scenes.data(),ds,count*sizeof(MockScene),cudaMemcpyDeviceToHost));
    for(unsigned i=0;i<count;++i){const auto source=selected?selection[i]:0;
      if(std::memcmp(&actual_worlds[i],&worlds[source],sizeof(MockWorld))||std::memcmp(&actual_scenes[i],&scenes[source],scene_bytes))
        throw std::runtime_error("tiled copy differs from source bytes");
      const auto* tail=reinterpret_cast<const unsigned char*>(&actual_scenes[i]);
      for(std::size_t j=scene_bytes;j<sizeof(MockScene);++j)if(tail[j]!=0xa5)throw std::runtime_error("tiled copy modified omitted tail");
    }
    std::vector<MockWorld> unchanged_worlds(source_count);std::vector<MockScene> unchanged_scenes(source_count);
    check(cudaMemcpy(unchanged_worlds.data(),sw,source_count*sizeof(MockWorld),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(unchanged_scenes.data(),ss,source_count*sizeof(MockScene),cudaMemcpyDeviceToHost));
    if(std::memcmp(unchanged_worlds.data(),worlds.data(),source_count*sizeof(MockWorld))||
       std::memcmp(unchanged_scenes.data(),scenes.data(),source_count*sizeof(MockScene)))throw std::runtime_error("copy helper changed source bytes");
    ++copy_cases;
  }
  std::printf("{\"schema\":\"irisu-search-acceleration-helper-qualifier-v1\",\"passed\":true,\"rank_cases\":%u,\"tiled_copy_cases\":%u,\"rank_wire_unchanged\":true,\"source_bytes_unchanged\":true,\"omitted_tails_unchanged\":true,\"scope\":\"Index sort and raw mock snapshot transport only; actual Game/Scene pointer rebind, continuation, native parity and performance require separate gates\"}\n",rank_cases,copy_cases);
  return 0;
}catch(const std::exception& error){std::fprintf(stderr,"search-accel-validate: %s\n",error.what());return 1;}}
