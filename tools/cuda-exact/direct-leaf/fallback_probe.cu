// Differential receipt driver for the isolated typed leaf and its fallback.
#include "../../../clone/cuda/guest.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>
namespace irisu::cuda_guest { __device__ void region_30224(Guest&,std::uint64_t&,unsigned); }
using irisu::cuda_guest::Guest;
constexpr unsigned count=13,capacity=524288;
__global__ void probe(Guest* states,std::uint64_t* budgets) {
  const auto i=threadIdx.x;if(i<count)irisu::cuda_guest::region_30224(states[i],budgets[i],0);
}
int main(int argc,char** argv){
 if(argc!=2)return 2;std::ifstream input(argv[1],std::ios::binary);std::vector<unsigned char> image((std::istreambuf_iterator<char>(input)),{});if(image.empty()||image.size()>capacity)return 2;
 std::vector<unsigned char> arena(count*capacity);std::array<Guest,count> states{};
 std::array<std::uint64_t,count> budgets{0,1,2,3,4,255,256,256,512,512,512,512,512};
 unsigned char* device_arena;Guest* device_states;std::uint64_t* device_budgets;
 auto check=[](cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}};
 check(cudaMalloc(&device_arena,arena.size()));check(cudaMalloc(&device_states,sizeof(states)));check(cudaMalloc(&device_budgets,sizeof(budgets)));
 for(unsigned i=0;i<count;++i){std::copy(image.begin(),image.end(),arena.begin()+i*capacity);std::fill(arena.begin()+i*capacity+512,arena.begin()+i*capacity+4096,0);auto& g=states[i];g.memory=arena.data()+i*capacity;g.capacity=capacity;g.pc=30224;g.r[1]=512;g.r[4]=3584;g.flags=0x845;g.status=0x4100;g.top=3;g.instructions=17;
  for(unsigned j=0;j<8;++j)g.fp[j]=j+.125;
  g.store<std::uint32_t>(3584,0);g.store<std::uint32_t>(516,1024);
  if(i==7){g.depth=1;}
  if(i>=8){g.store<std::uint32_t>(520,1);g.store<std::uint32_t>(1124,2048);g.store<std::uint32_t>(1128,2304);g.store<std::uint32_t>(1140,1);}
  if(i==9)g.store<std::uint32_t>(1112,0x7fc00001u); // NaN normal -> unsupported numeric domain.
  if(i==10)g.r[1]=capacity-4; // Invalid solver address: original fallback must fail closed.
  if(i==12)g.store<std::uint32_t>(1140,3); // Outside native point-count domain: fallback.
  if(i==11)g.error=irisu::cuda_guest::aborted;
  g.memory=device_arena+i*capacity;
 }
 check(cudaMemcpy(device_arena,arena.data(),arena.size(),cudaMemcpyHostToDevice));check(cudaMemcpy(device_states,states.data(),sizeof(states),cudaMemcpyHostToDevice));check(cudaMemcpy(device_budgets,budgets.data(),sizeof(budgets),cudaMemcpyHostToDevice));
 probe<<<1,32>>>(device_states,device_budgets);check(cudaDeviceSynchronize());
 check(cudaMemcpy(states.data(),device_states,sizeof(states),cudaMemcpyDeviceToHost));check(cudaMemcpy(budgets.data(),device_budgets,sizeof(budgets),cudaMemcpyDeviceToHost));check(cudaMemcpy(arena.data(),device_arena,arena.size(),cudaMemcpyDeviceToHost));
 for(unsigned i=0;i<count;++i){states[i].memory=nullptr;std::fprintf(stderr,"regs%u ecx=%u ebx=%u edx=%u esi=%u esp=%u faultpc=%u value=%g\n",i,states[i].r[1],states[i].r[3],states[i].r[2],states[i].r[6],states[i].r[4],states[i].fault_pc,states[i].fault_value);std::fprintf(stderr,"case %u error=%u pc=%u budget=%llu instructions=%llu\n",i,states[i].error,states[i].pc,(unsigned long long)budgets[i],(unsigned long long)states[i].instructions);}
 std::fwrite(states.data(),1,sizeof(states),stdout);std::fwrite(budgets.data(),1,sizeof(budgets),stdout);std::fwrite(arena.data(),1,arena.size(),stdout);
 check(cudaFree(device_states));check(cudaFree(device_budgets));check(cudaFree(device_arena));
}
