// The actual pinned MSVC physics instruction stream, executed on host or CUDA.
// Binary stdout matches physics_oracle.c: tick/body/(x,y,r,vx,vy) float words.
#include "../../clone/cuda/guest_program.cuh"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace irisu::cuda_guest {
IRISU_GUEST_HD void scenario(Guest& g,std::uint32_t* output,unsigned steps,bool mutations) {
  auto f=[] IRISU_GUEST_HD (float v){return float_word(v);};
  const auto world=call(g,msvc_b2d_world_create,f(-100),f(-100),f(100),f(100),f(10),f(1));
  auto floor=call(g,msvc_b2d_world_create_box,world,f(20),f(1),f(0),f(10),f(0),f(0),f(.3f),f(.1f));
  std::uint32_t bodies[]={
    call(g,msvc_b2d_world_create_circle,world,f(.5f),f(0),f(0),f(1),f(.3f),f(.1f)),
    call(g,msvc_b2d_world_create_box,world,f(1),f(1),f(.3f),f(-2),f(0),f(1),f(.3f),f(.1f)),
    call(g,msvc_b2d_world_create_triangle,world,f(1),f(1),f(-.2f),f(-4),f(0),f(1),f(.3f),f(.1f))};
  call(g,msvc_b2d_world_set_user_data,world,floor,1);
  for(unsigned i=0;i<3;++i)call(g,msvc_b2d_world_set_user_data,world,bodies[i],i+2);
  for(unsigned tick=1;tick<=steps&&!g.error;++tick) {
    if(mutations&&tick==200)call(g,msvc_b2d_world_set_v,world,bodies[0],f(1.25f),f(-3));
    if(mutations&&tick==300) {
      call(g,msvc_b2d_world_destroy_body,world,bodies[1]);
      bodies[1]=call(g,msvc_b2d_world_create_box,world,f(1),f(1),f(1),f(-3),f(.2f),f(1),f(.3f),f(.1f));
      call(g,msvc_b2d_world_set_user_data,world,bodies[1],3);
    }
    call(g,msvc_b2d_world_step,world,f(1.0f/60),10);
    for(unsigned i=0;i<3&&!g.error;++i) {
      auto* dst=output+(tick-1)*15+i*5;
      for(unsigned j=0;j<3;++j) {
        const std::uint32_t getters[]={msvc_b2d_world_get_x,msvc_b2d_world_get_y,msvc_b2d_world_get_r};
        call(g,getters[j],world,bodies[i]);
        dst[j]=float_word((float)g.st());g.fpop();
      }
      call(g,msvc_b2d_world_get_v,world,bodies[i],0x19000u,0x19004u);
      dst[3]=g.load<std::uint32_t>(0x19000);dst[4]=g.load<std::uint32_t>(0x19004);
    }
  }
}
} // namespace irisu::cuda_guest

#ifdef __CUDACC__
__global__ void run_guest(irisu::cuda_guest::Guest* guests,std::uint32_t* output,unsigned steps,bool mutations,unsigned lanes) {
  const auto lane=blockIdx.x*blockDim.x+threadIdx.x;
  if(lane<lanes){
    auto local=guests[lane];
    irisu::cuda_guest::scenario(local,output+lane*steps*15,steps,mutations);
    guests[lane]=local;
  }
}
#endif

int main(int argc,char** argv) {
  if(argc<2) {std::fprintf(stderr,"usage: physics-probe image.bin [steps] [--gpu] [--mutations] [--lanes N]\n");return 2;}
  unsigned steps=600,lanes=1;bool gpu=false,mutations=false;
  for(int i=2;i<argc;++i) {
    const std::string a=argv[i];
    if(a=="--gpu")gpu=true;
    else if(a=="--mutations")mutations=true;
    else if(a=="--lanes"&&i+1<argc)lanes=std::stoul(argv[++i]);
    else steps=std::stoul(a);
  }
  if(!steps||steps>100000||!lanes||lanes>8192)return 2;
  std::ifstream input(argv[1],std::ios::binary);
  const std::vector<unsigned char> image((std::istreambuf_iterator<char>(input)),{});
  if(image.empty()||image.size()>0x20000)return 2;
  constexpr unsigned capacity=2*1024*1024;
  std::vector<std::uint32_t> output(std::size_t(lanes)*steps*15);
  std::vector<irisu::cuda_guest::Guest> guests(lanes);
  std::vector<unsigned char> memory(gpu?0:std::size_t(capacity)*lanes);
#ifdef __CUDACC__
  unsigned char* device_memory=nullptr;std::uint32_t* device_output=nullptr;irisu::cuda_guest::Guest* device_guests=nullptr;
  auto check=[](cudaError_t code){if(code!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(code));std::exit(1);}};
  if(gpu) {
    check(cudaMalloc(&device_memory,std::size_t(capacity)*lanes));
    check(cudaMemset(device_memory,0,std::size_t(capacity)*lanes));
    for(unsigned i=0;i<lanes;++i)check(cudaMemcpy(device_memory+std::size_t(i)*capacity,image.data(),image.size(),cudaMemcpyHostToDevice));
    check(cudaMalloc(&device_output,output.size()*4));check(cudaMemset(device_output,0,output.size()*4));
    check(cudaMalloc(&device_guests,guests.size()*sizeof(guests[0])));
  }
#else
  if(gpu){std::fprintf(stderr,"CUDA was not compiled in\n");return 2;}
#endif
  for(unsigned i=0;i<lanes;++i) {
    auto& g=guests[i];g.capacity=capacity;g.heap=0x20000;g.heap_limit=capacity-0x10000;
#ifdef __CUDACC__
    if(gpu)g.memory=device_memory+std::size_t(i)*capacity;
    else
#endif
    {g.memory=memory.data()+std::size_t(i)*capacity;std::memcpy(g.memory,image.data(),image.size());}
  }
  double kernel_ms=0;
  const auto scenario_begin=std::chrono::steady_clock::now();
#ifdef __CUDACC__
  if(gpu) {
    cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
    cudaFuncAttributes attributes{};check(cudaFuncGetAttributes(&attributes,run_guest));
    check(cudaMemcpy(device_guests,guests.data(),guests.size()*sizeof(guests[0]),cudaMemcpyHostToDevice));
    check(cudaEventRecord(start));
    run_guest<<<(lanes+31)/32,32>>>(device_guests,device_output,steps,mutations,lanes);
    check(cudaGetLastError());check(cudaEventRecord(stop));check(cudaEventSynchronize(stop));
    float milliseconds=0;check(cudaEventElapsedTime(&milliseconds,start,stop));kernel_ms=milliseconds;
    cudaEventDestroy(start);cudaEventDestroy(stop);
    check(cudaMemcpy(guests.data(),device_guests,guests.size()*sizeof(guests[0]),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(output.data(),device_output,output.size()*4,cudaMemcpyDeviceToHost));
    cudaFree(device_guests);cudaFree(device_output);cudaFree(device_memory);
  }else
#endif
  for(unsigned i=0;i<lanes;++i)irisu::cuda_guest::scenario(guests[i],output.data()+std::size_t(i)*steps*15,steps,mutations);
  const auto elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-scenario_begin).count();
  unsigned max_heap=0;unsigned long long dispatches=0;
  for(unsigned i=0;i<lanes;++i) {
    const auto& g=guests[i];
    if(g.heap>max_heap)max_heap=g.heap;dispatches+=g.instructions;
    if(g.error) {std::fprintf(stderr,"lane=%u error=%u pc=0x%x heap=%u fpdepth=%u dispatches=%llu st=%.17g\n",i,g.error,g.fault_pc,g.heap,g.depth,(unsigned long long)g.instructions,g.fault_value);return 1;}
  }
  std::fprintf(stderr,"{\"backend\":\"%s\",\"lanes\":%u,\"steps\":%u,\"mutations\":%s,\"kernel_ms\":%.6f,\"elapsed_ms\":%.6f,\"bytesperlane\":%u,\"maxheapbytes\":%u,\"dispatches\":%llu,\"failures\":0}\n",gpu?"cuda":"translated-host",lanes,steps,mutations?"true":"false",kernel_ms,elapsed_ms,capacity,max_heap,dispatches);
  return std::fwrite(output.data(),4,output.size(),stdout)==output.size()?0:1;
}
