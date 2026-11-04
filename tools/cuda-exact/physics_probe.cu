// The actual pinned MSVC physics instruction stream, executed on host or CUDA.
// Binary stdout matches physics_oracle.c: tick/body/(x,y,r,vx,vy) float words.
#include "../../clone/cuda/guest_program.cuh"
#include "image_identity.hpp"
#include "game_digest.hpp"
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
__global__ void initialize_strided_image(unsigned char* memory,const unsigned char* image,
    unsigned image_bytes,unsigned capacity,unsigned lanes) {
  const auto index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const unsigned words=(image_bytes+3)/4;
  const auto cohort=index/(std::size_t(words)*32);
  if(cohort>=(lanes+31)/32)return;
  const auto slot=index%(std::size_t(words)*32);
  const unsigned lane=slot&31,word=slot/32;
  if(cohort*32+lane>=lanes)return;
  std::uint32_t value=0;
  for(unsigned byte=0;byte<4;++byte)if(word*4+byte<image_bytes)
    value|=std::uint32_t(image[word*4+byte])<<(8*byte);
  *reinterpret_cast<std::uint32_t*>(memory+cohort*capacity*32+std::size_t(word)*128+lane*4)=value;
}
__global__ void run_guest(irisu::cuda_guest::Guest* guests,std::uint32_t* output,unsigned steps,bool mutations,bool mixed,unsigned lanes) {
  const auto lane=blockIdx.x*blockDim.x+threadIdx.x;
  if(lane<lanes){
    auto local=guests[lane];
    irisu::cuda_guest::scenario(local,output+lane*steps*15,steps,mutations||(mixed&&(lane&1)));
    guests[lane]=local;
  }
}
#endif

int main(int argc,char** argv) {
  if(argc<2) {std::fprintf(stderr,"usage: physics-probe image.bin [steps] [--gpu] [--mutations] [--mixed-mutations] [--lanes N] [--arena-bytes N] [--word-stride 1|32] [--dump-state path]\n");return 2;}
  unsigned steps=600,lanes=1,capacity=2*1024*1024,stride=1;bool gpu=false,mutations=false,mixed=false;
  std::string state_path;
  for(int i=2;i<argc;++i) {
    const std::string a=argv[i];
    if(a=="--gpu")gpu=true;
    else if(a=="--mutations")mutations=true;
    else if(a=="--mixed-mutations")mixed=true;
    else if(a=="--lanes"&&i+1<argc)lanes=std::stoul(argv[++i]);
    else if(a=="--arena-bytes"&&i+1<argc)capacity=std::stoul(argv[++i]);
    else if(a=="--word-stride"&&i+1<argc)stride=std::stoul(argv[++i]);
    else if(a=="--dump-state"&&i+1<argc)state_path=argv[++i];
    else steps=std::stoul(a);
  }
  if(!steps||steps>100000||!lanes||lanes>65536 || capacity<0x80000 || capacity>0x40000000 || (capacity&15) || (stride!=1&&stride!=32))return 2;
  std::ifstream input(argv[1],std::ios::binary);
  const std::vector<unsigned char> image((std::istreambuf_iterator<char>(input)),{});
  try { irisu::cuda_exact::verify_image(image.data(),image.size()); }
  catch(const std::exception& error) { std::fprintf(stderr,"%s\n",error.what());return 2; }
  std::vector<std::uint32_t> output(std::size_t(lanes)*steps*15);
  std::vector<irisu::cuda_guest::Guest> guests(lanes);
  const unsigned padded_lanes=stride==32?(lanes+31)&~31u:lanes;
  const auto memory_bytes=std::size_t(capacity)*padded_lanes;
  auto arena_offset=[&](unsigned lane){return stride==1?std::size_t(lane)*capacity:
    std::size_t(lane/32)*capacity*32+(lane%32)*4;};
  std::vector<unsigned char> memory(gpu?0:memory_bytes);
#ifdef __CUDACC__
  unsigned char *device_memory=nullptr,*device_image=nullptr;std::uint32_t* device_output=nullptr;irisu::cuda_guest::Guest* device_guests=nullptr;
  auto check=[](cudaError_t code){if(code!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(code));std::exit(1);}};
  if(gpu) {
    check(cudaDeviceSetLimit(cudaLimitStackSize,16U<<10));
    check(cudaMalloc(&device_memory,memory_bytes));
    check(cudaMemset(device_memory,0,memory_bytes));
    if(stride==1)for(unsigned i=0;i<lanes;++i)check(cudaMemcpy(device_memory+std::size_t(i)*capacity,image.data(),image.size(),cudaMemcpyHostToDevice));
    else {
      check(cudaMalloc(&device_image,image.size()));check(cudaMemcpy(device_image,image.data(),image.size(),cudaMemcpyHostToDevice));
      const auto threads=std::size_t((lanes+31)/32)*((image.size()+3)/4)*32;
      initialize_strided_image<<<(threads+255)/256,256>>>(device_memory,device_image,image.size(),capacity,lanes);
      check(cudaGetLastError());
    }
    check(cudaMalloc(&device_output,output.size()*4));check(cudaMemset(device_output,0,output.size()*4));
    check(cudaMalloc(&device_guests,guests.size()*sizeof(guests[0])));
  }
#else
  if(gpu){std::fprintf(stderr,"CUDA was not compiled in\n");return 2;}
#endif
  for(unsigned i=0;i<lanes;++i) {
    auto& g=guests[i];g.capacity=capacity;g.heap=0x20000;g.heap_limit=capacity-0x10000;g.word_stride=stride;
#ifdef __CUDACC__
    if(gpu)g.memory=device_memory+arena_offset(i);
    else
#endif
    {g.memory=memory.data()+arena_offset(i);if(stride==1)std::memcpy(g.memory,image.data(),image.size());
      else for(unsigned a=0;a<image.size();++a)g.byte_at(a,image[a]);}
  }
  double kernel_ms=0;
  unsigned registers=0;std::size_t local_bytes=0,shared_bytes=0;
  const auto scenario_begin=std::chrono::steady_clock::now();
#ifdef __CUDACC__
  if(gpu) {
    cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
    cudaFuncAttributes attributes{};check(cudaFuncGetAttributes(&attributes,run_guest));
    registers=attributes.numRegs;local_bytes=attributes.localSizeBytes;shared_bytes=attributes.sharedSizeBytes;
    check(cudaMemcpy(device_guests,guests.data(),guests.size()*sizeof(guests[0]),cudaMemcpyHostToDevice));
    check(cudaEventRecord(start));
    run_guest<<<(lanes+31)/32,32>>>(device_guests,device_output,steps,mutations,mixed,lanes);
    check(cudaGetLastError());check(cudaEventRecord(stop));check(cudaEventSynchronize(stop));
    float milliseconds=0;check(cudaEventElapsedTime(&milliseconds,start,stop));kernel_ms=milliseconds;
    cudaEventDestroy(start);cudaEventDestroy(stop);
    check(cudaMemcpy(guests.data(),device_guests,guests.size()*sizeof(guests[0]),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(output.data(),device_output,output.size()*4,cudaMemcpyDeviceToHost));
    if(!state_path.empty()){memory.resize(memory_bytes);check(cudaMemcpy(memory.data(),device_memory,memory_bytes,cudaMemcpyDeviceToHost));}
    cudaFree(device_guests);cudaFree(device_output);cudaFree(device_memory);if(device_image)cudaFree(device_image);
  }else
#endif
  for(unsigned i=0;i<lanes;++i)irisu::cuda_guest::scenario(guests[i],output.data()+std::size_t(i)*steps*15,steps,mutations||(mixed&&(i&1)));
  const auto elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-scenario_begin).count();
  unsigned max_heap=0;unsigned long long dispatches=0;
  for(unsigned i=0;i<lanes;++i) {
    const auto& g=guests[i];
    if(g.heap>max_heap)max_heap=g.heap;dispatches+=g.instructions;
    if(g.error) {std::fprintf(stderr,"lane=%u error=%u pc=0x%x heap=%u fpdepth=%u dispatches=%llu st=%.17g\n",i,g.error,g.fault_pc,g.heap,g.depth,(unsigned long long)g.instructions,g.fault_value);return 1;}
  }
  if(!state_path.empty()) {
    auto* file=std::fopen(state_path.c_str(),"w");if(!file)return 2;
    std::fprintf(file,"{\"schema\":\"irisu-guest-logical-state-v1\",\"lanes\":[");
    std::vector<unsigned char> logical(capacity);const char* hex="0123456789abcdef";
    for(unsigned lane=0;lane<lanes;++lane){auto g=guests[lane];g.memory=memory.data()+arena_offset(lane);
      irisu::game_digest::Hash hash;
      for(auto value:g.r)hash.scalar(value);for(auto value:g.fp)hash.scalar(value);
#define FIELD(name) hash.scalar(g.name)
      FIELD(fault_value);FIELD(top);FIELD(depth);FIELD(status);FIELD(control);FIELD(flags);FIELD(pc);FIELD(error);FIELD(fault_pc);
      FIELD(heap);FIELD(heap_limit);FIELD(capacity);FIELD(instructions);FIELD(free_head);FIELD(allocation_reuses);
#undef FIELD
      for(unsigned a=0;a<capacity;++a)logical[a]=g.byte_at(a);
      unsigned char digest[SHA256_DIGEST_LENGTH];SHA256(logical.data(),logical.size(),digest);char encoded[65];
      for(unsigned i=0;i<32;++i){encoded[2*i]=hex[digest[i]>>4];encoded[2*i+1]=hex[digest[i]&15];}encoded[64]=0;
      std::fprintf(file,"%s{\"register_state_hash\":%llu,\"arena_sha256\":\"%s\",\"error\":%u}",lane?",":"",(unsigned long long)hash.value,encoded,g.error);
    }
    std::fprintf(file,"]}\n");if(std::fclose(file))return 2;
  }
  std::fprintf(stderr,"{\"backend\":\"%s\",\"lanes\":%u,\"padded_lanes\":%u,\"word_stride\":%u,\"steps\":%u,\"mutations\":%s,\"mixed_mutations\":%s,\"kernel_ms\":%.6f,\"elapsed_ms\":%.6f,\"bytesperlane\":%u,\"maxheapbytes\":%u,\"dispatches\":%llu,\"registers_per_thread\":%u,\"local_bytes_per_thread\":%zu,\"shared_bytes_per_block\":%zu,\"failures\":0}\n",gpu?"cuda":"translated-host",lanes,padded_lanes,stride,steps,mutations?"true":"false",mixed?"true":"false",kernel_ms,elapsed_ms,capacity,max_heap,dispatches,registers,local_bytes,shared_bytes);
  return std::fwrite(output.data(),4,output.size(),stdout)==output.size()?0:1;
}
