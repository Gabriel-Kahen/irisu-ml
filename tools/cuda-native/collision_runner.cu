#include "../../clone/cuda_native/collision.hpp"
#include "collision_wire.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace native=irisu::cuda_native;
struct Case {std::uint32_t words[22];};
NATIVE_HD float from_word(std::uint32_t word){union{std::uint32_t u;float f;}v{word};return v.f;}
NATIVE_HD void evaluate(const Case& in,native::Manifold& out,std::uint32_t& status){
  native::ShapeState a{},b{};
  a.position={from_word(in.words[0]),from_word(in.words[1])};a.radius=from_word(in.words[2]);
  b.position={from_word(in.words[3]),from_word(in.words[4])};b.radius=from_word(in.words[5]);
  std::memcpy(&out,in.words+7,sizeof(out));
  status=static_cast<std::uint32_t>(native::collide_circle(out,a,b,in.words[6]!=0).status);
}
NATIVE_HD native::ShapeState shape(const native::wire::ShapePacket& in){
  native::ShapeState out{};out.kind=static_cast<native::ShapeKind>(in.kind);out.vertex_count=in.vertex_count;
  out.position={in.position[0],in.position[1]};out.R={{in.R[0],in.R[1]},{in.R[2],in.R[3]}};out.radius=in.radius;
  for(unsigned i=0;i<8;++i){out.vertices[i]={in.vertices[2*i],in.vertices[2*i+1]};out.normals[i]={in.normals[2*i],in.normals[2*i+1]};}
  return out;
}
NATIVE_HD void evaluate(const native::wire::CollisionCaseV2& in,native::Manifold& out,std::uint32_t& status){
  const auto a=shape(in.first),b=shape(in.second);std::memcpy(&out,in.manifold,sizeof(out));
  const auto result=native::collide(out,a,b,in.conservative!=0);
  status=static_cast<std::uint32_t>(result.status);
}
template<class T> __global__ void collision_kernel(const T* in,native::Manifold* out,std::uint32_t* status,unsigned count){
  const auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)evaluate(in[i],out[i],status[i]);
}
void check(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
struct Allocation {
  void* value{};
  ~Allocation(){if(value)cudaFree(value);}
  template<class T>T* allocate(std::size_t count){check(cudaMalloc(&value,count*sizeof(T)));return static_cast<T*>(value);}
};
struct Event {cudaEvent_t value{};Event(){check(cudaEventCreate(&value));}~Event(){cudaEventDestroy(value);}};
template<class T> int run_cases(std::ifstream& file,unsigned count,bool gpu,unsigned repeats,
  unsigned warmups,unsigned threads,std::chrono::steady_clock::time_point started){
  std::vector<T> input(count);
  std::vector<native::Manifold> output(count);std::vector<std::uint32_t> status(count);
  file.read(reinterpret_cast<char*>(input.data()),std::streamsize(input.size()*sizeof(T)));
  if(!file||file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("wrong circle packet size");
  for(const auto& c:input){
    const auto flag=[&](){if constexpr(std::is_same_v<T,Case>)return c.words[6];else return c.conservative;}();
    if(flag>1)throw std::runtime_error("invalid conservative flag");
  }
  std::vector<double> samples;unsigned registers=0;std::size_t local_bytes=0;
  if(gpu){
    Allocation a,b,c;auto* in=a.allocate<T>(count);auto* out=b.allocate<native::Manifold>(count);
    auto* errors=c.allocate<std::uint32_t>(count);
    check(cudaMemcpy(in,input.data(),input.size()*sizeof(T),cudaMemcpyHostToDevice));
    cudaFuncAttributes attributes{};check(cudaFuncGetAttributes(&attributes,collision_kernel<T>));
    registers=attributes.numRegs;local_bytes=attributes.localSizeBytes;
    Event begin,end;
    for(unsigned run=0;run<warmups+repeats;++run){
      check(cudaEventRecord(begin.value));
      collision_kernel<T><<<(count+threads-1)/threads,threads>>>(in,out,errors,count);
      check(cudaGetLastError());check(cudaEventRecord(end.value));check(cudaEventSynchronize(end.value));
      float ms{};check(cudaEventElapsedTime(&ms,begin.value,end.value));if(run>=warmups)samples.push_back(ms);
    }
    check(cudaMemcpy(output.data(),out,output.size()*sizeof(output[0]),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(status.data(),errors,status.size()*sizeof(status[0]),cudaMemcpyDeviceToHost));
  }else{
    for(unsigned run=0;run<warmups+repeats;++run){const auto begin=std::chrono::steady_clock::now();
      for(unsigned i=0;i<count;++i)evaluate(input[i],output[i],status[i]);
      if(run>=warmups)samples.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());
    }
  }
  unsigned rejected=0,contacts=0;for(unsigned i=0;i<count;++i){rejected+=status[i]!=0;contacts+=!status[i]&&output[i].point_count>0;}
  auto sorted=samples;std::sort(sorted.begin(),sorted.end());
  const double median=(sorted[(sorted.size()-1)/2]+sorted[sorted.size()/2])*0.5;
  const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
  std::fprintf(stderr,"{\"schema\":\"irisu-native-collision-stage-v1\",\"backend\":\"%s\",\"scope\":\"collision stage only\",\"cases\":%u,\"contacts\":%u,\"rejected\":%u,\"warmups\":%u,\"repeats\":%u,\"median_stage_ms\":%.9f,\"elapsed_before_stdout_ms\":%.9f,\"registers\":%u,\"local_bytes\":%zu,\"promotion_eligible\":false,\"samples_ms\":[",gpu?"cuda":"typed-host",count,contacts,rejected,warmups,repeats,median,elapsed,registers,local_bytes);
  for(unsigned i=0;i<samples.size();++i)std::fprintf(stderr,"%s%.9f",i?",":"",samples[i]);std::fputs("]}\n",stderr);
  if(rejected)return 1;
  return std::fwrite(output.data(),sizeof(output[0]),output.size(),stdout)==output.size()?0:1;
}
int main(int argc,char** argv){try{
  const auto started=std::chrono::steady_clock::now();
  if(argc<2)throw std::runtime_error("usage: collision-runner cases.bin [--gpu] [--repeats N] [--warmups N] [--threads N]");
  bool gpu=false;unsigned repeats=3,warmups=1,threads=128;
  for(int i=2;i<argc;++i){const std::string arg=argv[i];
    if(arg=="--gpu")gpu=true;
    else if(arg=="--repeats"&&i+1<argc)repeats=std::stoul(argv[++i]);
    else if(arg=="--warmups"&&i+1<argc)warmups=std::stoul(argv[++i]);
    else if(arg=="--threads"&&i+1<argc)threads=std::stoul(argv[++i]);
    else throw std::runtime_error("unknown collision argument");
  }
  if(!repeats||repeats>20||warmups>10||!threads||threads>256)throw std::runtime_error("invalid bounded collision options");
  std::ifstream file(argv[1],std::ios::binary);native::wire::Header h{};file.read(reinterpret_cast<char*>(&h),sizeof(h));
  if(!file||!h.count||h.count>(1U<<22))throw std::runtime_error("invalid collision packet count");
  if(h.magic==0x31434e49&&h.version==1&&h.case_words==22)
    return run_cases<Case>(file,h.count,gpu,repeats,warmups,threads,started);
  if(h.magic==native::wire::collision_magic_v2&&h.version==2&&h.case_words==98)
    return run_cases<native::wire::CollisionCaseV2>(file,h.count,gpu,repeats,warmups,threads,started);
  throw std::runtime_error("unsupported collision packet format");
}catch(const std::exception& e){std::fprintf(stderr,"native circle: %s\n",e.what());return 1;}}
