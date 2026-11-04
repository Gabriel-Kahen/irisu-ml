#include "factory_wire.hpp"
#include "../../clone/cuda_native/factory.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
using namespace irisu::cuda_native;
namespace {
struct Result {std::uint64_t hash;std::uint32_t status;};
__global__ void construct(const FactoryInput* input,Result* output,unsigned count){
  const auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  const auto& in=input[i];BodyDefinition bd{};ShapeDefinition sd{};BodyState body{};ShapeState shape{};
  auto result=wrapper_definition(bd,sd,in.kind,in.width,in.height,in.x,in.y,in.angle,in.density,in.friction,in.restitution,in.scale);
  if(result)result=construct_body_shape(body,shape,bd,sd);
  output[i]={factory_hash(body,shape),std::uint32_t(result.status)};
}
void check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
}
int main(int argc,char** argv){try{
  if(argc!=2)throw std::runtime_error("usage: factory-runner native-factory-packets.bin");
  std::ifstream file(argv[1],std::ios::binary);if(!file)throw std::runtime_error("cannot open factory packets");
  auto read=[&](){std::uint32_t v{};if(!file.read(reinterpret_cast<char*>(&v),4))throw std::runtime_error("truncated factory packet");return v;};
  if(read()!=factory_magic||read()!=factory_version)throw std::runtime_error("factory packet identity mismatch");
  const unsigned count=read();if(!count||count>1000000||read()!=factory_case_words)throw std::runtime_error("invalid factory count/stride");
  std::vector<FactoryInput> input(count);std::vector<std::uint64_t> expected(count);std::vector<Result> output(count);
  for(unsigned i=0;i<count;++i){std::uint32_t words[10];for(auto& word:words)word=read();std::memcpy(&input[i],words,sizeof(FactoryInput));
    const auto low=read(),high=read();expected[i]=std::uint64_t(low)|(std::uint64_t(high)<<32);}
  if(file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing factory packet data");
  FactoryInput* device_input{};Result* device_output{};check(cudaMalloc(&device_input,count*sizeof(FactoryInput)));check(cudaMalloc(&device_output,count*sizeof(Result)));
  check(cudaMemcpy(device_input,input.data(),count*sizeof(FactoryInput),cudaMemcpyHostToDevice));
  construct<<<(count+127)/128,128>>>(device_input,device_output,count);check(cudaGetLastError());check(cudaDeviceSynchronize());
  check(cudaMemcpy(output.data(),device_output,count*sizeof(Result),cudaMemcpyDeviceToHost));cudaFree(device_input);cudaFree(device_output);
  for(unsigned i=0;i<count;++i)if(output[i].status||output[i].hash!=expected[i]){
    std::fprintf(stderr,"factory case%u kind%u status%u gpu%016llx native%016llx\n",i,input[i].kind,output[i].status,
      static_cast<unsigned long long>(output[i].hash),static_cast<unsigned long long>(expected[i]));throw std::runtime_error("native factory GPU mismatch");}
  cudaFuncAttributes a{};check(cudaFuncGetAttributes(&a,construct));
  std::printf("{\"pass\":true,\"native_factory_cases\":%u,\"all_consumed_state_hashes_equal\":true,\"registers_per_thread\":%d,\"local_bytes_per_thread\":%zu,\"proxy_allocation_compared\":false}\n",count,a.numRegs,a.localSizeBytes);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"factory-runner: %s\n",e.what());return 1;}}
