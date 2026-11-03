#define IRISU_REGION_WRITEBACK_NO_MAIN
#include "region_writeback_validate.cpp"
#include <cuda_runtime.h>

__global__ void run_cases(Guest* guests) {
  const auto test=threadIdx.x;if(test>=6)return;
  auto local=guests[test];exercise(local,test);guests[test]=local;
}
void check(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
int main(int argc,char** argv) {
 try {
  if(argc!=2)return 2;
  std::ifstream stream(argv[1],std::ios::binary);
  const std::vector<unsigned char> image((std::istreambuf_iterator<char>(stream)),{});
  irisu::cuda_exact::verify_image(image.data(),image.size());
  constexpr unsigned capacity=524288;
  unsigned char* arena{};Guest* device_guests{};
  check(cudaMalloc(&arena,6*capacity));check(cudaMemset(arena,0,6*capacity));
  check(cudaMalloc(&device_guests,6*sizeof(Guest)));
  Guest guests[6]{};
  for(unsigned test=0;test<6;++test) {
    auto& g=guests[test];g.memory=arena+test*capacity;g.capacity=capacity;g.heap=0x20000;g.heap_limit=capacity-0x10000;
    check(cudaMemcpy(g.memory,image.data(),image.size(),cudaMemcpyHostToDevice));
  }
  check(cudaMemcpy(device_guests,guests,sizeof(guests),cudaMemcpyHostToDevice));
  run_cases<<<1,6>>>(device_guests);check(cudaGetLastError());check(cudaDeviceSynchronize());
  check(cudaMemcpy(guests,device_guests,sizeof(guests),cudaMemcpyDeviceToHost));
  std::vector<unsigned char> memory(capacity);
  for(unsigned test=0;test<6;++test) {
    const auto& g=guests[test];
    const auto expected=test==4?bad_address:test==5?unsupported_fp:budget_exhausted;
    if(g.error!=expected)throw std::runtime_error("device fault/budget gate did not exercise expected failure");
    check(cudaMemcpy(memory.data(),g.memory,capacity,cudaMemcpyDeviceToHost));
    report(test==4?"bad-stack":test==5?"unsupported-fp":"budget",g,memory);
  }
  check(cudaFree(device_guests));check(cudaFree(arena));return 0;
 }catch(const std::exception& error){std::fprintf(stderr,"region-writeback-cuda: %s\n",error.what());return 1;}
}
