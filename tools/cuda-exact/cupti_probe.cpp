// Optional host-only CUPTI capability probe; creates no GPU kernels.
// g++ cupti_probe.cpp -I/opt/cuda/include -I/opt/cuda/extras/CUPTI/include \
//   -L/opt/cuda/lib64 -L/opt/cuda/extras/CUPTI/lib64 -lcuda -lcupti \
//   -Wl,-rpath,/opt/cuda/extras/CUPTI/lib64 -o cupti-probe
#include <cuda.h>
#include <cupti.h>
#include <cupti_pcsampling.h>
#include <cstdint>
#include <cstdio>

void report(const char* api,CUptiResult code){
  const char* name="unknown";cuptiGetResultString(code,&name);
  std::printf("{\"api\":\"%s\",\"result\":%u,\"name\":\"%s\"}\n",api,unsigned(code),name);
}
int main(){
  const auto init=cuInit(0);std::printf("{\"api\":\"cuInit\",\"result\":%u}\n",unsigned(init));if(init)return 1;
  CUdevice device;CUcontext context;if(cuDeviceGet(&device,0)||cuCtxCreate(&context,0,device))return 2;
  const auto trace=cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);report("cuptiActivityEnable(CONCURRENT_KERNEL)",trace);
  if(trace==CUPTI_SUCCESS)report("cuptiActivityDisable(CONCURRENT_KERNEL)",cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  std::uint32_t domains=0;report("cuptiDeviceGetNumEventDomains",cuptiDeviceGetNumEventDomains(device,&domains));
  CUpti_PCSamplingEnableParams enable{};enable.size=CUpti_PCSamplingEnableParamsSize;enable.ctx=context;
  const auto pc=cuptiPCSamplingEnable(&enable);report("cuptiPCSamplingEnable",pc);
  if(pc==CUPTI_SUCCESS){CUpti_PCSamplingDisableParams disable{};disable.size=CUpti_PCSamplingDisableParamsSize;disable.ctx=context;report("cuptiPCSamplingDisable",cuptiPCSamplingDisable(&disable));}
  cuCtxDestroy(context);return 0;
}
