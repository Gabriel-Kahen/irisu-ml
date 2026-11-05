// Diagnostic-only CUPTI injection. Compile as a host shared library, then set
// CUDA_INJECTION64_PATH and IRISU_CUPTI_OUTPUT for an existing CUDA executable.
// Whole-context continuous PC samples; timings under instrumentation are not
// benchmark results. Remaining/dropped records are reported explicitly.
#include <cuda.h>
#include <cupti.h>
#include <cupti_pcsampling.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <unordered_set>
#include <vector>

namespace {
std::ofstream output;
CUpti_SubscriberHandle subscriber{};
struct State {
  CUcontext context{};bool active{};
  CUpti_PCSamplingData data{};
  std::vector<CUpti_PCSamplingPCData> records;
  std::vector<CUpti_PCSamplingStallReason> stalls;
  std::vector<std::array<char,CUPTI_STALL_REASON_STRING_SIZE>> names;
  std::vector<char*> name_pointers;
  std::vector<std::uint32_t> indices;
};
State* state{};
bool check(const char* api,CUptiResult status){
  const char* message="unknown";cuptiGetResultString(status,&message);
  output<<"{\"kind\":\"api\",\"api\":"<<std::quoted(api)<<",\"status\":"<<unsigned(status)<<",\"message\":"<<std::quoted(message)<<"}\n";output.flush();
  return status==CUPTI_SUCCESS;
}
void finish(){
  if(!state||!state->active)return;auto& s=*state;s.active=false;
  CUpti_PCSamplingDisableParams disable{};disable.size=CUpti_PCSamplingDisableParamsSize;disable.ctx=s.context;
  if(!check("cuptiPCSamplingDisable",cuptiPCSamplingDisable(&disable))){
    output<<"{\"kind\":\"summary\",\"sampling_complete\":false,\"usable_pc_records\":0}\n";output.flush();return;
  }
  output<<"{\"kind\":\"summary\",\"pcs\":"<<s.data.totalNumPcs<<",\"samples\":"<<s.data.totalSamples
    <<",\"dropped_samples\":"<<s.data.droppedSamples<<",\"remaining_pcs\":"<<s.data.remainingNumPcs
    <<",\"hardware_buffer_full\":"<<unsigned(s.data.hardwareBufferFull)<<",\"non_user_samples\":"<<s.data.nonUsrKernelsTotalSamples<<"}\n";
  std::unordered_set<char*> strings;
  for(std::size_t i=0;i<s.data.totalNumPcs&&i<s.records.size();++i){const auto& pc=s.records[i];
    if(pc.functionName)strings.insert(pc.functionName);
    for(std::size_t j=0;j<pc.stallReasonCount;++j){const auto& reason=pc.stallReason[j];if(!reason.samples)continue;
      output<<"{\"kind\":\"pc\",\"function\":"<<std::quoted(pc.functionName?pc.functionName:"")<<",\"cubin_crc\":"<<pc.cubinCrc
        <<",\"pc_offset\":"<<pc.pcOffset<<",\"stall\":"<<reason.pcSamplingStallReasonIndex<<",\"samples\":"<<reason.samples<<"}\n";
    }
  }
  output.flush();for(auto name:strings)std::free(name);
}
void configure(CUcontext context){
  if(state){output<<"{\"kind\":\"error\",\"message\":\"multiple CUDA contexts unsupported by this diagnostic\"}\n";return;}
  state=new State;auto& s=*state;s.context=context;
  CUpti_PCSamplingEnableParams enable{};enable.size=CUpti_PCSamplingEnableParamsSize;enable.ctx=context;
  if(!check("cuptiPCSamplingEnable",cuptiPCSamplingEnable(&enable)))return;s.active=true;
  std::size_t count=0;CUpti_PCSamplingGetNumStallReasonsParams number{};
  number.size=CUpti_PCSamplingGetNumStallReasonsParamsSize;number.ctx=context;number.numStallReasons=&count;
  if(!check("cuptiPCSamplingGetNumStallReasons",cuptiPCSamplingGetNumStallReasons(&number)))return;
  s.names.resize(count);s.indices.resize(count);for(auto& name:s.names)s.name_pointers.push_back(name.data());
  CUpti_PCSamplingGetStallReasonsParams reasons{};reasons.size=CUpti_PCSamplingGetStallReasonsParamsSize;
  reasons.ctx=context;reasons.numStallReasons=count;reasons.stallReasonIndex=s.indices.data();reasons.stallReasons=s.name_pointers.data();
  if(!check("cuptiPCSamplingGetStallReasons",cuptiPCSamplingGetStallReasons(&reasons)))return;
  for(std::size_t i=0;i<count;++i)output<<"{\"kind\":\"stall\",\"index\":"<<s.indices[i]<<",\"name\":"<<std::quoted(s.names[i].data())<<"}\n";
  constexpr std::size_t capacity=65536;s.records.resize(capacity);s.stalls.resize(capacity*count);
  for(std::size_t i=0;i<capacity;++i){s.records[i].size=sizeof(CUpti_PCSamplingPCData);s.records[i].stallReason=s.stalls.data()+i*count;s.records[i].stallReasonCount=count;}
  s.data.size=sizeof(CUpti_PCSamplingData);s.data.collectNumPcs=capacity;s.data.pPcData=s.records.data();
  CUpti_PCSamplingConfigurationInfo settings[4]{};
  settings[0].attributeType=CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
  settings[0].attributeData.samplingDataBufferData.samplingDataBuffer=&s.data;
  settings[1].attributeType=CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
  settings[1].attributeData.samplingPeriodData.samplingPeriod=17;
  settings[2].attributeType=CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
  settings[2].attributeData.scratchBufferSizeData.scratchBufferSize=16U<<20;
  settings[3].attributeType=CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
  settings[3].attributeData.collectionModeData.collectionMode=CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS;
  CUpti_PCSamplingConfigurationInfoParams config{};config.size=CUpti_PCSamplingConfigurationInfoParamsSize;
  config.ctx=context;config.numAttributes=4;config.pPCSamplingConfigurationInfo=settings;
  check("cuptiPCSamplingSetConfigurationAttribute",cuptiPCSamplingSetConfigurationAttribute(&config));
}
void CUPTIAPI callback(void*,CUpti_CallbackDomain domain,CUpti_CallbackId id,const void* data){
  if(domain!=CUPTI_CB_DOMAIN_RESOURCE)return;
  const auto* resource=static_cast<const CUpti_ResourceData*>(data);
  if(id==CUPTI_CBID_RESOURCE_CONTEXT_CREATED)configure(resource->context);
  if(id==CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING)finish();
}
}
extern "C" int InitializeInjection(){
  const char* path=std::getenv("IRISU_CUPTI_OUTPUT");output.open(path?path:"irisu-cupti-pcs.jsonl");if(!output)return 0;
  if(!check("cuptiSubscribe",cuptiSubscribe(&subscriber,callback,nullptr)))return 0;
  if(!check("cuptiEnableDomain(resource)",cuptiEnableDomain(1,subscriber,CUPTI_CB_DOMAIN_RESOURCE)))return 0;
  std::atexit(finish);return 1;
}
