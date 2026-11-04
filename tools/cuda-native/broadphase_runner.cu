// Native-order fixed SAP history runner; snapshot transport is outside SAP.
#include "../../clone/cuda_native/broadphase.hpp"
#include "broadphase_wire.hpp"
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
namespace n=irisu::cuda_native;namespace wire=n::wire;
struct Record {wire::BroadResultHeader header{};std::uint32_t queries[512]{};wire::BroadEvent events[4096]{};std::uint32_t state[wire::native_broadphase_words]{};std::uint32_t error{};};
struct Callback {
 wire::BroadEvent* events{};std::uint32_t* count{};std::uint32_t* error{};std::uint32_t next_token{1};
 NATIVE_HD std::uint32_t PairAdded(std::uint32_t a,std::uint32_t b){const auto token=(a==0||b==0)?0:next_token++;append({1,a,b,token});return token;}
 NATIVE_HD void PairRemoved(std::uint32_t a,std::uint32_t b,std::uint32_t token){append({2,a,b,token});}
 NATIVE_HD void append(wire::BroadEvent e){if(*count>=4096){*error=unsigned(n::Status::capacity);return;}events[(*count)++]=e;}
};
NATIVE_HD void operation(n::BroadPhase& state,const wire::BroadOperation& op,std::uint16_t* slots,Callback& callback,Record& record,unsigned index){
 record.header={index,0,0,0,wire::native_broadphase_words};record.error=0;callback.events=record.events;callback.count=&record.header.event_count;callback.error=&record.error;
 const n::AABB aabb{{op.aabb[0],op.aabb[1]},{op.aabb[2],op.aabb[3]}};
 if(op.slot>=512||op.reserved){record.error=unsigned(n::Status::invalid_snapshot);return;}
 switch(op.kind){
 case 1:if(slots[op.slot]!=65535){record.error=unsigned(n::Status::invalid_snapshot);break;}slots[op.slot]=state.CreateProxy(aabb,op.user_or_limit,callback);record.header.result=slots[op.slot];break;
 case 2:state.MoveProxy(slots[op.slot],aabb);break;
 case 3:state.DestroyProxy(slots[op.slot],callback);slots[op.slot]=65535;break;
 case 4:state.Commit(callback);break;
 case 5:if(op.user_or_limit>512){record.error=unsigned(n::Status::invalid_snapshot);break;}record.header.query_count=state.Query(aabb,record.queries,op.user_or_limit);record.header.result=record.header.query_count;break;
 case 6:if(op.user_or_limit!=65535){record.error=unsigned(n::Status::invalid_snapshot);break;}state.m_timeStamp=65535;record.header.result=65535;break;
 default:record.error=unsigned(n::Status::invalid_snapshot);
 }
 if(state.error!=n::Status::ok)record.error=unsigned(state.error);
 if(!record.error&&!state.Validate())record.error=unsigned(n::Status::invalid_snapshot);
}
#ifdef __CUDACC__
__global__ void history_kernel(n::BroadPhase* state,const wire::BroadOperation* ops,Record* records,wire::BroadHeader header){
 __shared__ std::uint16_t slots[512];for(unsigned i=threadIdx.x;i<512;i+=blockDim.x)slots[i]=65535;
 if(!threadIdx.x)state->Initialize({{header.world[0],header.world[1]},{header.world[2],header.world[3]}});__syncthreads();Callback callback;
 for(unsigned i=0;i<header.count;++i){if(!threadIdx.x)operation(*state,ops[i],slots,callback,records[i],i);__syncthreads();
  const auto* source=reinterpret_cast<const unsigned char*>(state);for(unsigned j=threadIdx.x;j<wire::native_broadphase_words;j+=blockDim.x)std::memcpy(records[i].state+j,source+4*j,4);__syncthreads();
 }
}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
#endif
void emit(const void* data,std::size_t bytes){if(bytes&&std::fwrite(data,1,bytes,stdout)!=bytes)throw std::runtime_error("cannot write SAP output");}
int main(int argc,char** argv){try{
 if(argc<2||argc>3||(argc==3&&std::strcmp(argv[2],"--gpu")))throw std::runtime_error("usage: broadphase-runner operations.bin [--gpu]");const bool gpu=argc==3;
 std::ifstream input(argv[1],std::ios::binary);wire::BroadHeader header{};input.read(reinterpret_cast<char*>(&header),sizeof(header));if(!input||header.magic!=wire::broadphase_magic||header.version!=1||!header.count||header.count>4096||header.op_words!=8)throw std::runtime_error("invalid SAP header");
 std::vector<wire::BroadOperation> ops(header.count);input.read(reinterpret_cast<char*>(ops.data()),ops.size()*sizeof(ops[0]));if(!input||input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("invalid SAP input length");
 std::vector<Record> records(header.count);n::BroadPhase state{};
 if(gpu){
#ifdef __CUDACC__
  n::BroadPhase* ds;wire::BroadOperation* di;Record* dr;check(cudaMalloc(&ds,sizeof(state)));check(cudaMalloc(&di,ops.size()*sizeof(ops[0])));check(cudaMalloc(&dr,records.size()*sizeof(Record)));check(cudaMemcpy(ds,&state,sizeof(state),cudaMemcpyHostToDevice));check(cudaMemcpy(di,ops.data(),ops.size()*sizeof(ops[0]),cudaMemcpyHostToDevice));
  history_kernel<<<1,32>>>(ds,di,dr,header);check(cudaGetLastError());check(cudaDeviceSynchronize());check(cudaMemcpy(records.data(),dr,records.size()*sizeof(Record),cudaMemcpyDeviceToHost));check(cudaFree(ds));check(cudaFree(di));check(cudaFree(dr));
#else
  throw std::runtime_error("CUDA unavailable in host runner");
#endif
 }else{state.Initialize({{header.world[0],header.world[1]},{header.world[2],header.world[3]}});std::array<std::uint16_t,512>slots;slots.fill(65535);Callback callback;for(unsigned i=0;i<header.count;++i){operation(state,ops[i],slots.data(),callback,records[i],i);std::memcpy(records[i].state,&state,wire::native_broadphase_bytes);}}
 unsigned errors=0;for(const auto& record:records){errors+=record.error!=0;emit(&record.header,sizeof(record.header));emit(record.queries,record.header.query_count*4);emit(record.events,record.header.event_count*sizeof(record.events[0]));emit(record.state,wire::native_broadphase_bytes);}
 std::fprintf(stderr,"{\"backend\":\"%s\",\"operations\":%u,\"errors\":%u,\"state_bytes\":%u,\"scope\":\"fullSAPhistoryandnormalizedstate;noGuest\"}\n",gpu?"cuda-native":"typed-host",header.count,errors,wire::native_broadphase_bytes);return errors||std::fflush(stdout)?1:0;
}catch(const std::exception& e){std::fprintf(stderr,"broadphase: %s\n",e.what());return 2;}}
