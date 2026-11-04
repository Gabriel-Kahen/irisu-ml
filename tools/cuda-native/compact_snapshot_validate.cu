// Active-pool complete-call snapshots. No allocation/throughput improvement is
// claimed here: the kernel still executes the unchanged full-size Scene type.
#include "scene_dynamic_wire.hpp"
#include "../../clone/cuda_native/compact_snapshot.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace irisu::cuda_native;
namespace {
constexpr std::size_t persistent_bytes=offsetof(ScenePipelineState,scratch);
static_assert(std::is_standard_layout_v<ScenePipelineState>);
static_assert(alignof(ScenePipelineState)>=8&&sizeof(ScenePipelineState)%8==0);
static_assert(persistent_bytes>=offsetof(ScenePipelineState,broadphase)+sizeof(BroadPhase));
static_assert(sizeof(SceneScratch)%4==0);
struct Failure {unsigned count{},lane{},tick{},phase{},offset{};};
struct Counts {
  unsigned snapshots{},native_frames{},zero_steps{},warm_zero_steps{};
  unsigned pending_destroy{},origin_calls{},sleeping{},frozen{};
  std::uint64_t scene_transport_bytes{};
};
__device__ void fail(Failure* failure,unsigned lane,unsigned tick,unsigned phase,unsigned offset){
  if(!atomicAdd(&failure->count,1U)){failure->lane=lane;failure->tick=tick;failure->phase=phase;failure->offset=offset;}
}
__global__ void initialize_reference(ScenePipelineState* scenes,SceneActors* actors,const SceneCase* cases,
                                     unsigned count,Failure* failure){
  const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=count*4)return;
  const auto result=initialize_scene_case(scenes[lane],actors[lane],cases[lane/4]);
  if(!result)fail(failure,lane,0,1,unsigned(result.status));
}
// Odd modes snapshot after the public operation calls, including queue_destroy;
// even modes snapshot before them, so origin setters consume poisoned scratch.
__global__ void prior_operations(ScenePipelineState* scenes,SceneActors* actors,const SceneCase* cases,
                                 unsigned lanes,unsigned tick,Failure* failure){
  const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=lanes||!(lane&1))return;
  const auto result=apply_scene_ops(scenes[lane],actors[lane],cases[lane/4],tick);
  if(!result)fail(failure,lane,tick,2,unsigned(result.status));
}
__global__ void snapshot(const ScenePipelineState* source,const SceneActors* source_actors,
                         ScenePipelineState* full,SceneActors* full_actors,
                         ScenePipelineState* physical,SceneActors* physical_actors,
                         unsigned tick,bool virgin,compact_snapshot::Plan* plans,Failure* failure,Counts* counts){
  const unsigned lane=blockIdx.x;
  __shared__ unsigned valid;
  if(!threadIdx.x){
    auto result=compact_snapshot::build_plan(source[lane],plans[lane]);
    if(result&&(lane&1)&&tick)result=compact_snapshot::clear_reachable(physical[lane]);
    valid=bool(result);if(!result)fail(failure,lane,tick,9,unsigned(result.status));
    if(result)counts[lane].scene_transport_bytes+=compact_snapshot::transport_bytes(plans[lane]);
  }
  __syncthreads();if(!valid)return;
  const auto* input=reinterpret_cast<const unsigned char*>(source+lane);
  auto* reference=reinterpret_cast<unsigned char*>(full+lane);
  for(std::size_t i=threadIdx.x;i<sizeof(ScenePipelineState);i+=blockDim.x)reference[i]=input[i];
  if(!(lane&1)||!tick)compact_snapshot::clear_missing(physical[lane],plans[lane],threadIdx.x,blockDim.x);
  compact_snapshot::copy_active(physical[lane],source[lane],plans[lane],threadIdx.x,blockDim.x);
  const auto* mapping=reinterpret_cast<const unsigned char*>(source_actors+lane);
  auto* first=reinterpret_cast<unsigned char*>(full_actors+lane);
  auto* second=reinterpret_cast<unsigned char*>(physical_actors+lane);
  for(unsigned i=threadIdx.x;i<sizeof(SceneActors);i+=blockDim.x)first[i]=second[i]=mapping[i];
  if(!virgin){
    // Every float word is a quiet NaN; every index word is outside all pools.
    // Change both payload and sign each checkpoint, retaining those properties.
    const unsigned payload=(tick*257U+lane*17U)&0x003fffffU;
    auto* scratch=reinterpret_cast<unsigned char*>(&physical[lane].scratch);
    for(std::size_t i=threadIdx.x;i<sizeof(SceneScratch)/4;i+=blockDim.x){
      const unsigned word=((i+tick+lane)&1)?(0xffc00000U|payload):(0x7fc00000U|payload);
      for(unsigned byte=0;byte<4;++byte)scratch[4*i+byte]=static_cast<unsigned char>(word>>(8*byte));
    }
  }
}
__global__ void compare(const ScenePipelineState* full,const SceneActors* full_actors,
                        const ScenePipelineState* physical,const SceneActors* physical_actors,
                        unsigned tick,unsigned phase,Failure* failure,Counts* counts){
  const unsigned lane=blockIdx.x;
  if(!threadIdx.x){const auto offset=compact_snapshot::first_difference(full[lane],physical[lane]);
    if(offset!=null_index)fail(failure,lane,tick,phase,offset);
  }
  const auto* a=reinterpret_cast<const unsigned char*>(full_actors+lane);const auto* b=reinterpret_cast<const unsigned char*>(physical_actors+lane);
  for(unsigned i=threadIdx.x;i<sizeof(SceneActors);i+=blockDim.x)if(a[i]!=b[i]){
    fail(failure,lane,tick,phase,unsigned(persistent_bytes+i));break;
  }
  if(!threadIdx.x&&phase==3)++counts[lane].snapshots;
}
__global__ void continue_calls(ScenePipelineState* full,SceneActors* full_actors,
                               ScenePipelineState* physical,SceneActors* physical_actors,
                               const SceneCase* cases,const std::uint64_t* expected,
                               unsigned lanes,unsigned ticks,unsigned tick,float dt,
                               Failure* failure,Counts* counts){
  const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=lanes)return;
  auto& a=full[lane];auto& b=physical[lane];auto& seen=counts[lane];const auto& program=cases[lane/4];
  StageResult first{},second{};
  if(!(lane&1)){
    first=apply_scene_ops(a,full_actors[lane],program,tick);
    second=apply_scene_ops(b,physical_actors[lane],program,tick);
  }
  if(!first||!second){fail(failure,lane,tick,4,!first?unsigned(first.status):unsigned(second.status));return;}
  for(unsigned i=0;i<program.op_count;++i)seen.origin_calls+=program.ops[i].tick==tick&&program.ops[i].kind==SceneOpKind::origin;
  seen.pending_destroy+=a.world.destroy_head!=null_index;
  const unsigned iterations=(lane%4>=2&&tick>=ticks/2)?0:8;
  if(!iterations){
    ++seen.zero_steps;bool warm=false;
    for(auto c=a.world.contact_head;c!=null_index;c=a.world.contacts[c].next){const auto& contact=a.world.contacts[c];
      if(contact.manifold_count)for(int p=0;p<contact.manifold.point_count;++p)
        warm|=contact.manifold.points[p].normal_impulse!=0||contact.manifold.points[p].tangent_impulse!=0;
    }
    seen.warm_zero_steps+=warm;
  }
  first=a.step(dt,iterations);second=b.step(dt,iterations);
  if(!first||!second){fail(failure,lane,tick,5,!first?unsigned(first.status):unsigned(second.status));return;}
  for(auto body=a.world.body_head;body!=null_index;body=a.world.bodies[body].next){
    seen.sleeping+=(a.world.bodies[body].flags&sleeping_body)!=0;
    seen.frozen+=(a.world.bodies[body].flags&frozen_body)!=0;
  }
  const auto first_hash=dynamic_scene_hash(a),second_hash=dynamic_scene_hash(b);
  if(first_hash!=second_hash)fail(failure,lane,tick,6,0);
  if(iterations){
    ++seen.native_frames;
    if(first_hash!=expected[std::size_t(lane/4)*ticks+tick])fail(failure,lane,tick,7,0);
  }
}
void check(cudaError_t result){if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));}
struct Memory {
  std::vector<void*> pointers;~Memory(){for(auto* p:pointers)cudaFree(p);}
  template<class T>T* allocate(std::size_t count){T* p{};check(cudaMalloc(&p,count*sizeof(T)));pointers.push_back(p);return p;}
};
}
int main(int argc,char** argv){try{
  if(argc<2||argc>3||(argc==3&&std::string(argv[2])!="--virgin-scratch"))
    throw std::runtime_error("usage: compact-snapshot-validate native-dynamic-packets.bin [--virgin-scratch]");
  const bool virgin=argc==3;std::ifstream file(argv[1],std::ios::binary);
  if(!file)throw std::runtime_error("cannot open native dynamic packets");
  const auto read=[&](){std::uint32_t value{};if(!file.read(reinterpret_cast<char*>(&value),4))throw std::runtime_error("truncated dynamic packet");return value;};
  if(read()!=dynamic_magic||read()!=dynamic_version)throw std::runtime_error("dynamic packet identity mismatch");
  const unsigned count=read(),ticks=read();if(read()!=dynamic_case_words)throw std::runtime_error("dynamic case stride mismatch");
  union {std::uint32_t u;float f;}dt{read()};const unsigned iterations=read();
  if(read()||count!=16||ticks!=192||dt.u!=0x3c83126f||iterations!=8)throw std::runtime_error("expected qualified 16-case/192-step/8-iteration native corpus");
  std::vector<SceneCase> cases(count);std::vector<std::uint64_t> expected(std::size_t(count)*ticks);
  for(unsigned i=0;i<count;++i){
    std::uint32_t words[dynamic_case_words];for(auto& word:words)word=read();std::memcpy(&cases[i],words,sizeof(SceneCase));
    if(cases[i].actor_count>8||cases[i].op_count>16)throw std::runtime_error("dynamic case capacity exceeded");
    for(unsigned j=0;j<cases[i].op_count;++j)if(cases[i].ops[j].tick>=ticks||unsigned(cases[i].ops[j].kind)>3)throw std::runtime_error("invalid dynamic operation");
    for(unsigned j=0;j<ticks;++j){const auto low=read(),high=read();expected[std::size_t(i)*ticks+j]=std::uint64_t(low)|(std::uint64_t(high)<<32);}
  }
  if(file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing dynamic packet data");
  check(cudaDeviceSetLimit(cudaLimitStackSize,16U<<10));Memory memory;const unsigned lanes=4*count;
  auto* current=memory.allocate<ScenePipelineState>(lanes);auto* full=memory.allocate<ScenePipelineState>(lanes);
  auto* physical=memory.allocate<ScenePipelineState>(lanes);
  auto* current_actors=memory.allocate<SceneActors>(lanes);auto* full_actors=memory.allocate<SceneActors>(lanes);
  auto* physical_actors=memory.allocate<SceneActors>(lanes);auto* programs=memory.allocate<SceneCase>(count);
  auto* native_hashes=memory.allocate<std::uint64_t>(expected.size());auto* failures=memory.allocate<Failure>(1);
  auto* counters=memory.allocate<Counts>(lanes);auto* plans=memory.allocate<compact_snapshot::Plan>(lanes);
  // Match the actual fast path: persistent pool payload/padding is initialized
  // once; only SceneScratch stays virgin in --virgin-scratch mode. A new live
  // slot may retain memberwise-copy padding, which byte comparisons also read.
  const auto empty=std::make_unique<ScenePipelineState>();
  for(unsigned i=0;i<lanes;++i)check(cudaMemcpy(current+i,empty.get(),sizeof(*empty),cudaMemcpyHostToDevice));
  check(cudaMemset2D(physical,sizeof(ScenePipelineState),0,persistent_bytes,lanes));
  check(cudaMemset(current_actors,0,lanes*sizeof(SceneActors)));
  check(cudaMemset(failures,0,sizeof(Failure)));check(cudaMemset(counters,0,lanes*sizeof(Counts)));
  check(cudaMemcpy(programs,cases.data(),cases.size()*sizeof(SceneCase),cudaMemcpyHostToDevice));
  check(cudaMemcpy(native_hashes,expected.data(),expected.size()*sizeof(std::uint64_t),cudaMemcpyHostToDevice));
  initialize_reference<<<(lanes+31)/32,32>>>(current,current_actors,programs,count,failures);check(cudaGetLastError());
  const auto check_failure=[&](){
    Failure failure{};check(cudaMemcpy(&failure,failures,sizeof(failure),cudaMemcpyDeviceToHost));
    if(failure.count){std::fprintf(stderr,"lane%u tick%u phase%u offset/status%u failures%u\n",failure.lane,failure.tick,failure.phase,failure.offset,failure.count);throw std::runtime_error("compact snapshot qualification failed");}
  };
  check_failure();
  for(unsigned tick=0;tick<ticks;++tick){
    prior_operations<<<(lanes+31)/32,32>>>(current,current_actors,programs,lanes,tick,failures);check(cudaGetLastError());
    snapshot<<<lanes,256>>>(current,current_actors,full,full_actors,physical,physical_actors,tick,virgin,plans,failures,counters);check(cudaGetLastError());
    compare<<<lanes,256>>>(full,full_actors,physical,physical_actors,tick,3,failures,counters);check(cudaGetLastError());
    check_failure();
    continue_calls<<<(lanes+31)/32,32>>>(full,full_actors,physical,physical_actors,programs,native_hashes,lanes,ticks,tick,dt.f,failures,counters);check(cudaGetLastError());
    compare<<<lanes,256>>>(full,full_actors,physical,physical_actors,tick,8,failures,counters);check(cudaGetLastError());
    // Stop before a failed state could be used by a later list/hash traversal.
    check_failure();
    std::swap(current,full);std::swap(current_actors,full_actors);
  }
  std::vector<Counts> observed(lanes);check(cudaMemcpy(observed.data(),counters,lanes*sizeof(Counts),cudaMemcpyDeviceToHost));
  Counts total{};for(const auto& c:observed){
    total.snapshots+=c.snapshots;total.native_frames+=c.native_frames;total.zero_steps+=c.zero_steps;total.warm_zero_steps+=c.warm_zero_steps;
    total.pending_destroy+=c.pending_destroy;total.origin_calls+=c.origin_calls;total.sleeping+=c.sleeping;total.frozen+=c.frozen;
    total.scene_transport_bytes+=c.scene_transport_bytes;
  }
  if(total.snapshots!=lanes*ticks||total.native_frames!=count*ticks*3||!total.zero_steps||!total.warm_zero_steps||
     !total.pending_destroy||!total.origin_calls||!total.sleeping||!total.frozen)throw std::runtime_error("required snapshot edge-case witness missing");
  std::printf("{\"pass\":true,\"snapshot_profile\":\"active-World-and-full-SAP\",\"performance_claim\":false,\"virgin_scratch\":%s,\"virgin_scope\":\"%s\",\"persistent_prefix_initialized_once\":true,\"native_cases\":%u,\"ticks_each\":%u,\"lanes\":%u,\"snapshots\":%u,\"native_frame_checks\":%u,\"zero_iteration_steps\":%u,\"warm_zero_iteration_steps\":%u,\"pending_destroy_witnesses\":%u,\"origin_calls\":%u,\"sleeping_body_witnesses\":%u,\"frozen_body_witnesses\":%u,\"scene_bytes\":%zu,\"full_physical_prefix_bytes\":%zu,\"scratch_bytes\":%zu,\"all_active_bytes_and_metadata_equal_before_and_after\":true,\"inactive_payload_equality\":false,\"actor_mapping_bytes_equal\":true,\"pool_scan_and_reachable_list_clear\":true,\"summed_scene_transport_bytes\":%llu,\"fallback_calls\":0}\n",
    virgin?"true":"false",virgin?"SceneScratch only; persistent pool storage initialized once":"none; SceneScratch poisoned",count,ticks,lanes,total.snapshots,total.native_frames,total.zero_steps,total.warm_zero_steps,
    total.pending_destroy,total.origin_calls,total.sleeping,total.frozen,sizeof(ScenePipelineState),persistent_bytes,sizeof(SceneScratch),static_cast<unsigned long long>(total.scene_transport_bytes));
  return 0;
}catch(const std::exception& error){std::fprintf(stderr,"compact-snapshot-validate: %s\n",error.what());return 1;}}
