// GPU-only prefix preparation, bounded execution slots, and retained frontiers.
// Existing hybrid implementation remains an independent, read-only consumer.
#define main preserved_scalar_search_main
#include "search_runner.cu"
#undef main
#include "gpu_search_runtime.cuh"
#include <bit>
#include <limits>
#include <sstream>
#ifndef IRISU_CUDA_NATIVE_PERSISTENT_SOLVER
#define IRISU_CUDA_NATIVE_PERSISTENT_SOLVER 0
#endif
#ifndef IRISU_CUDA_NATIVE_COLLISION_CACHE
#define IRISU_CUDA_NATIVE_COLLISION_CACHE 0
#endif
#ifndef IRISU_CUDA_NATIVE_PARALLEL_PHASES
#define IRISU_CUDA_NATIVE_PARALLEL_PHASES 0
#endif
#if IRISU_CUDA_NATIVE_PERSISTENT_SOLVER
#include "../../clone/cuda_native/solver_persistent.hpp"
#endif
#if IRISU_CUDA_NATIVE_COLLISION_CACHE
#include "../../clone/cuda_native/collision_memo.cuh"
#endif
#if IRISU_CUDA_NATIVE_PARALLEL_PHASES
#include "../../clone/cuda_native/parallel_physics.cuh"
#endif

namespace gpu_first {
struct Config {
 unsigned roots{1},seed_step{1},prefix_step{},slots{},reserve_mib{512},collision_mib{};
 bool queue{},cooperative{},shared{},reuse_checkpoints{},mirror{},ordered{},counters{};Options options;
};
struct Timing {double preparation_ms{},clone_ms{},future_ms{},retention_ms{},rank_ms{},validation_ms{},total_ms{};};
unsigned number(const std::string& value,unsigned maximum,const char* name){
 std::size_t end{};const auto n=std::stoull(value,&end);if(end!=value.size()||n>maximum)throw std::runtime_error(std::string("invalid ")+name);return unsigned(n);
}
Config parse(int argc,char** argv){
 Config c;c.options.warmups=0;c.options.repeats=3;c.options.keep=2;c.options.threads=32;c.options.stack_kib=16;c.options.validate=true;
 for(int i=5;i<argc;++i){const std::string a=argv[i];auto value=[&](){if(++i>=argc)throw std::runtime_error("missing "+a);return std::string(argv[i]);};
  if(a=="--roots")c.roots=number(value(),wide::maximum_roots,"roots");
  else if(a=="--root-seed-step")c.seed_step=number(value(),UINT32_MAX,"root seed step");
  else if(a=="--root-prefix-step")c.prefix_step=number(value(),UINT32_MAX,"root prefix step");
  else if(a=="--slots")c.slots=number(value(),wide::maximum_frontier,"slots");
  else if(a=="--memory-reserve-mib")c.reserve_mib=number(value(),32768,"memory reserve");
  else if(a=="--collision-cache-mib")c.collision_mib=number(value(),16384,"collision cache");
  else if(a=="--threads"||a=="--block-threads")c.options.threads=number(value(),256,"threads");
  else if(a=="--stack-kib")c.options.stack_kib=number(value(),128,"stack");
  else if(a=="--keep")c.options.keep=number(value(),64,"keep");
  else if(a=="--warmups")c.options.warmups=number(value(),10,"warmups");
  else if(a=="--repeats")c.options.repeats=number(value(),100,"repeats");
  else if(a=="--scratch-poison")c.options.scratch_poison=number(value(),2,"scratch poison");
  else if(a=="--target")c.options.target=std::stoll(value());
  else if(a=="--final-only")c.options.trace=false;
  else if(a=="--skip-compaction-check")c.options.validate=false;
  else if(a=="--adaptive-shared-prefixes")c.shared=true;
  else if(a=="--reuse-checkpoints")c.reuse_checkpoints=true;
  else if(a=="--solver-mirror")c.mirror=true;
  else if(a=="--order-by-depth")c.ordered=true;
  else if(a=="--persistent-queue")c.queue=true;
  else if(a=="--cooperative-physics")c.cooperative=true;
  else if(a=="--diagnostic-counters")c.counters=true;
  else if(a=="--copy-u64"||a=="--reuse-prefix"||a=="--physical-snapshot"||a=="--compact-clone"||a=="--merge-rank"||a=="--tiled-selection"||a=="--parallel-prefix-guard"){}
  else throw std::runtime_error("unsupported GPU-first option: "+a);
 }
 if(!c.options.threads||c.options.threads%32||!c.options.keep||!c.options.repeats||c.options.stack_kib<4||!c.roots||c.queue&&c.cooperative)throw std::runtime_error("invalid bounded GPU-first configuration");
#if !IRISU_CUDA_NATIVE_COLLISION_CACHE
 if(c.collision_mib)throw std::runtime_error("collision memoization not compiled");
#endif
#if !IRISU_CUDA_NATIVE_PARALLEL_PHASES
 if(c.cooperative)throw std::runtime_error("cooperative physics not compiled");
#endif
#if !IRISU_CUDA_NATIVE_PERSISTENT_SOLVER
 if(c.mirror)throw std::runtime_error("persistent solver mirror not compiled");
#endif
 return c;
}
void print_timing(const Timing& t){std::fprintf(stderr,"{\"preparation_ms\":%.9f,\"branch_ms\":%.9f,\"rollout_ms\":%.9f,\"retention_ms\":%.9f,\"rank_compact_ms\":%.9f,\"validation_ms\":%.9f,\"total_ms\":%.9f}",t.preparation_ms,t.clone_ms,t.future_ms,t.retention_ms,t.rank_ms,t.validation_ms,t.total_ms);}
std::size_t mirror_bytes(unsigned slots){
#if IRISU_CUDA_NATIVE_PERSISTENT_SOLVER
 return std::size_t(physics::persistent_solver::words_required(slots))*sizeof(std::uint32_t);
#else
 (void)slots;return 0;
#endif
}
#if IRISU_CUDA_NATIVE_PARALLEL_PHASES
__global__ void cooperative_futures_kernel(Device d,unsigned begin,unsigned count){
 const unsigned slot=blockIdx.x;if(slot>=count)return;
 if(!threadIdx.x)physics::parallel_physics::begin_slot(slot,&d.scenes[slot].world);__syncthreads();
 if(!threadIdx.x){run_slot(d,slot,begin+slot);physics::parallel_physics::stop_slot(slot);}
 else if(threadIdx.x>=32)physics::parallel_physics::run_worker(slot);
}
#endif
std::size_t service_bytes(unsigned slots){
#if IRISU_CUDA_NATIVE_PARALLEL_PHASES
 return physics::parallel_physics::bytes_required(slots);
#else
 (void)slots;return 0;
#endif
}
} // namespace gpu_first

int main(int argc,char** argv){try{
 using namespace gpu_first;const auto started=std::chrono::steady_clock::now();if(argc<5)throw std::runtime_error("usage: gpu-search replay specs prefix horizon [--slots N] [--roots N] [--persistent-queue] [fast-profile options]");
 const auto input=wide::load(argv[1],argv[2],argv[3],argv[4]);auto config=parse(argc,argv);const auto plan=wide::plan(input,config.roots,config.seed_step,config.prefix_step,config.shared);
 const unsigned count=input.candidates.size(),bank_count=plan.checkpoints.size();config.options.keep=std::min(config.options.keep,count);const unsigned keep=config.options.keep;
 std::vector<Options> options;for(const auto& root:plan.roots){auto option=config.options;option.seed=root.seed;option.prefix=root.prefix;option.horizon=root.horizon;options.push_back(option);}
 check(cudaDeviceSetLimit(cudaLimitStackSize,std::size_t(config.options.stack_kib)*1024));std::size_t actual_stack{};check(cudaDeviceGetLimit(&actual_stack,cudaLimitStackSize));int device{};cudaDeviceProp properties{};check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&properties,device));
 Memory memory;Device d;d.roots=config.roots;d.checkpoints=bank_count;d.count=count;d.keep=keep;d.replay_size=input.replay.size();
 d.replay=memory.allocate<std::uint32_t>(input.replay.size());d.candidates=memory.allocate<wire::Candidate>(count);d.jobs=memory.allocate<wide::Job>(count);d.checkpoint=memory.allocate<wide::Checkpoint>(bank_count);d.options=memory.allocate<Options>(options.size());
 if(config.ordered){const auto order=wide::schedule(plan);std::vector<unsigned> inverse(count);for(unsigned i=0;i<count;++i)inverse[order[i]]=i;d.schedule=memory.allocate<unsigned>(count);d.inverse_schedule=memory.allocate<unsigned>(count);check(cudaMemcpy(const_cast<unsigned*>(d.schedule),order.data(),count*sizeof(unsigned),cudaMemcpyHostToDevice));check(cudaMemcpy(const_cast<unsigned*>(d.inverse_schedule),inverse.data(),count*sizeof(unsigned),cudaMemcpyHostToDevice));}
 d.bank=memory.allocate<World>(bank_count);d.bank_scenes=memory.allocate<Scene>(bank_count);d.states=memory.allocate<SharedState>(bank_count);d.game_plans=memory.allocate<compact::GamePlan>(bank_count);d.scene_plans=memory.allocate<compact::Plan>(bank_count);
 auto* saved_bank=memory.allocate<World>(bank_count);auto* saved_scenes=memory.allocate<Scene>(bank_count);auto* saved_states=memory.allocate<SharedState>(bank_count);
 d.results=memory.allocate<wire::Result>(count);auto* order=memory.allocate<unsigned>(count);d.errors=memory.allocate<Validation>(1);d.ticks=memory.allocate<unsigned long long>(1);d.processed=memory.allocate<unsigned>(1);
 auto* empty=memory.allocate<World>(1);auto* empty_scene=memory.allocate<Scene>(1);auto host_empty=cold_object<World>();auto host_scene=cold_object<Scene>();
 check(cudaMemcpy(empty,host_empty.get(),sizeof(World),cudaMemcpyHostToDevice));check(cudaMemcpy(empty_scene,host_scene.get(),sizeof(Scene),cudaMemcpyHostToDevice));
 World* retained[2]={memory.allocate<World>(keep),memory.allocate<World>(keep)};Scene* retained_scenes[2]={memory.allocate<Scene>(keep),memory.allocate<Scene>(keep)};unsigned* retained_ids[2]={memory.allocate<unsigned>(keep),memory.allocate<unsigned>(keep)};
 auto* references=memory.allocate<World>(keep);auto* reference_scenes=memory.allocate<Scene>(keep);auto* reference_plans=memory.allocate<compact::GamePlan>(keep);auto* selected=memory.allocate<Selection>(keep);auto* selected_slots=memory.allocate<unsigned>(keep);
 auto* cursor=memory.allocate<unsigned>(1);auto* lock=memory.allocate<unsigned>(1);std::size_t rank_bytes{};check(physics::search_accel::rank_workspace_bytes(d.results,order,count,rank_bytes));auto* rank_workspace=memory.allocate<unsigned char>(rank_bytes);
 unsigned memo_capacity=0;std::size_t memo_bytes=0;
#if IRISU_CUDA_NATIVE_COLLISION_CACHE
 physics::collision_memo::Counters* memo_counters{};
 if(config.collision_mib){const auto capacity=std::bit_floor(std::uint64_t(config.collision_mib)*1024*1024/sizeof(physics::collision_memo::Entry));if(capacity<4||capacity>UINT32_MAX)throw std::runtime_error("collision cache capacity");memo_capacity=unsigned(capacity);auto* entries=memory.allocate<physics::collision_memo::Entry>(memo_capacity);memo_bytes=std::size_t(memo_capacity)*sizeof(*entries);check(cudaMemset(entries,0,memo_bytes));if(config.counters){memo_counters=memory.allocate<physics::collision_memo::Counters>(1024);check(cudaMemset(memo_counters,0,1024*sizeof(*memo_counters)));}check(physics::collision_memo::configure({entries,memo_capacity,memo_counters,memo_counters?1024U:0U}));}
#endif
 std::size_t free_memory{},total_memory{};check(cudaMemGetInfo(&free_memory,&total_memory));const auto stack_reserve=actual_stack*std::size_t(properties.maxThreadsPerMultiProcessor)*properties.multiProcessorCount;
 // Conservative even when the driver already reserved some stacks. Refuse a
 // borderline launch rather than relying on lazy context-allocation behavior.
 const auto reserve=std::size_t(config.reserve_mib)*1024*1024+stack_reserve;
 auto fits=[&](unsigned slots){return free_memory>reserve&&std::uint64_t(slots)*(sizeof(World)+sizeof(Scene)+sizeof(unsigned))+(config.mirror?mirror_bytes(slots):0)+(config.cooperative?service_bytes(slots):0)<=free_memory-reserve;};
 unsigned slots=config.slots?config.slots:count;if(config.queue)slots=(slots+31U)/32U*32U;
 if(config.slots&&!fits(slots))throw std::runtime_error("requested execution slots exceed admitted free VRAM after runtime/fixed allocations");
 while(!fits(slots)&&!config.slots&&slots>32)slots-=std::min(32U,slots-32U);
 if(!fits(slots)||!slots||config.queue&&slots%32)throw std::runtime_error("no safe execution-slot capacity");d.slots=slots;d.worlds=memory.allocate<World>(slots);d.scenes=memory.allocate<Scene>(slots);d.initialized=memory.allocate<unsigned>(slots);check(cudaMemset(d.initialized,0,slots*sizeof(unsigned)));
#if IRISU_CUDA_NATIVE_PERSISTENT_SOLVER
 unsigned long long* mirror_counters{};
 if(config.mirror){auto* mirror=memory.allocate<std::uint32_t>(mirror_bytes(slots)/4);check(cudaMemset(mirror,0,mirror_bytes(slots)));if(config.counters){mirror_counters=memory.allocate<unsigned long long>(physics::persistent_solver::counter_count);check(cudaMemset(mirror_counters,0,physics::persistent_solver::counter_count*sizeof(unsigned long long)));}check(physics::persistent_solver::configure({mirror,reinterpret_cast<std::uintptr_t>(d.scenes),sizeof(Scene),slots,physics::max_bodies,physics::max_contacts,mirror_counters}));}
#endif
#if IRISU_CUDA_NATIVE_PARALLEL_PHASES
 physics::parallel_physics::Channel* service{};
 if(config.cooperative){service=memory.allocate<physics::parallel_physics::Channel>(slots);check(cudaMemset(service,0,service_bytes(slots)));check(physics::parallel_physics::configure({d.scenes,sizeof(Scene),slots,service,true,true,true,2}));}
#endif
 check(cudaMemset(d.worlds,0,std::size_t(slots)*sizeof(World)));check(cudaMemset2D(d.scenes,sizeof(Scene),0,physical_scene_bytes,slots));
 check(cudaMemset(d.bank,0,std::size_t(bank_count)*sizeof(World)));check(cudaMemset(d.bank_scenes,0,std::size_t(bank_count)*sizeof(Scene)));check(cudaMemset(d.states,0,std::size_t(bank_count)*sizeof(SharedState)));check(cudaMemset(d.game_plans,0,std::size_t(bank_count)*sizeof(compact::GamePlan)));check(cudaMemset(d.scene_plans,0,std::size_t(bank_count)*sizeof(compact::Plan)));
 check(cudaMemcpy(const_cast<std::uint32_t*>(d.replay),input.replay.data(),input.replay.size()*4,cudaMemcpyHostToDevice));check(cudaMemcpy(const_cast<wire::Candidate*>(d.candidates),input.candidates.data(),count*sizeof(wire::Candidate),cudaMemcpyHostToDevice));check(cudaMemcpy(const_cast<wide::Job*>(d.jobs),plan.jobs.data(),count*sizeof(wide::Job),cudaMemcpyHostToDevice));check(cudaMemcpy(const_cast<wide::Checkpoint*>(d.checkpoint),plan.checkpoints.data(),bank_count*sizeof(wide::Checkpoint),cudaMemcpyHostToDevice));check(cudaMemcpy(const_cast<Options*>(d.options),options.data(),options.size()*sizeof(Options),cudaMemcpyHostToDevice));
 Validation validation{};check(cudaMemcpy(d.errors,&validation,sizeof(validation),cudaMemcpyHostToDevice));Event prefix_a,prefix_b;prefix_a.record();roots_kernel<<<config.roots,256>>>(d,empty,empty_scene);plans_kernel<<<(config.roots+31)/32,32>>>(d,0,config.roots);check(cudaMemcpyAsync(saved_bank,d.bank,config.roots*sizeof(World),cudaMemcpyDeviceToDevice));check(cudaMemcpyAsync(saved_scenes,d.bank_scenes,config.roots*sizeof(Scene),cudaMemcpyDeviceToDevice));check(cudaMemcpyAsync(saved_states,d.states,config.roots*sizeof(SharedState),cudaMemcpyDeviceToDevice));prefix_b.record();check(cudaEventSynchronize(prefix_b.value));const double prefix_ms=elapsed(prefix_a,prefix_b);
 check(cudaMemcpy(&validation,d.errors,sizeof(validation),cudaMemcpyDeviceToHost));if(validation.errors)throw std::runtime_error("GPU root initialization rejected");
 const void* execution_kernel=config.queue?reinterpret_cast<const void*>(queue_kernel):reinterpret_cast<const void*>(futures_kernel);unsigned execution_threads=config.options.threads;
#if IRISU_CUDA_NATIVE_PARALLEL_PHASES
 if(config.cooperative){execution_kernel=reinterpret_cast<const void*>(cooperative_futures_kernel);execution_threads=physics::parallel_physics::block_threads;}
#endif
 cudaFuncAttributes attributes{};check(cudaFuncGetAttributes(&attributes,execution_kernel));int active_blocks{};check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,execution_kernel,execution_threads,0));
 std::vector<wire::Result> results(count),previous;std::vector<unsigned> ranks(count),previous_ranks,retained_host(keep),ranked_keep(keep);std::vector<Timing> samples;std::uint64_t measured_ticks=0,measured_jobs=0,preparation_ticks=0;unsigned guard_checks=0,preparation_executions=0;
 std::array<unsigned long long,7> service_counts{};
 for(unsigned run=0;run<config.options.warmups+config.options.repeats;++run){const auto batch_start=std::chrono::steady_clock::now();Timing sample;validation={};check(cudaMemcpy(d.errors,&validation,sizeof(validation),cudaMemcpyHostToDevice));check(cudaMemset(d.ticks,0,sizeof(unsigned long long)));check(cudaMemset(d.processed,0,sizeof(unsigned)));check(cudaMemset(retained_ids[0],0xff,keep*sizeof(unsigned)));check(cudaMemset(retained_ids[1],0xff,keep*sizeof(unsigned)));Event a,b;a.record();
  const bool prepare=bank_count>config.roots&&(!config.reuse_checkpoints||!run);
  if(prepare){prepare_clones_kernel<<<bank_count-config.roots,256>>>(d,run!=0);prepare_futures_kernel<<<(bank_count-config.roots+31)/32,32>>>(d);plans_kernel<<<(bank_count+31)/32,32>>>(d,0,bank_count);check(cudaMemcpyAsync(saved_bank+config.roots,d.bank+config.roots,(bank_count-config.roots)*sizeof(World),cudaMemcpyDeviceToDevice));check(cudaMemcpyAsync(saved_scenes+config.roots,d.bank_scenes+config.roots,(bank_count-config.roots)*sizeof(Scene),cudaMemcpyDeviceToDevice));check(cudaMemcpyAsync(saved_states+config.roots,d.states+config.roots,(bank_count-config.roots)*sizeof(SharedState),cudaMemcpyDeviceToDevice));++preparation_executions;}
  b.record();check(cudaEventSynchronize(b.value));sample.preparation_ms=elapsed(a,b);check(cudaMemcpy(&validation,d.errors,sizeof(validation),cudaMemcpyDeviceToHost));if(validation.errors)throw std::runtime_error("GPU checkpoint preparation rejected");if(prepare){unsigned long long ticks{};check(cudaMemcpy(&ticks,d.ticks,sizeof(ticks),cudaMemcpyDeviceToHost));preparation_ticks+=ticks;}unsigned bank=0;
  if(config.queue){check(cudaMemset(cursor,0,sizeof(unsigned)));check(cudaMemset(lock,0,sizeof(unsigned)));a.record();queue_kernel<<<(slots+config.options.threads-1)/config.options.threads,config.options.threads>>>(d,cursor,retained[0],retained_scenes[0],retained_ids[0],lock);b.record();check(cudaEventSynchronize(b.value));sample.future_ms=elapsed(a,b);}
  else for(unsigned begin=0;begin<count;begin+=slots){const unsigned active=std::min(slots,count-begin);a.record();clones_kernel<<<active,256>>>(d,begin,active,run!=0||begin!=0);if(config.options.scratch_poison)poison_scratch_kernel<<<active,256>>>(d.scenes,active,nullptr,config.options.scratch_poison);b.record();check(cudaEventSynchronize(b.value));sample.clone_ms+=elapsed(a,b);check(cudaMemcpy(&validation,d.errors,sizeof(validation),cudaMemcpyDeviceToHost));if(validation.errors)throw std::runtime_error("GPU slot restore rejected");a.record();
#if IRISU_CUDA_NATIVE_PARALLEL_PHASES
   if(config.cooperative)cooperative_futures_kernel<<<active,physics::parallel_physics::block_threads>>>(d,begin,active);else
#endif
   futures_kernel<<<(active+config.options.threads-1)/config.options.threads,config.options.threads>>>(d,begin,active);b.record();check(cudaEventSynchronize(b.value));sample.future_ms+=elapsed(a,b);
#if IRISU_CUDA_NATIVE_PARALLEL_PHASES
   if(config.cooperative&&config.counters){using Stats=physics::parallel_physics::Stats;static_assert(sizeof(Stats)==sizeof(service_counts));std::vector<Stats> stats(active);check(cudaMemcpy2D(stats.data(),sizeof(Stats),service,sizeof(physics::parallel_physics::Channel),sizeof(Stats),active,cudaMemcpyDeviceToHost));for(const auto& stat:stats){std::array<unsigned long long,7> values;std::memcpy(values.data(),&stat,sizeof(stat));for(unsigned i=0;i<7;++i)service_counts[i]+=values[i];}}
#endif
   if(slots<count){a.record();select_kernel<<<1,1>>>(d,begin,active,retained_ids[bank],selected);retain_kernel<<<keep,256>>>(d,selected,retained[bank],retained_scenes[bank],retained[bank^1],retained_scenes[bank^1],retained_ids[bank^1]);bank^=1;b.record();check(cudaEventSynchronize(b.value));sample.retention_ms+=elapsed(a,b);}}
  unsigned executed_jobs{};check(cudaMemcpy(&validation,d.errors,sizeof(validation),cudaMemcpyDeviceToHost));check(cudaMemcpy(&executed_jobs,d.processed,sizeof(executed_jobs),cudaMemcpyDeviceToHost));
  if(validation.errors||executed_jobs!=count)throw std::runtime_error("GPU execution rejected before ranking: errors="+std::to_string(validation.errors)+" section="+std::to_string(validation.first_section)+" jobs="+std::to_string(executed_jobs));
  a.record();check(physics::search_accel::rank_async(d.results,order,count,rank_workspace,rank_bytes));b.record();check(cudaEventSynchronize(b.value));sample.rank_ms=elapsed(a,b);
  if(!config.queue&&slots>=count){Options transport=config.options;transport.copy_u64=transport.physical_snapshot=transport.tiled_selection=true;a.record();const unsigned* source_slots=order;if(config.ordered){selected_slots_kernel<<<(keep+31)/32,32>>>(d,order,selected_slots);source_slots=selected_slots;}launch_clones(retained[0],retained_scenes[0],d.worlds,d.scenes,keep,source_slots,d.errors,transport);check(cudaMemcpyAsync(retained_ids[0],order,keep*sizeof(unsigned),cudaMemcpyDeviceToDevice));bank=0;b.record();check(cudaEventSynchronize(b.value));sample.retention_ms+=elapsed(a,b);}
  check(cudaMemcpy(retained_host.data(),retained_ids[bank],keep*sizeof(unsigned),cudaMemcpyDeviceToHost));check(cudaMemcpy(ranked_keep.data(),order,keep*sizeof(unsigned),cudaMemcpyDeviceToHost));
  for(unsigned k=0;k<keep;++k)if(retained_host[k]>=count||retained_host[k]!=ranked_keep[k])throw std::runtime_error("retained GPU frontier disagrees with global ranking before validation");a.record();
  if(config.options.validate){reference_clones_kernel<<<keep,256>>>(d,retained_ids[bank],references,reference_scenes);validate_kernel<<<(keep+31)/32,32>>>(d,retained_ids[bank],retained[bank],retained_scenes[bank],references,reference_scenes,reference_plans);poison_scratch_kernel<<<keep,256>>>(retained_scenes[bank],keep,nullptr,1);poison_scratch_kernel<<<keep,256>>>(reference_scenes,keep,nullptr,2);continuation_kernel<<<(keep+31)/32,32>>>(d,retained[bank],retained_scenes[bank],references,reference_scenes,reference_plans);}
  for(unsigned cp=0;cp<bank_count;++cp)check(physics::prefix_guard::check_prefix_async(d.bank+cp,d.bank_scenes+cp,d.states+cp,saved_bank+cp,saved_scenes+cp,saved_states+cp,d.errors));++guard_checks;b.record();check(cudaEventSynchronize(b.value));sample.validation_ms=elapsed(a,b);check(cudaGetLastError());
  check(cudaMemcpy(results.data(),d.results,count*sizeof(wire::Result),cudaMemcpyDeviceToHost));check(cudaMemcpy(ranks.data(),order,count*sizeof(unsigned),cudaMemcpyDeviceToHost));check(cudaMemcpy(&validation,d.errors,sizeof(validation),cudaMemcpyDeviceToHost));unsigned long long ticks{};unsigned jobs{};check(cudaMemcpy(&ticks,d.ticks,sizeof(ticks),cudaMemcpyDeviceToHost));check(cudaMemcpy(&jobs,d.processed,sizeof(jobs),cudaMemcpyDeviceToHost));
  if(validation.errors||jobs!=count)throw std::runtime_error("GPU-first validation failed: errors="+std::to_string(validation.errors)+" section="+std::to_string(validation.first_section)+" offset="+std::to_string(validation.first_offset)+" jobs="+std::to_string(jobs));for(unsigned k=0;k<keep;++k)if(retained_host[k]!=ranks[k])throw std::runtime_error("retained GPU frontier disagrees with global ranking");
  if(!previous.empty()&&(std::memcmp(previous.data(),results.data(),count*sizeof(wire::Result))||previous_ranks!=ranks))throw std::runtime_error("GPU-first repeated batch differs");previous=results;previous_ranks=ranks;sample.total_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-batch_start).count();if(run>=config.options.warmups){samples.push_back(sample);measured_ticks+=ticks;measured_jobs+=jobs;}
 }
 unsigned failed=0;std::uint64_t logical_ticks=0;for(const auto& result:results){failed+=bool(result.failed);logical_ticks+=result.simulated_ticks;}
 std::array<unsigned long long,7> memo_counts{};std::array<unsigned long long,6> mirror_counts{};
#if IRISU_CUDA_NATIVE_COLLISION_CACHE
 if(memo_counters){using Counter=physics::collision_memo::Counters;static_assert(sizeof(Counter)==sizeof(memo_counts));std::vector<Counter> counters(1024);check(cudaMemcpy(counters.data(),memo_counters,1024*sizeof(Counter),cudaMemcpyDeviceToHost));for(const auto& counter:counters){std::array<unsigned long long,7> values;std::memcpy(values.data(),&counter,sizeof(counter));for(unsigned i=0;i<7;++i)memo_counts[i]+=values[i];}}
#endif
#if IRISU_CUDA_NATIVE_PERSISTENT_SOLVER
 if(mirror_counters)check(cudaMemcpy(mirror_counts.data(),mirror_counters,sizeof(mirror_counts),cudaMemcpyDeviceToHost));
#endif
 std::ostringstream diagnostics;diagnostics<<"{\"enabled\":"<<(config.counters?"true":"false")<<",\"scope\":\"Diagnostic only; counts include prefix/checkpoint work and enabled validation\",\"mirror\":[";
 for(unsigned i=0;i<mirror_counts.size();++i){if(i)diagnostics<<',';diagnostics<<mirror_counts[i];}diagnostics<<"],\"mirror_names\":[\"islands\",\"body_records\",\"constraint_records\",\"unavailable\",\"pack_rejections\",\"failures\"],\"collision\":[";
 for(unsigned i=0;i<memo_counts.size();++i){if(i)diagnostics<<',';diagnostics<<memo_counts[i];}diagnostics<<"],\"collision_names\":[\"calls\",\"hits\",\"misses\",\"inserts\",\"busy\",\"full\",\"bypass\"],\"cooperative\":[";
 for(unsigned i=0;i<service_counts.size();++i){if(i)diagnostics<<',';diagnostics<<service_counts[i];}diagnostics<<"],\"cooperative_names\":[\"collision_calls\",\"velocity_calls\",\"position_calls\",\"bounds_calls\",\"collision_jobs\",\"body_jobs\",\"shape_jobs\"]}";
 std::fprintf(stderr,"{\"identity\":\"experimental-gpu-first-resident-search-v1\",\"backend\":\"cuda-native\",\"promotion_eligible\":false,\"candidates\":%u,\"prefix_ticks\":%u,\"horizon\":%u,\"root_count\":%u,\"prefix_executions\":%u,\"checkpoint_count\":%u,\"execution_slots\":%u,\"resident_root_bank\":true,\"persistent_queue\":%s,\"adaptive_shared_prefixes\":%s,\"cooperative_physics\":%s,\"host_prefix\":false,\"host_physics_calls\":0,\"guest_calls\":0,\"fallback_calls\":0,\"failed_candidates\":%u,\"compaction_errors\":%u,\"prefix_immutability_checks\":true,\"prefix_immutability_errors\":%u,\"checkpoint_guard_executions\":%u,\"selected_state_checks\":%s,\"retained_states_verified_against_independent_replay\":%s,\"keep\":%u,\"warmups\":%u,\"repeats\":%u,\"threads\":%u,\"registers\":%u,\"local_bytes\":%zu,\"actual_stack_bytes\":%zu,\"simulated_ticks\":%llu,\"planned_future_ticks_per_batch\":%llu,\"measured_gpu_executed_future_ticks\":%llu,\"measured_job_executions\":%llu,\"logical_future_ticks\":%llu,\"allocated_bytes\":%llu,\"free_bytes_before_slot_admission\":%zu,\"total_device_bytes\":%zu,\"memory_reserve_bytes\":%zu,\"world_bytes\":%zu,\"scene_bytes\":%zu,\"mirror_bytes\":%zu,\"multiprocessors\":%d,\"theoretical_active_blocks_per_sm\":%d,\"theoretical_active_warps_per_sm\":%d,\"occupancy_scope\":\"API resource ceiling, not measured achieved occupancy or hardware utilization\",\"prefix_ms\":%.9f,\"elapsed_before_stdout_ms\":%.9f,\"samples\":[",count,input.original.prefix,input.original.horizon,config.roots,config.roots,bank_count,slots,config.queue?"true":"false",config.shared?"true":"false",config.cooperative?"true":"false",failed,validation.errors,validation.prefix_errors,guard_checks,config.options.validate?"true":"false",config.options.validate?"true":"false",keep,config.options.warmups,config.options.repeats,execution_threads,attributes.numRegs,attributes.localSizeBytes,actual_stack,(unsigned long long)logical_ticks,(unsigned long long)plan.planned_ticks,(unsigned long long)measured_ticks,(unsigned long long)measured_jobs,(unsigned long long)(plan.logical_ticks*config.options.repeats),(unsigned long long)memory.bytes,free_memory,total_memory,reserve,sizeof(World),sizeof(Scene),(config.mirror?mirror_bytes(slots):0),properties.multiProcessorCount,active_blocks,active_blocks*int(execution_threads/32),prefix_ms,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());
 for(unsigned i=0;i<samples.size();++i){if(i)std::fputc(',',stderr);print_timing(samples[i]);}std::fprintf(stderr," ],\"reuse_checkpoints\":%s,\"order_by_depth\":%s,\"root_seed_step\":%u,\"root_prefix_step\":%u,\"diagnostics\":%s,\"collision_cache_entries\":%u,\"collision_cache_bytes\":%zu,\"stack_upper_bound_reserve_bytes\":%zu,\"solver_mirror_enabled\":%s,\"checkpoint_preparation_executions\":%u,\"checkpoint_preparation_gpu_ticks\":%llu,\"cooperative_service_bytes\":%zu,\"compile_profile\":{\"local_velocity\":%s,\"local_position\":%s,\"contact_anchors\":%s,\"persistent_solver\":%s,\"collision_memo\":%s,\"max_contacts\":%u}}\n",config.reuse_checkpoints?"true":"false",config.ordered?"true":"false",config.seed_step,config.prefix_step,diagnostics.str().c_str(),memo_capacity,memo_bytes,stack_reserve,config.mirror?"true":"false",preparation_executions,(unsigned long long)preparation_ticks,(config.cooperative?service_bytes(slots):0),IRISU_CUDA_NATIVE_LOCAL_VELOCITY?"true":"false",IRISU_CUDA_NATIVE_LOCAL_POSITION?"true":"false",IRISU_CUDA_NATIVE_CONTACT_ANCHORS?"true":"false",IRISU_CUDA_NATIVE_PERSISTENT_SOLVER?"true":"false",IRISU_CUDA_NATIVE_COLLISION_CACHE?"true":"false",physics::max_contacts);
 if(failed)return 2;wire::emit_results(results,ranks);return 0;
}catch(const std::exception& error){std::fprintf(stderr,"GPU-first search: %s\n",error.what());return 1;}}
