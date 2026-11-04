#pragma once
// Included after search_runner.cu: reuse its admitted arithmetic and snapshots.
#include "gpu_search_inputs.hpp"
#include <cuda/atomic>

namespace wide=irisu::gpu_search;
namespace gpu_first {
struct Selection {unsigned id{~0U},source{},index{};};
struct Device {
 const std::uint32_t* replay{};unsigned replay_size{},roots{},checkpoints{},count{},slots{},keep{};
 const Options* options{};const wide::Job* jobs{};const wide::Checkpoint* checkpoint{};const wire::Candidate* candidates{};const unsigned* schedule{};const unsigned* inverse_schedule{};
 World* bank{};Scene* bank_scenes{};SharedState* states{};compact::GamePlan* game_plans{};compact::Plan* scene_plans{};
 World* worlds{};Scene* scenes{};wire::Result* results{};Validation* errors{};
 unsigned long long* ticks{};unsigned* processed{};unsigned* initialized{};
};
__device__ inline void reject(Validation* errors,unsigned section,unsigned offset=0){atomicAdd(&errors->errors,1U);atomicCAS(&errors->first_section,0U,section);atomicMin(&errors->first_offset,offset);}
__device__ inline unsigned logical_id(const Device& d,unsigned position){return d.schedule?d.schedule[position]:position;}
__global__ void roots_kernel(Device d,const World* empty,const Scene* empty_scene){
 const unsigned root=blockIdx.x;if(root>=d.roots)return;
 compact::copy_bytes(reinterpret_cast<unsigned char*>(d.bank+root),reinterpret_cast<const unsigned char*>(empty),sizeof(World),threadIdx.x,blockDim.x);
 compact::copy_bytes(reinterpret_cast<unsigned char*>(d.bank_scenes+root),reinterpret_cast<const unsigned char*>(empty_scene),sizeof(Scene),threadIdx.x,blockDim.x);__syncthreads();
 if(!threadIdx.x)prepare_prefix(d.bank+root,d.bank_scenes+root,d.options[root],d.replay,&d.states[root].result);
}
__global__ void plans_kernel(Device d,unsigned begin,unsigned count){
 const unsigned i=begin+blockIdx.x*blockDim.x+threadIdx.x;if(i>=begin+count)return;
 d.game_plans[i].valid=false;
 if(!compact::build_plan(d.bank_scenes[i],d.scene_plans[i])||!compact::build_game_plan(d.bank[i],d.game_plans[i]))reject(d.errors,10,i);
}
__global__ void prepare_clones_kernel(Device d,bool initialized){
 const unsigned slot=d.roots+blockIdx.x;if(slot>=d.checkpoints)return;const auto root=d.checkpoint[slot].root;__shared__ bool valid;
 if(!threadIdx.x){valid=d.game_plans[root].valid&&(!initialized||bool(compact::clear_reachable(d.bank_scenes[slot])));if(!valid)reject(d.errors,11,slot);}__syncthreads();if(!valid)return;
 compact::copy_game(d.bank[slot],d.bank[root],d.game_plans[root],threadIdx.x,blockDim.x);
 compact::copy_active(d.bank_scenes[slot],d.bank_scenes[root],d.scene_plans[root],threadIdx.x,blockDim.x);__syncthreads();
 if(!threadIdx.x&&!compact::rebind_game(d.bank[slot],d.bank_scenes[slot],d.game_plans[root]))reject(d.errors,12,slot);
}
__global__ void prepare_futures_kernel(Device d){
 const unsigned i=d.roots+blockIdx.x*blockDim.x+threadIdx.x;if(i>=d.checkpoints)return;const auto cp=d.checkpoint[i];
 SharedState state{};state.result=d.states[cp.root].result;const auto before=state.result.simulated_ticks;
 shared_segment(d.bank[i],d.bank_scenes[i],d.options[cp.root],d.replay,d.replay_size,d.candidates[cp.candidate],0,cp.depth,state,false);
 d.states[i]=state;atomicAdd(d.ticks,(unsigned long long)(state.result.simulated_ticks-before));
}
__global__ void clones_kernel(Device d,unsigned begin,unsigned count,bool initialized){
 const unsigned slot=blockIdx.x;if(slot>=count)return;const auto job=d.jobs[logical_id(d,begin+slot)];const unsigned cp=job.checkpoint;__shared__ bool valid;
 if(!threadIdx.x){valid=d.game_plans[cp].valid&&(!initialized||bool(compact::clear_reachable(d.scenes[slot])));if(!valid)reject(d.errors,13,slot);}__syncthreads();if(!valid)return;
 compact::copy_game(d.worlds[slot],d.bank[cp],d.game_plans[cp],threadIdx.x,blockDim.x);
 compact::copy_active(d.scenes[slot],d.bank_scenes[cp],d.scene_plans[cp],threadIdx.x,blockDim.x);__syncthreads();
 if(!threadIdx.x&&!compact::rebind_game(d.worlds[slot],d.scenes[slot],d.game_plans[cp]))reject(d.errors,14,slot);
}
__device__ inline void run_slot(Device d,unsigned slot,unsigned logical){
 logical=logical_id(d,logical);const auto job=d.jobs[logical];SharedState state=d.states[job.checkpoint];const auto before=state.result.simulated_ticks;
 shared_segment(d.worlds[slot],d.scenes[slot],d.options[job.root],d.replay,d.replay_size,d.candidates[job.candidate],job.begin,d.options[job.root].horizon,state,true);
 d.results[logical]=state.result;atomicAdd(d.ticks,(unsigned long long)(state.result.simulated_ticks-before));atomicAdd(d.processed,1U);
}
__global__ void futures_kernel(Device d,unsigned begin,unsigned count){const unsigned slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot<count)run_slot(d,slot,begin+slot);}
__global__ void selected_slots_kernel(Device d,const unsigned* ranks,unsigned* slots){const unsigned k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.keep)slots[k]=d.inverse_schedule?d.inverse_schedule[ranks[k]]:ranks[k];}
__global__ void select_kernel(Device d,unsigned begin,unsigned count,const unsigned* previous,Selection* selected){
 if(blockIdx.x||threadIdx.x)return;
 for(unsigned k=0;k<d.keep;++k)selected[k]={previous[k],0,k};
 for(unsigned i=0;i<count;++i){const auto id=logical_id(d,begin+i);unsigned at=d.keep;
  for(unsigned k=0;k<d.keep;++k)if(selected[k].id==~0U||wire::better(d.results[id],id,d.results[selected[k].id],selected[k].id)){at=k;break;}
  if(at<d.keep){for(unsigned k=d.keep-1;k>at;--k)selected[k]=selected[k-1];selected[at]={id,1,i};}
 }
}
__global__ void retain_kernel(Device d,const Selection* selection,const World* previous,const Scene* previous_scenes,World* retained,Scene* retained_scenes,unsigned* ids){
 const unsigned k=blockIdx.x;if(k>=d.keep)return;const auto pick=selection[k];if(pick.id==~0U)return;
 const auto* source=pick.source?d.worlds+pick.index:previous+pick.index;const auto* scene=pick.source?d.scenes+pick.index:previous_scenes+pick.index;
 compact::copy_bytes(reinterpret_cast<unsigned char*>(retained+k),reinterpret_cast<const unsigned char*>(source),sizeof(World),threadIdx.x,blockDim.x);
 compact::copy_bytes(reinterpret_cast<unsigned char*>(retained_scenes+k),reinterpret_cast<const unsigned char*>(scene),physical_scene_bytes,threadIdx.x,blockDim.x);__syncthreads();
 if(!threadIdx.x){ids[k]=pick.id;if(!retained[k].rebind(retained_scenes+k,body_base(*source)))reject(d.errors,15,k);}
}
__device__ inline bool clone_serial(Device d,unsigned slot,unsigned logical,bool initialized){
 const auto cp=d.jobs[logical_id(d,logical)].checkpoint;
 if(!d.game_plans[cp].valid||(initialized&&!compact::clear_reachable(d.scenes[slot]))){reject(d.errors,16,slot);return false;}
 compact::copy_game(d.worlds[slot],d.bank[cp],d.game_plans[cp]);compact::copy_active(d.scenes[slot],d.bank_scenes[cp],d.scene_plans[cp]);
 if(!compact::rebind_game(d.worlds[slot],d.scenes[slot],d.game_plans[cp])){reject(d.errors,17,slot);return false;}return true;
}
__device__ inline bool copy_state_serial(World& out,Scene& scene,const World& source,const Scene& source_scene){
 for(std::size_t i=0;i<sizeof(World);i+=8)copy_word64<>(reinterpret_cast<unsigned char*>(&out)+i,reinterpret_cast<const unsigned char*>(&source)+i);
 for(std::size_t i=0;i<physical_scene_bytes;i+=8)copy_word64<>(reinterpret_cast<unsigned char*>(&scene)+i,reinterpret_cast<const unsigned char*>(&source_scene)+i);
 return out.rebind(&scene,body_base(source));
}
// Only lane0 of each warp takes the lock; no lock owner shares its warp with a
// spinning contender. Slots remain unchanged until their warp releases it.
__device__ inline void queue_retain(Device d,unsigned first,unsigned slot_first,unsigned active,World* retained,Scene* scenes,unsigned* ids,unsigned* lock){
 if(first>=d.count||active>d.count-first||slot_first>d.slots||active>d.slots-slot_first){reject(d.errors,26,first);return;}
 cuda::atomic_ref<unsigned,cuda::thread_scope_device> gate(*lock);unsigned expected=0,spins=0;
 while(!gate.compare_exchange_weak(expected,1U,cuda::memory_order_acquire,cuda::memory_order_relaxed)){
  if(++spins==(1U<<24)){reject(d.errors,27,first);return;}expected=0;__nanosleep(spins<64?64:1024);
 }
 for(unsigned k=0;k<d.keep;++k)if(ids[k]!=~0U&&ids[k]>=d.count){reject(d.errors,28,k);gate.store(0U,cuda::memory_order_release);return;}
 for(unsigned j=0;j<active;++j){const auto id=logical_id(d,first+j);unsigned at=d.keep;
  for(unsigned k=0;k<d.keep;++k)if(ids[k]==~0U||wire::better(d.results[id],id,d.results[ids[k]],ids[k])){at=k;break;}
  if(at==d.keep)continue;
  for(unsigned k=d.keep-1;k>at;--k)if(ids[k-1]!=~0U){if(!copy_state_serial(retained[k],scenes[k],retained[k-1],scenes[k-1]))reject(d.errors,24,k);ids[k]=ids[k-1];}
  if(!copy_state_serial(retained[at],scenes[at],d.worlds[slot_first+j],d.scenes[slot_first+j]))reject(d.errors,25,at);ids[at]=id;
 }
 gate.store(0U,cuda::memory_order_release);
}
__global__ void queue_kernel(Device d,unsigned* cursor,World* retained,Scene* retained_scenes,unsigned* ids,unsigned* lock){
 const unsigned slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=d.slots)return;const unsigned lane=threadIdx.x&31U,slot_first=slot-lane;
 while(true){unsigned first=0;if(!lane)first=atomicAdd(cursor,32U);first=__shfl_sync(0xffffffffU,first,0);if(first>=d.count)break;
  const unsigned logical=first+lane;if(logical<d.count){if(clone_serial(d,slot,logical,d.initialized[slot]!=0)){d.initialized[slot]=1;run_slot(d,slot,logical);}else{wire::Result failed{};failed.failed=1;d.results[logical_id(d,logical)]=failed;atomicAdd(d.processed,1U);}}
  __syncwarp();if(!lane)queue_retain(d,first,slot_first,min(32U,d.count-first),retained,retained_scenes,ids,lock);__syncwarp();
 }
}
__global__ void reference_clones_kernel(Device d,const unsigned* retained_ids,World* references,Scene* scenes){
 const unsigned k=blockIdx.x;if(k>=d.keep)return;const auto root=d.jobs[retained_ids[k]].root;
 compact::copy_bytes(reinterpret_cast<unsigned char*>(references+k),reinterpret_cast<const unsigned char*>(d.bank+root),sizeof(World),threadIdx.x,blockDim.x);
 compact::copy_bytes(reinterpret_cast<unsigned char*>(scenes+k),reinterpret_cast<const unsigned char*>(d.bank_scenes+root),physical_scene_bytes,threadIdx.x,blockDim.x);__syncthreads();
 if(!threadIdx.x&&!references[k].rebind(scenes+k,body_base(d.bank[root])))reject(d.errors,18,k);
}
__device__ inline void compare_states(Device d,World& a,Scene& sa,World& b,Scene& sb,compact::GamePlan& plan,unsigned k){
 if(!compact::build_game_plan(a,plan)){reject(d.errors,19,k);return;}
 constexpr std::uintptr_t canonical=0x100000;bool valid=a.relocate_body_pointers(body_base(a),canonical)&&b.relocate_body_pointers(body_base(b),canonical);a.bind(nullptr);b.bind(nullptr);
 const auto ga=compact::first_game_difference(a,b,plan),sc=compact::first_difference(sa,sb);valid=a.rebind(&sa,canonical)&&b.rebind(&sb,canonical)&&valid;
 if(!valid||ga!=physics::null_index||sc!=physics::null_index)reject(d.errors,20,k);
}
__global__ void validate_kernel(Device d,const unsigned* retained_ids,World* retained,Scene* retained_scenes,World* references,Scene* reference_scenes,compact::GamePlan* plans){
 const unsigned k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=d.keep)return;const auto id=retained_ids[k],root=d.jobs[id].root;wire::Result expected;
 future(references[k],reference_scenes[k],d.options[root],d.replay,d.replay_size,d.candidates[d.jobs[id].candidate],d.states[root].result,expected);
 const auto* a=reinterpret_cast<const unsigned char*>(&expected);const auto* b=reinterpret_cast<const unsigned char*>(d.results+id);
 for(unsigned i=0;i<sizeof(expected);++i)if(a[i]!=b[i]){reject(d.errors,21,k);break;}
 compare_states(d,references[k],reference_scenes[k],retained[k],retained_scenes[k],plans[k],k);
}
__global__ void continuation_kernel(Device d,World* retained,Scene* retained_scenes,World* references,Scene* reference_scenes,compact::GamePlan* plans){
 const unsigned k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=d.keep)return;
 const auto a=retained[k].step({}),b=references[k].step({});const auto fa=retained[k].frame(a),fb=references[k].frame(b);
 const auto* aa=reinterpret_cast<const unsigned char*>(&fa);const auto* bb=reinterpret_cast<const unsigned char*>(&fb);
 for(unsigned i=0;i<sizeof(fa);++i)if(aa[i]!=bb[i]){reject(d.errors,22,k);break;}
 if(retained[k].failed()||references[k].failed())reject(d.errors,23,k);
 compare_states(d,references[k],reference_scenes[k],retained[k],retained_scenes[k],plans[k],k);
}
} // namespace gpu_first
