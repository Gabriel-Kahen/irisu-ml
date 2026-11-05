// Resident typed-CUDA search service. Original benchmark executables are unchanged.
#include "../../clone/cuda_native/game_world.cuh"
#include "common.hpp"
#include "../../clone/cuda_native/compact_game_snapshot.cuh"
#include <cuda_runtime.h>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
namespace game=irisu::cuda_game;
namespace physics=irisu::cuda_native;
using World=game::NativeGameWorld;using Scene=physics::ScenePipelineState;
namespace compact=physics::compact_snapshot;
namespace {
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
template<class T>struct Device {T* p{};explicit Device(std::size_t n){check(cudaMalloc(&p,n*sizeof(T)));}~Device(){cudaFree(p);}Device(const Device&)=delete;};
struct Failure {
 unsigned long long tick{};
 unsigned physics_status{},simulator_failed{},body_overflow{},group_overflow{},active_contacts_overflow{};
 unsigned body_count{},shape_count{},contact_count{},game_bodies{},game_groups{},active_contacts{},summary_capacity{};
};
ORCH_HD void failure(const World& world,const Scene& scene,Failure& out){
 const auto& s=world.simulator;out.tick=s.tick_;out.physics_status=unsigned(scene.world.status);out.simulator_failed=s.failed_;
 out.body_overflow=s.bodies_.overflow;out.group_overflow=s.groups_.overflow;out.active_contacts_overflow=s.active_contact_pairs_.overflow;
 out.body_count=scene.world.body_count;out.shape_count=scene.world.shape_count;out.contact_count=scene.world.contact_count;
 out.game_bodies=s.bodies_.size();out.game_groups=s.groups_.size();out.active_contacts=s.active_contact_pairs_.size();out.summary_capacity=!world.failed();
}
const char* failure_status(const Failure& f){
 static constexpr const char* names[]={"ok","capacity","invalid_body","invalid_shape","unsupported_numeric","unsupported_collision","unsupported_joint","bridge_required","invalid_snapshot"};
 if(f.physics_status)return f.physics_status<std::size(names)?names[f.physics_status]:"unknown_physics_status";
 return f.summary_capacity?"summary_capacity":f.body_overflow?"game_body_capacity":f.group_overflow?"game_group_capacity":f.active_contacts_overflow?"active_contact_capacity":"game_failure";
}
void emit_failure(const Failure& f){
 std::cout<<"{\"status\":"<<std::quoted(failure_status(f))<<",\"physics_status\":"<<f.physics_status<<",\"tick\":"<<f.tick
  <<",\"simulator_failed\":"<<(f.simulator_failed?"true":"false")<<",\"game_body_overflow\":"<<(f.body_overflow?"true":"false")
  <<",\"game_group_overflow\":"<<(f.group_overflow?"true":"false")<<",\"active_contacts_overflow\":"<<(f.active_contacts_overflow?"true":"false")
  <<",\"body_count\":"<<f.body_count<<",\"shape_count\":"<<f.shape_count<<",\"contact_count\":"<<f.contact_count
  <<",\"game_bodies\":"<<f.game_bodies<<",\"game_groups\":"<<f.game_groups<<",\"active_contacts\":"<<f.active_contacts
  <<",\"max_bodies\":"<<physics::max_bodies<<",\"max_contacts\":"<<physics::max_contacts<<'}';
}
ORCH_HD void summarize(const World& world,unsigned seed,orch::Summary& out){
 const auto& s=world.simulator;out.seed=seed;out.tick=s.tick_;out.score=s.score_;out.gauge=s.gauge_;out.level=s.level_;out.qualifying=s.qualifying_clear_count_;out.highest_chain=s.highest_chain_;out.terminated=s.terminated_;out.truncated=s.truncated_;out.failed=world.failed();out.config_hash=s.config_hash_;out.left_held=s.previous_left_level_;out.right_held=s.previous_right_level_;out.canonical_score=s.recorded_final_score_;out.canonical_level=s.recorded_final_level_;out.canonical_chain=s.recorded_final_highest_chain_;out.body_count=out.group_count=0;
 for(const auto& b:s.bodies_){if(b.lifecycle==game::Lifecycle::Deleted)continue;if(out.body_count==orch::max_bodies){out.failed=true;return;}auto& v=out.bodies[out.body_count++];v={b.id,unsigned(b.kind),b.chain_id,b.projectile_hits,unsigned(b.rot_timer),b.color,b.position.x,b.position.y,b.velocity.x,b.velocity.y,b.size,b.remaining_lifetime,b.successful_clear_pending,b.delete_marked||b.pending_delete,b.special,b.physics_owned};
  if(b.chain_id){bool found=false;for(unsigned k=0;k<out.group_count;++k)found|=out.groups[k].id==b.chain_id;if(!found){for(const auto& g:s.groups_)if(g.id==b.chain_id){if(out.group_count==orch::max_groups){out.failed=true;return;}out.groups[out.group_count++]={g.id,g.chain,g.num};break;}}}
 }
 for(unsigned i=1;i<out.group_count;++i){const auto group=out.groups[i];unsigned j=i;while(j&&out.groups[j-1].id>group.id){out.groups[j]=out.groups[j-1];--j;}out.groups[j]=group;}
}
ORCH_HD void load_metrics(const World& world,orch::Summary& out){
 out.metrics={};out.metrics.new_groups=world.simulator.groups_.size();out.metrics.qualifications=world.simulator.qualifying_clear_count_;
 for(const auto& g:world.simulator.groups_){out.metrics.recruitment_events+=g.chain;if(g.chain>out.metrics.max_new_chain)out.metrics.max_new_chain=g.chain;}
}
__global__ void initialize(World* worlds,Scene* scenes,unsigned slot,unsigned seed,const unsigned* words,unsigned count,unsigned* executed,orch::Summary* summaries){
 if(blockIdx.x||threadIdx.x)return;auto& w=worlds[slot];w.initialize(scenes+slot,seed);unsigned n=0;while(n<count&&!w.failed()&&!w.simulator.terminated_&&!w.simulator.truncated_){w.step(game::native_replay_action(words[n],n));++n;}*executed=n;summarize(w,seed,summaries[0]);load_metrics(w,summaries[0]);
}
__global__ void rebind(World* worlds,Scene* scenes,unsigned source,const unsigned* slots,unsigned count){
 const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=count)return;
 worlds[slots[lane]].rebind(scenes+slots[lane],reinterpret_cast<std::uintptr_t>(worlds[source].simulator.bodies_.data()));
}
__global__ void clone_plan(const World* worlds,const Scene* scenes,unsigned parent,compact::GamePlan* game_plan,compact::Plan* scene_plan,unsigned* errors){
 if(!threadIdx.x&&!blockIdx.x&&(!compact::build_game_plan(worlds[parent],*game_plan)||!compact::build_plan(scenes[parent],*scene_plan)))atomicAdd(errors,1U);
}
__global__ void clone_checkpoints(World* worlds,Scene* scenes,unsigned parent,const unsigned* slots,const unsigned* initialized,unsigned count,const compact::GamePlan* game_plan,const compact::Plan* scene_plan,unsigned* errors){
 const auto lane=blockIdx.x;if(lane>=count)return;const auto slot=slots[lane];__shared__ bool valid;
 if(!threadIdx.x){valid=game_plan->valid&&(!initialized[lane]||bool(compact::clear_reachable(scenes[slot])));if(!valid)atomicAdd(errors,1U);}__syncthreads();if(!valid)return;
 compact::copy_game(worlds[slot],worlds[parent],*game_plan,threadIdx.x,blockDim.x);compact::copy_active(scenes[slot],scenes[parent],*scene_plan,threadIdx.x,blockDim.x);__syncthreads();
 if(!threadIdx.x&&!compact::rebind_game(worlds[slot],scenes[slot],*game_plan))atomicAdd(errors,1U);
}
__global__ void rollouts(World* worlds,const Scene* scenes,unsigned seed,const unsigned* slots,unsigned count,unsigned horizon,const unsigned* modes,const unsigned* randoms,const unsigned* lengths,const unsigned* inputs,unsigned* trace,unsigned* executed,orch::Summary* summaries,Failure* failures){
 const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=count)return;auto& w=worlds[slots[lane]];auto& summary=summaries[lane];summarize(w,seed,summary);summary.group_delta_count=summary.group_count;for(unsigned k=0;k<summary.group_count;++k){const auto& g=summary.groups[k];summary.group_deltas[k]={g.id,g.chain,g.num,g.chain,g.num,0};}const unsigned mode=modes[lane],random=randoms[lane],old_groups=w.simulator.groups_.size();std::uint64_t old_recruited=0;for(const auto& g:w.simulator.groups_)old_recruited+=g.chain;const auto old_qualifications=w.simulator.qualifying_clear_count_;unsigned n=0;
 while(n<horizon&&!w.failed()&&!w.simulator.terminated_&&!w.simulator.truncated_){unsigned word=0;
  if(n<lengths[lane])word=inputs[lane*horizon+n];
  else if(mode){const unsigned interval=mode==2?16U+(random%17U):6U+(random%13U);if((w.simulator.tick_+random)%interval==0){summarize(w,seed,summary);word=orch::policy(summary,mode,random);}}
  trace[lane*horizon+n]=word;w.step(game::native_replay_action(word,w.simulator.tick_));++n;
 }
 executed[lane]=n;summarize(w,seed,summary);summary.metrics={};summary.metrics.new_groups=w.simulator.groups_.size()-old_groups;summary.metrics.qualifications=w.simulator.qualifying_clear_count_-old_qualifications;unsigned group_index=0;for(const auto& g:w.simulator.groups_){summary.metrics.recruitment_events+=g.chain;if(group_index++>=old_groups&&g.chain>summary.metrics.max_new_chain)summary.metrics.max_new_chain=g.chain;}summary.metrics.recruitment_events-=old_recruited;
 if(summary.failed)failure(w,scenes[slots[lane]],failures[lane]);
 for(unsigned k=0;k<summary.group_delta_count;++k){auto& d=summary.group_deltas[k];for(const auto& g:w.simulator.groups_)if(g.id==d.id){d.chain_after=g.chain;d.num_after=g.num;break;}for(unsigned b=0;b<summary.body_count;++b)d.members_after+=summary.bodies[b].chain_id==d.id;}

}
__global__ void rebind_prefix(World* worlds,Scene* scenes,unsigned slot,std::uintptr_t old_base){
 if(!threadIdx.x&&!blockIdx.x)worlds[slot].rebind(scenes+slot,old_base);
}
struct Node {unsigned slot{},seed{};};
class Service {
 unsigned capacity_,next_{1},launch_threads_{},multiprocessors_{1};bool host_prefix_{};Device<World> worlds_;Device<Scene> scenes_;std::map<unsigned,Node> nodes_;std::vector<unsigned> free_,initialized_;std::unique_ptr<World> empty_;
 unsigned allocate(){if(free_.empty())throw std::runtime_error("checkpoint capacity exhausted; release handles");const auto slot=free_.back();free_.pop_back();return slot;}
 void reset_slot(unsigned slot){
  check(cudaMemcpy(worlds_.p+slot,empty_.get(),sizeof(World),cudaMemcpyHostToDevice));check(cudaMemset(scenes_.p+slot,0,sizeof(Scene)));initialized_[slot]=0;
 }
public:
 explicit Service(unsigned capacity,bool host_prefix,unsigned launch_threads):capacity_(capacity),launch_threads_(launch_threads),host_prefix_(host_prefix),worlds_(capacity),scenes_(capacity),initialized_(capacity,0){
  int device=0;cudaDeviceProp properties{};check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&properties,device));multiprocessors_=unsigned(properties.multiProcessorCount);
  check(cudaMemset(worlds_.p,0,capacity*sizeof(World)));check(cudaMemset(scenes_.p,0,capacity*sizeof(Scene)));
  // Simulator has a configuration constructor. Copy a cold, uninitialized
  // object before device initialization, as in the established CUDA runner.
  empty_=std::make_unique<World>();std::memset(empty_.get(),0,sizeof(World));::new(static_cast<void*>(empty_.get()))World();
  for(unsigned i=0;i<capacity;++i)check(cudaMemcpy(worlds_.p+i,empty_.get(),sizeof(World),cudaMemcpyHostToDevice));
  for(unsigned i=capacity;i>0;--i)free_.push_back(i-1);
 }
 unsigned rollout_threads(unsigned count)const{
  if(launch_threads_)return launch_threads_;
  // Small frontiers need independent CTAs, not one warp sharing a single SM.
  // Increase threads only when there are enough worlds for two CTAs per SM.
  const auto target=std::max(1U,2U*multiprocessors_);unsigned threads=1;
  while(threads<32U&&threads*2U<=count/target)threads*=2U;return threads;
 }
 void identity(){std::cout<<"{\"ok\":true,\"identity\":{\"protocol\":1,\"backend\":\"cuda-native-resident\",\"checkpoint_mode\":\"resident-device-compact\",\"promotion_eligible\":false,\"max_checkpoints\":"<<capacity_<<",\"launch_threads\":"<<launch_threads_<<",\"partial_batches\":true,\"max_contacts\":"<<physics::max_contacts<<",\"max_bodies\":"<<physics::max_bodies<<",\"adaptive_launch_threads\":"<<(!launch_threads_?"true":"false")<<",\"multiprocessors\":"<<multiprocessors_<<",\"host_prefix\":"<<(host_prefix_?"true":"false")<<",\"max_horizon\":"<<orch::max_horizon<<",\"checkpoint_bytes\":"<<sizeof(World)+sizeof(Scene)<<",\"numeric_profile\":\""<<(IRISU_CUDA_NATIVE_DEVICE_TRIG?"approximate-device-trig":IRISU_CUDA_NATIVE_BOUNDED_MATH?"bounded":"checked")<<"\"}}\n";}
 void load(unsigned seed,const std::vector<unsigned>& words){
  const auto slot=allocate();try{Device<unsigned> input(std::max<std::size_t>(1,words.size())),executed(1);Device<orch::Summary> summaries(1);if(!words.empty())check(cudaMemcpy(input.p,words.data(),words.size()*4,cudaMemcpyHostToDevice));
   orch::Summary summary;unsigned count=0;
   if(host_prefix_){
    auto world=std::make_unique<World>();auto scene=std::make_unique<Scene>();
    std::memset(world.get(),0,sizeof(World));::new(static_cast<void*>(world.get()))World();
    std::memset(scene.get(),0,sizeof(Scene));::new(static_cast<void*>(scene.get()))Scene();
    world->initialize(scene.get(),seed);
    while(count<words.size()&&!world->failed()&&!world->simulator.terminated_&&!world->simulator.truncated_){world->step(game::native_replay_action(words[count],count));++count;}
    summarize(*world,seed,summary);load_metrics(*world,summary);
    const auto base=reinterpret_cast<std::uintptr_t>(world->simulator.bodies_.data());
    check(cudaMemcpy(worlds_.p+slot,world.get(),sizeof(World),cudaMemcpyHostToDevice));check(cudaMemcpy(scenes_.p+slot,scene.get(),sizeof(Scene),cudaMemcpyHostToDevice));
    rebind_prefix<<<1,1>>>(worlds_.p,scenes_.p,slot,base);check(cudaGetLastError());check(cudaDeviceSynchronize());
   }else{
    check(cudaMemset(summaries.p,0,sizeof(orch::Summary)));
    initialize<<<1,1>>>(worlds_.p,scenes_.p,slot,seed,input.p,words.size(),executed.p,summaries.p);check(cudaGetLastError());check(cudaDeviceSynchronize());check(cudaMemcpy(&summary,summaries.p,sizeof(summary),cudaMemcpyDeviceToHost));check(cudaMemcpy(&count,executed.p,4,cudaMemcpyDeviceToHost));
   }
   if(summary.failed)throw std::runtime_error("CUDA initialization/replay failed");initialized_[slot]=1;const auto handle=next_++;nodes_[handle]={slot,seed};std::cout<<"{\"ok\":true,\"result\":{\"handle\":"<<handle<<",\"summary\":";orch::emit_summary(summary);std::cout<<",\"metrics\":";orch::emit_metrics(summary.metrics);std::cout<<",\"words\":";orch::emit_words(std::vector<unsigned>(words.begin(),words.begin()+count));std::cout<<"}}\n";
  }catch(...){free_.push_back(slot);throw;}
 }
 void batch(unsigned parent,unsigned horizon,const std::vector<orch::Candidate>& candidates){
  const auto found=nodes_.find(parent);if(found==nodes_.end())throw std::runtime_error("unknown checkpoint");const auto count=unsigned(candidates.size());if(free_.size()<count)throw std::runtime_error("checkpoint capacity exhausted; release handles");const auto start=std::chrono::steady_clock::now();std::vector<unsigned> slots,handles(count,0),modes,randoms,lengths,inputs(std::size_t(count)*horizon),trace(inputs.size()),executed(count);std::vector<orch::Summary> summaries(count);std::vector<Failure> failures(count);
  for(const auto& c:candidates){modes.push_back(c.mode);randoms.push_back(c.search_seed);lengths.push_back(c.words.size());std::copy(c.words.begin(),c.words.end(),inputs.begin()+(modes.size()-1)*horizon);}
  try{for(unsigned i=0;i<count;++i)slots.push_back(allocate());std::vector<unsigned> initialized;for(auto slot:slots)initialized.push_back(initialized_[slot]);Device<unsigned> dinitialized(count),errors(1);Device<compact::GamePlan> game_plan(1);Device<compact::Plan> scene_plan(1);Device<unsigned> dslots(count),dmodes(count),drandoms(count),dlengths(count),dinputs(inputs.size()),dtrace(inputs.size()),dexecuted(count);Device<orch::Summary> dsummaries(count);Device<Failure> dfailures(count);check(cudaMemset(dfailures.p,0,count*sizeof(Failure)));
   // Full host readbacks include inactive summary fields and early-stop trace tails.
   check(cudaMemset(dsummaries.p,0,count*sizeof(orch::Summary)));check(cudaMemset(dtrace.p,0,trace.size()*4));
   check(cudaMemset(errors.p,0,4));check(cudaMemcpy(dinitialized.p,initialized.data(),count*4,cudaMemcpyHostToDevice));check(cudaMemcpy(dslots.p,slots.data(),count*4,cudaMemcpyHostToDevice));check(cudaMemcpy(dmodes.p,modes.data(),count*4,cudaMemcpyHostToDevice));check(cudaMemcpy(drandoms.p,randoms.data(),count*4,cudaMemcpyHostToDevice));check(cudaMemcpy(dlengths.p,lengths.data(),count*4,cudaMemcpyHostToDevice));check(cudaMemcpy(dinputs.p,inputs.data(),inputs.size()*4,cudaMemcpyHostToDevice));
   clone_plan<<<1,1>>>(worlds_.p,scenes_.p,found->second.slot,game_plan.p,scene_plan.p,errors.p);
   clone_checkpoints<<<count,256>>>(worlds_.p,scenes_.p,found->second.slot,dslots.p,dinitialized.p,count,game_plan.p,scene_plan.p,errors.p);
   unsigned clone_errors=0;check(cudaMemcpy(&clone_errors,errors.p,4,cudaMemcpyDeviceToHost));if(clone_errors)throw std::runtime_error("compact checkpoint clone rejected");for(auto slot:slots)initialized_[slot]=1;
   const auto threads=rollout_threads(count);rollouts<<<(count+threads-1)/threads,threads>>>(worlds_.p,scenes_.p,found->second.seed,dslots.p,count,horizon,dmodes.p,drandoms.p,dlengths.p,dinputs.p,dtrace.p,dexecuted.p,dsummaries.p,dfailures.p);check(cudaGetLastError());check(cudaDeviceSynchronize());check(cudaMemcpy(executed.data(),dexecuted.p,count*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(trace.data(),dtrace.p,trace.size()*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(summaries.data(),dsummaries.p,count*sizeof(orch::Summary),cudaMemcpyDeviceToHost));check(cudaMemcpy(failures.data(),dfailures.p,count*sizeof(Failure),cudaMemcpyDeviceToHost));
   // Invalid candidates never acquire handles. Reset their storage before reuse;
   // compact clear_reachable deliberately rejects a failed physics world.
   for(unsigned i=0;i<count;++i)if(summaries[i].failed){reset_slot(slots[i]);}else{const auto handle=next_++;handles[i]=handle;nodes_[handle]={slots[i],found->second.seed};}
   std::cout<<"{\"ok\":true,\"results\":[";bool comma=false;
   for(unsigned i=0;i<count;++i)if(handles[i]){if(comma)std::cout<<',';comma=true;std::cout<<"{\"candidate_index\":"<<i<<",\"handle\":"<<handles[i]<<",\"summary\":";orch::emit_summary(summaries[i]);std::cout<<",\"group_deltas\":";orch::emit_group_deltas(summaries[i]);std::cout<<",\"metrics\":";orch::emit_metrics(summaries[i].metrics);std::cout<<",\"words\":";orch::emit_words(std::vector<unsigned>(trace.begin()+i*horizon,trace.begin()+i*horizon+executed[i]));std::cout<<'}';}
   std::cout<<"],\"rejected\":[";comma=false;
   for(unsigned i=0;i<count;++i)if(!handles[i]){if(comma)std::cout<<',';comma=true;std::cout<<"{\"candidate_index\":"<<i<<",\"diagnostic\":";emit_failure(failures[i]);std::cout<<",\"words\":";orch::emit_words(std::vector<unsigned>(trace.begin()+i*horizon,trace.begin()+i*horizon+executed[i]));std::cout<<'}';}
   std::cout<<"],\"simulated_ticks\":"<<std::accumulate(executed.begin(),executed.end(),std::uint64_t{})<<",\"launch_threads\":"<<threads<<",\"rollout_blocks\":"<<(count+threads-1)/threads<<",\"wall_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
   for(unsigned i=0;i<count;++i)if(!handles[i])free_.push_back(slots[i]);
  }catch(...){for(auto handle:handles)nodes_.erase(handle);for(auto slot:slots)free_.push_back(slot);throw;}
 }
 void release(const std::vector<unsigned>& handles){for(auto handle:handles){const auto found=nodes_.find(handle);if(found!=nodes_.end()){free_.push_back(found->second.slot);nodes_.erase(found);}}std::cout<<"{\"ok\":true}\n";}
};
}
int main(int argc,char** argv){try{
 unsigned capacity=64,launch_threads=0;bool host_prefix=false;for(int i=1;i<argc;++i){if(std::string(argv[i])=="--host-prefix")host_prefix=true;else if(std::string(argv[i])=="--threads"&&i+1<argc){launch_threads=std::stoul(argv[++i]);if(!launch_threads||launch_threads>128)throw std::runtime_error("threads must be 1..128");}else if(std::string(argv[i])=="--max-checkpoints"&&i+1<argc)capacity=std::stoul(argv[++i]);else throw std::runtime_error("usage: cuda-engine [--max-checkpoints N] [--host-prefix] [--threads 1..128]");}if(!capacity||capacity>1024)throw std::runtime_error("checkpoint limit must be 1..1024");check(cudaDeviceSetLimit(cudaLimitStackSize,16U<<10));const std::uint16_t control=0x027f;asm volatile("fldcw %0"::"m"(control));Service service(capacity,host_prefix,launch_threads);std::string line;
 while(std::getline(std::cin,line)){try{std::istringstream in(line);std::string command;in>>command;
  if(command=="identity"){orch::end(in);service.identity();}
  else if(command=="load"){const auto seed=orch::number(in,UINT32_MAX,"seed"),count=orch::number(in,10000000,"word count");service.load(seed,orch::words(in,count));}
  else if(command=="batch"){const auto parent=orch::number(in,UINT32_MAX,"parent"),horizon=orch::number(in,orch::max_horizon,"horizon"),count=orch::number(in,1024,"candidate count");orch::end(in);if(!horizon||!count)throw std::runtime_error("empty batch");std::vector<std::string> lines;for(unsigned i=0;i<count;++i){std::string candidate;if(!std::getline(std::cin,candidate))throw std::runtime_error("short batch");lines.push_back(candidate);}std::vector<orch::Candidate> candidates;for(const auto& candidate:lines)candidates.push_back(orch::candidate(candidate,horizon));service.batch(parent,horizon,candidates);}
  else if(command=="release"){const auto count=orch::number(in,4096,"handle count");std::vector<unsigned> handles;for(unsigned i=0;i<count;++i)handles.push_back(orch::number(in,UINT32_MAX,"handle"));orch::end(in);service.release(handles);}
  else if(command=="quit"){orch::end(in);break;}else throw std::runtime_error("unknown command");std::cout<<std::flush;
 }catch(const std::exception& error){orch::error(error);}}
 return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
