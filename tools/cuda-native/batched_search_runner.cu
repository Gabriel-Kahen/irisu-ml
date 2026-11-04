// Independent launch-group scheduler alongside the scalar runner. Reuse its
// rules, kernels and wire without changing the original executable's defaults.
#define main scalar_search_entry
#include "search_runner.cu"
#undef main

struct GroupSample {Sample time;unsigned batches{};};
__global__ void group_clone_kernel(World* worlds,Scene* scenes,const World* source,const Scene* source_scene,unsigned lanes,const compact::GamePlan* gp,const compact::Plan* sp,unsigned initialized,Validation* errors){
 const unsigned lane=blockIdx.x;if(lane>=lanes)return;__shared__ bool valid;
 if(!threadIdx.x){valid=lane>=initialized||bool(compact::clear_reachable(scenes[lane]));if(!valid)atomicAdd(&errors->errors,1U);}__syncthreads();if(!valid)return;
 compact::copy_game(worlds[lane],*source,*gp,threadIdx.x,blockDim.x);compact::copy_active(scenes[lane],*source_scene,*sp,threadIdx.x,blockDim.x);__syncthreads();
 if(!threadIdx.x&&!compact::rebind_game(worlds[lane],scenes[lane],*gp))atomicAdd(&errors->errors,1U);
}
__global__ void group_future_kernel(World* worlds,Scene* scenes,Options options,const std::uint32_t* replay,unsigned replay_size,const wire::Candidate* candidates,unsigned count,unsigned lanes,const wire::Result* prefix,wire::Result* results,const unsigned* permutation){
 const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane<lanes){const unsigned index=permutation?permutation[lane%count]:lane%count;future(worlds[lane],scenes[lane],options,replay,replay_size,candidates[index],*prefix,results[(lane/count)*count+index]);}
}
__global__ void map_selected_ranks(const unsigned* ranks,const unsigned* inverse,unsigned* mapped,unsigned keep){const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<keep)mapped[i]=inverse[ranks[i]];}
int main(int argc,char** argv){try{
 const auto started=std::chrono::steady_clock::now();if(argc<5)throw std::runtime_error("usage: batched-search replay specs prefix horizon [--concurrent-batches 1|2] [scalar fast-profile options]");
 auto input=wire::load_inputs(argv[1],argv[2],argv[3],argv[4]);Options options;options.seed=input.options.seed;options.prefix=input.options.prefix;options.horizon=input.options.horizon;options.warmups=0;
 options.copy_u64=options.reuse_prefix=options.physical_snapshot=options.compact_clone=options.merge_rank=options.tiled_selection=options.parallel_guard=true;
 const unsigned count=input.candidates.size();options.keep=std::min(2U,count);unsigned parallel=2;bool solver_counters=true,order_candidates=false;
 for(int i=5;i<argc;++i){const std::string a=argv[i];
  if(a=="--order-candidates")order_candidates=true;else if(a=="--host-prefix")options.host_prefix=true;else if(a=="--final-only")options.trace=false;else if(a=="--skip-compaction-check")options.validate=false;else if(a=="--skip-solver-counters")solver_counters=false;
  else if(a=="--copy-u64"||a=="--reuse-prefix"||a=="--physical-snapshot"||a=="--compact-clone"||a=="--merge-rank"||a=="--tiled-selection"||a=="--parallel-prefix-guard"){}
  else if(i+1<argc&&a=="--concurrent-batches")parallel=std::stoul(argv[++i]);
  else if(i+1<argc&&a=="--repeats")options.repeats=std::stoul(argv[++i]);else if(i+1<argc&&a=="--warmups")options.warmups=std::stoul(argv[++i]);
  else if(i+1<argc&&a=="--keep")options.keep=std::stoul(argv[++i]);else if(i+1<argc&&a=="--threads")options.threads=std::stoul(argv[++i]);
  else if(i+1<argc&&a=="--stack-kib")options.stack_kib=std::stoul(argv[++i]);else if(i+1<argc&&a=="--target")options.target=std::stoll(argv[++i]);
  else if(i+1<argc&&a=="--scratch-poison")options.scratch_poison=std::stoul(argv[++i]);else throw std::runtime_error("unsupported batched-search option: "+a);
 }
 if(parallel<1||parallel>2||!options.repeats||options.repeats>20||options.warmups>10||!options.keep||options.keep>count||!options.threads||options.threads>256||options.stack_kib<4||options.stack_kib>128||options.scratch_poison>2)throw std::runtime_error("invalid bounded launch-group options");
 std::vector<unsigned> permutation(count),inverse(count);std::iota(permutation.begin(),permutation.end(),0U);
 if(order_candidates)std::stable_sort(permutation.begin(),permutation.end(),[&](unsigned a,unsigned b){
  for(unsigned offset=0;offset<options.horizon;++offset){const unsigned tick=options.prefix+offset,base=tick<input.replay.size()?input.replay[tick]:0;const auto av=wire::input_at(input.candidates[a],offset,base),bv=wire::input_at(input.candidates[b],offset,base);if(av!=bv)return av<bv;}return a<b;});
 for(unsigned i=0;i<count;++i)inverse[permutation[i]]=i;
 const unsigned capacity=count*parallel;std::vector<wire::Result> output(capacity),previous;std::vector<unsigned> order(capacity),previous_order;std::vector<GroupSample> samples;
 Validation validation{};unsigned initialized=0,prefix_executions=0;std::uint64_t observed_ticks=0,allocated=0;std::size_t stack=0;unsigned registers=0;std::size_t local_bytes=0;
 compact::GamePlan host_gp{};compact::Plan host_sp{};
#if IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER
 physics::interleaved_solver::Counters solver_counts{};
#endif
 {
  check(cudaDeviceSetLimit(cudaLimitStackSize,std::size_t(options.stack_kib)*1024));check(cudaDeviceGetLimit(&stack,cudaLimitStackSize));Memory memory;
  auto* replay=memory.allocate<std::uint32_t>(input.replay.size());auto* candidates=memory.allocate<wire::Candidate>(count);
  auto* prefix=memory.allocate<World>(1);auto* prefix_scene=memory.allocate<Scene>(1);auto* prefix_result=memory.allocate<wire::Result>(1);
  auto* saved=memory.allocate<World>(1);auto* saved_scene=memory.allocate<Scene>(1);auto* saved_result=memory.allocate<wire::Result>(1);
  auto* worlds=memory.allocate<World>(capacity);auto* scenes=memory.allocate<Scene>(capacity);
  auto* results=memory.allocate<wire::Result>(capacity);auto* ranks=memory.allocate<unsigned>(capacity);
  unsigned *device_permutation=nullptr,*device_inverse=nullptr,*mapped_ranks=nullptr;
  if(order_candidates){device_permutation=memory.allocate<unsigned>(count);device_inverse=memory.allocate<unsigned>(count);mapped_ranks=memory.allocate<unsigned>(options.keep*parallel);check(cudaMemcpy(device_permutation,permutation.data(),count*sizeof(unsigned),cudaMemcpyHostToDevice));check(cudaMemcpy(device_inverse,inverse.data(),count*sizeof(unsigned),cudaMemcpyHostToDevice));}
  auto* selected=memory.allocate<World>(options.keep*parallel);auto* selected_scenes=memory.allocate<Scene>(options.keep*parallel);auto* errors=memory.allocate<Validation>(1);
  auto* gp=memory.allocate<compact::GamePlan>(1);auto* sp=memory.allocate<compact::Plan>(1);
  check(cudaMemset(worlds,0,capacity*sizeof(World)));check(cudaMemset2D(scenes,sizeof(Scene),0,physical_scene_bytes,capacity));check(cudaMemset(gp,0,sizeof(*gp)));check(cudaMemset(sp,0,sizeof(*sp)));
#if IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER
  namespace inter=physics::interleaved_solver;static_assert(offsetof(Scene,world)==0);
  auto* words=memory.allocate<std::uint32_t>(inter::words_required(capacity));auto* counters=memory.allocate<unsigned long long>(inter::counter_count);check(cudaMemset(counters,0,sizeof(solver_counts)));
  const inter::Descriptor descriptor{words,reinterpret_cast<std::uintptr_t>(scenes),sizeof(Scene),capacity,32,64,solver_counters?counters:nullptr};check(cudaMemcpyToSymbol(inter::device_workspace,&descriptor,sizeof(descriptor)));
#endif
  std::size_t rank_bytes{};check(physics::search_accel::rank_workspace_bytes(results,ranks,count,rank_bytes));auto* workspace=memory.allocate<unsigned char>(rank_bytes);
  World* empty{};Scene* empty_scene{};if(!options.host_prefix){auto w=cold_object<World>();auto s=cold_object<Scene>();empty=memory.allocate<World>(1);empty_scene=memory.allocate<Scene>(1);check(cudaMemcpy(empty,w.get(),sizeof(World),cudaMemcpyHostToDevice));check(cudaMemcpy(empty_scene,s.get(),sizeof(Scene),cudaMemcpyHostToDevice));}
  allocated=memory.bytes;check(cudaMemcpy(replay,input.replay.data(),input.replay.size()*4,cudaMemcpyHostToDevice));check(cudaMemcpy(candidates,input.candidates.data(),count*sizeof(wire::Candidate),cudaMemcpyHostToDevice));
  cudaFuncAttributes attr{};check(cudaFuncGetAttributes(&attr,group_future_kernel));registers=attr.numRegs;local_bytes=attr.localSizeBytes;
  for(unsigned run=0;run<options.warmups+options.repeats;){
   const unsigned boundary=run<options.warmups?options.warmups:options.warmups+options.repeats;
   const unsigned batches=std::min(parallel,boundary-run),lanes=batches*count;GroupSample sample;sample.batches=batches;const auto begin=std::chrono::steady_clock::now();validation={};check(cudaMemcpy(errors,&validation,sizeof(validation),cudaMemcpyHostToDevice));
   std::uintptr_t old_base{};
   if(!run&&options.host_prefix){Environment environment;auto w=cold_object<World>();auto s=cold_object<Scene>();wire::Result initial;prepare_prefix(w.get(),s.get(),options,input.replay.data(),&initial);old_base=body_base(*w);check(cudaMemcpy(prefix,w.get(),sizeof(World),cudaMemcpyHostToDevice));check(cudaMemcpy(prefix_scene,s.get(),sizeof(Scene),cudaMemcpyHostToDevice));check(cudaMemcpy(prefix_result,&initial,sizeof(initial),cudaMemcpyHostToDevice));sample.time.host_prefix_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();}
   Event a,b,c,d,e,f;a.record();
   if(!run){if(options.host_prefix)rebind_prefix_kernel<<<1,1>>>(prefix,prefix_scene,old_base,errors);else prefix_kernel<<<1,256>>>(prefix,prefix_scene,empty,empty_scene,options,replay,prefix_result);
    ++prefix_executions;check(cudaMemcpyAsync(saved,prefix,sizeof(World),cudaMemcpyDeviceToDevice));check(cudaMemcpyAsync(saved_scene,prefix_scene,sizeof(Scene),cudaMemcpyDeviceToDevice));check(cudaMemcpyAsync(saved_result,prefix_result,sizeof(wire::Result),cudaMemcpyDeviceToDevice));compact_plan_kernel<<<1,1>>>(prefix,prefix_scene,gp,sp,errors);check(cudaMemcpy(&validation,errors,sizeof(validation),cudaMemcpyDeviceToHost));if(validation.errors)throw std::runtime_error("launch-group prefix rejected");}
   b.record();group_clone_kernel<<<lanes,256>>>(worlds,scenes,prefix,prefix_scene,lanes,gp,sp,initialized,errors);initialized=std::max(initialized,lanes);if(options.scratch_poison)poison_scratch_kernel<<<lanes,256>>>(scenes,lanes,nullptr,options.scratch_poison);c.record();
   group_future_kernel<<<(lanes+options.threads-1)/options.threads,options.threads>>>(worlds,scenes,options,replay,input.replay.size(),candidates,count,lanes,prefix_result,results,device_permutation);d.record();
   for(unsigned batch=0;batch<batches;++batch){auto* r=ranks+batch*count;check(physics::search_accel::rank_async(results+batch*count,r,count,workspace,rank_bytes));auto* selection=r;if(order_candidates){selection=mapped_ranks+batch*options.keep;map_selected_ranks<<<(options.keep+31)/32,32>>>(r,device_inverse,selection,options.keep);}launch_clones(selected+batch*options.keep,selected_scenes+batch*options.keep,worlds+batch*count,scenes+batch*count,options.keep,selection,errors,options);}e.record();
   for(unsigned batch=0;batch<batches;++batch){auto* w=worlds+batch*count;auto* s=scenes+batch*count;auto* r=order_candidates?mapped_ranks+batch*options.keep:ranks+batch*count;auto* chosen=selected+batch*options.keep;auto* chosen_scene=selected_scenes+batch*options.keep;
    if(options.validate){verify_kernel<<<options.keep,256>>>(chosen,chosen_scene,w,s,r,options.keep,errors,physical_scene_bytes);poison_scratch_kernel<<<options.keep,256>>>(chosen_scene,options.keep,nullptr,1);poison_scratch_kernel<<<options.keep,256>>>(s,options.keep,r,2);continue_kernel<<<(options.keep+31)/32,32>>>(chosen,w,r,options.keep,errors);verify_kernel<<<options.keep,256>>>(chosen,chosen_scene,w,s,r,options.keep,errors,physical_scene_bytes);}
    check(physics::prefix_guard::check_prefix_async(prefix,prefix_scene,prefix_result,saved,saved_scene,saved_result,errors));
   }
   check(cudaGetLastError());f.record();check(cudaEventSynchronize(f.value));check(cudaMemcpy(output.data(),results,lanes*sizeof(wire::Result),cudaMemcpyDeviceToHost));check(cudaMemcpy(order.data(),ranks,lanes*sizeof(unsigned),cudaMemcpyDeviceToHost));check(cudaMemcpy(&validation,errors,sizeof(validation),cudaMemcpyDeviceToHost));
   if(validation.errors)throw std::runtime_error("launch-group snapshot check failed");
   for(unsigned batch=0;batch<batches;++batch){auto* values=output.data()+batch*count;auto* indices=order.data()+batch*count;
    if(!previous.empty()&&(std::memcmp(previous.data(),values,count*sizeof(wire::Result))||std::memcmp(previous_order.data(),indices,count*sizeof(unsigned))))throw std::runtime_error("independent launch-group packets differ");
    previous.assign(values,values+count);previous_order.assign(indices,indices+count);
    if(run>=options.warmups)for(unsigned i=0;i<count;++i)observed_ticks+=values[i].simulated_ticks;
   }
   sample.time.prefix_ms=run?0:elapsed(a,b);sample.time.branch_ms=elapsed(b,c);sample.time.rollout_ms=elapsed(c,d);sample.time.rank_compact_ms=elapsed(d,e);sample.time.validation_ms=elapsed(e,f);sample.time.total_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();if(run>=options.warmups)samples.push_back(sample);run+=batches;
  }
  check(cudaMemcpy(&host_gp,gp,sizeof(host_gp),cudaMemcpyDeviceToHost));check(cudaMemcpy(&host_sp,sp,sizeof(host_sp),cudaMemcpyDeviceToHost));
#if IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER
  check(cudaMemcpy(&solver_counts,counters,sizeof(solver_counts),cudaMemcpyDeviceToHost));const inter::Descriptor cleared{};check(cudaMemcpyToSymbol(inter::device_workspace,&cleared,sizeof(cleared)));
#endif
 }
 unsigned failed=0;std::uint64_t ticks=0;for(const auto& r:previous){failed+=bool(r.failed);ticks+=r.simulated_ticks;}
 std::fprintf(stderr,"{\"identity\":\"experimental-independent-launch-group-search-v1\",\"backend\":\"cuda-native\",\"numeric_profile\":\"%s\",\"promotion_eligible\":false,\"candidates\":%u,\"prefix_ticks\":%u,\"horizon\":%u,\"simulated_ticks\":%llu,\"total_future_ticks\":%llu,\"failed_candidates\":%u,\"trace_every_tick\":%s,\"host_prefix\":%s,\"concurrent_batches\":%u,\"world_pool_lanes\":%u,\"max_contacts\":%u,\"prefix_executions\":%u,\"prefix_immutability_checks\":true,\"prefix_immutability_errors\":%u,\"guest_calls\":0,\"fallback_calls\":0,\"compaction_errors\":%u,\"selected_state_checks\":%s,\"physical_snapshot\":true,\"compact_clone\":true,\"merge_rank\":true,\"tiled_selection\":true,\"parallel_prefix_guard\":true,\"reuse_prefix\":true,\"keep\":%u,\"warmups\":%u,\"repeats\":%u,\"world_bytes\":%zu,\"scene_bytes\":%zu,\"allocated_bytes\":%llu,\"registers\":%u,\"local_bytes\":%zu,\"effective_stack_bytes\":%zu,\"clone_game_bytes\":%u,\"clone_scene_bytes\":%zu,\"sample_scope\":\"launch-group; divide phase times by sample.batches for per-batch attribution\",\"elapsed_before_stdout_ms\":%.9f,\"samples\":[",(IRISU_CUDA_NATIVE_DEVICE_TRIG==1?"approximate-device-trig-library":IRISU_CUDA_NATIVE_DEVICE_TRIG==2?"approximate-device-trig-intrinsic":IRISU_CUDA_NATIVE_BOUNDED_MATH?"bounded-device-add-sub-mul":"checked"),count,options.prefix,options.horizon,(unsigned long long)ticks,(unsigned long long)observed_ticks,failed,options.trace?"true":"false",options.host_prefix?"true":"false",parallel,capacity,physics::max_contacts,prefix_executions,validation.prefix_errors,validation.errors,options.validate?"true":"false",options.keep,options.warmups,options.repeats,sizeof(World),sizeof(Scene),(unsigned long long)allocated,registers,local_bytes,stack,host_gp.bytes,compact::transport_bytes(host_sp),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());
 for(unsigned i=0;i<samples.size();++i){if(i)std::fputc(',',stderr);const auto& s=samples[i];const auto& t=s.time;std::fprintf(stderr,"{\"batches\":%u,\"host_prefix_ms\":%.9f,\"prefix_ms\":%.9f,\"branch_ms\":%.9f,\"rollout_ms\":%.9f,\"rank_compact_ms\":%.9f,\"validation_ms\":%.9f,\"total_ms\":%.9f}",s.batches,t.host_prefix_ms,t.prefix_ms,t.branch_ms,t.rollout_ms,t.rank_compact_ms,t.validation_ms,t.total_ms);}
 std::fprintf(stderr,"],\"interleaved_solver\":{\"counters_enabled\":%s,\"compiled\":%s",solver_counters?"true":"false",
#if IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER
 "true");std::fprintf(stderr,",\"velocity_islands\":%llu,\"position_islands\":%llu,\"capacity_fallbacks\":%llu,\"outside_pool_fallbacks\":%llu,\"errors\":%llu",solver_counts.velocity_islands,solver_counts.position_islands,solver_counts.capacity_fallbacks,solver_counts.outside_pool_fallbacks,solver_counts.errors);
#else
 "false");
#endif
 std::fprintf(stderr,"},\"order_candidates\":%s,\"compile_profile\":{\"bounded_math\":%s,\"bounded_store\":%s,\"f32_div_store\":%s,\"device_trig\":%u}}\n",order_candidates?"true":"false",IRISU_CUDA_NATIVE_BOUNDED_MATH?"true":"false",IRISU_CUDA_NATIVE_BOUNDED_STORE?"true":"false",IRISU_CUDA_NATIVE_F32_DIV_STORE?"true":"false",IRISU_CUDA_NATIVE_DEVICE_TRIG);if(failed||validation.errors)return 2;wire::emit_results(previous,previous_order);return 0;
}catch(const std::exception& error){std::fprintf(stderr,"batched-search: %s\n",error.what());return 1;}}
