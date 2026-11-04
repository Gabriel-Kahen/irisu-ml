// Included inside namespace irisu::cuda_search after the one-pass kernels.
// The parent beam and child pools survive every dependent selection depth.
namespace {
__global__ void session_parent_indices(std::uint32_t* indices,unsigned count,unsigned fanout) {
  const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;
  if(lane<count)indices[lane]=lane/fanout;
}
__global__ void session_future_kernel(World* worlds,Options options,const std::uint32_t* replay,
 unsigned replay_size,const Candidate* specs,unsigned fanout,unsigned count,
 const Result* parents,const Result* root,Result* results,std::uint32_t* lifecycle,std::uint64_t end_tick,
 unsigned long long* actual_ticks,std::uint32_t* peak_heap) {
  const unsigned thread=blockIdx.x*blockDim.x+threadIdx.x;
  if(options.isolated_warps&&(thread&31))return;
  const unsigned lane=options.isolated_warps?thread/32:thread;if(lane>=count)return;
  auto& world=worlds[lane];auto result=parents[lane/fanout];
  const auto& spec=specs[lane%fanout];unsigned ticks=0;
  cuda_game::StepResult step{};GuestCache cache(world,options.cache_guest);
  if(!result.failed&&!world.simulator.terminated_&&!world.simulator.truncated_)
    for(unsigned offset=0;offset<options.horizon&&!world.failed();++offset) {
      const auto tick=world.simulator.tick_;
      const auto base=tick<replay_size?replay[tick]:0;
      step=world.step(cuda_game::replay_action(input_at(spec,offset,base),tick));
      if(options.trace_every_tick)fold(result,world.frame(step));
      ++result.simulated_ticks;++ticks;
      result.natural=step.terminated&&!step.truncated;
      result.canonical_level=step.diagnostics.recorded_final_level;
      result.canonical_score=step.diagnostics.recorded_final_score;
      result.failed|=result.natural&&!step.diagnostics.terminal_metadata_recorded;
      if(step.terminated||step.truncated)break;
    }
  cache.finish();
  if(!options.trace_every_tick&&ticks){result.trace_hash=root->trace_hash;fold(result,world.frame(step));}
  result.clears=world.simulator.qualifying_clear_count_;
  result.failed|=world.failed();result.guest_error=world.guest.error;result.fault_pc=world.guest.fault_pc;
  rank_keys(result,end_tick,options.target_score);results[lane]=result;
  lifecycle[lane]=(world.simulator.terminated_?1U:0U)|(world.simulator.truncated_?2U:0U);
  atomicAdd(actual_ticks,static_cast<unsigned long long>(ticks));
  if(ticks)atomicAdd(actual_ticks+1,1ULL);
  atomicMax(peak_heap,world.guest.heap);
}
__global__ void session_compact_results(Result* parents,const Result* children,
 const std::uint32_t* selected,unsigned keep) {
  const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;
  if(lane<keep)parents[lane]=children[selected[lane]];
}
__global__ void session_select_children(const Result* results,const std::uint32_t* ranks,
 const std::uint32_t* lifecycle,unsigned count,unsigned keep,unsigned fanout,
 bool greedy,bool protect,std::uint32_t* protected_parent,
 std::uint32_t* selected,std::uint32_t* protected_history,unsigned* errors) {
  if(threadIdx.x||blockIdx.x)return;
  selected[0]=ranks[0];unsigned filled=1;
  if(!greedy&&keep>1)for(unsigned i=1;i<count&&filled<keep;++i){
    const auto child=ranks[i];
    if(!lifecycle[child]&&!results[child].failed)selected[filled++]=child;
  }
  for(unsigned i=1;i<count&&filled<keep;++i){
    const auto child=ranks[i];bool duplicate=false;
    for(unsigned j=0;j<filled;++j)duplicate|=selected[j]==child;
    if(!duplicate)selected[filled++]=child;
  }
  if(!protect){*protected_history=UINT32_MAX;return;}
  const auto child=(*protected_parent)*fanout;
  if(child>=count){atomicAdd(errors,1U);*protected_history=UINT32_MAX;return;}
  unsigned position=keep;
  for(unsigned i=0;i<keep;++i)if(selected[i]==child)position=i;
  if(position==keep){position=keep-1;selected[position]=child;}
  *protected_parent=position;*protected_history=position;
}
__global__ void session_winner_info(const World* children,const std::uint32_t* ranks,std::uint64_t* chain) {
  if(!threadIdx.x&&!blockIdx.x)*chain=children[ranks[0]].simulator.recorded_final_highest_chain_;
}
}

void run_session(const SessionOptions& options,const unsigned char* image,
 std::size_t image_bytes,const std::uint32_t* replay,std::uint32_t replay_size,
 const Candidate* candidates,std::uint32_t fanout,SessionOutput* output) {
  const auto& game=options.game;
  if(!output||!image||!image_bytes||image_bytes>0x20000||!replay||!replay_size||!candidates||
     !fanout||fanout>16384||!options.depths||options.depths>16||!options.beam||options.beam>64||
     std::uint64_t(fanout)*options.beam>16384||game.prefix>replay_size||
     !game.horizon||game.horizon>100000||game.arena_bytes<0x40000||game.arena_bytes>(512U<<20)||
     !game.block_threads||game.block_threads>256||
     (game.isolated_warps&&game.block_threads>32&&game.block_threads%32)||
     game.stack_bytes<(16U<<10)||game.stack_bytes>(128U<<10)||
     (game.arena_word_stride!=1&&game.arena_word_stride!=32)||
     (game.arena_word_stride==32&&(game.arena_bytes&3))||
     std::uint64_t(game.prefix)+std::uint64_t(game.horizon)*options.depths>UINT32_MAX)
    throw std::invalid_argument("invalid bounded resident session arguments");
  for(unsigned i=0;i<fanout;++i)if(!valid(candidates[i],game.horizon))
    throw std::invalid_argument("invalid session candidate");
  if(options.protect_incumbent&&candidates[0].count)
    throw std::invalid_argument("protected incumbent requires an empty Candidate0");
  cuda_exact::verify_image(image,image_bytes);
  const auto begin=std::chrono::steady_clock::now();*output=SessionOutput{};
  std::vector<std::size_t> offsets,rank_offsets,selection_offsets;
  std::size_t total=0,total_ranks=0,total_selected=0;unsigned parents_count=1;
  for(unsigned d=0;d<options.depths;++d){offsets.push_back(total);
    const auto count=parents_count*fanout;const auto keep=std::min(options.beam,count);
    output->round_counts.push_back(count);output->round_kept.push_back(keep);rank_offsets.push_back(total_ranks);
    selection_offsets.push_back(total_selected);total_selected+=keep;
    total+=count;total_ranks+=padded_rank_count(count);parents_count=keep;}
  const unsigned capacity=fanout*options.beam;
  check(cudaDeviceSetLimit(cudaLimitStackSize,game.stack_bytes));Memory memory;
  auto* image_gpu=memory.allocate<unsigned char>(image_bytes);
  auto* replay_gpu=memory.allocate<std::uint32_t>(replay_size);
  auto* specs_gpu=memory.allocate<Candidate>(fanout);
  auto* parents=memory.allocate<World>(options.beam);check(cudaMemset(parents,0,sizeof(World)));
  auto* parent_arenas=memory.allocate<unsigned char>(std::size_t(options.beam)*game.arena_bytes);
  auto* parent_results=memory.allocate<Result>(options.beam);
  auto* root_result=memory.allocate<Result>(1);
  auto* children=memory.allocate<World>(capacity);
  auto* child_arenas=memory.allocate<unsigned char>(std::size_t(arena_storage_lanes(capacity,game.arena_word_stride))*game.arena_bytes);
  auto* indices=memory.allocate<std::uint32_t>(capacity);
  auto* all_results=memory.allocate<Result>(total);
  auto* all_lifecycle=memory.allocate<std::uint32_t>(total);
  auto* all_ranks=memory.allocate<std::uint32_t>(total_ranks);
  auto* all_selected=memory.allocate<std::uint32_t>(total_selected);
  auto* protected_parent=memory.allocate<std::uint32_t>(1);check(cudaMemset(protected_parent,0,4));
  auto* protected_history=memory.allocate<std::uint32_t>(options.depths);
  auto* actual_ticks=memory.allocate<unsigned long long>(2);check(cudaMemset(actual_ticks,0,16));
  auto* peak_heap=memory.allocate<std::uint32_t>(1);check(cudaMemset(peak_heap,0,4));
  auto* errors=memory.allocate<unsigned>(1);check(cudaMemset(errors,0,4));
  auto* chain=memory.allocate<std::uint64_t>(1);
  check(cudaMemcpy(image_gpu,image,image_bytes,cudaMemcpyHostToDevice));
  check(cudaMemcpy(replay_gpu,replay,std::size_t(replay_size)*4,cudaMemcpyHostToDevice));
  check(cudaMemcpy(specs_gpu,candidates,std::size_t(fanout)*sizeof(Candidate),cudaMemcpyHostToDevice));
  double host_prefix_ms=0;
  if(game.host_prefix){
    const auto start=std::chrono::steady_clock::now();auto world=std::make_unique<World>();
    std::memset(world.get(),0,sizeof(World));std::vector<unsigned char> arena(game.arena_bytes);Result result;
    struct Environment {std::fenv_t prior;Environment(){
      if(std::fegetenv(&prior)||std::fesetenv(FE_DFL_ENV))throw std::runtime_error("cannot install session floating point environment");
#if defined(__i386__) || defined(__x86_64__)
      const std::uint16_t control=0x027f;__asm__ __volatile__("fldcw %0"::"m"(control));
#endif
    }~Environment(){std::fesetenv(&prior);}} environment;
    prepare_prefix(world.get(),arena.data(),game,image,image_bytes,replay,&result);
    check(cudaMemcpy(parents,world.get(),sizeof(World),cudaMemcpyHostToDevice));
    check(cudaMemcpy(parent_arenas,arena.data(),game.arena_bytes,cudaMemcpyHostToDevice));
    check(cudaMemcpy(root_result,&result,sizeof(Result),cudaMemcpyHostToDevice));
    host_prefix_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  }
  Event start,prefix_end,end,validation_end;start.record();
  if(game.host_prefix)bind_prefix_kernel<<<1,1>>>(parents,parent_arenas);
  else prefix_kernel<<<1,1>>>(parents,parent_arenas,game,image_gpu,image_bytes,replay_gpu,root_result);
  check(cudaMemcpyAsync(parent_results,root_result,sizeof(Result),cudaMemcpyDeviceToDevice));
  prefix_end.record();
  struct RoundEvents {Event before,branched,simulated,selected;};
  std::vector<std::unique_ptr<RoundEvents>> events;
  for(unsigned depth=0;depth<options.depths;++depth){
    auto event=std::make_unique<RoundEvents>();event->before.record();
    const auto count=output->round_counts[depth],keep=output->round_kept[depth];
    auto* results=all_results+offsets[depth];auto* ranks=all_ranks+rank_offsets[depth];
    auto* lifecycle=all_lifecycle+offsets[depth];
    auto* selected=all_selected+selection_offsets[depth];
    session_parent_indices<<<(count+127)/128,128>>>(indices,count,fanout);
    clone_layout(children,child_arenas,parents,parent_arenas,game.arena_bytes,count,indices,1,game.arena_word_stride);
    event->branched.record();const auto threads=count*(game.isolated_warps?32U:1U);
    const auto block=game.isolated_warps?std::max(32U,game.block_threads):game.block_threads;
    session_future_kernel<<<(threads+block-1)/block,block>>>(children,game,replay_gpu,replay_size,specs_gpu,
      fanout,count,parent_results,root_result,results,lifecycle,std::uint64_t(game.prefix)+std::uint64_t(depth+1)*game.horizon,actual_ticks,peak_heap);
    event->simulated.record();rank_results(results,count,ranks);
    session_select_children<<<1,1>>>(results,ranks,lifecycle,count,keep,fanout,options.greedy_beam,options.protect_incumbent,
      protected_parent,selected,protected_history+depth,errors);
    clone_layout(parents,parent_arenas,children,child_arenas,game.arena_bytes,keep,selected,game.arena_word_stride,1);
    session_compact_results<<<(keep+31)/32,32>>>(parent_results,results,selected,keep);
    event->selected.record();events.push_back(std::move(event));
  }
  session_winner_info<<<1,1>>>(children,all_ranks+rank_offsets.back(),chain);end.record();
  if(game.validate_compaction){const auto keep=output->round_kept.back();const auto* selected=all_selected+selection_offsets.back();
    verify_compact_kernel<<<keep,256>>>(parents,parent_arenas,children,child_arenas,selected,keep,game.arena_bytes,errors,game.arena_word_stride,1);
    continue_compact_kernel<<<(keep+31)/32,32>>>(parents,children,selected,keep,errors,game.cache_guest);
    verify_compact_kernel<<<keep,256>>>(parents,parent_arenas,children,child_arenas,selected,keep,game.arena_bytes,errors,game.arena_word_stride,1);
  }
  check(cudaGetLastError());validation_end.record();check(cudaEventSynchronize(validation_end.value));
  output->results.resize(total);output->ranks.resize(total);
  output->lifecycle_flags.resize(total);
  output->selected_children.resize(total_selected);output->protected_ranks.resize(options.depths);
  check(cudaMemcpy(output->results.data(),all_results,total*sizeof(Result),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(output->lifecycle_flags.data(),all_lifecycle,total*4,cudaMemcpyDeviceToHost));
  for(unsigned d=0;d<options.depths;++d)
    check(cudaMemcpy(output->ranks.data()+offsets[d],all_ranks+rank_offsets[d],std::size_t(output->round_counts[d])*4,cudaMemcpyDeviceToHost));
  check(cudaMemcpy(output->selected_children.data(),all_selected,total_selected*4,cudaMemcpyDeviceToHost));
  check(cudaMemcpy(output->protected_ranks.data(),protected_history,std::size_t(options.depths)*4,cudaMemcpyDeviceToHost));
  unsigned compact_errors{},heap{};unsigned long long counts[2]{};
  check(cudaMemcpy(&compact_errors,errors,4,cudaMemcpyDeviceToHost));
  check(cudaMemcpy(&heap,peak_heap,4,cudaMemcpyDeviceToHost));
  check(cudaMemcpy(counts,actual_ticks,16,cudaMemcpyDeviceToHost));
  check(cudaMemcpy(&output->winner_canonical_chain,chain,8,cudaMemcpyDeviceToHost));
  output->actual_simulated_ticks=counts[0];output->actual_simulated_branches=counts[1];
  output->winner_candidates.resize(options.depths);
  auto child=output->ranks[offsets.back()];
  for(unsigned depth=options.depths;depth-->0;){
    output->winner_candidates[depth]=child%fanout;
    if(depth)child=output->selected_children[selection_offsets[depth-1]+child/fanout];}
  const auto& winner=output->results[offsets.back()+output->ranks[offsets.back()]];
  if(winner.final.tick>std::uint64_t(game.prefix)+std::uint64_t(game.horizon)*options.depths)
    throw std::runtime_error("selected session tick exceeds budget");
  output->winner_words.resize(winner.final.tick);
  for(std::size_t i=0;i<output->winner_words.size();++i)output->winner_words[i]=i<replay_size?replay[i]:0;
  for(unsigned d=0;d<options.depths;++d){const auto& c=candidates[output->winner_candidates[d]];
    for(unsigned i=0;i<c.count;++i){const auto tick=std::uint64_t(game.prefix)+std::uint64_t(d)*game.horizon+c.mutations[i].offset;
      if(tick<output->winner_words.size())output->winner_words[tick]=c.mutations[i].input;}}
  auto& t=output->timings;t.host_prefix_ms=host_prefix_ms;t.prefix_ms=elapsed(start,prefix_end);
  for(const auto& e:events){t.branch_ms+=elapsed(e->before,e->branched);t.rollout_ms+=elapsed(e->branched,e->simulated);t.rank_compact_ms+=elapsed(e->simulated,e->selected);}
  t.validation_ms=elapsed(end,validation_end);t.compaction_errors=compact_errors;t.peak_guest_bytes=heap;
  t.allocation_bytes=std::uint64_t(capacity+options.beam)*sizeof(World)+
    (std::uint64_t(arena_storage_lanes(capacity,game.arena_word_stride))+options.beam)*game.arena_bytes+image_bytes+
    std::uint64_t(replay_size)*4+std::uint64_t(fanout)*sizeof(Candidate)+std::uint64_t(options.beam)*sizeof(Result)+
    std::uint64_t(capacity)*4+total*(sizeof(Result)+4)+total_ranks*4+32+sizeof(Result)+
    total_selected*4+std::uint64_t(options.depths)*4+4;
  t.total_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
}
