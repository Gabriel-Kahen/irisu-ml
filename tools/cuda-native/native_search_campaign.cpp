// Actual native ELF32 control: one retained prefix, repeated fork/COW futures.
// Never links the independent typed physics implementation.
#include "irisu/simulator.hpp"
#include "../cuda-exact/search_io.hpp"
#include <algorithm>
#include <chrono>
#include <csignal>
#include <new>
#include <numeric>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace irisu::cuda_search;
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
irisu::Action action(std::uint32_t word,std::uint32_t tick){
  return {static_cast<irisu::ActionKind>(word&3),double((word>>2)&1023),double((word>>12)&511),1,tick<2};
}
irisu::game_digest::Frame frame(const irisu::Simulator& sim,const irisu::StepResult& step){
  return irisu::game_digest::make_frame(sim.bodies(),sim.observation(),
    irisu::game_digest::native_events(step.events),step.events.size(),step);
}
template<class T,bool inherited> struct Mapping {
  T* data;std::size_t count;
  explicit Mapping(std::size_t size):count(size){
    data=static_cast<T*>(mmap(nullptr,count*sizeof(T),PROT_READ|PROT_WRITE,
      (inherited?MAP_SHARED:MAP_PRIVATE)|MAP_ANONYMOUS,-1,0));
    if(data==MAP_FAILED)throw std::runtime_error("shared result mmap failed");
    if constexpr(!inherited)if(madvise(data,count*sizeof(T),MADV_DONTFORK)){
      const auto error=errno;munmap(data,count*sizeof(T));
      throw std::runtime_error(std::string("MADV_DONTFORK failed: ")+std::strerror(error));
    }
    for(std::size_t i=0;i<count;++i)::new(static_cast<void*>(data+i)) T{};
  }
  ~Mapping(){munmap(data,count*sizeof(T));}
  Mapping(const Mapping&)=delete;Mapping& operator=(const Mapping&)=delete;
};
template<class T>using Shared=Mapping<T,true>;
template<class T>using ParentOnly=Mapping<T,false>;
struct ChildMetric {double rollout_ms{};std::uint32_t ready{};};
struct Children {
  std::vector<pid_t> running;
  ~Children(){
    for(const auto pid:running)kill(pid,SIGKILL);
    for(const auto pid:running)while(waitpid(pid,nullptr,0)<0&&errno==EINTR){}
  }
  double reap(){
    const auto start=Clock::now();int status{};pid_t pid;
    do{pid=waitpid(-1,&status,0);}while(pid<0&&errno==EINTR);
    const auto it=std::find(running.begin(),running.end(),pid);
    if(pid<0||it==running.end())throw std::runtime_error("cannot reap native candidate child");
    running.erase(it);
    if(!WIFEXITED(status)||WEXITSTATUS(status))throw std::runtime_error("native candidate child failed");
    return elapsed(start);
  }
};
struct Timing {
  double initialize_ms{},fork_syscall_ms{},wait_ms{},fork_and_rollout_ms{},child_rollout_sum_ms{},child_rollout_max_ms{};
  double rank_ms{},verification_ms{},batch_ms{};
  std::uint64_t ticks{};
};
template<class T> void hash_bytes(irisu::game_digest::Hash& hash,const T* values,std::size_t count){
  const auto* bytes=reinterpret_cast<const unsigned char*>(values);
  for(std::size_t i=0;i<count*sizeof(T);++i)hash.byte(bytes[i]);
}
std::uint64_t packet_hash(const Result* results,const std::uint32_t* ranks,std::size_t count){
  irisu::game_digest::Hash hash;const std::uint32_t header[]={result_magic,result_version,std::uint32_t(count),sizeof(Result)};
  hash_bytes(hash,header,4);hash_bytes(hash,results,count);hash_bytes(hash,ranks,count);return hash.value;
}
std::uint64_t input_hash(const Inputs& in){
  irisu::game_digest::Hash hash;hash_bytes(hash,in.replay.data(),in.replay.size());
  hash_bytes(hash,in.candidates.data(),in.candidates.size());return hash.value;
}
unsigned bounded_count(const char* value,unsigned maximum,const char* label){
  std::size_t end{};const auto count=std::stoull(value,&end);
  if(value[0]=='-'||value[end]||!count||count>maximum)throw std::runtime_error(label);
  return unsigned(count);
}
}
int main(int argc,char** argv){
  const auto process_start=Clock::now();
  try{
    if(argc<8)throw std::runtime_error("usage: native-search-campaign replay.rpy specs.bin prefix horizon workers batches target-score [--final-only]");
    auto loaded=load_inputs(argv[1],argv[2],argv[3],argv[4]);
    const unsigned workers=bounded_count(argv[5],64,"workers must be 1..64");
    const unsigned batches=bounded_count(argv[6],10000,"batches must be 1..10000");
    std::size_t end{};loaded.options.target_score=std::stoll(argv[7],&end);
    if(argv[7][end])throw std::runtime_error("invalid target score");
    for(int i=8;i<argc;++i){if(std::string(argv[i])=="--final-only")loaded.options.trace_every_tick=false;
      else throw std::runtime_error("unknown native campaign flag");}
    const auto in=std::move(loaded);const auto count=in.candidates.size();
    const auto loaded_input_hash=input_hash(in);
    Shared<Result> output(count);Shared<ChildMetric> metrics(count);
    // Dedicated, page-owned mappings only. Children never consume ranking,
    // verification or reporting storage, so do not copy those VMAs/PTEs at fork.
    ParentOnly<Result> baseline(count);ParentOnly<std::uint32_t> ranks(count),baseline_ranks(count);
    ParentOnly<Timing> timings(batches);
    const auto parent_only_bytes=count*(sizeof(Result)+2*sizeof(std::uint32_t))+batches*sizeof(Timing);
    const std::uint16_t cw=0x027f;__asm__ __volatile__("fldcw %0"::"m"(cw));
    const auto prefix_start=Clock::now();
    irisu::Simulator sim;sim.reset(in.options.seed);irisu::StepResult prefix_step{};
    for(std::uint32_t tick=0;tick<in.options.prefix;++tick){
      if(prefix_step.terminated||prefix_step.truncated)throw std::runtime_error("prefix extends past terminal");
      prefix_step=sim.step(action(in.replay[tick],tick));
    }
    if(prefix_step.terminated||prefix_step.truncated)throw std::runtime_error("cannot branch terminal prefix");
    const auto prefix_frame=frame(sim,prefix_step);const double prefix_ms=elapsed(prefix_start);
    const pid_t owner=getpid();
    std::uint64_t all_ticks=0,baseline_packet_hash=0;
    for(unsigned batch=0;batch<batches;++batch){
      Timing timing;const auto batch_start=Clock::now();
      for(std::size_t i=0;i<count;++i){output.data[i]=Result{};metrics.data[i]=ChildMetric{};}
      timing.initialize_ms=elapsed(batch_start);const auto future_start=Clock::now();
      {
        Children children;children.running.reserve(std::min<std::size_t>(workers,count));
        for(std::size_t i=0;i<count;++i){
          if(children.running.size()>=workers)timing.wait_ms+=children.reap();
          const auto fork_start=Clock::now();const auto child=fork();
          if(child<0)throw std::runtime_error("fork failed");
          if(!child){
            // All candidates are direct children. This covers owner death even
            // between fork and installing the death signal; no keeper adoption.
            if(prctl(PR_SET_PDEATHSIG,SIGKILL)<0||getppid()!=owner)_exit(92);
            try{
              const auto start=Clock::now();auto step=prefix_step;Result result;fold(result,prefix_frame);
              for(std::uint32_t offset=0;offset<in.options.horizon;++offset){
                const auto tick=in.options.prefix+offset;
                const auto base=tick<in.replay.size()?in.replay[tick]:0;
                step=sim.step(action(input_at(in.candidates[i],offset,base),tick));
                if(in.options.trace_every_tick)fold(result,frame(sim,step));
                ++result.simulated_ticks;if(step.terminated||step.truncated)break;
              }
              if(!in.options.trace_every_tick)fold(result,frame(sim,step));
              const auto observation=sim.observation();result.clears=observation.qualifying_clear_count;
              result.natural=step.terminated&&!step.truncated;
              result.canonical_level=step.diagnostics.recorded_final_level;
              result.canonical_score=step.diagnostics.recorded_final_score;
              if(result.natural&&!step.diagnostics.terminal_metadata_recorded)
                throw std::runtime_error("native terminal lacks canonical metadata");
              rank_keys(result,std::uint64_t(in.options.prefix)+in.options.horizon,in.options.target_score);
              const auto rollout_ms=elapsed(start);output.data[i]=result;
              metrics.data[i]={rollout_ms,batch+1};_exit(0);
            }catch(const std::exception& e){std::fprintf(stderr,"candidate %zu: %s\n",i,e.what());_exit(1);}
            catch(...){_exit(1);}
          }
          timing.fork_syscall_ms+=elapsed(fork_start);children.running.push_back(child);
        }
        while(!children.running.empty())timing.wait_ms+=children.reap();
      }
      timing.fork_and_rollout_ms=elapsed(future_start);
      const auto rank_start=Clock::now();std::iota(ranks.data,ranks.data+count,0);
      std::sort(ranks.data,ranks.data+count,[&](auto a,auto b){return better(output.data[a],a,output.data[b],b);});
      timing.rank_ms=elapsed(rank_start);const auto verify_start=Clock::now();
      for(std::size_t i=0;i<count;++i){
        if(metrics.data[i].ready!=batch+1||output.data[i].failed)throw std::runtime_error("incomplete or failed native candidate result");
        timing.ticks+=output.data[i].simulated_ticks;timing.child_rollout_sum_ms+=metrics.data[i].rollout_ms;
        timing.child_rollout_max_ms=std::max(timing.child_rollout_max_ms,metrics.data[i].rollout_ms);
      }
      if(!batch){std::memcpy(baseline.data,output.data,count*sizeof(Result));
        std::memcpy(baseline_ranks.data,ranks.data,count*sizeof(std::uint32_t));
        baseline_packet_hash=packet_hash(output.data,ranks.data,count);}
      else if(std::memcmp(output.data,baseline.data,count*sizeof(Result))||
              std::memcmp(ranks.data,baseline_ranks.data,count*sizeof(std::uint32_t)))
        throw std::runtime_error("repeated native batch packet/ranks changed");
      const auto checkpoint_frame=frame(sim,prefix_step);
      if(std::memcmp(&checkpoint_frame,&prefix_frame,sizeof(prefix_frame))||input_hash(in)!=loaded_input_hash)
        throw std::runtime_error("parent checkpoint or loaded candidate inputs changed");
      timing.verification_ms=elapsed(verify_start);timing.batch_ms=elapsed(batch_start);
      all_ticks+=timing.ticks;timings.data[batch]=timing;
    }
    // The shared wire helper takes vectors; materialize them only AFTER the last
    // child is reaped, so no future fork inherits these private heap buffers.
    const auto output_start=Clock::now();
    const std::vector<Result> final_results(output.data,output.data+count);
    const std::vector<std::uint32_t> final_ranks(ranks.data,ranks.data+count);
    emit_results(final_results,final_ranks);const double output_ms=elapsed(output_start);
    const double process_ms=elapsed(process_start);
    std::fprintf(stderr,"{\"identity\":\"experimental-native-exact-cached-prefix-campaign-v2\",\"candidates\":%zu,\"workers\":%u,\"batches\":%u,\"prefix_ticks\":%u,\"prefix_preparations\":1,\"horizon\":%u,\"simulated_future_ticks\":%llu,\"trace_every_tick\":%s,\"prefix_ms\":%.9f,\"cold_process_wall_ms\":%.9f,\"cold_wall_scope\":\"main entry through last packet flush; excludes dynamic-loader startup and final stderr reporting\",\"output_ms\":%.9f,\"parent_only_buffers\":\"dedicated MAP_PRIVATE MADV_DONTFORK mappings\",\"parent_only_bytes_excluded_from_fork\":%zu,\"all_batch_packets_and_ranks_identical\":true,\"parent_checkpoint_frame_unchanged\":true,\"loaded_inputs_unchanged\":true,\"packet_fnv1a64\":\"%016llx\",\"loaded_inputs_fnv1a64\":\"%016llx\",\"packet_bytes\":%zu,\"winner\":%u,\"timing_scope\":\"fork_syscall/wait and child rollout times overlap fork_and_rollout wall; do not add them\",\"batch_timings\":[",
      count,workers,batches,in.options.prefix,in.options.horizon,(unsigned long long)all_ticks,
      in.options.trace_every_tick?"true":"false",prefix_ms,process_ms,output_ms,parent_only_bytes,
      (unsigned long long)baseline_packet_hash,(unsigned long long)loaded_input_hash,16+count*(sizeof(Result)+4),ranks.data[0]);
    for(unsigned i=0;i<batches;++i){const auto& t=timings.data[i];
      std::fprintf(stderr,"%s{\"batch\":%u,\"initialize_ms\":%.9f,\"fork_syscall_ms\":%.9f,\"wait_ms\":%.9f,\"fork_and_rollout_ms\":%.9f,\"child_rollout_sum_ms\":%.9f,\"child_rollout_max_ms\":%.9f,\"rank_ms\":%.9f,\"verification_ms\":%.9f,\"batch_wall_ms\":%.9f,\"simulated_ticks\":%llu}",
        i?",":"",i,t.initialize_ms,t.fork_syscall_ms,t.wait_ms,t.fork_and_rollout_ms,t.child_rollout_sum_ms,
        t.child_rollout_max_ms,t.rank_ms,t.verification_ms,t.batch_ms,(unsigned long long)t.ticks);}
    std::fprintf(stderr,"]}\n");return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"native-search-campaign: %s\n",e.what());return 1;}
}
