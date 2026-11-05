// Independent authoritative native exact baseline. Snapshot branching uses
// fork/COW in this single-threaded ELF32 process, preserving the native heap.
#include "irisu/simulator.hpp"
#include "search_io.hpp"
#include <algorithm>
#include <chrono>
#include <csignal>
#include <numeric>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace irisu::cuda_search;
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point start) {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
irisu::Action action(std::uint32_t input,std::uint32_t tick) {
  return {static_cast<irisu::ActionKind>(input&3),double((input>>2)&1023),double((input>>12)&511),1,tick<2};
}
irisu::game_digest::Frame frame(const irisu::Simulator& sim,const irisu::StepResult& step) {
  return irisu::game_digest::make_frame(sim.bodies(),sim.observation(),
    irisu::game_digest::native_events(step.events),step.events.size(),step);
}
struct Children {
  std::vector<pid_t> running;
  ~Children(){
    for(auto pid:running)kill(pid,SIGKILL);
    for(auto pid:running){while(waitpid(pid,nullptr,0)<0&&errno==EINTR){}}
  }
  void reap(){
    int status{};pid_t pid;
    do{pid=waitpid(-1,&status,0);}while(pid<0&&errno==EINTR);
    const auto it=std::find(running.begin(),running.end(),pid);
    if(pid<0||it==running.end())throw std::runtime_error("cannot reap native candidate child");
    running.erase(it);
    if(!WIFEXITED(status)||WEXITSTATUS(status))throw std::runtime_error("native candidate child failed");
  }
};
}
int main(int argc,char** argv) {
  try {
    if(argc<7) throw std::runtime_error("usage: search-oracle replay.rpy specs.bin prefix horizon workers target-score [--final-only] [--emit-lifecycle path]");
    auto in=load_inputs(argv[1],argv[2],argv[3],argv[4]);
    const auto workers=std::stoul(argv[5]);in.options.target_score=std::stoll(argv[6]);
    const char* lifecycle_path=nullptr;
    for(int i=7;i<argc;++i){const std::string flag=argv[i];
      if(flag=="--final-only")in.options.trace_every_tick=false;
      else if(flag=="--emit-lifecycle"&&i+1<argc)lifecycle_path=argv[++i];
      else throw std::runtime_error("unknown native search flag");}
    if(!workers||workers>64) throw std::runtime_error("workers must be 1..64");
    const auto count=in.candidates.size();
    auto* output=static_cast<Result*>(mmap(nullptr,count*sizeof(Result),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0));
    if(output==MAP_FAILED) throw std::runtime_error("mmap failed");
    std::uint32_t* lifecycle=nullptr;
    if(lifecycle_path){lifecycle=static_cast<std::uint32_t*>(mmap(nullptr,count*4,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0));
      if(lifecycle==MAP_FAILED)throw std::runtime_error("lifecycle mmap failed");}
    const std::uint16_t cw=0x027f;__asm__ __volatile__("fldcw %0"::"m"(cw));
    const auto begin=Clock::now();
    irisu::Simulator sim;sim.reset(in.options.seed);irisu::StepResult step{};
    for(std::uint32_t tick=0;tick<in.options.prefix;++tick) {
      if(step.terminated||step.truncated) throw std::runtime_error("prefix extends past terminal");
      step=sim.step(action(in.replay[tick],tick));
    }
    if(step.terminated||step.truncated) throw std::runtime_error("cannot branch terminal prefix");
    const auto prefix_ms=ms(begin);const auto prefix_frame=frame(sim,step);
    const auto branch_begin=Clock::now();
    Children children;
    for(std::size_t i=0;i<count;++i) {
      if(children.running.size()>=workers)children.reap();
      const auto child=fork();if(child<0) throw std::runtime_error("fork failed");
      if(!child) {
        try {
          Result result;fold(result,prefix_frame);
          for(std::uint32_t offset=0;offset<in.options.horizon;++offset) {
            const auto tick=in.options.prefix+offset;
            const auto base=tick<in.replay.size()?in.replay[tick]:0;
            step=sim.step(action(input_at(in.candidates[i],offset,base),tick));
            if(in.options.trace_every_tick)fold(result,frame(sim,step));
            ++result.simulated_ticks;
            if(step.terminated||step.truncated) break;
          }
          if(!in.options.trace_every_tick)fold(result,frame(sim,step));
          const auto observation=sim.observation();
          result.clears=observation.qualifying_clear_count;
          result.natural=step.terminated&&!step.truncated;
          result.canonical_level=step.diagnostics.recorded_final_level;
          result.canonical_score=step.diagnostics.recorded_final_score;
          if(result.natural&&!step.diagnostics.terminal_metadata_recorded)throw std::runtime_error("native terminal lacks canonical metadata");
          rank_keys(result,std::uint64_t(in.options.prefix)+in.options.horizon,in.options.target_score);
          if(lifecycle)lifecycle[i]=(step.terminated?1U:0U)|(step.truncated?2U:0U);
          output[i]=result;_exit(0);
        } catch(...) {_exit(1);}
      }
      children.running.push_back(child);
    }
    while(!children.running.empty())children.reap();
    if(lifecycle){std::ofstream stream(lifecycle_path,std::ios::binary);
      stream.write(reinterpret_cast<const char*>(lifecycle),count*4);stream.flush();
      if(!stream)throw std::runtime_error("cannot write lifecycle sidecar");
      munmap(lifecycle,count*4);}
    std::vector<Result> results(output,output+count);munmap(output,count*sizeof(Result));
    std::vector<std::uint32_t> ranks(count);std::iota(ranks.begin(),ranks.end(),0);
    std::sort(ranks.begin(),ranks.end(),[&](auto a,auto b){return better(results[a],a,results[b],b);});
    std::uint64_t ticks=0;for(const auto& r:results) ticks+=r.simulated_ticks;
    std::fprintf(stderr,"{\"identity\":\"experimental-native-exact-search-oracle-v2\",\"candidates\":%zu,\"workers\":%lu,\"prefix_ticks\":%u,\"horizon\":%u,\"simulated_ticks\":%llu,\"trace_every_tick\":%s,\"prefix_ms\":%.9f,\"branch_rollout_rank_ms\":%.9f,\"total_ms\":%.9f,\"winner\":%u}\n",count,workers,in.options.prefix,in.options.horizon,(unsigned long long)ticks,in.options.trace_every_tick?"true":"false",prefix_ms,ms(branch_begin),ms(begin),ranks[0]);
    emit_results(results,ranks);return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
