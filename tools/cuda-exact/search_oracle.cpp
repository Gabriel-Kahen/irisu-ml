// Independent authoritative native exact baseline. Snapshot branching uses
// fork/COW in this single-threaded ELF32 process, preserving the native heap.
#include "irisu/simulator.hpp"
#include "search_io.hpp"
#include <algorithm>
#include <chrono>
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
}
int main(int argc,char** argv) {
  try {
    if(argc!=7) throw std::runtime_error("usage: search-oracle replay.rpy specs.bin prefix horizon workers target-score");
    auto in=load_inputs(argv[1],argv[2],argv[3],argv[4]);
    const auto workers=std::stoul(argv[5]);in.options.target_score=std::stoll(argv[6]);
    if(!workers||workers>64) throw std::runtime_error("workers must be 1..64");
    const auto count=in.candidates.size();
    auto* output=static_cast<Result*>(mmap(nullptr,count*sizeof(Result),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0));
    if(output==MAP_FAILED) throw std::runtime_error("mmap failed");
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
    std::vector<pid_t> running;
    auto reap=[&]() {
      int status{};const auto pid=waitpid(running.front(),&status,0);
      running.erase(running.begin());
      if(pid<0||!WIFEXITED(status)||WEXITSTATUS(status)) throw std::runtime_error("native candidate child failed");
    };
    for(std::size_t i=0;i<count;++i) {
      if(running.size()>=workers) reap();
      const auto child=fork();if(child<0) throw std::runtime_error("fork failed");
      if(!child) {
        try {
          Result result;fold(result,prefix_frame);
          for(std::uint32_t offset=0;offset<in.options.horizon;++offset) {
            const auto tick=in.options.prefix+offset;
            const auto base=tick<in.replay.size()?in.replay[tick]:0;
            step=sim.step(action(input_at(in.candidates[i],offset,base),tick));
            fold(result,frame(sim,step));++result.simulated_ticks;
            if(step.terminated||step.truncated) break;
          }
          const auto observation=sim.observation();
          result.clears=observation.qualifying_clear_count;
          result.natural=step.terminated&&!step.truncated;
          result.canonical_level=step.diagnostics.latest_final_level;
          rank_keys(result,std::uint64_t(in.options.prefix)+in.options.horizon,in.options.target_score);
          output[i]=result;_exit(0);
        } catch(...) {_exit(1);}
      }
      running.push_back(child);
    }
    while(!running.empty()) reap();
    std::vector<Result> results(output,output+count);munmap(output,count*sizeof(Result));
    std::vector<std::uint32_t> ranks(count);std::iota(ranks.begin(),ranks.end(),0);
    std::sort(ranks.begin(),ranks.end(),[&](auto a,auto b){return better(results[a],a,results[b],b);});
    std::uint64_t ticks=0;for(const auto& r:results) ticks+=r.simulated_ticks;
    std::fprintf(stderr,"{\"identity\":\"experimental-native-exact-search-oracle-v1\",\"candidates\":%zu,\"workers\":%lu,\"prefix_ticks\":%u,\"horizon\":%u,\"simulated_ticks\":%llu,\"prefix_ms\":%.9f,\"branch_rollout_rank_ms\":%.9f,\"total_ms\":%.9f,\"winner\":%u}\n",count,workers,in.options.prefix,in.options.horizon,(unsigned long long)ticks,prefix_ms,ms(branch_begin),ms(begin),ranks[0]);
    emit_results(results,ranks);return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
