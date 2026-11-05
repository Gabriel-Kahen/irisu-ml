// Experimental native resident beam: selected fork/COW children become the next
// depth's checkpoint keepers. No physics reconstruction or per-tick IPC.
#include "irisu/simulator.hpp"
#include "search_io.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <fcntl.h>
#include <numeric>
#include <poll.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace irisu::cuda_search;
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
irisu::Action action(std::uint32_t word,std::uint32_t tick){
  return {static_cast<irisu::ActionKind>(word&3),double((word>>2)&1023),double((word>>12)&511),1,tick<2};
}
irisu::game_digest::Frame frame(const irisu::Simulator& sim,const irisu::StepResult& step){
  return irisu::game_digest::make_frame(sim.bodies(),sim.observation(),
    irisu::game_digest::native_events(step.events),step.events.size(),step);
}
struct Slot {
  sem_t command,ack;
  std::atomic<unsigned> ready{};
  unsigned request{},quit{},error{},lifecycle{},ticks{},depth{},candidate{};
  pid_t pid{};
  Result result{};
};
static_assert(std::atomic<unsigned>::is_always_lock_free);
struct Shared {sem_t completed;Result prefix;double prefix_ms;std::atomic<unsigned> failure{};};
void post(sem_t& semaphore){if(sem_post(&semaphore))_exit(90);}
void failed(Shared* shared){shared->failure.store(1,std::memory_order_release);post(shared->completed);}
bool owner_gone(int descriptor){pollfd item{descriptor,POLLIN,0};return poll(&item,1,0)>0&&(item.revents&(POLLHUP|POLLERR));}
void child_wait(sem_t& semaphore,Shared* shared,int owner){
  for(;;){if(owner_gone(owner))_exit(0);
    int status;pid_t pid;
    while((pid=waitpid(-1,&status,WNOHANG))>0)if(!WIFEXITED(status)||WEXITSTATUS(status)){failed(shared);_exit(98);}
    timespec deadline{};clock_gettime(CLOCK_REALTIME,&deadline);++deadline.tv_sec;
    if(!sem_timedwait(&semaphore,&deadline))return;
    if(errno!=EINTR&&errno!=ETIMEDOUT){failed(shared);_exit(91);}
  }
}
struct Processes {
  pid_t group{};int previous_subreaper{},owner[2]{-1,-1};Shared* shared{};
  Processes(Shared* state):shared(state){
    if(prctl(PR_GET_CHILD_SUBREAPER,&previous_subreaper)||prctl(PR_SET_CHILD_SUBREAPER,1))throw std::runtime_error("cannot enable owned-child reaping");
    if(pipe2(owner,O_CLOEXEC))throw std::runtime_error("cannot create owner liveness pipe");
  }
  ~Processes(){
    if(group){kill(-group,SIGKILL);int status;while(waitpid(-group,&status,0)>0||errno==EINTR){}}
    for(auto fd:owner)if(fd>=0)close(fd);
    prctl(PR_SET_CHILD_SUBREAPER,previous_subreaper);
  }
  void reap(){
    int status;pid_t pid;
    while((pid=waitpid(-group,&status,WNOHANG))>0)
      if(!WIFEXITED(status)||WEXITSTATUS(status))throw std::runtime_error("native resident child failed");
    if(pid<0&&errno!=ECHILD&&errno!=EINTR)throw std::runtime_error("cannot reap native resident children");
  }
  void wait(sem_t& semaphore){
    for(;;){if(shared->failure.load(std::memory_order_acquire))throw std::runtime_error("native resident worker reported failure");
      timespec deadline{};clock_gettime(CLOCK_REALTIME,&deadline);++deadline.tv_sec;
      if(!sem_timedwait(&semaphore,&deadline)){
        if(shared->failure.load(std::memory_order_acquire))throw std::runtime_error("native resident worker reported failure");return;
      }
      if(errno!=EINTR&&errno!=ETIMEDOUT)throw std::runtime_error("native resident semaphore failed");
      reap();
    }
  }
};
void keeper(unsigned index,Slot* slots,Shared* shared,const Inputs& in,
            irisu::Simulator& sim,irisu::StepResult& step,Result result,int owner){
  for(;;){auto& self=slots[index];child_wait(self.command,shared,owner);if(self.quit)_exit(0);
    const auto next=self.request;const pid_t child=fork();
    if(child<0){self.error=1;post(self.ack);continue;}
    if(child){slots[next].pid=child;post(self.ack);continue;}
    // A child owns the inherited exact native heap. It will remain frozen at
    // the segment boundary unless selected to fork the following depth.
    try {
      auto& out=slots[next];unsigned ticks=0;
      const auto start_tick=static_cast<std::uint32_t>(result.final.tick);
      if(!step.terminated&&!step.truncated){
        const auto& spec=in.candidates[out.candidate];
        for(unsigned offset=0;offset<in.options.horizon;++offset){
          if(!(offset&255)&&owner_gone(owner))_exit(0);
          const auto tick=start_tick+offset;
          const auto base=tick<in.replay.size()?in.replay[tick]:0;
          step=sim.step(action(input_at(spec,offset,base),tick));
          if(in.options.trace_every_tick)fold(result,frame(sim,step));
          ++ticks;++result.simulated_ticks;
          result.natural=step.terminated&&!step.truncated;
          result.canonical_level=step.diagnostics.recorded_final_level;
          result.canonical_score=step.diagnostics.recorded_final_score;
          if(result.natural&&!step.diagnostics.terminal_metadata_recorded)throw std::runtime_error("terminal metadata missing");
          if(step.terminated||step.truncated)break;
        }
      }
      if(!in.options.trace_every_tick&&ticks){result.trace_hash=shared->prefix.trace_hash;fold(result,frame(sim,step));}
      result.clears=sim.observation().qualifying_clear_count;
      rank_keys(result,std::uint64_t(in.options.prefix)+std::uint64_t(out.depth+1)*in.options.horizon,in.options.target_score);
      out.result=result;out.ticks=ticks;out.lifecycle=(step.terminated?1U:0U)|(step.truncated?2U:0U);
      out.ready.store(1,std::memory_order_release);post(shared->completed);
      index=next;
    }catch(...){slots[next].error=1;slots[next].ready.store(1,std::memory_order_release);failed(shared);_exit(93);}
  }
}
void emit_word(unsigned value){if(std::fwrite(&value,4,1,stdout)!=1)throw std::runtime_error("cannot write native session header");}
}
int main(int argc,char** argv){
  try {
    if(argc<9)throw std::runtime_error("usage: search-session-oracle replay specs prefix horizon workers beam depths target [--final-only] [--no-protect-incumbent] [--greedy-beam]");
    auto in=load_inputs(argv[1],argv[2],argv[3],argv[4]);
    const auto workers=std::stoul(argv[5]),beam=std::stoul(argv[6]),depths=std::stoul(argv[7]);
    in.options.target_score=std::stoll(argv[8]);bool protect=true,greedy=false;
    for(int i=9;i<argc;++i){const std::string flag=argv[i];
      if(flag=="--final-only")in.options.trace_every_tick=false;
      else if(flag=="--no-protect-incumbent")protect=false;
      else if(flag=="--greedy-beam")greedy=true;
      else throw std::runtime_error("unknown native resident flag");}
    const unsigned fanout=in.candidates.size();
    if(!workers||workers>64||!beam||beam>64||!depths||depths>16||beam*fanout>1024||
       std::uint64_t(in.options.prefix)+std::uint64_t(in.options.horizon)*depths>UINT32_MAX||
       (protect&&in.candidates[0].count))throw std::runtime_error("invalid bounded native resident dimensions");
    std::vector<unsigned> counts,offsets;unsigned nodes=0,parents=1;
    for(unsigned d=0;d<depths;++d){offsets.push_back(nodes+1);const auto count=parents*fanout;counts.push_back(count);nodes+=count;parents=std::min<unsigned>(beam,count);}
    auto* shared=static_cast<Shared*>(mmap(nullptr,sizeof(Shared),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0));
    auto* slots=static_cast<Slot*>(mmap(nullptr,std::size_t(nodes+1)*sizeof(Slot),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0));
    if(shared==MAP_FAILED||slots==MAP_FAILED)throw std::runtime_error("native resident mmap failed");
    ::new(shared)Shared{};if(sem_init(&shared->completed,1,0))throw std::runtime_error("shared semaphore initialization failed");
    for(unsigned i=0;i<=nodes;++i){::new(slots+i)Slot{};
      if(sem_init(&slots[i].command,1,0)||sem_init(&slots[i].ack,1,0))throw std::runtime_error("keeper semaphore initialization failed");}
    const std::uint16_t cw=0x027f;__asm__ __volatile__("fldcw %0"::"m"(cw));
    Processes processes(shared);const auto begin=Clock::now();const pid_t root=fork();
    if(root<0)throw std::runtime_error("cannot fork native prefix keeper");
    if(!root){
      close(processes.owner[1]);
      if(setpgid(0,0))_exit(94);
      try {irisu::Simulator sim;sim.reset(in.options.seed);irisu::StepResult step{};
        for(unsigned tick=0;tick<in.options.prefix;++tick){
          if(!(tick&255)&&owner_gone(processes.owner[0]))_exit(0);
          if(step.terminated||step.truncated)_exit(95);step=sim.step(action(in.replay[tick],tick));
        }
        if(step.terminated||step.truncated)_exit(96);
        Result result;fold(result,frame(sim,step));shared->prefix=result;shared->prefix_ms=ms(begin);
        post(shared->completed);keeper(0,slots,shared,in,sim,step,result,processes.owner[0]);_exit(0);
      }catch(...){failed(shared);_exit(97);}
    }
    processes.group=root;if(setpgid(root,root)&&errno!=EACCES&&errno!=ESRCH)throw std::runtime_error("cannot own native process group");
    processes.wait(shared->completed);std::vector<unsigned> parent_slots{0};unsigned protected_parent=0;
    std::vector<std::vector<unsigned>> all_ranks,all_selected;std::vector<unsigned> protected_history;
    std::uint64_t ticks=0,active=0;double branch_ms=0,rank_ms=0;
    for(unsigned d=0;d<depths;++d){const auto count=counts[d],base=offsets[d],keep=std::min<unsigned>(beam,count);const auto start=Clock::now();
      unsigned launched=0,finished=0;std::vector<unsigned> running;
      while(finished<count){
        while(launched<count&&running.size()<workers){const auto ordinal=launched++,child=base+ordinal,parent=parent_slots[ordinal/fanout];
          slots[child].depth=d;slots[child].candidate=ordinal%fanout;slots[parent].request=child;post(slots[parent].command);
          processes.wait(slots[parent].ack);if(slots[parent].error)throw std::runtime_error("native resident fork failed");running.push_back(child);}
        processes.wait(shared->completed);const auto it=std::find_if(running.begin(),running.end(),[&](auto i){return slots[i].ready.load(std::memory_order_acquire)!=0;});
        if(it==running.end())throw std::runtime_error("native resident completion mismatch");
        ticks+=slots[*it].ticks;active+=slots[*it].ticks!=0;running.erase(it);++finished;
      }
      branch_ms+=ms(start);const auto ranking=Clock::now();std::vector<unsigned> ranks(count);std::iota(ranks.begin(),ranks.end(),0);
      std::sort(ranks.begin(),ranks.end(),[&](auto a,auto b){return better(slots[base+a].result,a,slots[base+b].result,b);});
      std::vector<unsigned> selected{ranks[0]};
      if(!greedy)for(unsigned i=1;i<count&&selected.size()<keep;++i){const auto child=ranks[i];if(!slots[base+child].lifecycle&&!slots[base+child].result.failed)selected.push_back(child);}
      for(unsigned i=1;i<count&&selected.size()<keep;++i)if(std::find(selected.begin(),selected.end(),ranks[i])==selected.end())selected.push_back(ranks[i]);
      if(protect){const auto child=protected_parent*fanout;auto found=std::find(selected.begin(),selected.end(),child);
        if(found==selected.end()){selected.back()=child;protected_parent=keep-1;}else protected_parent=found-selected.begin();
        protected_history.push_back(protected_parent);
      }else protected_history.push_back(UINT32_MAX);
      // Selected children are already frozen at the exact native boundary.
      // Exit every previous parent and discarded child; selected children are
      // adopted by our subreaper and become the following depth's keepers.
      for(auto i:parent_slots){slots[i].quit=1;post(slots[i].command);}
      parent_slots.clear();for(auto i:selected)parent_slots.push_back(base+i);
      for(unsigned i=0;i<count;++i)if(std::find(selected.begin(),selected.end(),i)==selected.end()){slots[base+i].quit=1;post(slots[base+i].command);}
      processes.reap();rank_ms+=ms(ranking);all_ranks.push_back(std::move(ranks));all_selected.push_back(std::move(selected));
    }
    const auto total=ms(begin);emit_word(4);emit_word(depths);
    for(unsigned d=0;d<depths;++d){emit_word(counts[d]);emit_word(all_selected[d].size());std::vector<Result> results;std::vector<unsigned> lifecycle;
      for(unsigned i=0;i<counts[d];++i){results.push_back(slots[offsets[d]+i].result);lifecycle.push_back(slots[offsets[d]+i].lifecycle);}
      emit_results(results,all_ranks[d]);
      if(std::fwrite(all_selected[d].data(),4,all_selected[d].size(),stdout)!=all_selected[d].size())throw std::runtime_error("cannot write native selection");
      emit_word(protected_history[d]);if(std::fwrite(lifecycle.data(),4,lifecycle.size(),stdout)!=lifecycle.size())throw std::runtime_error("cannot write native lifecycle");}
    const auto winner=all_ranks.back()[0];
    std::fprintf(stderr,"{\"identity\":\"experimental-native-exact-resident-search-v1\",\"workers\":%lu,\"depths\":%lu,\"beam\":%lu,\"fanout\":%u,\"candidate_nodes\":%u,\"actual_simulated_branches\":%llu,\"actual_simulated_ticks\":%llu,\"prefix_ms\":%.9f,\"branch_rollout_ms\":%.9f,\"rank_selection_ms\":%.9f,\"total_ms\":%.9f,\"winner\":%u}\n",workers,depths,beam,fanout,nodes,(unsigned long long)active,(unsigned long long)ticks,shared->prefix_ms,branch_ms,rank_ms,total,winner);
    if(std::fflush(stdout))throw std::runtime_error("cannot flush native session");return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"native-session-oracle: %s\n",e.what());return 1;}
}
