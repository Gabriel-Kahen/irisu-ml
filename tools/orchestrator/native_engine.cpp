// Native authoritative backend: immutable checkpoints are Linux fork/COW keepers.
#include "irisu/simulator.hpp"
#include "common.hpp"
#include <cerrno>
#include <csignal>
#include <cstring>
#include <map>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#ifndef IRISU_ORCHESTRATOR_BACKEND
#define IRISU_ORCHESTRATOR_BACKEND "portable-gnu-r58"
#endif
namespace {
using Clock=std::chrono::steady_clock;
struct Tracker {
 std::map<unsigned,orch::Group> groups;std::map<unsigned,unsigned> members;
 void add(const irisu::StepResult& step){for(const auto& e:step.events){
  if(e.kind==irisu::EventKind::ChainJoined){auto id=unsigned(e.value);auto& g=groups[id];g.id=id;++g.chain;members[e.a]=id;}
  if(e.kind==irisu::EventKind::Confirmed){const auto found=members.find(e.a);if(found!=members.end())groups[found->second].num=unsigned(e.value);}
 }}
};
irisu::Action action(unsigned word,unsigned long long tick){return {static_cast<irisu::ActionKind>(word&3),double((word>>2)&1023),double((word>>12)&511),1,tick<2};}
orch::Summary summarize(const irisu::Simulator& sim,const Tracker& tracker,unsigned seed,const irisu::StepResult& step){
 orch::Summary out{};const auto o=sim.observation();out.seed=seed;out.tick=o.tick;out.score=o.score;out.gauge=o.gauge;out.level=o.level;out.qualifying=o.qualifying_clear_count;out.highest_chain=o.highest_chain;out.terminated=o.terminated;out.truncated=o.truncated;out.config_hash=sim.config_hash();out.left_held=o.left_held;out.right_held=o.right_held;
 out.canonical_score=step.diagnostics.recorded_final_score;out.canonical_level=step.diagnostics.recorded_final_level;out.canonical_chain=step.diagnostics.recorded_final_highest_chain;
 for(const auto& b:sim.bodies()){if(b.lifecycle==irisu::Lifecycle::Deleted)continue;if(out.body_count==orch::max_bodies)throw std::runtime_error("summary body capacity");auto& v=out.bodies[out.body_count++];v={b.id,unsigned(b.kind),b.chain_id,b.projectile_hits,unsigned(b.rot_timer),b.color,b.position.x,b.position.y,b.velocity.x,b.velocity.y,b.size,b.remaining_lifetime,b.successful_clear_pending,b.delete_marked||b.pending_delete,b.special,b.physics_owned};}
 for(const auto& [id,g]:tracker.groups){bool live=false;for(unsigned i=0;i<out.body_count;++i)live|=out.bodies[i].chain_id==id;if(live){if(out.group_count==orch::max_groups)throw std::runtime_error("summary group capacity");out.groups[out.group_count++]=g;}}
 return out;
}
void exact_read(int fd,void* target,std::size_t n){auto* p=static_cast<char*>(target);while(n){const auto r=read(fd,p,n);if(r<0&&errno==EINTR)continue;if(r<=0)throw std::runtime_error("checkpoint pipe closed");p+=r;n-=std::size_t(r);}}
void exact_write(int fd,const void* source,std::size_t n){const auto* p=static_cast<const char*>(source);while(n){const auto r=write(fd,p,n);if(r<0&&errno==EINTR)continue;if(r<=0)throw std::runtime_error("checkpoint pipe write failed");p+=r;n-=std::size_t(r);}}
void send_text(int fd,const std::string& text){const auto n=std::uint32_t(text.size());exact_write(fd,&n,4);exact_write(fd,text.data(),n);}
std::string read_text(int fd){std::uint32_t n;exact_read(fd,&n,4);if(n>128U*1024U*1024U)throw std::runtime_error("oversized keeper message");std::string out(n,' ');exact_read(fd,out.data(),n);return out;}
struct Node {int fd;pid_t pid;};
void send_node(int fd,Node node){char control[CMSG_SPACE(sizeof(int))]{};iovec io{&node.pid,sizeof(node.pid)};msghdr message{};message.msg_iov=&io;message.msg_iovlen=1;message.msg_control=control;message.msg_controllen=sizeof(control);auto* c=CMSG_FIRSTHDR(&message);c->cmsg_level=SOL_SOCKET;c->cmsg_type=SCM_RIGHTS;c->cmsg_len=CMSG_LEN(sizeof(int));std::memcpy(CMSG_DATA(c),&node.fd,sizeof(int));if(sendmsg(fd,&message,0)!=sizeof(node.pid))throw std::runtime_error("cannot send checkpoint descriptor");}
Node read_node(int fd){Node out{-1,0};char control[CMSG_SPACE(sizeof(int))]{};iovec io{&out.pid,sizeof(out.pid)};msghdr message{};message.msg_iov=&io;message.msg_iovlen=1;message.msg_control=control;message.msg_controllen=sizeof(control);if(recvmsg(fd,&message,MSG_WAITALL)!=sizeof(out.pid))throw std::runtime_error("cannot receive checkpoint descriptor");auto* c=CMSG_FIRSTHDR(&message);if(!c||c->cmsg_type!=SCM_RIGHTS||message.msg_flags&MSG_CTRUNC)throw std::runtime_error("missing checkpoint descriptor");std::memcpy(&out.fd,CMSG_DATA(c),sizeof(int));return out;}
void close_other_fds(int retained){
#ifdef SYS_close_range
 const auto first=retained>3?syscall(SYS_close_range,3U,unsigned(retained-1),0U):0L;
 const auto second=syscall(SYS_close_range,unsigned(retained+1),~0U,0U);
 if(first==0&&second==0)return;
#endif
 const long maximum=sysconf(_SC_OPEN_MAX);for(int fd=3;fd<std::min<long>(maximum,65536);++fd)if(fd!=retained)close(fd);
}
std::string output(const irisu::Simulator& sim,const Tracker& tracker,unsigned seed,const irisu::StepResult& step,const std::vector<unsigned>& words,const orch::Summary* before=nullptr,const orch::Metrics* metrics=nullptr){std::ostringstream out;auto* previous=std::cout.rdbuf(out.rdbuf());try{std::cout<<"{\"summary\":";auto final=summarize(sim,tracker,seed,step);orch::emit_summary(final);if(before){final.group_delta_count=before->group_count;for(unsigned i=0;i<before->group_count;++i){const auto& g=before->groups[i];auto& d=final.group_deltas[i];d.id=g.id;d.chain_before=g.chain;d.num_before=g.num;const auto found=tracker.groups.find(g.id);if(found==tracker.groups.end())throw std::runtime_error("lost group history");d.chain_after=found->second.chain;d.num_after=found->second.num;for(unsigned j=0;j<final.body_count;++j)d.members_after+=final.bodies[j].chain_id==g.id;}}std::cout<<",\"group_deltas\":";orch::emit_group_deltas(final);std::cout<<",\"metrics\":";orch::Metrics aggregate;if(!metrics){aggregate.new_groups=tracker.groups.size();aggregate.qualifications=final.qualifying;for(const auto& [id,g]:tracker.groups){(void)id;aggregate.recruitment_events+=g.chain;aggregate.max_new_chain=std::max<std::uint64_t>(aggregate.max_new_chain,g.chain);}metrics=&aggregate;}orch::emit_metrics(*metrics);std::cout<<",\"words\":";orch::emit_words(words);std::cout<<'}';std::cout.rdbuf(previous);}catch(...){std::cout.rdbuf(previous);throw;}return out.str();}
[[noreturn]] void keeper(int fd,irisu::Simulator& sim,Tracker& tracker,unsigned seed,irisu::StepResult step){
 std::signal(SIGCHLD,SIG_IGN);
 for(;;){try{
  const auto text=read_text(fd);std::istringstream in(text);const auto horizon=orch::number(in,orch::max_horizon,"horizon");std::string rest;std::getline(in,rest);const auto c=orch::candidate(rest,horizon);
  int sockets[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,sockets))throw std::runtime_error("socketpair failed");const pid_t pid=fork();if(pid<0)throw std::runtime_error("checkpoint fork failed");
  if(!pid){close(sockets[0]);close_other_fds(sockets[1]);std::vector<unsigned> trace;trace.reserve(horizon);try{const auto before=summarize(sim,tracker,seed,step);const auto last_group=tracker.groups.empty()?0U:tracker.groups.rbegin()->first;std::uint64_t recruited_before=0;for(const auto& [id,g]:tracker.groups){(void)id;recruited_before+=g.chain;}
    for(unsigned i=0;i<horizon&&!step.terminated&&!step.truncated;++i){const auto summary=summarize(sim,tracker,seed,step);const auto w=i<c.words.size()?c.words[i]:orch::policy(summary,c.mode,c.search_seed);step=sim.step(action(w,summary.tick));tracker.add(step);trace.push_back(w);}
    orch::Metrics metrics;metrics.qualifications=sim.observation().qualifying_clear_count-before.qualifying;for(const auto& [id,g]:tracker.groups){metrics.recruitment_events+=g.chain;if(id>last_group){++metrics.new_groups;metrics.max_new_chain=std::max<std::uint64_t>(metrics.max_new_chain,g.chain);}}metrics.recruitment_events-=recruited_before;
    send_text(sockets[1],output(sim,tracker,seed,step,trace,&before,&metrics));keeper(sockets[1],sim,tracker,seed,step);
   }catch(const std::exception& e){try{send_text(sockets[1],"!"+std::string(e.what()));}catch(...){} _exit(1);}}
  close(sockets[1]);try{send_node(fd,{sockets[0],pid});}catch(...){kill(pid,SIGKILL);close(sockets[0]);throw;}close(sockets[0]);
 }catch(...){close(fd);_exit(0);}}
}
struct LoadRequest {unsigned seed{},count{};};
// The pinned legacy allocator leaves one broad-phase constructor counter
// indeterminate. Keep its fresh-root allocation environment pristine: this
// factory forks before it sees any variable-size input, and children construct
// and reset their Simulator before allocating the replay buffer. Do not patch
// the pinned original physics library or fork roots from the dirty RPC heap.
Node start_factory(){
 int sockets[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,sockets))throw std::runtime_error("factory socketpair failed");const auto pid=fork();if(pid<0)throw std::runtime_error("factory fork failed");
 if(!pid){close(sockets[0]);close_other_fds(sockets[1]);std::signal(SIGCHLD,SIG_IGN);
  for(;;){try{
   LoadRequest request;exact_read(sockets[1],&request,sizeof(request));
   int child_sockets[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,child_sockets))throw std::runtime_error("load socketpair failed");const auto child=fork();if(child<0)throw std::runtime_error("load fork failed");
   if(!child){close(child_sockets[0]);close_other_fds(child_sockets[1]);try{
    irisu::Simulator sim;sim.reset(request.seed);
    std::vector<unsigned> values(request.count);if(request.count)exact_read(child_sockets[1],values.data(),std::size_t(request.count)*sizeof(unsigned));
    Tracker tracker;irisu::StepResult step;std::vector<unsigned> trace;trace.reserve(request.count);
    for(const auto word:values){if(step.terminated||step.truncated)break;step=sim.step(action(word,trace.size()));tracker.add(step);trace.push_back(word);}
    send_text(child_sockets[1],output(sim,tracker,request.seed,step,trace));keeper(child_sockets[1],sim,tracker,request.seed,step);
   }catch(const std::exception& e){try{send_text(child_sockets[1],"!"+std::string(e.what()));}catch(...){} _exit(1);}}
   close(child_sockets[1]);try{send_node(sockets[1],{child_sockets[0],child});}catch(...){kill(child,SIGKILL);close(child_sockets[0]);throw;}close(child_sockets[0]);
  }catch(...){close(sockets[1]);_exit(0);}}
 }
 close(sockets[1]);return {sockets[0],pid};
}
Node load(const Node& factory,unsigned seed,const std::vector<unsigned>& words){
 const LoadRequest request{seed,unsigned(words.size())};exact_write(factory.fd,&request,sizeof(request));auto node=read_node(factory.fd);
 try{if(!words.empty())exact_write(node.fd,words.data(),words.size()*sizeof(unsigned));}catch(...){close(node.fd);kill(node.pid,SIGTERM);throw;}return node;
}
void emit_result(unsigned handle,const std::string& json){if(json.empty()||json[0]=='!')throw std::runtime_error(json.empty()?"empty keeper result":json.substr(1));std::cout<<"{\"handle\":"<<handle<<','<<json.substr(1);}
}
int main(int argc,char** argv){
 unsigned limit=256;for(int i=1;i<argc;++i){if(std::string(argv[i])=="--max-checkpoints"&&i+1<argc)limit=std::stoul(argv[++i]);else{std::cerr<<"usage: native-engine [--max-checkpoints N]\n";return 1;}}
 if(!limit||limit>4096)return 1;std::signal(SIGPIPE,SIG_IGN);std::signal(SIGCHLD,SIG_IGN);prctl(PR_SET_CHILD_SUBREAPER,1);
#if defined(__i386__) || defined(__x86_64__)
 const std::uint16_t control=0x027f;asm volatile("fldcw %0"::"m"(control));
#endif
 const auto factory=start_factory();std::map<unsigned,Node> nodes;unsigned next=1;std::string line;
 while(std::getline(std::cin,line)){try{std::istringstream in(line);std::string command;in>>command;
  if(command=="identity"){orch::end(in);std::cout<<"{\"ok\":true,\"identity\":{\"protocol\":1,\"backend\":\""<<IRISU_ORCHESTRATOR_BACKEND<<"\",\"checkpoint_mode\":\"resident-fork-cow-pristine-factory\",\"max_checkpoints\":"<<limit<<",\"max_horizon\":"<<orch::max_horizon<<"}}\n";}
  else if(command=="load"){const auto seed=orch::number(in,UINT32_MAX,"seed"),count=orch::number(in,10000000,"word count");const auto words=orch::words(in,count);if(nodes.size()>=limit)throw std::runtime_error("checkpoint capacity exhausted; release handles");auto node=load(factory,seed,words);const auto handle=next++;nodes.emplace(handle,node);const auto result=read_text(node.fd);if(result.empty()||result[0]=='!'){close(node.fd);nodes.erase(handle);throw std::runtime_error(result.substr(1));}std::cout<<"{\"ok\":true,\"result\":";emit_result(handle,result);std::cout<<"}\n";}
  else if(command=="batch"){const auto parent=orch::number(in,UINT32_MAX,"parent"),horizon=orch::number(in,orch::max_horizon,"horizon"),count=orch::number(in,1024,"candidate count");orch::end(in);if(!horizon||!count)throw std::runtime_error("empty batch");std::vector<std::string> requests;for(unsigned i=0;i<count;++i){std::string request;if(!std::getline(std::cin,request))throw std::runtime_error("short batch");requests.push_back(request);}for(auto& request:requests){orch::candidate(request,horizon);request=std::to_string(horizon)+" "+request;}const auto found=nodes.find(parent);if(found==nodes.end())throw std::runtime_error("unknown checkpoint");if(nodes.size()+count>limit)throw std::runtime_error("checkpoint capacity exhausted; release handles");const auto start=Clock::now();std::vector<unsigned> handles;
   try{for(const auto& request:requests){send_text(found->second.fd,request);const auto child=read_node(found->second.fd);const auto handle=next++;nodes.emplace(handle,child);handles.push_back(handle);}std::vector<std::string> results;for(auto h:handles){auto result=read_text(nodes.at(h).fd);if(result.empty()||result[0]=='!')throw std::runtime_error(result.empty()?"empty keeper result":result.substr(1));results.push_back(std::move(result));}std::cout<<"{\"ok\":true,\"results\":[";for(unsigned i=0;i<handles.size();++i){if(i)std::cout<<',';emit_result(handles[i],results[i]);}std::cout<<"],\"requested_ticks\":"<<std::uint64_t(horizon)*count<<",\"wall_seconds\":"<<std::chrono::duration<double>(Clock::now()-start).count()<<"}\n";}catch(...){for(auto h:handles){close(nodes.at(h).fd);kill(nodes.at(h).pid,SIGTERM);nodes.erase(h);}throw;}}
  else if(command=="release"){const auto count=orch::number(in,4096,"handle count");for(unsigned i=0;i<count;++i){const auto h=orch::number(in,UINT32_MAX,"handle");const auto found=nodes.find(h);if(found!=nodes.end()){close(found->second.fd);nodes.erase(found);}}orch::end(in);std::cout<<"{\"ok\":true}\n";}
  else if(command=="quit"){orch::end(in);break;}else throw std::runtime_error("unknown command");
  std::cout<<std::flush;
 }catch(const std::exception& e){orch::error(e);}}
 for(const auto& [h,node]:nodes){(void)h;close(node.fd);kill(node.pid,SIGTERM);}close(factory.fd);kill(factory.pid,SIGTERM);return 0;
}
