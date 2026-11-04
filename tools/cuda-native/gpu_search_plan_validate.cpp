#include "gpu_search_inputs.hpp"
#include <iostream>
#include <random>
#include <set>

void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
 using namespace irisu;std::mt19937 random(921483);unsigned cases=0;std::uint64_t memberships=0;
 for(unsigned trial=0;trial<1000;++trial){
  gpu_search::Inputs input;input.original={12345,1,unsigned(1+random()%12)};input.replay.assign(64,0);
  for(auto& word:input.replay)word=random()%5;
  const unsigned count=1+random()%64,roots=1+random()%std::min(4U,count);input.candidates.resize(count);
  for(auto& candidate:input.candidates){std::set<unsigned> used;for(unsigned j=0,n=random()%5;j<n;++j){const unsigned offset=random()%input.original.horizon;if(used.insert(offset).second)candidate.mutations[candidate.count++]={offset,unsigned(random()%10)};}}
  for(bool sharing:{false,true}){const auto plan=gpu_search::plan(input,roots,7,2,sharing);
   require(plan.roots.size()==roots&&plan.jobs.size()==count,"plan sizes");require(plan.planned_ticks<=plan.logical_ticks,"sharing increases planned work");
   std::uint64_t ticks=0;for(const auto& cp:plan.checkpoints)ticks+=cp.depth;
   for(unsigned i=0;i<count;++i){const auto job=plan.jobs[i];require(job.root==i%roots&&job.candidate==i,"logical mapping");require(job.checkpoint<plan.checkpoints.size(),"checkpoint bounds");const auto cp=plan.checkpoints[job.checkpoint];
    require(cp.root==job.root&&cp.depth==job.begin,"checkpoint root/depth");require(plan.roots[job.root].seed==12345+7*job.root&&plan.roots[job.root].prefix==1+2*job.root,"root parameters");
    for(unsigned t=0;t<job.begin;++t){const auto word=input.replay[plan.roots[job.root].prefix+t];require(cuda_search::input_at(input.candidates[i],t,word)==cuda_search::input_at(input.candidates[cp.candidate],t,word),"shared effective input mismatch");++memberships;}
    ticks+=input.original.horizon-job.begin;
   }
   require(ticks==plan.planned_ticks&&plan.logical_ticks==std::uint64_t(count)*input.original.horizon,"tick accounting");++cases;
   const auto schedule=gpu_search::schedule(plan);auto sorted=schedule;std::sort(sorted.begin(),sorted.end());for(unsigned i=0;i<count;++i)require(sorted[i]==i,"dispatch schedule is not a bijection");
   for(unsigned i=1;i<count;++i){const auto a=plan.jobs[schedule[i-1]],b=plan.jobs[schedule[i]];require(plan.roots[a.root].horizon-a.begin>=plan.roots[b.root].horizon-b.begin,"remaining-depth order");}
  }
 }
 gpu_search::Inputs input;input.original={UINT32_MAX,1,4};input.replay.assign(8,0);input.candidates.resize(2);bool rejected=false;
 try{gpu_search::plan(input,2,1,0,true);}catch(const std::runtime_error&){rejected=true;}require(rejected,"seed overflow accepted");
 input.original.seed=1;rejected=false;try{gpu_search::plan(input,2,0,8,true);}catch(const std::runtime_error&){rejected=true;}require(rejected,"prefix overflow accepted");
 std::cout<<"{\"passed\":true,\"plan_cases\":"<<cases<<",\"shared_input_memberships\":"<<memberships<<",\"overflow_rejections\":2}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
