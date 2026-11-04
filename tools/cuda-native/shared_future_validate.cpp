// Host differential for segment carry, terminal boundaries and final-only hash.
#define main irisu_search_runner_main
#include "search_runner.cu"
#undef main
#include <cassert>
int main(){
 Environment environment;unsigned cases=0;
 for(bool trace:{false,true})for(unsigned stop:{0U,101U,102U,103U,0xffffffffU,0xfffffffeU,0xfffffffdU})for(unsigned split:{1U,2U,4U}){
  Options options;options.seed=1298144938;options.prefix=100;options.horizon=4;options.trace=trace;
  std::vector<std::uint32_t> replay(104,0);auto root=cold_object<World>();auto root_scene=cold_object<Scene>();wire::Result prefix;
  prepare_prefix(root.get(),root_scene.get(),options,replay.data(),&prefix);if(stop==0xffffffffU)root->simulator.terminated_=true;else if(stop==0xfffffffeU)root->simulator.failed_=true;else if(stop==0xfffffffdU)root_scene->world.status=physics::Status::capacity;else if(stop)root->simulator.config_.max_episode_ticks=stop;
  auto direct=cold_object<World>();auto direct_scene=cold_object<Scene>();auto cached=cold_object<World>();auto cached_scene=cold_object<Scene>();auto leaf=cold_object<World>();auto leaf_scene=cold_object<Scene>();
  host_clone(*direct,*direct_scene,*root,*root_scene,sizeof(Scene));host_clone(*cached,*cached_scene,*root,*root_scene,sizeof(Scene));
  wire::Candidate candidate{};candidate.count=1;candidate.mutations[0]={3,(350U<<12)|(250U<<2)|1U};wire::Result expected;
  future(*direct,*direct_scene,options,replay.data(),replay.size(),candidate,prefix,expected);
  SharedState state{};state.result=prefix;shared_segment(*cached,*cached_scene,options,replay.data(),replay.size(),candidate,0,split,state,false);
  host_clone(*leaf,*leaf_scene,*cached,*cached_scene,sizeof(Scene));shared_segment(*leaf,*leaf_scene,options,replay.data(),replay.size(),candidate,split,options.horizon,state,true);
  assert(!std::memcmp(&expected,&state.result,sizeof(expected)));assert(host_equal(*direct,*direct_scene,*leaf,*leaf_scene,sizeof(Scene)));++cases;
 }
 std::printf("shared future segment differential: %u cases passed\n",cases);
}
