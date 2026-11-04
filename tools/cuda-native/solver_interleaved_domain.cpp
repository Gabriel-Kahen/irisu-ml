#include "../../clone/cuda_native/solver_interleaved.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace n=irisu::cuda_native;
namespace s=n::interleaved_solver;
void require(bool value){if(!value)throw std::runtime_error("interleaved workspace domain gate");}
int main(){try{
 constexpr unsigned lanes=3;alignas(8) unsigned char scenes[lanes][128]{};
 std::vector<std::uint32_t> words(s::words_required(lanes));
 s::Descriptor descriptor{words.data(),reinterpret_cast<std::uintptr_t>(scenes),128,lanes,32,64,nullptr};
 for(unsigned lane=0;lane<lanes;++lane){auto view=descriptor.view(scenes[lane],2,1);require(view.valid()&&view.lane==lane&&view.lanes==lanes);}
 require(!descriptor.view(scenes[0]+1,2,1).words);
 require(!descriptor.view(scenes+lanes,2,1).words);
 require(!descriptor.view(nullptr,2,1).words);
 require(!descriptor.view(scenes[0],33,1).valid());
 require(!descriptor.view(scenes[0],2,65).valid());
 n::BodyState bodies[2]{};n::ContactConstraint constraints[1]{};constraints[0].body1=0;constraints[0].body2=1;
 std::uint32_t order[2]{1,0};auto view=descriptor.view(scenes[0],2,1);
 require(bool(s::pack(view,bodies,2,order,constraints)));
 auto copy=bodies[0];const auto original=constraints[0];
 auto too_small=view;too_small.body_capacity=1;
 require(s::pack(too_small,bodies,2,order,constraints).status==n::Status::capacity);
 require(!std::memcmp(&copy,bodies,sizeof(copy))&&!std::memcmp(&original,constraints,sizeof(original)));
 order[0]=order[1]=0;require(s::pack(view,bodies,2,order,constraints).status==n::Status::invalid_snapshot);
 order[0]=0;order[1]=2;require(s::pack(view,bodies,2,order,constraints).status==n::Status::invalid_body);
 order[1]=1;constraints[0].body2=2;require(s::pack(view,bodies,2,order,constraints).status==n::Status::invalid_body);
 constraints[0]=original;constraints[0].point_count=3;require(s::pack(view,bodies,2,order,constraints).status==n::Status::invalid_snapshot);
 constraints[0]=original;bool used=true;
 require(bool(s::try_iterate<s::Phase::velocity>(scenes[0],bodies,2,order,2,constraints,1,8,used))&&!used);
 require(!std::memcmp(&copy,bodies,sizeof(copy))&&!std::memcmp(&original,constraints,sizeof(original)));
 std::puts("{\"schema\":\"irisu-interleaved-solver-domain-v1\",\"passed\":true,\"scene_pool_lanes\":3,\"checked\":\"Exact-stride pool admission, bounded capacity, duplicate/missing body rejection, invalid point count, host unchanged-path bridge and source immutability\"}");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
