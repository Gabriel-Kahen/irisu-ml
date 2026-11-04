// The preserved scalar/adaptive runner with only its input admission widened.
// Its physics, cloning, scheduling, ranking and validation remain unchanged.
#include "gpu_search_inputs.hpp"
namespace irisu::cuda_search {
inline Inputs gpu_wide_load(const char* replay,const char* specs,const char* prefix,const char* horizon){
 auto wide=::irisu::gpu_search::load(replay,specs,prefix,horizon);Inputs input;
 input.options.seed=wide.original.seed;input.options.prefix=wide.original.prefix;input.options.horizon=wide.original.horizon;
 input.replay=std::move(wide.replay);input.candidates=std::move(wide.candidates);return input;
}
}
#define load_inputs gpu_wide_load
#include "search_runner.cu"
#undef load_inputs
