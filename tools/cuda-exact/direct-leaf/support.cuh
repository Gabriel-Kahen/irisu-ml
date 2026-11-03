#pragma once
#include "../../../clone/cuda/guest.hpp"
#include "../x87_numeric.cuh"
namespace irisu::cuda_direct {
IRISU_GUEST_HD inline float word_float(std::uint32_t value){union{std::uint32_t u;float f;}bits{value};return bits.f;}
IRISU_GUEST_HD inline void fault(cuda_guest::Guest& g,cuda_guest::Error error,double value){if(!g.error){g.error=error;g.fault_pc=g.pc;g.fault_value=value;}}
}
