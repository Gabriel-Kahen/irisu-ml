#pragma once

#include "guest.hpp"
#include "../../tools/cuda-exact/x87_numeric.cuh"

namespace irisu::cuda_guest {
#include "guest_exports.inc"
#include "guest_callbacks.inc"
#include "guest_blocks.inc"

IRISU_GUEST_HD inline std::uint32_t invoke(Guest& g,std::uint32_t function,
                                    const std::uint32_t* args,unsigned n) {
  if(g.error)return 0;
  g.r[4]=g.capacity-16;
  for(unsigned i=n;i>0;--i)g.push(args[i-1]);
  g.push(0);g.pc=function;execute(g);
  return g.r[0];
}
template<class... T> IRISU_GUEST_HD inline std::uint32_t call(Guest& g,std::uint32_t f,T... args) {
  const std::uint32_t values[]={std::uint32_t(args)...};
  return invoke(g,f,values,sizeof...(T));
}

} // namespace irisu::cuda_guest
