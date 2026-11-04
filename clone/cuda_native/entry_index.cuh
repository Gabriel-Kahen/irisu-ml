#pragma once
#include "../cuda/game_support.cuh"

namespace irisu::cuda_game {
// Optional lookup cache only: entry storage/order remains owned by the adapter.
// Plain arrays avoid expanding a large device aggregate constructor; reset()
// establishes the stamps before any lookup. All bytes remain checkpoint state.
struct EntryIndex {
  static constexpr std::uint32_t capacity=1024,missing=0xffffffffU;
  std::uint32_t keys[capacity],indices[capacity],stamps[capacity],generation{};
  GAME_HD void reset(){for(auto& stamp:stamps)stamp=0;generation=1;}
  GAME_HD void rebuild(){if(++generation==0)reset();}
  GAME_HD static std::uint32_t bucket(std::uint32_t key){return (key*2654435761U)&(capacity-1);}
  GAME_HD bool insert(std::uint32_t key,std::uint32_t index){
    if(!generation)reset();const auto first=bucket(key);
    for(unsigned probe=0;probe<capacity;++probe){const auto slot=(first+probe)&(capacity-1);
      if(stamps[slot]!=generation){keys[slot]=key;indices[slot]=index;stamps[slot]=generation;return true;}
      if(keys[slot]==key)return true; // First entry wins, matching linear find.
    }
    return false;
  }
  GAME_HD std::uint32_t find(std::uint32_t key)const{
    if(!generation)return missing;const auto first=bucket(key);
    for(unsigned probe=0;probe<capacity;++probe){const auto slot=(first+probe)&(capacity-1);
      if(stamps[slot]!=generation)return missing;if(keys[slot]==key)return indices[slot];
    }
    return missing;
  }
};
} // namespace irisu::cuda_game
