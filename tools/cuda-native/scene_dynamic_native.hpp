#pragma once
#include "scene_native.hpp"
#include "scene_dynamic_wire.hpp"
namespace scene_oracle {
inline void import_broadphase(BroadPhase& output,void* handle,const Mapping& map){
  void* world=*static_cast<void**>(handle);
  const auto* native=pointer<unsigned char>(at<std::uint32_t>(world,102868));
  std::memcpy(&output,native,offsetof(BroadPhase,error));output.error=Status::ok;
  output.m_pairManager.m_broadPhase=output.m_pairManager.m_callback=0;
  for(auto& proxy:output.m_proxyPool)if(proxy.IsValid())proxy.userData=Mapping::index(map.shapes,proxy.userData)+1;
  const auto null_contact=reinterpret_cast<std::uintptr_t>(world)+102876;
  for(auto& pair:output.m_pairManager.m_pairs)if(pair.proxyId1!=65535)
    pair.userData=pair.userData==null_contact?0:Mapping::index(map.contacts,pair.userData)+1;
}
} // namespace scene_oracle
