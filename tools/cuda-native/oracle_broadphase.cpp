// Exact pinned native SAP operation/callback/state oracle. The zeroed backing
// is an explicit initial condition: original PairManager leaves bufferCount
// uninitialized, unlike the later portable-source safety patch.
#include "oracle_native.h"
#include "broadphase_wire.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
namespace wire=irisu::cuda_native::wire;
struct Callback {void** vtable;std::vector<wire::BroadEvent>* events;std::uint32_t next_token;};
void* ORACLE_THISCALL deleting_destructor(void* self,unsigned){return self;}
void* ORACLE_THISCALL added(void* self,void* first,void* second){
  auto& c=*static_cast<Callback*>(self);const auto token=first&&second?c.next_token++:0U;
  c.events->push_back({1,static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(first)),static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(second)),token});
  return reinterpret_cast<void*>(static_cast<std::uintptr_t>(token));
}
void ORACLE_THISCALL removed(void* self,void* first,void* second,void* token){
  auto& c=*static_cast<Callback*>(self);c.events->push_back({2,static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(first)),static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(second)),static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(token))});
}
void emit(const void* data,std::size_t bytes){if(std::fwrite(data,1,bytes,stdout)!=bytes)oracle_die("cannot write broad-phase snapshot");}
void valid_aabb(const float* aabb){for(unsigned i=0;i<4;++i)if(!std::isfinite(aabb[i]))oracle_die("nonfinite broad-phase bound");if(!(aabb[0]<aabb[2]&&aabb[1]<aabb[3]))oracle_die("broad-phase AABB must have positive extent");}
int main(int argc,char** argv){
  if(argc<3||argc>4||(argc==4&&std::strcmp(argv[3],"--capacity-512")))oracle_die("usage: oracle-broadphase native.so operations.bin [--capacity-512]");
  const unsigned maximum_live=argc==4?512:64;auto native=oracle_open(argv[1]);
  FILE* file=std::fopen(argv[2],"rb");wire::BroadHeader header{};
  if(!file||std::fread(&header,sizeof(header),1,file)!=1||header.magic!=wire::broadphase_magic||header.version!=1||!header.count||header.count>4096||header.op_words!=8)oracle_die("invalid broad-phase header");valid_aabb(header.world);
  auto ctor=oracle_broad_constructor(&native);auto create=oracle_proxy_create(&native);auto move=oracle_proxy_move(&native);auto destroy=oracle_proxy_destroy(&native);auto commit=oracle_broad_commit(&native);auto query=oracle_broad_query(&native);
  std::vector<wire::BroadEvent> events;events.reserve(4096);void* table[]={reinterpret_cast<void*>(deleting_destructor),reinterpret_cast<void*>(added),reinterpret_cast<void*>(removed)};Callback callback{table,&events,1};
  std::vector<std::uint32_t> state(wire::native_broadphase_words,0),snapshot(state.size());std::array<std::uint16_t,512> slots;slots.fill(0xffff);
  const std::uint16_t cw=0x027f;__asm__ __volatile__("fninit; fldcw %0"::"m"(cw));ctor(state.data(),header.world,&callback);
  unsigned active=0,adds=0,removes=0;
  for(unsigned index=0;index<header.count;++index){wire::BroadOperation op{};
    if(std::fread(&op,sizeof(op),1,file)!=1||op.reserved||op.slot>=512)oracle_die("invalid/short broad-phase operation");
    events.clear();std::uint32_t result=0;std::array<void*,512> matches{};unsigned count=0;
    if(op.kind>=1&&op.kind<=3){
      const auto pairs=state[0xc00c/4],buffered=state[0x10010/4];
      if(pairs>4096||buffered>4096||pairs+active>4096||buffered+2*active>4096)
        oracle_die("conservative pair/buffer capacity guard refused native operation");
    }
    switch(op.kind){
      case 1:if(slots[op.slot]!=0xffff||active>=maximum_live)oracle_die("create exceeds configured live-proxy scope");valid_aabb(op.aabb);slots[op.slot]=create(state.data(),op.aabb,reinterpret_cast<void*>(static_cast<std::uintptr_t>(op.user_or_limit)));result=slots[op.slot];if(result==0xffff)oracle_die("native proxy allocation failed");++active;break;
      case 2:if(slots[op.slot]==0xffff)oracle_die("move of inactive logical slot");valid_aabb(op.aabb);move(state.data(),slots[op.slot],op.aabb);break;
      case 3:if(slots[op.slot]==0xffff)oracle_die("destroy of inactive logical slot");destroy(state.data(),slots[op.slot]);slots[op.slot]=0xffff;--active;break;
      case 4:commit(state.data());break;
      case 5:if(op.user_or_limit>512)oracle_die("invalid query limit");valid_aabb(op.aabb);result=query(state.data(),op.aabb,matches.data(),op.user_or_limit);if(result>op.user_or_limit)oracle_die("invalid native query count");count=result;break;
      case 6:if(op.user_or_limit!=65535)oracle_die("only valid timestamp wrap fixture supported");std::memcpy(reinterpret_cast<unsigned char*>(state.data())+0x17438,&op.user_or_limit,2);result=65535;break;
      default:oracle_die("unknown broad-phase operation");
    }
    const wire::BroadResultHeader out{index,result,count,static_cast<std::uint32_t>(events.size()),wire::native_broadphase_words};emit(&out,sizeof(out));
    for(unsigned i=0;i<count;++i){const auto tag=static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(matches[i]));emit(&tag,4);}
    for(const auto& event:events){emit(&event,sizeof(event));adds+=event.kind==1;removes+=event.kind==2;}
    snapshot=state;snapshot[0]=snapshot[1]=0;emit(snapshot.data(),wire::native_broadphase_bytes);
  }
  if(std::fgetc(file)!=EOF||std::ferror(file))oracle_die("trailing broad-phase operation data");std::fclose(file);oracle_broad_destructor(&native)(state.data());oracle_close(&native);
  std::fprintf(stderr,"{\"schema\":\"irisu-native-broadphase-oracle-v1\",\"operations\":%u,\"add_events\":%u,\"remove_events\":%u,\"active_proxies\":%u,\"state_bytes\":%u,\"calloc_initial_condition\":true,\"scope_max_live_proxies\":%u,\"conservative_pair_capacity_guard\":true}\n",header.count,adds,removes,active,wire::native_broadphase_bytes,maximum_live);return std::fflush(stdout)?1:0;
}
