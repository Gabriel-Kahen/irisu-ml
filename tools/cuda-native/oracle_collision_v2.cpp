// Geometry-wire v2 native collision oracle. No typed collision implementation
// is linked: all results come from the pinned hidden MSVC functions.
#include "oracle_native.h"
#include "collision_wire.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
namespace wire=irisu::cuda_native::wire;
using Collision=void (*)(void*,const void*,const void*,int);
using NativeShape=std::array<std::uint32_t,75>;
NativeShape shape(const wire::ShapePacket& input){
  if(input.kind>1||input.vertex_count>8||(input.kind==1&&input.vertex_count<3))oracle_die("invalid shape packet kind/count");
  for(float v:input.position)if(!std::isfinite(v))oracle_die("nonfinite position");
  for(float v:input.R)if(!std::isfinite(v))oracle_die("nonfinite rotation matrix");
  if(!std::isfinite(input.radius)||input.radius<0)oracle_die("invalid radius");
  for(unsigned i=0;i<2*input.vertex_count;++i)if(!std::isfinite(input.vertices[i])||!std::isfinite(input.normals[i]))oracle_die("nonfinite polygon geometry");
  NativeShape result;std::memset(result.data(),0xcd,sizeof(result));
  std::memcpy(result.data()+2,input.R,16);std::memcpy(result.data()+6,input.position,8);result[8]=input.kind?2:0;
  if(input.kind){std::memcpy(result.data()+26,input.vertices,64);result[58]=input.vertex_count;std::memcpy(result.data()+59,input.normals,64);}
  else std::memcpy(result.data()+18,&input.radius,4);
  return result;
}
int main(int argc,char** argv){
  if(argc!=3)oracle_die("usage: oracle-collision-v2 native.so cases.bin");NativeOracle native=oracle_open(argv[1]);
  Collision circle=reinterpret_cast<Collision>(oracle_symbol(&native,"?b2CollideCircle@@YAXPAUb2Manifold@@PAVb2CircleShape@@1_N@Z",0x45f0));
  Collision mixed=reinterpret_cast<Collision>(oracle_symbol(&native,"?b2CollidePolyAndCircle@@YAXPAUb2Manifold@@PBVb2PolyShape@@PBVb2CircleShape@@_N@Z",0x4730));
  Collision polygon=reinterpret_cast<Collision>(oracle_symbol(&native,"?b2CollidePoly@@YAXPAUb2Manifold@@PBVb2PolyShape@@1_N@Z",0x56e0));
  FILE* input=std::fopen(argv[2],"rb");wire::Header header{};
  if(!input||std::fread(&header,sizeof(header),1,input)!=1||header.magic!=wire::collision_magic_v2||header.version!=2||!header.count||header.count>1000000||header.case_words!=98)oracle_die("invalid v2 collision header");
  const std::uint16_t cw=0x027f;__asm__ __volatile__("fninit; fldcw %0"::"m"(cw));
  unsigned kinds[3]{},contacts=0,shape_errors=0;
  for(unsigned i=0;i<header.count;++i){wire::CollisionCaseV2 packet{};
    if(std::fread(&packet,sizeof(packet),1,input)!=1||packet.conservative>1)oracle_die("invalid/short v2 collision case");
    auto a=shape(packet.first),b=shape(packet.second);const auto saved_a=a,saved_b=b;
    Collision fn=nullptr;unsigned kind=0;
    if(!packet.first.kind&&!packet.second.kind)fn=circle;
    else if(packet.first.kind&&!packet.second.kind){fn=mixed;kind=1;}
    else if(packet.first.kind&&packet.second.kind){fn=polygon;kind=2;}
    else oracle_die("circle/polygon order unsupported; put polygon first");
    fn(packet.manifold,a.data(),b.data(),packet.conservative);++kinds[kind];contacts+=packet.manifold[14]!=0;
    if(packet.manifold[14]>2)oracle_die("native invalid manifold count");shape_errors+=a!=saved_a||b!=saved_b;
    if(std::fwrite(packet.manifold,60,1,stdout)!=1)oracle_die("cannot write native v2 manifold");
  }
  if(std::fgetc(input)!=EOF||std::ferror(input))oracle_die("trailing v2 packet data");std::fclose(input);oracle_close(&native);
  std::fprintf(stderr,"{\"schema\":\"irisu-native-collision-v2\",\"cases\":%u,\"circle_cases\":%u,\"poly_circle_cases\":%u,\"poly_cases\":%u,\"contacts\":%u,\"shape_mutation_errors\":%u,\"native_sha256\":\"%s\"}\n",header.count,kinds[0],kinds[1],kinds[2],contacts,shape_errors,ORACLE_PIN);
  return shape_errors||std::fflush(stdout)?1:0;
}
