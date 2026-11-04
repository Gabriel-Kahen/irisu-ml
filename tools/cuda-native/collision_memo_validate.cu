#include "../../clone/cuda_native/collision_memo.cuh"
#include "collision_wire.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace p=irisu::cuda_native;namespace memo=p::collision_memo;
struct Case {p::ShapeState first{},second{};p::Manifold manifold{};bool conservative{};};
struct Outcome {p::Manifold reference{},cached{};p::StageResult first{},second{};};
p::ShapeState shape(const p::wire::ShapePacket& packet){
 p::ShapeState out{};out.kind=p::ShapeKind(packet.kind);out.vertex_count=packet.vertex_count;out.position={packet.position[0],packet.position[1]};out.R={{packet.R[0],packet.R[1]},{packet.R[2],packet.R[3]}};out.radius=packet.radius;
 for(unsigned i=0;i<8;++i){out.vertices[i]={packet.vertices[2*i],packet.vertices[2*i+1]};out.normals[i]={packet.normals[2*i],packet.normals[2*i+1]};}return out;
}
template<class T>void read(std::ifstream& input,T& value){if(!input.read(reinterpret_cast<char*>(&value),sizeof(value)))throw std::runtime_error("short collision packet");}
std::vector<Case> load(const char* path,unsigned limit){
 std::ifstream input(path,std::ios::binary);p::wire::Header header;read(input,header);if(!header.count||header.count>(1U<<22))throw std::runtime_error("invalid collision count");
 std::vector<Case> result;
 for(unsigned i=0;i<header.count;++i){Case c;
  if(header.magic==p::wire::collision_magic_v2&&header.version==2&&header.case_words==98){p::wire::CollisionCaseV2 value;read(input,value);c.first=shape(value.first);c.second=shape(value.second);std::memcpy(&c.manifold,value.manifold,sizeof(c.manifold));c.conservative=value.conservative!=0;}
  else if(header.magic==0x31434e49&&header.version==1&&header.case_words==22){unsigned words[22];read(input,words);auto f=[&](unsigned i){float x;std::memcpy(&x,words+i,4);return x;};c.first.position={f(0),f(1)};c.first.radius=f(2);c.second.position={f(3),f(4)};c.second.radius=f(5);c.conservative=words[6]!=0;std::memcpy(&c.manifold,words+7,sizeof(c.manifold));}
  else throw std::runtime_error("unsupported collision packet");
  if(i<limit)result.push_back(c);
 }
 if(input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing collision packet");return result;
}
void flip(float& value){unsigned bits;std::memcpy(&bits,&value,4);bits^=1;std::memcpy(&value,&bits,4);}
std::vector<Case> variants(const std::vector<Case>& native){
 auto cases=native;const unsigned base=unsigned(native.size()<16?native.size():16);
 for(unsigned i=0;i<base;++i){const auto& original=native[i];
  auto conservative=original;conservative.conservative=!conservative.conservative;cases.push_back(conservative);
  for(unsigned word=0;word<15;++word){auto c=original;unsigned value;std::memcpy(&value,reinterpret_cast<unsigned char*>(&c.manifold)+4*word,4);value^=1;std::memcpy(reinterpret_cast<unsigned char*>(&c.manifold)+4*word,&value,4);cases.push_back(c);}
  for(unsigned side=0;side<2;++side){
   const auto& source=side?original.second:original.first;unsigned count=source.kind==p::ShapeKind::circle?3:6+4*source.vertex_count;
   for(unsigned field=0;field<count;++field){auto c=original;auto& s=side?c.second:c.first;
    if(field<2)flip(field?s.position.y:s.position.x);
    else if(s.kind==p::ShapeKind::circle)flip(s.radius);
    else if(field<6){float* a[]={&s.R.col1.x,&s.R.col1.y,&s.R.col2.x,&s.R.col2.y};flip(*a[field-2]);}
    else {const unsigned index=(field-6)/4,component=(field-6)%4;float* a[]={&s.vertices[index].x,&s.vertices[index].y,&s.normals[index].x,&s.normals[index].y};flip(*a[component]);}
    cases.push_back(c);
   }
  }
  auto invalid=original;invalid.first.kind=p::ShapeKind(99);cases.push_back(invalid);
  auto numeric=original;numeric.first.position.x=__builtin_inff();cases.push_back(numeric);
  auto no_contact=original;no_contact.second.position.x=100000.0f;cases.push_back(no_contact);
 }
 // Duplicated keys exercise simultaneous publication followed by guaranteed
 // warm reads; distinct variants force exact full-key discrimination.
 const auto first=cases;cases.insert(cases.end(),first.begin(),first.end());return cases;
}
NATIVE_HD Outcome evaluate(const Case& c){Outcome out;std::memcpy(&out.reference,&c.manifold,sizeof(c.manifold));std::memcpy(&out.cached,&c.manifold,sizeof(c.manifold));out.first=p::collide(out.reference,c.first,c.second,c.conservative);out.second=memo::collide_exact(out.cached,c.first,c.second,c.conservative);return out;}
#ifdef __CUDACC__
__global__ void qualify(const Case* cases,Outcome* output,unsigned count){const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane<count)output[lane]=evaluate(cases[lane]);}
__global__ void force_hash_collision(const Case* cases,memo::Entry* entries,unsigned capacity,unsigned* forced){
 if(threadIdx.x||blockIdx.x)return;memo::Key key;if(!memo::make_key(key,cases[0].manifold,cases[0].first,cases[0].second,cases[0].conservative))return;
 const auto hash=memo::hash_key(key);key.words[0]^=1U;p::Manifold wrong{};p::StageResult status{p::Status::invalid_shape};
 memo::publish(entries[unsigned(hash)&(capacity-1)],key,hash,wrong,status);*forced=1;
}
__global__ void virgin_initialize(Case* cases){
 if(threadIdx.x||blockIdx.x)return;
 // Circle/circle, polygon/circle, polygon/polygon: write only consumed fields.
 // Inactive vertices, cold fields, and ABI padding remain virgin.
 for(unsigned c=0;c<3;++c){
  for(unsigned i=0;i<2;++i){auto* s=i?&cases[c].second:&cases[c].first;const bool polygon=c==2||(c==1&&!i);s->kind=polygon?p::ShapeKind::polygon:p::ShapeKind::circle;s->position={float(i),0};
   if(!polygon)s->radius=1;
   else {s->vertex_count=3;s->R={{1,0},{0,1}};s->vertices[0]={-1,-1};s->vertices[1]={1,-1};s->vertices[2]={0,1};s->normals[0]={0,-1};s->normals[1]={.894427191f,.447213595f};s->normals[2]={-.894427191f,.447213595f};}
  }
  for(auto& point:cases[c].manifold.points){point.position={0,0};point.separation=0;point.normal_impulse=2;point.tangent_impulse=3;point.id=7;}
  cases[c].manifold.normal={0,1};cases[c].manifold.point_count=1;cases[c].conservative=false;
 }
}
void check(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
#endif
int main(int argc,char** argv){try{
 if(argc<2)throw std::runtime_error("usage: collision-memo-validate native-cases.bin [--gpu] [--limit N] [--capacity N] [--expected native.bin] [--virgin-shapes]");
 bool gpu=false,virgin=false;unsigned limit=512,capacity=1024;std::string expected;
 for(int i=2;i<argc;++i){const std::string arg=argv[i];if(arg=="--gpu")gpu=true;else if(arg=="--virgin-shapes")virgin=true;else if(arg=="--limit"&&i+1<argc)limit=unsigned(std::stoul(argv[++i]));else if(arg=="--capacity"&&i+1<argc)capacity=unsigned(std::stoul(argv[++i]));else if(arg=="--expected"&&i+1<argc)expected=argv[++i];else throw std::runtime_error("unknown collision-memo argument");}
 if(!limit||limit>65536||capacity<4||capacity>(1U<<22)||(capacity&(capacity-1)))throw std::runtime_error("invalid bounded collision-memo options");
 const auto native=load(argv[1],limit);if(native.empty())throw std::runtime_error("empty collision corpus");auto cases=variants(native);std::vector<Outcome> output(cases.size());memo::Counters total{};
 unsigned key_cases=0;for(const auto& c:cases){memo::Key first,second;if(memo::make_key(first,c.manifold,c.first,c.second,c.conservative)){auto ignored=c;ignored.first.id^=123;ignored.second.user_data^=55;if(!memo::make_key(second,ignored.manifold,ignored.first,ignored.second,ignored.conservative)||!memo::equal_key(first,second))throw std::runtime_error("unconsumed fields changed key");++key_cases;}}
 if(gpu){
#ifdef __CUDACC__
  Case* input{};Outcome* device_output{};memo::Entry* entries{};memo::Counters* counts{};unsigned* forced{};
  const unsigned count=virgin?3:unsigned(cases.size());output.resize(count);check(cudaMalloc(&input,count*sizeof(Case)));check(cudaMalloc(&device_output,count*sizeof(Outcome)));check(cudaMalloc(&entries,std::size_t(capacity)*sizeof(memo::Entry)));check(cudaMalloc(&counts,128*sizeof(memo::Counters)));check(cudaMemset(entries,0,std::size_t(capacity)*sizeof(memo::Entry)));check(cudaMemset(counts,0,128*sizeof(memo::Counters)));
  if(virgin)virgin_initialize<<<1,1>>>(input);else check(cudaMemcpy(input,cases.data(),count*sizeof(Case),cudaMemcpyHostToDevice));
  check(cudaMalloc(&forced,4));check(cudaMemset(forced,0,4));force_hash_collision<<<1,1>>>(input,entries,capacity,forced);
  check(memo::configure({entries,capacity,counts,128}));
  for(unsigned repeat=0;repeat<3;++repeat){qualify<<<(count+63)/64,64>>>(input,device_output,count);check(cudaGetLastError());check(cudaDeviceSynchronize());}
  check(cudaMemcpy(output.data(),device_output,count*sizeof(Outcome),cudaMemcpyDeviceToHost));memo::Counters shards[128];check(cudaMemcpy(shards,counts,sizeof(shards),cudaMemcpyDeviceToHost));
  for(const auto& s:shards){total.calls+=s.calls;total.hits+=s.hits;total.misses+=s.misses;total.inserts+=s.inserts;total.busy+=s.busy;total.full+=s.full;total.bypass+=s.bypass;}
  unsigned injected{};check(cudaMemcpy(&injected,forced,4,cudaMemcpyDeviceToHost));
  check(memo::configure({}));check(cudaFree(forced));check(cudaFree(counts));check(cudaFree(entries));check(cudaFree(device_output));check(cudaFree(input));if(!total.hits||!total.inserts||!injected)throw std::runtime_error("no hit/publication/hash-collision witnesses");
#else
  throw std::runtime_error("CUDA unavailable");
#endif
 }else{if(virgin)throw std::runtime_error("virgin shapes needs --gpu");for(unsigned i=0;i<cases.size();++i)output[i]=evaluate(cases[i]);}
 unsigned failures=0,rejections=0;for(unsigned i=0;i<output.size();++i){const auto& o=output[i];rejections+=!o.first;if(o.first.status!=o.second.status||o.first.body!=o.second.body||std::memcmp(&o.reference,&o.cached,sizeof(p::Manifold))){++failures;std::fprintf(stderr,"case%u status%u/%u\n",i,unsigned(o.first.status),unsigned(o.second.status));}}
 unsigned native_matches=0;if(!expected.empty()&&!virgin){std::ifstream source(expected,std::ios::binary);for(unsigned i=0;i<native.size();++i){p::Manifold reference;read(source,reference);if(std::memcmp(&reference,&output[i].reference,sizeof(reference)))throw std::runtime_error("actual native collision packet differs");++native_matches;}}
 std::printf("{\"passed\":%s,\"gpu\":%s,\"virgin_shapes\":%s,\"cases\":%zu,\"key_cases\":%u,\"native_matches\":%u,\"rejections\":%u,\"mismatches\":%u,\"capacity\":%u,\"entry_bytes\":%zu,\"calls\":%llu,\"hits\":%llu,\"misses\":%llu,\"inserts\":%llu,\"busy\":%llu,\"full\":%llu,\"bypass\":%llu}\n",failures?"false":"true",gpu?"true":"false",virgin?"true":"false",output.size(),key_cases,native_matches,rejections,failures,capacity,sizeof(memo::Entry),total.calls,total.hits,total.misses,total.inserts,total.busy,total.full,total.bypass);
 return failures?1:0;
}catch(const std::exception& error){std::fprintf(stderr,"collision-memo-validate: %s\n",error.what());return 2;}}
