// Host-only exact comparison of cached-R branches against their unchanged
// canonical_rotation=false path. No native or GPU execution in this source.
#ifdef IRISU_CUDA_NATIVE_CACHED_ROTATION
#undef IRISU_CUDA_NATIVE_CACHED_ROTATION
#endif
#define IRISU_CUDA_NATIVE_CACHED_ROTATION 1
#ifdef IRISU_CUDA_NATIVE_F32_STORE
#undef IRISU_CUDA_NATIVE_F32_STORE
#endif
#define IRISU_CUDA_NATIVE_F32_STORE 0
#include "../../clone/cuda_native/world_pipeline.hpp"
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <new>
namespace n=irisu::cuda_native;
struct Counts {unsigned integration{},position{},rejected{},default_calls{},mismatches{};} counts;
float from_bits(unsigned u){union{unsigned u;float f;}v{u};return v.f;}
unsigned random_word(unsigned x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16);}
void initialize(n::BodyState& b,float angle){std::memset(&b,0,sizeof b);::new(&b)n::BodyState();b.inv_mass=1;b.inv_i=.5f;b.id=17;b.position={3,-4};b.velocity={1,-2};b.rotation=angle;n::rotation_matrix(b);}
bool equal(const n::SolverResult& a,const n::SolverResult& b){return a.status==b.status&&a.constraint==b.constraint&&a.point==b.point&&a.converged==b.converged;}
int main(){if(std::fesetenv(FE_DFL_ENV))return 1;
#if defined(__i386__)||defined(__x86_64__)
 const unsigned short cw=0x027f;asm volatile("fldcw %0"::"m"(cw));
#endif
 const unsigned edges[]={0,0x80000000U,1,0x80000001U,0x33000000U,0xb3000000U,0x3f800000U,0xbf800000U,0x48fffffeU,0xc8fffffeU,0x49000000U,0xc9000000U,0x7f800000U,0xff800000U,0x7fc01234U};
 for(unsigned i=0;i<20000;++i){const float angle=i<sizeof(edges)/sizeof(edges[0])?from_bits(edges[i]):float(int(random_word(i)%1000000)-500000)*.01f;n::BodyState base;initialize(base,angle);base.angular_velocity=i%3?float(int(random_word(i+1)%2000)-1000)*.01f:0;
  n::BodyState a,b;std::memcpy(&a,&base,sizeof a);std::memcpy(&b,&base,sizeof b);const auto ar=n::integrate_position(a,1.0f/60,false),br=n::integrate_position(b,1.0f/60,true);++counts.integration;counts.rejected+=ar.status!=n::Status::ok;
  if(ar.status!=br.status||ar.body!=br.body||(ar&&std::memcmp(&a,&b,sizeof a)))++counts.mismatches;
  if(!n::math::sincos_input_supported(angle)){
   n::BodyState x[2],y[2];initialize(x[0],angle);initialize(x[1],0);x[0].inv_mass=x[0].inv_i=0;std::memcpy(y,x,sizeof y);
   n::ContactConstraint ca{},cb{};ca.body1=0;ca.body2=1;ca.point_count=1;ca.normal={0,1};ca.points[0].normal_mass=1;std::memcpy(&cb,&ca,sizeof cb);
   const auto xr=n::solver_position(x,2,&ca,1,.2f,false),yr=n::solver_position(y,2,&cb,1,.2f,true);++counts.position;++counts.rejected;if(!equal(xr,yr)||xr.status==n::Status::ok)++counts.mismatches;
   continue;
  }
  n::BodyState reference[2],cached[2];initialize(reference[0],angle);initialize(reference[1],-angle);reference[1].position={4,-2};reference[1].inv_mass=i%2?0:1;reference[1].inv_i=i%2?0:.7f;std::memcpy(cached,reference,sizeof cached);
  n::ContactConstraint ca{},cb{};ca.body1=0;ca.body2=1;ca.point_count=2;ca.normal={.6f,.8f};
  for(unsigned j=0;j<2;++j){auto& p=ca.points[j];p.local_anchor1={1,.5f};p.local_anchor2={.2f,-.4f};p.normal_mass=1;p.separation=j?-.03f:.01f;}
  std::memcpy(&cb,&ca,sizeof cb);
  for(unsigned step=0;step<3;++step){const float beta=i%4?.2f:0;const auto x=n::solver_position(reference,2,&ca,1,beta,false),y=n::solver_position(cached,2,&cb,1,beta,true);++counts.position;if(!equal(x,y)||(x&&(std::memcmp(reference,cached,sizeof cached)||std::memcmp(&ca,&cb,sizeof ca))))++counts.mismatches;}
 }
 // The ordinary standalone call must repair an arbitrary R even in an opt-in
 // translation unit; +0 default Mat22 differs from canonical -sin(+0).
 n::BodyState standalone;initialize(standalone,0);standalone.R={{42,17},{-3,25}};const auto result=n::integrate_position(standalone,0);n::BodyState expected=standalone;n::rotation_matrix(expected);++counts.default_calls;if(!result||std::memcmp(&standalone.R,&expected.R,sizeof expected.R))++counts.mismatches;
 std::printf("{\"integration_cases\":%u,\"position_cases\":%u,\"rejected_numeric_cases\":%u,\"standalone_default_cases\":%u,\"mismatches\":%u}\n",counts.integration,counts.position,counts.rejected,counts.default_calls,counts.mismatches);return counts.mismatches?2:0;
}
