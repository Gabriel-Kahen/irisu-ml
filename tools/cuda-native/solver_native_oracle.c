// Actual unmodified ELF32 MSVC solver stages; no Guest/translated reference.
#include "oracle_native.h"
#include "solver_wire.h"
static uint32_t word(const void* p,unsigned offset){uint32_t v;memcpy(&v,(const char*)p+offset,4);return v;}
typedef void* (ORACLE_THISCALL *ManifoldGetter)(void*);
static void* native_manifolds(void* c){return ((ManifoldGetter)(uintptr_t)word((void*)(uintptr_t)word(c,0),0))(c);}
static unsigned pointer_id(void** pointers,unsigned* count,void* p){for(unsigned i=0;i<*count;++i)if(pointers[i]==p)return i;if(*count>=64)oracle_die("fixture pointer capacity");pointers[*count]=p;return (*count)++;}
static void write_words(FILE* f,const void* p,size_t words){if(words&&fwrite(p,4,words,f)!=words)oracle_die("cannot write solver packet");}
static void body_packet(FILE* f,const void* body){const unsigned offsets[12]={4,8,12,16,20,24,28,44,48,52,108,116};for(unsigned i=0;i<12;++i){uint32_t value=word(body,offsets[i]);write_words(f,&value,1);}}
static void state(FILE* f,void** bodies,unsigned nb,const void* constraints,unsigned nc,void** manifolds,unsigned nm,int initial){
 for(unsigned i=0;i<nb;++i)body_packet(f,bodies[i]);
 for(unsigned i=0;i<nc;++i){uint32_t data[30];if(initial){for(unsigned j=0;j<30;++j)data[j]=0xcdcdcdcdu;data[24]=data[25]=data[26]=0;}else{
   memcpy(data,(const char*)constraints+i*120,sizeof(data));unsigned ignored=nb;data[25]=pointer_id(bodies,&ignored,(void*)(uintptr_t)data[25]);if(ignored!=nb)oracle_die("new solver body");ignored=nb;data[26]=pointer_id(bodies,&ignored,(void*)(uintptr_t)data[26]);if(ignored!=nb)oracle_die("new solver body");ignored=nm;data[24]=pointer_id(manifolds,&ignored,(void*)(uintptr_t)data[24]);if(ignored!=nm)oracle_die("new solver manifold");
  }write_words(f,data,30);
 }
 for(unsigned i=0;i<nm;++i)write_words(f,manifolds[i],15);
}
static void contacts(FILE* f,void** list,unsigned count,void** bodies,unsigned nb,void** manifolds,unsigned nm){
 for(unsigned i=0;i<count;++i){const void* contact=list[i];uint32_t data[5];unsigned n=nb;data[0]=pointer_id(bodies,&n,(void*)(uintptr_t)word((void*)(uintptr_t)word(contact,48),40));n=nb;data[1]=pointer_id(bodies,&n,(void*)(uintptr_t)word((void*)(uintptr_t)word(contact,52),40));
  if((int32_t)word(contact,56)>0){n=nm;data[2]=pointer_id(manifolds,&n,native_manifolds((void*)contact));if(n!=nm)oracle_die("unrecognized manifold address");}else data[2]=nm-1;
  data[3]=word(contact,60);data[4]=word(contact,64);write_words(f,data,5);
 }
}
int main(int argc,char** argv){
 if(argc!=3)oracle_die("usage: solver-native-oracle pinned.so cases.bin");NativeOracle oracle=oracle_open(argv[1]);
 void* (*create)(float,float,float,float,float,float)=dlsym(oracle.library,"b2d_world_create");void (*destroy)(void*)=dlsym(oracle.library,"b2d_world_destroy");
 void* (*circle)(void*,float,float,float,float,float,float)=dlsym(oracle.library,"b2d_world_create_circle");void* (*box)(void*,float,float,float,float,float,float,float,float)=dlsym(oracle.library,"b2d_world_create_box");void* (*triangle)(void*,float,float,float,float,float,float,float,float)=dlsym(oracle.library,"b2d_world_create_triangle");void (*step)(void*,float,int)=dlsym(oracle.library,"b2d_world_step");void (*set_v)(void*,void*,float,float)=dlsym(oracle.library,"b2d_world_set_v");
 if(!create||!destroy||!circle||!box||!triangle||!step||!set_v)oracle_die("missing native scene exports");
 NativeSolverCtor construct=oracle_solver_constructor(&oracle);NativeMethod0 pre=oracle_pre_solve(&oracle),velocity=oracle_velocity_solve(&oracle),post=oracle_post_solve(&oracle),destruct=oracle_solver_destructor(&oracle);NativePositionSolve position=oracle_position_solve(&oracle);
 NativeMethod0 stack_construct=oracle_stack_constructor(&oracle),stack_destruct=oracle_stack_destructor(&oracle);
 const unsigned fixtures=32,stages=5;FILE* cases=fopen(argv[2],"wb");uint32_t header[3]={SOLVER_INPUT_MAGIC,1,fixtures*stages};if(!cases)oracle_die("cannot create cases");write_words(cases,header,3);header[0]=SOLVER_OUTPUT_MAGIC;write_words(stdout,header,3);
 const uint16_t cw=0x027f;__asm__ __volatile__("fninit; fldcw %0"::"m"(cw));unsigned active=0,total_constraints=0,total_points=0;
 for(unsigned fixture=0;fixture<fixtures;++fixture){
  void* world_handle=create(-100,-100,100,100,10,1);if(!world_handle)oracle_die("create world");box(world_handle,20,1,0,10,0,0,.3f,.1f);
  void* first=circle(world_handle,.5f,(float)(fixture%4)*.15f,8.0f,1,.3f,.1f);box(world_handle,1,1,.3f,-2,fixture%3*.2f,1,.3f,.1f);triangle(world_handle,1,1,-.2f,-4,0,1,.3f,.1f);
  if(fixture&1)set_v(world_handle,first,1.25f,-3);
  for(unsigned tick=0;tick<24+fixture*5;++tick)step(world_handle,1.0f/60.0f,10);
  void* world=(void*)(uintptr_t)word(world_handle,0);void* contact=(void*)(uintptr_t)word(world,102956);void* list[64];unsigned contact_count=0;
  while(contact){if(contact_count>=63)oracle_die("too many contacts");list[contact_count++]=contact;contact=(void*)(uintptr_t)word(contact,12);}
  void* body_list[64];void* manifold_list[64];unsigned nb=0,nm=0;
  for(unsigned i=0;i<contact_count;++i){pointer_id(body_list,&nb,(void*)(uintptr_t)word((void*)(uintptr_t)word(list[i],48),40));pointer_id(body_list,&nb,(void*)(uintptr_t)word((void*)(uintptr_t)word(list[i],52),40));if(word(list[i],56)>1)oracle_die("multiple manifolds per contact outside packet contract");if((int32_t)word(list[i],56)>0)pointer_id(manifold_list,&nm,native_manifolds(list[i]));}
  // Inactive contacts refer to a zero-count sentinel manifold, not a guess at
  // their subclass storage. The native virtual GetManifolds method resolves active storage.
  uint32_t empty_manifold[15]={0};manifold_list[nm++]=empty_manifold;
  unsigned nc=0;for(unsigned i=0;i<contact_count;++i)nc+=word(list[i],56);
  uint32_t solver[3]={0};unsigned char* allocator=malloc(102800);if(!allocator)oracle_die("allocator malloc");memset(allocator,0xcd,102800);stack_construct(allocator);
  SolverWireCase meta={0,nb,nc,nm,contact_count,fixture%2,0};float beta=.2f;memcpy(&meta.beta,&beta,4);*(uint32_t*)(oracle.base+99272)=meta.warm;
  write_words(cases,&meta,7);state(cases,body_list,nb,NULL,nc,manifold_list,nm,1);contacts(cases,list,contact_count,body_list,nb,manifold_list,nm);
  construct(solver,list,contact_count,allocator);if(solver[2]!=nc)oracle_die("native constraint count mismatch");void* constraint_data=(void*)(uintptr_t)solver[1];
  SolverWireResult result={0,nc,1};write_words(stdout,&result,3);state(stdout,body_list,nb,constraint_data,nc,manifold_list,nm,0);
  for(unsigned stage=1;stage<5;++stage){meta.stage=stage;write_words(cases,&meta,7);state(cases,body_list,nb,constraint_data,nc,manifold_list,nm,0);contacts(cases,list,contact_count,body_list,nb,manifold_list,nm);
   if(stage==1)pre(solver);if(stage==2)velocity(solver);if(stage==3)result.converged=position(solver,beta);if(stage==4){post(solver);result.converged=1;}
   write_words(stdout,&result,3);state(stdout,body_list,nb,constraint_data,nc,manifold_list,nm,0);
  }
  if(nc)++active;total_constraints+=nc;for(unsigned i=0;i<nc;++i)total_points+=word((char*)constraint_data+i*120,116);
  destruct(solver);stack_destruct(allocator);free(allocator);destroy(world_handle);
 }
 fclose(cases);oracle_close(&oracle);fprintf(stderr,"{\"identity\":\"native-solver-stage-v1\",\"native_sha256\":\"%s\",\"fixtures\":%u,\"stage_cases\":%u,\"active_fixtures\":%u,\"constraints\":%u,\"points\":%u,\"order_scope\":\"contact-list stage fixture; not original island traversal\"}\n",ORACLE_PIN,fixtures,fixtures*stages,active,total_constraints,total_points);return fflush(stdout)?2:0;
}
