// Execute selected original v2.03 contact/update components, never its entrypoint.
#define _GNU_SOURCE
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

typedef struct { uint8_t bytes[256]; } Actor;
static uint32_t actor_vt[32], scene_vt[32], manager_vt[32], manager[1], scene[1];
static uint32_t chains[16][3], allocations;
static uint32_t u32(const void *p) { uint32_t v; memcpy(&v,p,4); return v; }
static uint16_t u16(const void *p) { uint16_t v; memcpy(&v,p,2); return v; }
static void put(void *p, uint32_t v) { memcpy(p,&v,4); }
#define U(a,o) u32((a)->bytes+(o))
#define W(a,o,v) put((a)->bytes+(o),(uint32_t)(v))
#define B(a,o) ((a)->bytes[(o)])
static void die(const char *s) { fprintf(stderr,"%s\n",s); exit(1); }
static void check(int valid,const char *s) { if(!valid) die(s); }
static uint32_t norma=6;
static void *get_config(void *map,void *type,unsigned size,unsigned length,const char *key) {
  (void)map; (void)type;
  check(size==4 && length==18 && !memcmp(key,"statge_level_norma",18),"unexpected config lookup");
  return &norma;
}
static void jump(uintptr_t address, uintptr_t target) {
  uint8_t *p=(uint8_t*)address; p[0]=0xe9; put(p+1,(uint32_t)(target-address-5));
}
__attribute__((used)) static void *new_chain(void) { check(allocations<16,"chain capacity"); return chains[allocations++]; }
__attribute__((naked)) static void allocate_chain(void) {
  __asm__("call new_chain; mov %eax,%edx; mov $1,%eax; ret");
}
__attribute__((naked)) static void noop(void) { __asm__("ret"); }
__attribute__((naked)) static void yes(void) { __asm__("mov $1,%eax; ret"); }
__attribute__((naked)) static void no(void) { __asm__("xor %eax,%eax; ret"); }
__attribute__((naked)) static void no2(void) { __asm__("xor %eax,%eax; ret $8"); }
__attribute__((naked)) static void no3(void) { __asm__("xor %eax,%eax; ret $12"); }
__attribute__((naked)) static void destroy(void) { __asm__("movl $1,0x68(%eax); ret"); }
static uint32_t run(uintptr_t fn, void *self, unsigned n, uintptr_t a,uintptr_t b,uintptr_t c,uintptr_t d) {
  uint32_t result; uintptr_t args[4]={a,b,c,d};
  (void)n;
  // Extra trailing arguments are ignored; restore ESP after the callee's own pop.
  __asm__ volatile("mov %%esp,%%ebx; push 12(%%esi); push 8(%%esi); push 4(%%esi); push (%%esi); call *%%edi; mov %%ebx,%%esp"
    : "=a"(result) : "a"(self),"S"(args),"D"(fn) : "ebx","ecx","edx","cc","memory");
  return result;
}
static void load(const char *path) {
  FILE *f=fopen(path,"rb"); if(!f) die("open");
  fseek(f,0,SEEK_END); long length=ftell(f); rewind(f);
  check(length>0x400&&length<1000000,"image size");
  uint8_t *file=malloc((size_t)length); check(file&&fread(file,1,(size_t)length,f)==(size_t)length,"read"); fclose(f);
  uint32_t pe=u32(file+0x3c); check(pe+24<(uint32_t)length&&!memcmp(file+pe,"PE\0\0",4),"PE");
  uint8_t *opt=file+pe+24; check(u16(opt)==0x10b&&u32(opt+28)==0x400000,"ABI");
  uint32_t image_size=u32(opt+56); check(image_size<=0x100000,"image extent");
  uint8_t *image=mmap((void*)0x400000,image_size,7,MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED_NOREPLACE,-1,0);
  check(image!=MAP_FAILED,"map");
  uint8_t *s=opt+u16(file+pe+20);
  for(unsigned i=0;i<u16(file+pe+6);++i,s+=40) {
    check(s+40<=file+length,"section table"); uint32_t rva=u32(s+12),size=u32(s+16),off=u32(s+20);
    check(rva<=image_size&&size<=image_size-rva&&off<=(uint32_t)length&&size<=(uint32_t)length-off,"section extent");
    memcpy(image+rva,file+off,size);
  }
  free(file);
  jump(0x416268,(uintptr_t)get_config);
  jump(0x4160e0,(uintptr_t)noop); // Original this-pointer assertion helper only.
  jump(0x418964,(uintptr_t)noop); // Debug output only.
  jump(0x419080,(uintptr_t)allocate_chain); // Zeroed CHAIN allocation only.
  jump(0x4077c4,(uintptr_t)noop); // Base transform/lifetime update excluded.
  jump(0x407c98,(uintptr_t)noop); // Particles excluded, clear flag routine retained.
  jump(0x431946,(uintptr_t)no2); jump(0x43194c,(uintptr_t)no3); // Audio APIs.
  memcpy(actor_vt,(void*)0x432390,sizeof actor_vt);
  actor_vt[0x1c/4]=(uintptr_t)destroy; actor_vt[0x54/4]=(uintptr_t)yes; actor_vt[0x5c/4]=(uintptr_t)noop;
  memcpy(scene_vt,(void*)0x434820,sizeof scene_vt);
  scene[0]=(uintptr_t)scene_vt;
  manager_vt[0x20/4]=(uintptr_t)no2; manager_vt[0x48/4]=(uintptr_t)yes;
  manager[0]=(uintptr_t)manager_vt; put((void*)0x4428a0,(uintptr_t)manager); put((void*)0x4428d0,(uintptr_t)manager);
  put((void*)0x436a38,120); put((void*)0x4428c8,1); put((void*)0x4428cc,0);
  unsigned short cw=0x027f; __asm__ volatile("fninit; fldcw %0"::"m"(cw));
}
static void reset_actor(Actor *a,int ex) {
  memset(a,0,sizeof *a); W(a,0,(uintptr_t)actor_vt); W(a,0x58,-1); W(a,0x5c,150);
  W(a,0xc8,2); W(a,0xcc,4); B(a,0xf0)=ex; W(a,0xf4,-1);
}
static void emit(const char *name,Actor *a,uint32_t ret) {
  uint32_t *g=(uint32_t*)(uintptr_t)U(a,0xd8);
  printf("%s\t%u\t%u\t%u\t%u\t%u\t%d\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\n",name,ret,
   B(a,0xc4),B(a,0xd4),B(a,0xd5),U(a,0xd0),(int32_t)U(a,0xf4),U(a,0xf8),B(a,0xe4),
   U(a,0x68),U(a,0xdc),g?g[0]:0,g?g[1]:0,g?g[2]:0,u32((void*)0x4428e4),u32((void*)0x4428cc));
}
int main(int argc,char **argv) {
  if(argc!=2) die("usage: original_contacts_probe irisu.exe");
  setbuf(stdout,NULL);
  load(argv[1]);
  Actor a,b; int sound; uint32_t ret;
  puts("case\treturn\tdynamic\tgrouped\tclear_pending\trot_timer\tex_timer\tdynamic_updates\tgate_seen\tdelete\thits\tchain\tshadow\tclears\tgauge\tqualifying_clears");
  reset_actor(&a,1); B(&a,0xc4)=1; W(&a,0xf4,120); W(&a,0xd8,(uintptr_t)chains[0]); chains[0][0]=2;
  for(unsigned i=0;i<8;++i) run(0x4033d0,&a,0,0,0,0,0);
  check(U(&a,0xf4)==112&&U(&a,0xf8)==8&&!B(&a,0xd4),"countdown112"); emit("countdown_8",&a,0);
  reset_actor(&b,1); ret=run(0x40a35c,scene,2,(uintptr_t)&a,(uintptr_t)&b,0,0);
  check(B(&b,0xc4)&&U(&b,0xf4)==120&&!B(&b,0xd4)&&chains[0][0]==3&&!ret,"propagate112"); emit("propagate_112",&b,ret);
  reset_actor(&b,1); W(&a,0xf4,113); ret=run(0x40a35c,scene,2,(uintptr_t)&a,(uintptr_t)&b,0,0);
  check(!B(&b,0xc4)&&(int32_t)U(&b,0xf4)==-1&&!ret,"no propagate113"); emit("propagate_113_rejected",&b,ret);
  W(&a,0xf4,1); run(0x4033d0,&a,0,0,0,0,0); check(B(&a,0xd4)&&U(&a,0xf4)==0,"countdown expiry"); emit("countdown_expired",&a,0);
  reset_actor(&a,0); B(&a,0xd4)=1; W(&a,0xd8,(uintptr_t)chains[1]); chains[1][0]=4; chains[1][1]=4;
  reset_actor(&b,1); W(&b,0xf4,57); ret=run(0x40a35c,scene,2,(uintptr_t)&a,(uintptr_t)&b,0,0);
  check(U(&b,0xf4)==120&&B(&b,0xc4)&&!B(&b,0xd4)&&ret==5,"mixed reset"); emit("mixed_group_resets_countdown",&b,ret);
  reset_actor(&a,1); reset_actor(&b,0); W(&b,0xcc,2); ret=run(0x40a6f0,scene,2,(uintptr_t)&a,(uintptr_t)&b,0,0);
  check(ret&&B(&a,0xe4),"EX bottom gate"); emit("ex_bottom_gate",&a,ret);
  reset_actor(&a,1); W(&b,0xcc,6); ret=run(0x40a6f0,scene,2,(uintptr_t)&a,(uintptr_t)&b,0,0);
  check(!ret&&!B(&a,0xe4),"EX top open"); emit("ex_top_open",&a,ret);
  reset_actor(&a,1); reset_actor(&b,0); sound=-1; run(0x40a7d8,scene,3,(uintptr_t)&sound,(uintptr_t)&b,(uintptr_t)&a,0);
  check(B(&a,0xc4)&&sound==0,"EX ordinary activation"); emit("ex_activated_by_scripted_ordinary",&a,0);
  reset_actor(&a,1); B(&b,0xc4)=1; sound=-1; run(0x40a7d8,scene,3,(uintptr_t)&sound,(uintptr_t)&b,(uintptr_t)&a,0);
  check(!B(&a,0xc4),"EX dynamic ordinary rejection"); emit("ex_dynamic_ordinary_rejected",&a,0);
  reset_actor(&a,1); W(&b,0xcc,5); run(0x40a7d8,scene,3,(uintptr_t)&sound,(uintptr_t)&b,(uintptr_t)&a,0);
  check(B(&a,0xc4)&&U(&b,0x68)==1,"EX projectile activation"); emit("ex_activated_by_projectile",&a,0);
  reset_actor(&a,1); reset_actor(&b,0); W(&b,0xcc,2); W(&a,0xf8,4); run(0x40a66c,scene,2,(uintptr_t)&b,(uintptr_t)&a,0,0);
  check(!U(&a,0xd0),"rot before5"); emit("ex_rot_before_5_rejected",&a,0);
  W(&a,0xf8,5); run(0x40a66c,scene,2,(uintptr_t)&b,(uintptr_t)&a,0,0); check(U(&a,0xd0)==1,"rot at5"); emit("ex_rot_at_5",&a,0);
  for(unsigned i=0;i<130;++i) run(0x4033d0,&a,0,0,0,0,0);
  check(U(&a,0xc8)==2&&U(&a,0xd0)==1,"EX cannot rot"); emit("ex_rot_does_not_advance",&a,0);
  reset_actor(&a,1); reset_actor(&b,0); ret=run(0x40a958,scene,2,(uintptr_t)&a,(uintptr_t)&b,0,0);
  check(!ret,"mixed scripted contact allowed"); emit("mixed_scripted_contact_allowed",&a,ret);
  B(&b,0xf0)=1; ret=run(0x40a958,scene,2,(uintptr_t)&a,(uintptr_t)&b,0,0); check(ret,"same EX scripted rejected"); emit("same_ex_scripted_short_circuit",&a,ret);
  reset_actor(&a,1); B(&a,0xc4)=1; B(&a,0xd4)=1; W(&a,0xd8,(uintptr_t)chains[2]); chains[2][0]=7;
  reset_actor(&b,1); sound=-1; int other_flag=0; put((void*)0x4428e4,1000);
  run(0x40a510,scene,4,(uintptr_t)&sound,(uintptr_t)&other_flag,(uintptr_t)&b,(uintptr_t)&a);
  check(B(&a,0xd5)&&chains[2][2]==1&&u32((void*)0x4428e4)==1120&&u32((void*)0x4428cc)==1,"EX source burst"); emit("ex_source_bursts_mature_ex",&a,0);
  reset_actor(&b,0); W(&b,0xcc,5); W(&b,0xe0,2); reset_actor(&a,1); run(0x40a89c,scene,2,(uintptr_t)&a,(uintptr_t)&b,0,0);
  check(!U(&b,0x68),"EX hit retains projectile"); emit("ex_direct_hit_retains_projectile",&b,0);
  B(&a,0xf0)=0; run(0x40a89c,scene,2,(uintptr_t)&a,(uintptr_t)&b,0,0); check(U(&b,0x68),"ordinary hit deletes projectile"); emit("ordinary_direct_hit_deletes_projectile",&b,0);
  Actor orb,dead; reset_actor(&a,1); W(&a,0xbc,3); reset_actor(&orb,0); W(&orb,0xbc,-2);
  B(&orb,0xc5)=1; B(&orb,0xe5)=1; B(&orb,0xc4)=1;
  reset_actor(&dead,0); W(&dead,0xbc,3); B(&dead,0x54)=1;
  uintptr_t pool[3]={(uintptr_t)&a,(uintptr_t)&orb,(uintptr_t)&dead};
  put((void*)0x442620,3); put((void*)0x442624,(uintptr_t)pool); put((void*)0x4428e4,1000);
  sound=-1; other_flag=0;
  run(0x40a748,scene,4,(uintptr_t)&other_flag,(uintptr_t)&sound,(uintptr_t)&a,(uintptr_t)&orb);
  check(U(&a,0x68)&&U(&orb,0x68)&&U(&dead,0x68)&&u32((void*)0x4428e4)==1000,"Metsu special no reward");
  emit("metsu_orb_clears_color_without_gauge",&a,0);
  manager_vt[0x48/4]=(uintptr_t)no; run(0x4032d0,(void*)3,0,0,0,0,0);
  check(u32((void*)0x4428e4)==1240,"normal special positive control");
  emit("normal_color_clear_positive_control",&a,0);
  reset_actor(&a,0); reset_actor(&b,0); memset(chains,0,sizeof chains); allocations=0;
  ret=run(0x40a35c,scene,2,(uintptr_t)&a,(uintptr_t)&b,0,0);
  check(ret==1&&B(&b,0xd4)&&allocations==1,"fresh chain allocation"); emit("fresh_ordinary_chain_control",&b,ret);
  return 0;
}
