/* Original-game observer. This includes the existing transparent ABI forwarder;
 * gameplay and physics remain the authentic executable and shipped DLL. */
#define DllMainCRTStartup ForwarderDllMain
#include "../../reference/trace-proxy/src/box2d_trace_proxy.c"
#undef DllMainCRTStartup

DLLIMPORT BOOL WINAPI VirtualProtect(void *,DWORD,DWORD,DWORD *);
DLLIMPORT BOOL WINAPI FlushInstructionCache(HANDLE,const void *,DWORD);
DLLIMPORT DWORD WINAPI GetEnvironmentVariableA(const char *,char *,DWORD);
DLLIMPORT void WINAPI ExitProcess(DWORD);

static HANDLE state_file=INVALID_HANDLE_VALUE;
static DWORD previous_frame=0xffffffffUL,observed_world,trace_scene,bootstrap_done;
static char replay_path[1024],launch_mode[32],limit_text[32],replay_sha256[65];
static DWORD finish_calls,first_finish_score,first_finish_level,first_finish_chain,first_finish_clears;
static DWORD frame_limit,selected_mode=1,title_passes,title_warmup=60,first_title_cw,eager_real;
static DWORD read32(DWORD address) { return *(volatile DWORD *)address; }
static unsigned int read8(DWORD address) { return *(volatile unsigned char *)address; }
static void state_write(const char *text) {
  DWORD written;
  if(state_file!=INVALID_HANDLE_VALUE) WriteFile(state_file,text,string_length(text),&written,0);
}
static unsigned short pre_load_cw;
static void state_open(void) {
  char path[512],line[768];
  __asm__ volatile("fnstcw %0":"=m"(pre_load_cw));
  ensure_loaded();
  __asm__ volatile("fldcw %0"::"m"(pre_load_cw));
  if(state_file==INVALID_HANDLE_VALUE&&format&&sibling_path(path,sizeof(path),"metsu-state.jsonl")) {
    state_file=CreateFileA(path,GENERIC_WRITE,FILE_SHARE_READ,0,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,0);
    if(state_file==INVALID_HANDLE_VALUE) ExitProcess(91);
    GetEnvironmentVariableA("IRISU_METSU_REPLAY_SHA256",replay_sha256,sizeof(replay_sha256));
    format(line,sizeof(line),"{\"type\":\"metadata\",\"schema\":\"irisu-metsu-state-trace-v1\",\"source\":\"original\",\"execution\":\"original-process\",\"game_exe_sha256\":\"0636d3e44439d88807d0c00aeb1bb072316c69fc13a21f79d67e53affad28255\",\"mode\":%lu,\"boundary\":\"after-actor-pass\",\"original_executable_sha256\":\"0636d3e44439d88807d0c00aeb1bb072316c69fc13a21f79d67e53affad28255\",\"gameplay_stubs\":false,\"observer_preload_cw\":\"%04x\",\"replay_sha256\":\"%s\",\"eager_real_dll\":%lu}\n",selected_mode,pre_load_cw,replay_sha256,eager_real);state_write(line);
  }
}
static DWORD call0(DWORD address,DWORD self) {
  DWORD result;
  __asm__ volatile("call *%2":"=a"(result):"a"(self),"r"(address):"ecx","edx","memory","cc");
  return result;
}
static void set_mode(DWORD mode) {
  DWORD self=read32(0x4428d0),address=0x408594;
  __asm__ volatile("push %2;call *%1":"+a"(self):"r"(address),"r"(mode):"ecx","edx","memory","cc");
}
static DWORD allocate_play(void) {
  DWORD result,address=0x418ec8;
  __asm__ volatile("push $0x4347d0;call *%1;add $4,%%esp":"=a"(result):"r"(address):"ecx","edx","memory","cc");
  return result;
}
/* Called before the authentic pool update, at the same main-loop phase as the
 * title's replay/start selection. Only mode/path and scene selection differ. */
__attribute__((used,noinline)) static void before_actor(DWORD game) {
  DWORD title,play,length;
  char line[256];
  if(bootstrap_done||!read32(0x4428d0)||!read32(0x4428a4)) return;
  title=read32(game+8);
  if(!title||read32(title)!=0x435c10) return;
  if(!title_passes) first_title_cw=x87_control_word();
  if(title_passes++<title_warmup) return;
  state_open();
  format(line,sizeof(line),"{\"type\":\"bootstrap\",\"title_passes\":%lu,\"first_title_cw\":\"%04lx\",\"warm_title_cw\":\"%04x\"}\n",title_passes,first_title_cw,x87_control_word());state_write(line);
  length=GetEnvironmentVariableA("IRISU_METSU_REPLAY",replay_path,sizeof(replay_path));
  GetEnvironmentVariableA("IRISU_METSU_LAUNCH",launch_mode,sizeof(launch_mode));
  if(!length&&launch_mode[0]!='l') return;
  if(length>=sizeof(replay_path)) ExitProcess(92);
  bootstrap_done=1;
  set_mode(selected_mode);
  if(length) { *(DWORD*)0x4428e8=length; *(DWORD*)0x4428ec=(DWORD)replay_path; }
  call0(0x412268,0); /* Actual mode profile and both ticket builders. */
  play=allocate_play();
  if(!play) ExitProcess(93);
  if(length) *(unsigned char*)(play+0x0c)=1;
  trace_scene=play;
  call0(0x40941c,play); /* Actual ScenePlay init invokes actual replay loader. */
  *(DWORD*)(title+8)=play; /* Original title selection's next-scene field. */
  call0(0x40eeec,title); /* Authentic title teardown, outer loop 0x406636. */
  *(DWORD*)(game+8)=play; /* Complete outer transition 0x40663f before pool pass. */
}
__attribute__((used,noinline)) static void after_actor(void) {
  char line[1536];
  DWORD input,frame,count,pool,i,actor,role,group,first=1;
  if(!trace_scene||!world_number) return;
  state_open();
  frame=read32(0x4428e0); input=read32(0x4428a4);
  if(world_number==observed_world&&frame==previous_frame) return;
  observed_world=world_number; previous_frame=frame;
  format(line,sizeof(line),"{\"type\":\"frame\",\"phase\":\"%s\",\"frame\":%lu,\"tick\":%lu,\"input_index\":%lu,\"input_count\":%lu,\"playback\":%u,\"seed\":%lu,\"score\":%ld,\"gauge\":%ld,\"level\":%lu,\"qualifying_clears\":%lu,\"highest_chain\":%lu,\"rng_calls\":%lu,\"active_scene_state\":%lu,\"mode\":%lu,\"x87_cw\":\"%04x\",\"recording_enabled\":%u,\"ordinary_spawn_interval\":%lu,\"ex_spawn_interval\":%lu,\"score_scale_bits\":\"%08lx%08lx\",\"finish_call_count\":%lu,\"terminal_metadata_recorded\":%s,\"recorded_final_score\":%ld,\"recorded_final_level\":%lu,\"recorded_final_chain\":%lu,\"recorded_final_clears\":%lu,\"actors\":[",
    frame?"tick":"reset",frame,frame,read32(input+0x94),read32(input+0x80),read8(input+0x98),
    read8(input+0x98)?read32(input+0x4c):read32(trace_scene+0x38),(long)read32(0x4428c0),(long)read32(0x4428e4),read32(0x4428c8),
    read32(0x4428cc),read32(0x4428c4),read32(0x442854),read32(trace_scene+0x30),read32(read32(read32(0x4428d0)+8)+0x1c),x87_control_word(),read8(input+0x48),read32(0x436a24),read32(0x436a28),read32(0x436a34),read32(0x436a30),finish_calls,finish_calls?"true":"false",(long)first_finish_score,first_finish_level,first_finish_chain,first_finish_clears);
  state_write(line);
  count=read32(0x442620);pool=read32(0x442624);
  if(count>200||!pool) ExitProcess(94);
  for(i=0;i<count;++i) {
    actor=read32(pool+4*i); role=read32(actor+0xcc);
    if(read8(actor+0x54)||(role!=4&&role!=5)) continue;
    group=read32(actor+0xd8);
    format(line,sizeof(line),"%s{\"slot\":%lu,\"role\":%lu,\"ex\":%u,\"ex_timer\":%ld,\"physics_updates\":%lu,\"color\":%ld,\"size_slot\":%lu,\"x_bits\":\"%08lx\",\"y_bits\":\"%08lx\",\"angle_bits\":\"%08lx\",\"vx_bits\":\"%08lx\",\"vy_bits\":\"%08lx\",\"size_bits\":\"%08lx\",\"age\":%lu,\"remaining_lifetime\":%ld,\"freshness\":%lu,\"grouped\":%u,\"physics_owned\":%u,\"rot_timer\":%lu,\"projectile_hits\":%lu,\"clear_pending\":%u,\"delete_marked\":%lu,\"gate_seen\":%u,\"gate_enabled\":%u,\"non_wall_contacts\":%lu,\"special\":%u,\"group_ptr\":%lu,\"group_chain\":%lu,\"group_shadow\":%lu,\"group_num\":%lu}",
      first?"":",",i,role,read8(actor+0xf0),(long)read32(actor+0xf4),read32(actor+0xf8),
      (long)read32(actor+0xbc),read32(actor+0xc0),read32(actor+0x0c),read32(actor+0x10),read32(actor+0x24),read32(actor+0x14),read32(actor+0x18),read32(actor+0x34),read32(actor+0x5c),(long)read32(actor+0x58),read32(actor+0xc8),read8(actor+0xd4),read8(actor+0xc4),read32(actor+0xd0),read32(actor+0xdc),read8(actor+0xd5),read32(actor+0x68),read8(actor+0xe4),read8(actor+0xe5),read32(actor+0xe0),read8(actor+0xc5),group,group?read32(group):0,group?read32(group+4):0,group?read32(group+8):0);
    state_write(line);first=0;
  }
  state_write("],\"pool_colors\":[");
  for(i=0;i<count;++i) { actor=read32(pool+4*i);format(line,sizeof(line),"%s%ld",i?",":"",(long)read32(actor+0xbc));state_write(line); }
  state_write("]}\n"); FlushFileBuffers(state_file);
  if((frame_limit&&frame>=frame_limit)||read32(trace_scene+0x30)==1) {
    format(line,sizeof(line),"{\"type\":\"complete\",\"frame\":%lu,\"expected_frame\":%lu,\"reason\":\"%s\"}\n",frame,frame_limit,read32(trace_scene+0x30)==1?"original-finished":"frame-limit");
    state_write(line);FlushFileBuffers(state_file);ExitProcess(0);
  }
}
__attribute__((used,noinline)) static void finish_observe(void) {
  char line[512];
  DWORD input=read32(0x4428a4);
  ++finish_calls;
  if(finish_calls==1) {
    first_finish_score=read32(0x4428c0);first_finish_level=read32(0x4428c8);
    first_finish_chain=read32(0x4428c4);first_finish_clears=read32(0x4428cc);
  }
  format(line,sizeof(line),"{\"type\":\"finish\",\"finish_call\":%lu,\"scene_frame\":%lu,\"input_index\":%lu,\"score\":%ld,\"gauge\":%ld,\"level\":%lu,\"highest_chain\":%lu,\"qualifying_clears\":%lu}\n",
    finish_calls,read32(0x4428e0),read8(input+0x98)?read32(input+0x94):read32(input+0x80),
    (long)read32(0x4428c0),(long)read32(0x4428e4),read32(0x4428c8),read32(0x4428c4),read32(0x4428cc));
  state_write(line);FlushFileBuffers(state_file);
}
__attribute__((naked)) static void finish_hook(void) {
  __asm__ volatile("pushfl;pushal;subl $528,%esp;leal 15(%esp),%eax;andl $-16,%eax;fxsave (%eax);pushl %eax;call _finish_observe;popl %eax;fxrstor (%eax);addl $528,%esp;popal;popfl;jmpl *%cs:_original_finish;");
}
__attribute__((used)) static DWORD original_finish=0x40adc4;
/* Preserve the complete original machine context around integer-only observer
 * callbacks; the authentic actor-pool call itself is not replaced by a stub. */
__attribute__((naked)) static void actor_hook(void) {
  __asm__ volatile(
    "pushfl;pushal;subl $528,%esp;leal 15(%esp),%eax;andl $-16,%eax;fxsave (%eax);pushl %eax;pushl %ebx;call _before_actor;addl $4,%esp;popl %eax;fxrstor (%eax);addl $528,%esp;popal;popfl;"
    "calll *%cs:_original_actor;"
    "pushfl;pushal;subl $528,%esp;leal 15(%esp),%eax;andl $-16,%eax;fxsave (%eax);pushl %eax;call _after_actor;popl %eax;fxrstor (%eax);addl $528,%esp;popal;popfl;retl");
}
__attribute__((used)) static DWORD original_actor=0x4031ec;
static BOOL install_hook(void) {
  unsigned char *site=(unsigned char*)0x40666d;
  DWORD old,ignored,relative;
  if(site[0]!=0xe8||read32((DWORD)site+1)!=0xffffcb7aUL||read32(0x434888)!=0x40adc4) return 0;
  if(!VirtualProtect(site,5,0x40,&old)) return 0;
  relative=(DWORD)actor_hook-(DWORD)site-5;
  *(DWORD*)(site+1)=relative;
  FlushInstructionCache((HANDLE)-1,site,5);
  VirtualProtect(site,5,old,&ignored);
  if(!VirtualProtect((void*)0x434888,4,0x40,&old)) return 0;
  *(DWORD*)0x434888=(DWORD)finish_hook;
  VirtualProtect((void*)0x434888,4,old,&ignored);
  return 1;
}
BOOL WINAPI DllMainCRTStartup(void *instance,DWORD reason,void *reserved) {
  if(reason==DLL_PROCESS_ATTACH) {
    DWORD i;
    ForwarderDllMain(instance,reason,reserved);
    GetEnvironmentVariableA("IRISU_METSU_EAGER_REAL",limit_text,sizeof(limit_text));
    eager_real=limit_text[0]=='1';
    if(eager_real) { ensure_loaded();if(!proxy_ready) return 0; }
    limit_text[0]=0;
    GetEnvironmentVariableA("IRISU_ORIGINAL_MODE",limit_text,sizeof(limit_text));
    if(limit_text[0]=='0') selected_mode=0;
    limit_text[0]=0;
    GetEnvironmentVariableA("IRISU_METSU_TITLE_WARMUP",limit_text,sizeof(limit_text));
    if(limit_text[0]) { title_warmup=0;for(i=0;limit_text[i]>='0'&&limit_text[i]<='9';++i) title_warmup=title_warmup*10+limit_text[i]-'0'; }
    limit_text[0]=0;
    GetEnvironmentVariableA("IRISU_METSU_MAX_FRAME",limit_text,sizeof(limit_text));
    for(i=0;limit_text[i]>='0'&&limit_text[i]<='9';++i) frame_limit=frame_limit*10+limit_text[i]-'0';
    return install_hook();
  }
  if(reason==DLL_PROCESS_DETACH&&state_file!=INVALID_HANDLE_VALUE) { FlushFileBuffers(state_file);CloseHandle(state_file);state_file=INVALID_HANDLE_VALUE; }
  return ForwarderDllMain(instance,reason,reserved);
}
