#pragma once
// Experimental ELF32-only native oracle access. The original library stays
// unchanged; local MSVC methods are resolved from its pinned ELF symbol table.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <openssl/sha.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define ORACLE_PIN "ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5"
#define ORACLE_THISCALL __attribute__((thiscall))
typedef struct {int fd;void* library;uintptr_t base;unsigned char* image;size_t bytes;} NativeOracle;
static void oracle_die(const char* message){fprintf(stderr,"native-stage-oracle: %s\n",message);exit(1);}
static NativeOracle oracle_open(const char* path){
  NativeOracle result={0};struct stat info;unsigned char digest[32];char actual[65],fd_path[64];unsigned i;
  if(sizeof(void*)!=4)oracle_die("requires ELF32 build");
  result.fd=open(path,O_RDONLY);if(result.fd<0||fstat(result.fd,&info)||info.st_size<52)oracle_die("cannot open native library");
  result.bytes=info.st_size;result.image=(unsigned char*)mmap(NULL,result.bytes,PROT_READ,MAP_PRIVATE,result.fd,0);
  if(result.image==MAP_FAILED||!SHA256(result.image,result.bytes,digest))oracle_die("cannot hash native library");
  for(i=0;i<32;++i)sprintf(actual+2*i,"%02x",digest[i]);
  if(strcmp(actual,ORACLE_PIN))oracle_die("native library SHA-256 pin mismatch");
  const Elf32_Ehdr* elf=(const Elf32_Ehdr*)result.image;
  if(memcmp(elf->e_ident,ELFMAG,4)||elf->e_ident[EI_CLASS]!=ELFCLASS32||elf->e_ident[EI_DATA]!=ELFDATA2LSB||elf->e_machine!=EM_386)oracle_die("expected pinned little-endian i386 ELF");
  snprintf(fd_path,sizeof(fd_path),"/proc/self/fd/%d",result.fd);
  result.library=dlopen(fd_path,RTLD_NOW|RTLD_LOCAL);if(!result.library)oracle_die(dlerror());
  struct link_map* mapping=NULL;
  if(dlinfo(result.library,RTLD_DI_LINKMAP,&mapping)||!mapping)oracle_die("cannot attest mapped native library");
  result.base=(uintptr_t)mapping->l_addr;return result;
}
static void* oracle_symbol(const NativeOracle* oracle,const char* name,uint32_t expected){
  const Elf32_Ehdr* elf=(const Elf32_Ehdr*)oracle->image;
  if(elf->e_shoff+(size_t)elf->e_shnum*sizeof(Elf32_Shdr)>oracle->bytes)oracle_die("invalid section table");
  const Elf32_Shdr* sections=(const Elf32_Shdr*)(oracle->image+elf->e_shoff);
  for(unsigned i=0;i<elf->e_shnum;++i)if(sections[i].sh_type==SHT_SYMTAB){
    const Elf32_Shdr* symbols=sections+i;if(symbols->sh_link>=elf->e_shnum)oracle_die("invalid symbol string table");
    const Elf32_Shdr* strings=sections+symbols->sh_link;
    const Elf32_Sym* entries=(const Elf32_Sym*)(oracle->image+symbols->sh_offset);
    for(unsigned j=0;j<symbols->sh_size/sizeof(Elf32_Sym);++j){
      const Elf32_Sym* entry=entries+j;
      if(entry->st_name<strings->sh_size&&!strcmp((const char*)oracle->image+strings->sh_offset+entry->st_name,name)){
        if(ELF32_ST_TYPE(entry->st_info)!=STT_FUNC||entry->st_value!=expected)oracle_die("native stage symbol/layout mismatch");
        return (void*)(oracle->base+entry->st_value);
      }
    }
  }
  oracle_die("native stage symbol missing");return NULL;
}
static void oracle_close(NativeOracle* oracle){dlclose(oracle->library);munmap(oracle->image,oracle->bytes);close(oracle->fd);}
typedef void (ORACLE_THISCALL *NativeMethod0)(void*);
typedef void (ORACLE_THISCALL *NativeSolverCtor)(void*,void**,int,void*);
typedef unsigned char (ORACLE_THISCALL *NativePositionSolve)(void*,float);
static NativeSolverCtor oracle_solver_constructor(const NativeOracle* o){return (NativeSolverCtor)oracle_symbol(o,"??0b2ContactSolver@@QAE@PAPAVb2Contact@@HPAVb2StackAllocator@@@Z",0x6e20);}
static NativeMethod0 oracle_pre_solve(const NativeOracle* o){return (NativeMethod0)oracle_symbol(o,"?PreSolve@b2ContactSolver@@QAEXXZ",0x72f0);}
static NativeMethod0 oracle_velocity_solve(const NativeOracle* o){return (NativeMethod0)oracle_symbol(o,"?SolveVelocityConstraints@b2ContactSolver@@QAEXXZ",0x7610);}
static NativePositionSolve oracle_position_solve(const NativeOracle* o){return (NativePositionSolve)oracle_symbol(o,"?SolvePositionConstraints@b2ContactSolver@@QAE_NM@Z",0x7b60);}
static NativeMethod0 oracle_post_solve(const NativeOracle* o){return (NativeMethod0)oracle_symbol(o,"?PostSolve@b2ContactSolver@@QAEXXZ",0x7510);}
static NativeMethod0 oracle_solver_destructor(const NativeOracle* o){return (NativeMethod0)oracle_symbol(o,"??1b2ContactSolver@@QAE@XZ",0x72e0);}
static NativeMethod0 oracle_stack_constructor(const NativeOracle* o){return (NativeMethod0)oracle_symbol(o,"??0b2StackAllocator@@QAE@XZ",0x13460);}
static NativeMethod0 oracle_stack_destructor(const NativeOracle* o){return (NativeMethod0)oracle_symbol(o,"??1b2StackAllocator@@QAE@XZ",0x13480);}
// Broad-phase ABI: native MSVC callback vtable has deleting-dtor/add/remove
// slots at byte0/4/8. Build it explicitly; GCC virtual class layout differs.
typedef void (ORACLE_THISCALL *NativeBroadCtor)(void*,const void*,void*);
typedef uint16_t (ORACLE_THISCALL *NativeProxyCreate)(void*,const void*,void*);
typedef void (ORACLE_THISCALL *NativeProxyDestroy)(void*,int);
typedef void (ORACLE_THISCALL *NativeProxyMove)(void*,int,const void*);
typedef int (ORACLE_THISCALL *NativeBroadQuery)(void*,const void*,void**,int);
typedef void (ORACLE_THISCALL *NativeComputeBounds)(void*,uint16_t*,uint16_t*,const void*);
static NativeBroadCtor oracle_broad_constructor(const NativeOracle* o){return (NativeBroadCtor)oracle_symbol(o,"??0b2BroadPhase@@QAE@ABUb2AABB@@PAVb2PairCallback@@@Z",0x3580);}
static NativeMethod0 oracle_broad_destructor(const NativeOracle* o){return (NativeMethod0)oracle_symbol(o,"??1b2BroadPhase@@QAE@XZ",0x3010);}
static NativeProxyCreate oracle_proxy_create(const NativeOracle* o){return (NativeProxyCreate)oracle_symbol(o,"?CreateProxy@b2BroadPhase@@QAEGABUb2AABB@@PAX@Z",0x37c0);}
static NativeProxyDestroy oracle_proxy_destroy(const NativeOracle* o){return (NativeProxyDestroy)oracle_symbol(o,"?DestroyProxy@b2BroadPhase@@QAEXH@Z",0x39f0);}
static NativeProxyMove oracle_proxy_move(const NativeOracle* o){return (NativeProxyMove)oracle_symbol(o,"?MoveProxy@b2BroadPhase@@QAEXHABUb2AABB@@@Z",0x3c30);}
static NativeMethod0 oracle_broad_commit(const NativeOracle* o){return (NativeMethod0)oracle_symbol(o,"?Commit@b2BroadPhase@@QAEXXZ",0x3390);}
static NativeBroadQuery oracle_broad_query(const NativeOracle* o){return (NativeBroadQuery)oracle_symbol(o,"?Query@b2BroadPhase@@QAEHABUb2AABB@@PAPAXH@Z",0x4100);}
static NativeComputeBounds oracle_compute_bounds(const NativeOracle* o){return (NativeComputeBounds)oracle_symbol(o,"?ComputeBounds@b2BroadPhase@@AAEXPAG0ABUb2AABB@@@Z",0x3670);}
