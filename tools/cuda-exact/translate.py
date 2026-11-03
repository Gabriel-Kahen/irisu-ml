#!/usr/bin/env python3
"""Lower the pinned i386 MSVC Box2D image to bounded C++/CUDA basic blocks.

This is an experimental execution engine, not an exact-runtime attestation.
The generated manifest enumerates unsupported instructions; executing one
invalidates its lane. Never accept its output without CPU differential gates.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from capstone import Cs, CS_ARCH_X86, CS_MODE_32
from capstone.x86 import X86_OP_IMM, X86_OP_MEM, X86_OP_REG
from elftools.elf.elffile import ELFFile

PIN = 'ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5'
REGS = ['eax', 'ecx', 'edx', 'ebx', 'esp', 'ebp', 'esi', 'edi']
COND = dict(je=0,jne=1,jb=2,jae=3,jbe=4,ja=5,jl=6,jge=7,jle=8,jg=9,jp=10,jnp=11,js=12,jns=13)
TYPES = {1:'std::uint8_t',2:'std::uint16_t',4:'std::uint32_t',8:'std::uint64_t'}


def write_generated(path: Path, content: str | bytes | bytearray) -> None:
    data = content.encode() if isinstance(content, str) else content
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


class Lower:
    def __init__(self, ins, flags_live=True):
        self.i=ins
        self.o=ins.operands
        self.flags_live=flags_live

    def reg(self, op):
        name=self.i.reg_name(op.reg)
        if name.startswith('st('): return f'g.st({name[3:-1]})'
        if name=='st': return 'g.st()'
        if name in REGS: return f'g.r[{REGS.index(name)}]'
        base={'ax':'eax','cx':'ecx','dx':'edx','bx':'ebx','sp':'esp','bp':'ebp','si':'esi','di':'edi',
              'al':'eax','cl':'ecx','dl':'edx','bl':'ebx','ah':'eax','ch':'ecx','dh':'edx','bh':'ebx'}[name]
        r=f'g.r[{REGS.index(base)}]'
        return f'(({r} >> 8)&255)' if name.endswith('h') else f'({r}&{65535 if op.size==2 else 255}u)'

    def addr(self, op):
        m=op.mem
        parts=[str(m.disp)]
        if m.base: parts.append(f'g.r[{REGS.index(self.i.reg_name(m.base))}]')
        if m.index: parts.append(f'g.r[{REGS.index(self.i.reg_name(m.index))}]*{m.scale}u')
        if m.segment and self.i.mnemonic!='lea': raise ValueError('segment override')
        return 'std::uint32_t('+ '+'.join(parts)+')'

    def read(self, op, fp=False):
        if op.type==X86_OP_REG: return self.reg(op)
        if op.type==X86_OP_IMM: return f'{op.imm&0xffffffff}u'
        if op.type==X86_OP_MEM:
            t=({4:'float',8:'double'} if fp else TYPES)[op.size]
            return f'g.load<{t}>({self.addr(op)})'
        raise ValueError('operand')

    def write(self, op, value, fp=False):
        if op.type==X86_OP_MEM:
            t=({4:'float',8:'double'} if fp else TYPES)[op.size]
            return f'g.store<{t}>({self.addr(op)},({t})({value}));'
        name=self.i.reg_name(op.reg)
        if op.size==4 or name.startswith('st'): return f'{self.reg(op)}={value};'
        base={'ax':'eax','cx':'ecx','dx':'edx','bx':'ebx','sp':'esp','bp':'ebp','si':'esi','di':'edi',
              'al':'eax','cl':'ecx','dl':'edx','bl':'ebx','ah':'eax','ch':'ecx','dh':'edx','bh':'ebx'}[name]
        r=f'g.r[{REGS.index(base)}]'
        shift=8 if name.endswith('h') else 0
        mask=(1<<(8*op.size))-1
        return f'{r}=({r}&~{mask<<shift}u)|((std::uint32_t({value})&{mask}u)<<{shift});'

    def emit(self):
        m=self.i.mnemonic; o=self.o; rd=self.read; wr=self.write
        nxt=self.i.address+self.i.size
        if m in ('nop','endbr32'): return ''
        if m in ('mov','movzx'): return wr(o[0],rd(o[1]))
        if m=='lea': return wr(o[0],self.addr(o[1]))
        if m=='push': return f'g.push({rd(o[0])});'
        if m=='pop': return wr(o[0],'g.pop()')
        if m=='popfd': return 'g.flags=g.pop();'
        if m=='leave': return 'g.r[4]=g.r[5];g.r[5]=g.pop();'
        if m=='ret':
            return 'g.pc=g.pop();'+(f'g.r[4]+={o[0].imm};' if o else '')+'continue;'
        if m=='call': return f'g.push({nxt}u);g.pc={rd(o[0])};continue;'
        if m=='jmp': return f'g.pc={rd(o[0])};continue;'
        if m in COND: return f'g.pc=g.condition({COND[m]})?{rd(o[0])}:{nxt}u;continue;'
        if m.startswith('set'):
            return wr(o[0],f'g.condition({COND["j"+m[3:]]})')
        if not self.flags_live:
            if m in ('cmp','test'): return ''
            if m in ('add','sub','and','or','xor'):
                symbol=dict(add='+',sub='-',**{'and':'&','or':'|','xor':'^'})[m]
                return wr(o[0],f'({rd(o[0])}){symbol}({rd(o[1])})')
            if m in ('inc','dec'): return wr(o[0],f'({rd(o[0])}){"+" if m=="inc" else "-"}1u')
            if m=='neg': return wr(o[0],f'0u-({rd(o[0])})')
        if m in ('cmp','test'): return f'g.alu({1 if m=="cmp" else 2},{rd(o[0])},{rd(o[1])},{o[0].size*8});'
        if m in ('add','sub','and','or','xor'):
            return wr(o[0],f'g.alu({dict(add=0,sub=1,**{"and":2,"or":3,"xor":4})[m]},{rd(o[0])},{rd(o[1])},{o[0].size*8})')
        if m in ('inc','dec'):
            return '{auto c=g.flags&1;'+wr(o[0],f'g.alu({0 if m=="inc" else 1},{rd(o[0])},1,{o[0].size*8})')+'g.flags=(g.flags&~1u)|c;}'
        if m=='neg': return wr(o[0],f'g.alu(1,0,{rd(o[0])},{o[0].size*8})')
        if m=='not': return wr(o[0],f'~({rd(o[0])})')
        if m in ('shl','shr','sar'): return wr(o[0],f'g.shift({dict(shl=0,shr=1,sar=2)[m]},{rd(o[0])},{rd(o[1])},{o[0].size*8})')
        if m in ('adc','sbb'):
            return wr(o[0],f'g.carry({0 if m=="adc" else 1},{rd(o[0])},{rd(o[1])},{o[0].size*8})')
        if m=='xchg': return '{auto v='+rd(o[0])+';'+wr(o[0],rd(o[1]))+wr(o[1],'v')+'}'
        if m=='imul':
            a,b=(o[0],o[1]) if len(o)==2 else (o[1],o[2])
            return '{auto v=std::int64_t(std::int32_t('+rd(a)+'))*std::int32_t('+rd(b)+');'+wr(o[0],'std::uint32_t(v)')+'bool ov=v!=std::int64_t(std::int32_t(v));g.flags=(g.flags&~2049u)|(ov?2049u:0);}'
        if m=='cdq': return 'g.r[2]=std::int32_t(g.r[0])<0?0xffffffffu:0;'
        if m=='idiv': return '{auto d=std::int32_t('+rd(o[0])+');auto n=std::int64_t((std::uint64_t(g.r[2])<<32)|g.r[0]);if(!d||(n==INT64_MIN&&d==-1)){g.fail(invalid_division);break;}auto q=n/d;if(q<INT32_MIN||q>INT32_MAX){g.fail(invalid_division);break;}g.r[0]=q;g.r[2]=n%d;}'
        if m=='fld': return f'g.fpush({rd(o[0],True)});'
        if m=='fldz': return 'g.fpush(0.0);'
        if m=='fld1': return 'g.fpush(1.0);'
        if m in ('fst','fstp'): return wr(o[0],'g.st()',True)+('g.fpop();' if m=='fstp' else '')
        if m=='fxch': return '{auto v=g.st();g.st()='+rd(o[-1])+';'+wr(o[-1],'v')+'}'
        if m=='fchs': return 'g.st()=-g.st();'
        if m=='fabs': return 'g.st()=fabs(g.st());'
        if m.startswith(('fadd','fsub','fmul','fdiv')):
            operation=m[:4]; pop=m.endswith('p'); rev='r' in m[4:]
            if not o: a,b='g.st(1)','g.st()'; dst='g.st(1)'
            elif len(o)==1 and pop: a,b=rd(o[0],True),'g.st()';dst=a
            elif len(o)==1: a,b='g.st()',rd(o[0],True);dst='g.st()'
            else:a,b=rd(o[0],True),rd(o[1],True);dst=a
            if rev:a,b=b,a
            helper=dict(fadd='add',fsub='sub',fmul='mul',fdiv='div')[operation]
            return f'{{auto v=irisu::cuda_exact::{helper}({a},{b});if(!v.supported)g.fail(unsupported_fp);{dst}=v.value;}}'+('g.fpop();' if pop else '')
        if m in ('fcom','fcomp','fcompp','fucompp'):
            b=rd(o[0],True) if o else 'g.st(1)'
            return f'g.compare_fp(g.st(),{b});'+('g.fpop();' if m.endswith('p') else '')+('g.fpop();' if m.endswith('pp') else '')
        if m=='fnstsw': return wr(o[0],'g.status | (g.top<<11)')
        if m=='fnstcw': return wr(o[0],'g.control')
        if m=='fldcw': return f'g.control={rd(o[0])};if((g.control&0x300)!=0x200)g.fail(unsupported_fp);'
        if m=='fistp':
            return '{auto v=g.st();auto rm=(g.control>>10)&3;auto z=rm==3?trunc(v):rm==1?floor(v):rm==2?ceil(v):nearbyint(v);if(!(z>=INT32_MIN&&z<=INT32_MAX))g.fail(unsupported_fp);'+wr(o[0],'std::int32_t(z)')+'g.fpop();}'
        if m=='fsqrt': return '{auto v=irisu::cuda_exact::square_root(g.st());if(!v.supported)g.fail(unsupported_fp);g.st()=v.value;}'
        if m in ('fsin','fcos','fsincos'):
            s='{auto v=irisu::cuda_exact::sincos_reloaded_float(g.st());if(!v.supported)g.fail(unsupported_fp);'
            return s+('g.st()=v.sine;g.fpush(v.cosine);}' if m=='fsincos' else f'g.st()=v.{"sine" if m=="fsin" else "cosine"};}}')
        if m in ('rep stosd','rep movsd','lodsb','stosd'):
            if m=='lodsb': return wr(type('Op',(),dict(type=X86_OP_REG,reg=2,size=1))(),'0') if False else 'g.r[0]=(g.r[0]&~255u)|g.load<std::uint8_t>(g.r[6]);g.r[6]+=(g.flags&1024)?-1:1;'
            direction='((g.flags&1024)?-4:4)'
            s='g.store<std::uint32_t>(g.r[7],'+('g.r[0]' if 'stos' in m else 'g.load<std::uint32_t>(g.r[6])')+');g.r[7]+='+direction+';'
            if 'movs' in m:s+='g.r[6]+='+direction+';'
            return 'while(g.r[1]&&!g.error){'+s+'--g.r[1];}' if m.startswith('rep') else s
        raise ValueError(m)


def generate(source: Path, output: Path, region_shift: int = 10):
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=PIN: raise SystemExit(f'input host hash is {digest}; expected {PIN}')
    output.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as stream:
        elf=ELFFile(stream)
        segments=[s for s in elf.iter_segments() if s['p_type']=='PT_LOAD']
        size=max(s['p_vaddr']+s['p_memsz'] for s in segments)
        image=bytearray((size+4095)&~4095)
        for s in segments:image[s['p_vaddr']:s['p_vaddr']+s['p_filesz']]=s.data()
        symbol_list=list(elf.get_section_by_name('.symtab').iter_symbols())
        symbols={s.name:s['st_value'] for s in symbol_list if s.name}
        dynamic=elf.get_section_by_name('.dynsym')
        imports={}
        plt=elf.get_section_by_name('.plt')
        for section in elf.iter_sections():
            if section['sh_type'] not in ('SHT_REL','SHT_RELA'):continue
            for rel_index,rel in enumerate(section.iter_relocations()):
                kind=rel['r_info_type'];at=rel['r_offset']
                if kind==8:continue # base-zero RELATIVE already contains the addend
                sym=dynamic.get_symbol(rel['r_info_sym'])
                if sym['st_shndx']=='SHN_UNDEF':
                    if sym.name in ('malloc','free','memcpy','memmove','memset','abort'):
                        marker=0xff000000+len(imports)*16
                        imports[marker]=sym.name
                        if section.name=='.rel.plt':imports[plt['sh_addr']+16*(rel_index+1)]=sym.name
                        value=(int.from_bytes(image[at:at+4],'little')+marker-at) if kind==2 else marker
                        image[at:at+4]=(value&0xffffffff).to_bytes(4,'little')
                    elif sym['st_info']['bind']=='STB_WEAK':image[at:at+4]=bytes(4)
                    else:raise ValueError(f'unhandled import {sym.name}')
                else:raise ValueError(f'unhandled relocation {kind} {sym.name}')
        text=elf.get_section_by_name('.text'); decoder=Cs(CS_ARCH_X86,CS_MODE_32);decoder.detail=True
        low,high=text['sh_addr'],text['sh_addr']+text['sh_size']
        # Decode from symbol entries and control-flow edges. Linear sweep would
        # misinterpret inline jump tables as instructions and lose alignment.
        pending=[s['st_value'] for s in symbol_list if s['st_info']['type']=='STT_FUNC' and low<=s['st_value']<high]
        rdata=elf.get_section_by_name('.rdata')
        for at in range(rdata['sh_addr'],rdata['sh_addr']+rdata['sh_size']-3,4):
            value=int.from_bytes(image[at:at+4],'little')
            if low<=value<high:pending.append(value)
        entries=set(pending)
        decoded={}
        while pending:
            at=pending.pop()
            while low<=at<high and at not in decoded:
                i=next(decoder.disasm(bytes(image[at:at+16]),at,count=1),None)
                if i is None:break
                decoded[at]=i
                if i.mnemonic.startswith('j') or i.mnemonic=='call':
                    if i.operands and i.operands[0].type==X86_OP_IMM:pending.append(i.operands[0].imm)
                    elif (i.mnemonic=='jmp' and i.operands and i.operands[0].type==X86_OP_MEM
                          and not i.operands[0].mem.base and i.operands[0].mem.scale==4):
                        table=i.operands[0].mem.disp
                        for offset in range(0,4096,4):
                            pos=table+offset
                            if pos<0 or pos+4>len(image):break
                            target=int.from_bytes(image[pos:pos+4],'little')
                            if not low<=target<high:break
                            pending.append(target);entries.add(target)
                    if i.mnemonic=='jmp':break
                if i.mnemonic=='ret':break
                at+=i.size
        instructions=sorted(decoded.values(),key=lambda i:i.address)
        starts={i.address for i in instructions if i.mnemonic.startswith('j') or i.mnemonic in ('call','ret')}
        starts.update(entries)
        starts.update(v for v in symbols.values() if text['sh_addr']<=v<text['sh_addr']+text['sh_size'])
        for i in instructions:
            if i.mnemonic.startswith('j') or i.mnemonic in ('call','ret'):
                starts.add(i.address+i.size)
                if i.operands and i.operands[0].type==X86_OP_IMM:starts.add(i.operands[0].imm)
        # Flags are materialized only when some reachable consumer can read
        # them before the next full writer. Calls/indirect edges stay conservative.
        flag_writers={'add','sub','and','or','xor','cmp','test','neg','popfd'}
        live={i.address:True for i in instructions}
        live_after={}
        changed=True
        while changed:
            changed=False
            for i in reversed(instructions):
                m=i.mnemonic;nxt=i.address+i.size
                boundary=m in ('call','ret') or (m=='jmp' and i.operands[0].type!=X86_OP_IMM)
                if boundary: after=True
                else:
                    successors=([i.operands[0].imm] if m=='jmp' else
                                [nxt,i.operands[0].imm] if m in COND else [nxt])
                    after=any(live.get(target,True) for target in successors)
                before=(m in COND or m.startswith('set') or m in ('adc','sbb','inc','dec') or
                        (after and m not in flag_writers))
                live_after[i.address]=after
                if live[i.address]!=before: live[i.address]=before;changed=True
        # Basic-block entries support direct/virtual calls and recovered jump
        # tables; contiguous instructions fall through without dispatch overhead.
        code=['// Generated from the pinned exact host; do not edit.',
              '#ifdef __CUDACC__', '#define IRISU_GUEST_NOINLINE __noinline__',
              '#else', '#define IRISU_GUEST_NOINLINE __attribute__((noinline))', '#endif',
              '#if defined(IRISU_GUEST_SEPARATE_REGIONS)', '#include "guest_region_declarations.inc"',
              '#else', '#define IRISU_GUEST_REGION_INLINE inline']
        unsupported=[]
        region=None
        regions=[]
        grouped={}
        for i in instructions: grouped.setdefault(i.address>>region_shift,[]).append(i)
        region_sources=[]
        for region, group in grouped.items():
            region_start=len(code)
            regions.append(region);starts.add(group[0].address)
            code.extend([f'IRISU_GUEST_HD IRISU_GUEST_NOINLINE IRISU_GUEST_REGION_INLINE void region_{region}(Guest& g, std::uint64_t& budget) {{',
                         'dispatch:',f'if(g.error || (g.pc>>{region_shift})!={region}u) return;', 'switch(g.pc) {'])
            code.extend(f'case {i.address}u: goto block_{i.address};' for i in group if i.address in starts)
            code.extend(['default:g.fail(bad_pc);return;','}'])
            def edge(target):
                if target>>region_shift==region and target in decoded:
                    return f'goto block_{target};'
                return f'g.pc={target}u;return;'
            for i in group:
                if i.address in starts:
                    code.append(f'block_{i.address}: g.pc={i.address}u; if(g.error)return;if(!budget){{g.fail(budget_exhausted);return;}}--budget;++g.instructions;')
                try:
                    if i.mnemonic in COND:
                        s=f'if(g.condition({COND[i.mnemonic]})){{{edge(i.operands[0].imm)}}}'+edge(i.address+i.size)
                    elif i.mnemonic=='jmp' and i.operands[0].type==X86_OP_IMM:
                        s=edge(i.operands[0].imm)
                    elif i.mnemonic=='call' and i.operands[0].type==X86_OP_IMM:
                        s=f'g.push({i.address+i.size}u);'+edge(i.operands[0].imm)
                    else:
                        s=Lower(i,live_after[i.address]).emit().replace('continue;','goto dispatch;')
                except (ValueError,KeyError,IndexError) as exc:
                    unsupported.append(dict(pc=i.address,mnemonic=i.mnemonic,operands=i.op_str,reason=str(exc)))
                    s='g.fail(unsupported_instruction);return;'
                # Error paths generated for division exit the region, not a
                # surrounding switch (the dispatcher is no longer a loop).
                s=s.replace('g.fail(invalid_division);break;','g.fail(invalid_division);return;')
                if s:code.append('  '+s)
            next_address=group[-1].address+group[-1].size
            code.extend([f'g.pc={next_address}u;return;','}'])
            region_name=f'guest_region_{region}'
            write_generated(output/f'{region_name}.inc', '\n'.join(code[region_start:])+'\n')
            code[region_start:]=[f'#include "{region_name}.inc"']
            root=Path(__file__).resolve().parents[2]
            includes=[root/'clone/cuda/guest.hpp',root/'tools/cuda-exact/x87_numeric.cuh']
            unit='\n'.join('#include '+json.dumps(str(p)) for p in includes)+'\n'
            unit+='#ifdef __CUDACC__\n#define IRISU_GUEST_NOINLINE __noinline__\n#else\n#define IRISU_GUEST_NOINLINE __attribute__((noinline))\n#endif\n'
            unit+='#define IRISU_GUEST_REGION_INLINE\nnamespace irisu::cuda_guest {\n'
            unit+=f'#include "{region_name}.inc"\n}}\n'
            write_generated(output/f'{region_name}.cu', unit)
            region_sources.append(f'{region_name}.cu')
        write_generated(output/'guest_region_declarations.inc', '\n'.join(
            f'IRISU_GUEST_HD IRISU_GUEST_NOINLINE void region_{r}(Guest&, std::uint64_t&);' for r in regions)+'\n')
        write_generated(output/'region_sources.cmake', 'set(IRISU_CUDA_GENERATED_REGION_SOURCES\n'+
            '\n'.join(f'  "${{CMAKE_CURRENT_LIST_DIR}}/{p}"' for p in region_sources)+'\n)\n')
        code.append('#endif')
        code.extend(['IRISU_GUEST_HD IRISU_GUEST_NOINLINE inline void execute(Guest& g,std::uint64_t budget=100000000) {',
                     f'while(g.pc&&!g.error&&budget){{if(callback(g)){{--budget;++g.instructions;continue;}}switch(g.pc>>{region_shift}){{'])
        code.extend(f'case {r}:region_{r}(g,budget);break;' for r in regions)
        code.extend(['default:g.fail(bad_pc);break;','}}if(g.pc&&!g.error)g.fail(budget_exhausted);','}'])
        write_generated(output/'guest_blocks.inc', '\n'.join(code)+'\n')
        write_generated(output/'guest_image.bin', image)
        image_sha=hashlib.sha256(image).hexdigest()
        identity='#pragma once\n#include <cstdint>\nnamespace irisu::cuda_guest {\n'
        identity+=f'inline constexpr std::uint32_t image_bytes={len(image)}u;\n'
        identity+='inline constexpr unsigned char image_sha256[32]={'+','.join(
            f'0x{x:02x}' for x in bytes.fromhex(image_sha))+'};\n'
        identity+=f'inline constexpr const char image_sha256_hex[]="{image_sha}";\n'
        identity+=f'inline constexpr const char source_sha256[]="{digest}";\n}}\n'
        write_generated(output/'guest_image_identity.inc', identity)
        public={k:v for k,v in symbols.items() if k.startswith('msvc_b2d_world_')}
        manifest=dict(version=1,region_shift=region_shift,region_sources=region_sources,source_sha256=digest,image_bytes=len(image),image_sha256=image_sha,exports=public,imports=imports,
                      instructions=len(instructions),dead_flag_writes=sum(not live_after[i.address] and i.mnemonic in flag_writers for i in instructions),mnemonics=dict(Counter(i.mnemonic for i in instructions)),unsupported=unsupported,
                      attestation='experimental; CPU comparison required; no promotion authority')
        write_generated(output/'manifest.json', json.dumps(manifest,indent=2)+'\n')
        # Export addresses allow callers to build ABI stacks without x86 decoding.
        write_generated(output/'guest_exports.inc', '\n'.join(f'constexpr std::uint32_t {k}={v}u;' for k,v in public.items())+'\n')
        callbacks=['IRISU_GUEST_HD inline bool callback(Guest& g) { switch(g.pc) {']
        for address,name in imports.items():
            pre=f'case {address}u:{{auto a=g.load<std::uint32_t>(g.r[4]+4);'
            if name=='malloc':body='g.r[0]=g.alloc(a);'
            elif name=='free':body='g.release(a);'
            elif name=='abort':body='g.fail(aborted);'
            else:
                body='auto b=g.load<std::uint32_t>(g.r[4]+8);auto n=g.load<std::uint32_t>(g.r[4]+12);'
                if name=='memset':body+='g.fill(a,b,n);'
                else:body+='g.copy(a,b,n);'
                body+='g.r[0]=a;'
            callbacks.append(pre+body+'g.pc=g.pop();return true;}')
        callbacks+=['default:return false;} }']
        write_generated(output/'guest_callbacks.inc', '\n'.join(callbacks)+'\n')
        print(json.dumps(dict(image_bytes=len(image),instructions=len(instructions),unsupported=len(unsupported))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--region-shift', type=int, choices=range(8,15), default=10, help='log2 guest-code bytes per optimizer function')
    args=parser.parse_args()
    generate(args.source,args.output,args.region_shift)
