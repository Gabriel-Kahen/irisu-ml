#!/usr/bin/env python3
"""Experimental scalar lift of the pinned contact velocity solver (host first).

Unlike the general backend, x87 slots and stack words have compile-time indices.
Only an empty incoming x87 stack is admitted; other calls retain the interpreter.
"""
from __future__ import annotations
import argparse,collections,hashlib,importlib.util,json,re,struct
from pathlib import Path
from capstone import Cs,CS_ARCH_X86,CS_MODE_32
from capstone.x86 import X86_OP_IMM,X86_OP_MEM,X86_OP_REG
from elftools.elf.elffile import ELFFile
ROOT=Path(__file__).resolve().parents[3]
SPEC=importlib.util.spec_from_file_location('baseline_lower',ROOT/'tools/cuda-exact/translate.py')
BASE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(BASE)
ENTRY=0x7610;END=0x7b60

def effect(i,sp,depth):
 m=i.mnemonic;o=i.operands
 if m=='push':sp-=4
 elif m=='pop':sp+=4
 elif m in ('add','sub') and i.op_str.startswith('esp,'):
  assert o[1].type==X86_OP_IMM;sp+=o[1].imm*(1 if m=='add' else -1)
 if m in ('fld','fldz','fld1'):depth+=1
 elif m=='fstp' or m in ('fcomp','faddp','fmulp','fsubp','fsubrp','fdivp','fdivrp'):depth-=1
 elif m in ('fcompp','fucompp'):depth-=2
 assert 0<=depth<=8,(hex(i.address),depth)
 return sp,depth

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('host',type=Path);p.add_argument('output',type=Path);p.add_argument('--baseline-generated',type=Path,required=True);a=p.parse_args()
 assert hashlib.sha256(a.host.read_bytes()).hexdigest()==BASE.PIN
 with a.host.open('rb') as f:
  e=ELFFile(f);text=e.get_section_by_name('.text');raw=text.data();low=text['sh_addr'];rdata=e.get_section_by_name('.rdata');rlo=rdata['sh_addr'];rb=rdata.data()
 d=Cs(CS_ARCH_X86,CS_MODE_32);d.detail=True;instructions=list(d.disasm(raw[ENTRY-low:END-low],ENTRY));lookup={i.address:i for i in instructions};states={ENTRY:(0,0)};work=[ENTRY]
 while work:
  pc=work.pop();i=lookup[pc];state=effect(i,*states[pc])
  if i.mnemonic=='ret':assert state==(0,0),state;continue
  if i.mnemonic=='jmp':successors=[i.operands[0].imm]
  elif i.mnemonic.startswith('j'):successors=[i.operands[0].imm,i.address+i.size]
  else:successors=[i.address+i.size]
  for target in successors:
   assert target in lookup,(hex(pc),hex(target))
   if target in states:assert states[target]==state,(hex(target),states[target],state)
   else:states[target]=state;work.append(target)
 assert not any(i.mnemonic=='call' for i in instructions)
 offsets=set()
 for i in instructions:
  if i.address not in states:continue
  sp,depth=states[i.address]
  for op in i.operands:
   if op.type==X86_OP_MEM and i.reg_name(op.mem.base)=='esp':
    assert op.size==4 and not op.mem.index;offsets.add(sp+op.mem.disp)
  if i.mnemonic=='push':offsets.add(sp-4)
  if i.mnemonic=='pop':offsets.add(sp)
 assert all(x<0 and x%4==0 for x in offsets)
 slots={x:f's{n}' for n,x in enumerate(sorted(offsets))}
 def fp(depth,index=0):return f'f{(index-depth)&7}'
 class Lower(BASE.Lower):
  def reg(self,op):
   name=self.i.reg_name(op.reg)
   if name=='st':return fp(self.depth)
   if name.startswith('st('):return fp(self.depth,int(name[3:-1]))
   return super().reg(op)
  def read(self,op,fp=False):
   if op.type==X86_OP_MEM:
    if self.i.reg_name(op.mem.base)=='esp':
     value=slots[self.sp+op.mem.disp];return f'word_float({value})' if fp else value
    if not op.mem.base and not op.mem.index and fp and rlo<=op.mem.disp<rlo+len(rb):
     value=struct.unpack_from('<f' if op.size==4 else '<d',rb,op.mem.disp-rlo)[0]
     assert abs(value)<1e100;return float(value).hex()
   return super().read(op,fp)
  def write(self,op,value,fp=False):
   if op.type==X86_OP_MEM and self.i.reg_name(op.mem.base)=='esp':
    value=f'cuda_guest::float_word(static_cast<float>({value}))' if fp else f'std::uint32_t({value})'
    return f'{slots[self.sp+op.mem.disp]}={value};'
   return super().write(op,value,fp)
 starts=set()
 for inc in a.baseline_generated.glob('guest_region_*.inc'):
  starts.update(int(x) for x in re.findall(r'block_(\d+):',inc.read_text()))
 lines=['#pragma once','#include '+json.dumps(str(Path(__file__).with_name('support.cuh'))),'namespace irisu::cuda_direct {',
 'IRISU_GUEST_HD inline bool velocity( cuda_guest::Guest& g) {',
 'if(g.depth || g.error || g.r[4]<256 || g.r[4]>g.capacity-4) return false;',
 'const auto entry_sp=g.r[4],initial_top=g.top; unsigned exit_depth=0;',
 'if(g.r[1]<=g.capacity-12 && g.load<std::int32_t>(g.r[1]+8)<=0) {',
 'g.store<std::uint32_t>(entry_sp-192,0);g.r[0]=0;',
 'g.alu(0,entry_sp-208,208,32);g.r[4]=entry_sp+4;g.pc=g.load<std::uint32_t>(entry_sp);',
 f'g.instructions+={sum(pc in starts for pc in [ENTRY,0x761f,0x7b56,0x7b5c])};return true;}}',
 'cuda_guest::Guest state{};state.flags=g.flags;state.status=g.status;']
 for n in range(8):lines.append(f'auto r{n}=g.r[{n}];double f{n}=g.fp[(initial_top+{n})&7];')
 for off,name in slots.items():lines.append(f'auto {name}=g.load<std::uint32_t>(entry_sp{off});')
 lines.append(f'goto block_{ENTRY};')
 for i in instructions:
  if i.address not in states:continue
  sp,depth=states[i.address];lower=Lower(i);lower.sp=sp;lower.depth=depth
  lines.append(f'block_{i.address}:')
  if i.address in starts:lines.append(f'g.pc={i.address}u;++g.instructions;if(g.error){{exit_depth={depth};goto commit;}}')
  m=i.mnemonic
  if m in BASE.COND:s=f'if(state.condition({BASE.COND[m]}))goto block_{i.operands[0].imm};goto block_{i.address+i.size};'
  elif m=='jmp':s=f'goto block_{i.operands[0].imm};'
  elif m=='push':s=f'r4-=4;{slots[sp-4]}={lower.read(i.operands[0])};'
  elif m=='pop':s=lower.write(i.operands[0],slots[sp])+'r4+=4;'
  elif m=='ret':s=f'g.pc=g.load<std::uint32_t>(r4);r4+=4;exit_depth={depth};goto commit;'
  elif m=='fld':s=f'{fp(depth+1)}={lower.read(i.operands[0],True)};'
  elif m in ('fldz','fld1'):s=f'{fp(depth+1)}={0 if m=="fldz" else 1}.0;'
  else:s=lower.emit().replace('g.fpop();','')
  for n in range(8):s=s.replace(f'g.r[{n}]',f'r{n}')
  for n in range(7,-1,-1):s=s.replace(f'g.st({n})',fp(depth,n))
  s=s.replace('g.st()',fp(depth));s=s.replace('g.top',f'((initial_top-{depth})&7)')
  for method in ['alu','condition','compare_fp','carry','shift']:s=s.replace(f'g.{method}(',f'state.{method}(')
  s=s.replace('g.flags','state.flags').replace('g.status','state.status')
  s=s.replace('g.fail(unsupported_fp)',f'fault(g,cuda_guest::unsupported_fp,{fp(depth)})')
  lines.append('  '+s)
 lines.append('commit:')
 for off,name in slots.items():lines.append(f'g.store<std::uint32_t>(entry_sp{off},{name});')
 for n in range(8):lines.append(f'g.r[{n}]=r{n};g.fp[(initial_top+{n})&7]=f{n};')
 lines+=['g.top=(initial_top-exit_depth)&7;g.depth=exit_depth;g.flags=state.flags;g.status=state.status;return true;','}','}']
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'velocity.cuh').write_text('\n'.join(lines)+'\n')
 (a.output/'manifest.json').write_text(json.dumps({'source_sha256':BASE.PIN,'entry':ENTRY,'end':END,'decoded':len(instructions),'reachable':len(states),'shadow_words':len(slots),'max_depth':max(x[1] for x in states.values()),'register_scalarization':True,'fp_scalarization':True,'stack_scalarization':True,'scope':'experimental leaf only; host/GPU differential required'},indent=2)+'\n')
 print((a.output/'manifest.json').read_text())
if __name__=='__main__':main()
