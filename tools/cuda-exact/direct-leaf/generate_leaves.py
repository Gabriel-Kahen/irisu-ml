#!/usr/bin/env python3
"""Conservative experimental scalar lowering of pinned, call-free x86 functions.

Generated leaves retain exact per-block budget/error checks and scalarize only
fixed ESP-relative four-byte scratch words. Escaped stack addresses are rejected.
A successful static screen is admission for differential tests, not attestation.
"""
from __future__ import annotations
import argparse,collections,hashlib,importlib.util,json,re,struct
from pathlib import Path
from capstone import Cs,CS_ARCH_X86,CS_MODE_32,CS_AC_WRITE
from capstone.x86 import X86_OP_IMM,X86_OP_MEM,X86_OP_REG
from elftools.elf.elffile import ELFFile
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('velocity_generator',Path(__file__).with_name('generate_velocity.py'))
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V);BASE=V.BASE

class Rejected(ValueError):pass

def require(condition,reason):
 if not condition:raise Rejected(reason)

def analyze(instructions,entry):
 lookup={i.address:i for i in instructions};states={entry:(0,0)};work=[entry];returns=[]
 while work:
  pc=work.pop();require(pc in lookup,'control flow outside function');i=lookup[pc];m=i.mnemonic;o=i.operands;sp,depth=states[pc]
  require(m!='call','contains call')
  require(m not in ('leave','popfd','idiv','fldcw','fnstcw','fsincos','rep stosd','rep movsd','lodsb','stosd'),'unsupported scalar form: '+m)
  for op in o:
   if op.type==X86_OP_MEM:
    require(not op.mem.segment,'segment override')
    if m=='lea':require(i.reg_name(op.mem.base) not in ('esp','ebp') and i.reg_name(op.mem.index) not in ('esp','ebp'),'escaped stack address')
    if i.reg_name(op.mem.base)=='esp':
     require(m!='lea','escaped stack address');require(op.size==4 and not op.mem.index,'non-word/indexed stack operand')
     if sp+op.mem.disp>=0:require(not op.access&CS_AC_WRITE,'write to caller stack')
   if op.type==X86_OP_REG and i.reg_name(op.reg)=='esp':
    require(m in ('add','sub') and op is o[0] and len(o)>1 and o[1].type==X86_OP_IMM,'escaped or nonconstant ESP')
  try:after=V.effect(i,sp,depth)
  except AssertionError:raise Rejected('inconsistent/unsupported x87 depth')
  if m=='ret':require(after[0]==0 and after[1] in (0,1),'unbalanced return stack');returns.append(after[1]);continue
  if m=='jmp' or m in BASE.COND:
   require(o[0].type==X86_OP_IMM,'indirect branch');targets=[o[0].imm]+([] if m=='jmp' else [pc+i.size])
  else:targets=[pc+i.size]
  for target in targets:
   require(target in lookup,'control flow outside function')
   if target in states:require(states[target]==after,'inconsistent ESP/x87 merge')
   else:states[target]=after;work.append(target)
 require(bool(returns),'no return');require(len(set(returns))==1,'variable return FP depth')
 offsets=set()
 for pc,(sp,depth) in states.items():
  i=lookup[pc]
  for op in i.operands:
   if op.type==X86_OP_MEM and i.reg_name(op.mem.base)=='esp' and sp+op.mem.disp<0:offsets.add(sp+op.mem.disp)
  if i.mnemonic=='push':offsets.add(sp-4)
  if i.mnemonic=='pop':require(sp<0,'pop from caller stack');offsets.add(sp)
 require(all(x%4==0 for x in offsets),'unaligned shadow stack')
 # Match the baseline dead-flag pass, including conservative return boundaries.
 live={pc:True for pc in states};live_after={};changed=True
 while changed:
  changed=False
  for pc in sorted(states,reverse=True):
   i=lookup[pc];m=i.mnemonic;nxt=pc+i.size
   targets=[] if m=='ret' else [i.operands[0].imm] if m=='jmp' else [nxt,i.operands[0].imm] if m in BASE.COND else [nxt]
   after=m=='ret' or any(live.get(t,True) for t in targets)
   before=m in BASE.COND or m.startswith('set') or m in ('adc','sbb','inc','dec') or (after and m not in {'add','sub','and','or','xor','cmp','test','neg','popfd'})
   live_after[pc]=after
   if live[pc]!=before:live[pc]=before;changed=True
 return states,offsets,returns[0],live_after

def lower(instructions,entry,states,offsets,live_after,starts,rlo,rb):
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
    if self.i.reg_name(op.mem.base)=='esp' and self.sp+op.mem.disp in slots:
     v=slots[self.sp+op.mem.disp];return f'word_float({v})' if fp else v
    if not op.mem.base and not op.mem.index and fp and rlo<=op.mem.disp<rlo+len(rb):
     value=struct.unpack_from('<f' if op.size==4 else '<d',rb,op.mem.disp-rlo)[0]
     require(abs(value)<1e100,'nonfinite constant');return float(value).hex()
   return super().read(op,fp)
  def write(self,op,value,fp=False):
   if op.type==X86_OP_MEM and self.i.reg_name(op.mem.base)=='esp' and self.sp+op.mem.disp in slots:
    value=f'cuda_guest::float_word(static_cast<float>({value}))' if fp else f'std::uint32_t({value})'
    return f'{slots[self.sp+op.mem.disp]}={value};'
   return super().write(op,value,fp)
 lines=['#pragma once','#include '+json.dumps(str(Path(__file__).with_name('support.cuh'))),'namespace irisu::cuda_direct {',f'IRISU_GUEST_HD inline bool leaf_{entry}(cuda_guest::Guest& g,std::uint64_t& budget) {{',f'if(g.depth||g.error||g.r[4]<{-min(offsets,default=0)}u||g.capacity<4||g.r[4]>g.capacity-4)return false;', 'const auto entry_sp=g.r[4],initial_top=g.top;unsigned exit_depth=0;auto remaining=budget;std::uint64_t executed=0;', 'cuda_guest::Guest state{};state.flags=g.flags;state.status=g.status;']
 for n in range(8):lines.append(f'auto r{n}=g.r[{n}];double f{n}=g.fp[(initial_top+{n})&7];')
 for off,name in slots.items():lines.append(f'auto {name}=g.load<std::uint32_t>(entry_sp{off});')
 lines.append(f'goto block_{entry};')
 for i in instructions:
  if i.address not in states:continue
  sp,depth=states[i.address];m=i.mnemonic;lo=Lower(i,live_after[i.address]);lo.sp=sp;lo.depth=depth
  lines.append(f'block_{i.address}:')
  if i.address in starts:lines.append(f'g.pc={i.address}u;if(g.error){{exit_depth={depth};goto commit;}}if(!remaining){{fault(g,cuda_guest::budget_exhausted,{fp(depth)});exit_depth={depth};goto commit;}}--remaining;++executed;')
  if m in BASE.COND:s=f'if(state.condition({BASE.COND[m]}))goto block_{i.operands[0].imm};goto block_{i.address+i.size};'
  elif m=='jmp':s=f'goto block_{i.operands[0].imm};'
  elif m=='push':s=f'r4-=4;{slots[sp-4]}={lo.read(i.operands[0])};'
  elif m=='pop':s=lo.write(i.operands[0],slots[sp])+'r4+=4;'
  elif m=='ret':s=f'g.pc=g.load<std::uint32_t>(r4);r4+={4+(i.operands[0].imm if i.operands else 0)};exit_depth={depth};goto commit;'
  elif m=='fld':s=f'{fp(depth+1)}={lo.read(i.operands[0],True)};'
  elif m in ('fldz','fld1'):s=f'{fp(depth+1)}={0 if m=="fldz" else 1}.0;'
  else:s=lo.emit().replace('g.fpop();','')
  require('g.fpush(' not in s and 'break;' not in s and 'continue;' not in s,'unsupported scalar emission: '+m)
  for n in range(8):s=s.replace(f'g.r[{n}]',f'r{n}')
  for n in range(7,-1,-1):s=s.replace(f'g.st({n})',fp(depth,n))
  s=s.replace('g.st()',fp(depth)).replace('g.top',f'((initial_top-{depth})&7)')
  for method in ['alu','condition','compare_fp','carry','shift']:s=s.replace(f'g.{method}(',f'state.{method}(')
  s=s.replace('g.flags','state.flags').replace('g.status','state.status')
  s=re.sub(r'g.fail\((\w+)\)',lambda m:f'fault(g,cuda_guest::{m[1]},{fp(depth)})',s)
  lines.append('  '+s)
 lines.append('commit:budget=remaining;g.instructions+=executed;')
 for off,name in slots.items():lines.append(f'g.store<std::uint32_t>(entry_sp{off},{name});')
 for n in range(8):lines.append(f'g.r[{n}]=r{n};g.fp[(initial_top+{n})&7]=f{n};')
 lines+=['g.top=(initial_top-exit_depth)&7;g.depth=exit_depth;g.flags=state.flags;g.status=state.status;return true;','}','}']
 return '\n'.join(lines)+'\n'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('host',type=Path);p.add_argument('output',type=Path);p.add_argument('--baseline-generated',type=Path,required=True);p.add_argument('--entry',type=lambda x:int(x,0));p.add_argument('--end',type=lambda x:int(x,0));a=p.parse_args();require((a.entry is None)==(a.end is None),'entry and end must be paired')
 require(hashlib.sha256(a.host.read_bytes()).hexdigest()==BASE.PIN,'host SHA pin')
 with a.host.open('rb') as f:
  elf=ELFFile(f);text=elf.get_section_by_name('.text');raw=text.data();low=text['sh_addr'];high=low+len(raw);rdata=elf.get_section_by_name('.rdata');rlo=rdata['sh_addr'];rb=rdata.data();symbols=list(elf.get_section_by_name('.symtab').iter_symbols());names=collections.defaultdict(list)
  for s in symbols:
   if s['st_info']['type']=='STT_FUNC' and low<=s['st_value']<high:names[s['st_value']].append(s.name)
 starts=set()
 for file in a.baseline_generated.glob('guest_region_*.inc'):starts.update(int(x) for x in re.findall(r'block_(\d+):',file.read_text()))
 entries=sorted(set(names)|{low});bounds=list(zip(entries,entries[1:]+[high]));bounds=[(a.entry,a.end)] if a.entry is not None else bounds
 d=Cs(CS_ARCH_X86,CS_MODE_32);d.detail=True;a.output.mkdir(parents=True,exist_ok=True);rows=[];headers=[]
 for entry,end in bounds:
  row={'entry':entry,'end':end,'symbols':names[entry]}
  try:
   instructions=list(d.disasm(raw[entry-low:end-low],entry));states,offsets,return_depth,live=analyze(instructions,entry);code=lower(instructions,entry,states,offsets,live,starts,rlo,rb)
   file=a.output/f'leaf_{entry}.cuh';file.write_text(code);headers.append(file);row.update(admitted=True,reachable_instructions=len(states),shadow_words=len(offsets),return_fp_depth=return_depth,static_blocks=sum(pc in starts for pc in states),max_fp_depth=max(x[1] for x in states.values()))
  except (Rejected,ValueError,KeyError,IndexError,AssertionError) as exc:row.update(admitted=False,reason=str(exc) or type(exc).__name__)
  rows.append(row)
 (a.output/'all.cuh').write_text('#pragma once\n'+''.join('#include '+json.dumps(str(f.resolve()))+'\n' for f in headers))
 report={'schema':'irisu-scalar-leaf-static-screen-v1','source_sha256':BASE.PIN,'scope':'Static admission only; full postcall Guest/arena/budget differential mandatory before use. No perf claim.','functions':rows,'totals':{'functions':len(rows),'admitted':len(headers),'rejected':len(rows)-len(headers),'rejection_reasons':dict(collections.Counter(x['reason'] for x in rows if not x['admitted']))}}
 (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['totals'],indent=2))
if __name__=='__main__':main()
