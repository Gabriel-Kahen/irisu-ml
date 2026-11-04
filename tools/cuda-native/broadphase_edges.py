#!/usr/bin/env python3
"""Write explicit SAP cancellation and PC53 quantization boundary fixtures."""
import argparse,json,struct
from pathlib import Path
OP=struct.Struct('<3I4fI')
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def neighbor(x,step):
 u=struct.unpack('<I',struct.pack('<f',x))[0];return struct.unpack('<f',struct.pack('<I',u+(step if x>=0 else -step)))[0]
def generate(mode):
 world=list(map(f32,[-37.12345,-105.03125,231.77777,327.12345]));ops=[];labels=[];witnesses=[]
 def add(kind,slot=0,user=0,box=(0,0,1,1),label=''):
  ops.append(OP.pack(kind,slot,user,*box,0));labels.append(label)
 if mode=='cancellation':
  add(1,0,101,(-1,-1,1,1),'create-first');add(1,1,202,(4,4,5,5),'create-disjoint')
  add(2,1,box=(-.5,-.5,.5,.5),label='buffer-new-add');add(2,1,box=(4,4,5,5),label='cancel-before-first-commit');add(4,label='no-callback-new-pair-cancellation')
  add(2,1,box=(-.5,-.5,.5,.5),label='buffer-recycled-add');add(4,label='first-final-add')
  add(2,1,box=(4,4,5,5),label='buffer-remove-final');add(2,1,box=(-.5,-.5,.5,.5),label='cancel-remove-final');add(4,label='no-callback-final-pair-cancellation')
  add(1,2,303,(-.25,-.25,.25,.25),'create-overlap-ties');add(2,2,box=(6,6,7,7),label='buffer-third-remove');add(3,0,label='destroy-commits-other-buffered-pairs');add(3,1,label='destroy-second');add(3,2,label='destroy-third');add(5,user=512,box=(-100,-100,100,100),label='empty-final-query')
 else:
  add(1,0,101,(-1,-1,1,1),'reference-proxy');add(1,1,202,(1,1,2,2),'moving-proxy')
  for axis in range(2):
   low,high=world[axis],world[axis+2];factor=f32(65535/f32(high-low));selected=[]
   for bucket in range(2,65534,2):
    center=f32(low+bucket/factor)
    for step in [-1,0,1]:
     x=neighbor(center,step) if step else center
     exact=int((float(x)-float(low))*factor)&65534
     overrounded=int(f32(float(x)-float(low))*factor)&65534
     if exact!=overrounded:selected.append((bucket,x,exact,overrounded));break
    if len(selected)==16:break
   if len(selected)!=16:raise RuntimeError('could not construct rounding witnesses')
   for bucket,x,exact,overrounded in selected:
    witnesses.append({'axis':axis,'bucket':bucket,'coordinate':x,'correct_lower':exact,'incorrect_float_difference_lower':overrounded})
    for step in [-1,0,1]:
     coordinate=neighbor(x,step) if step else x;box=[0.,0.,1.,1.];box[axis]=coordinate;box[axis+2]=f32(coordinate+.25)
     add(2,1,box=box,label=f'axis{axis}-bucket{bucket}-ulp{step}');add(4,label='commit-boundary-step')
  add(5,user=512,box=world,label='whole-world-query');add(3,0,label='destroy-reference');add(3,1,label='destroy-moving')
 header=struct.pack('<4I4f',0x3153424e,1,len(ops),8,*world)
 return header+b''.join(ops),{'mode':mode,'operations':len(ops),'world':world,'labels':labels,'quantization_witnesses':witnesses}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['quantization','cancellation'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();packet,metadata=generate(a.mode);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(packet);a.output.with_suffix('.labels.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps({'operations':metadata['operations'],'witnesses':len(metadata['quantization_witnesses'])}))
