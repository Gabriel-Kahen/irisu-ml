#!/usr/bin/env python3
"""Wrap statically admitted leaves in a COPIED generated function-call tree."""
import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline',type=Path);p.add_argument('typed',type=Path);p.add_argument('output',type=Path);p.add_argument('--entries',type=int,nargs='*');a=p.parse_args();shutil.copytree(a.baseline,a.output,dirs_exist_ok=True)
manifest=json.loads((a.typed/'manifest.json').read_text());headers=[]
for row in manifest['functions']:
 if not row['admitted'] or (a.entries is not None and row['entry'] not in a.entries):continue
 entry=row['entry'];file=a.output/f'guest_region_{entry}.inc'
 if not file.exists():continue
 original=file.read_text().replace(f'void region_{entry}(',f'void original_region_{entry}(',1)
 wrapper=f'''\nIRISU_GUEST_HD IRISU_GUEST_NOINLINE IRISU_GUEST_REGION_INLINE void region_{entry}(Guest& g,std::uint64_t& budget,unsigned native_depth) {{
#if !defined(__CUDA_ARCH__) && defined(IRISU_DIRECT_LEAF_VERIFY)
 ::irisu::cuda_direct::verified_leaf<{entry}>(g,budget,native_depth,original_region_{entry},::irisu::cuda_direct::leaf_{entry});
#else
 if(g.pc=={entry}u && ::irisu::cuda_direct::leaf_{entry}(g,budget))return;
 original_region_{entry}(g,budget,native_depth);
#endif
}}
'''
 file.write_text(original+wrapper);header=(a.typed/f'leaf_{entry}.cuh').resolve();unit=file.with_suffix('.cu');unit.write_text('#include '+json.dumps(str(header))+'\n'+unit.read_text());headers.append(header)
(a.output/'typed-prefix.hpp').write_text('#pragma once\n#include '+json.dumps(str(Path(__file__).with_name('verification.hpp').resolve()))+'\n'+''.join('#include '+json.dumps(str(f))+'\n' for f in headers))
print(f'Patched {len(headers)} copied leaf definitions; production untouched')
