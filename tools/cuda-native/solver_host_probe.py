#!/usr/bin/env python3
"""Create an isolated function-call fixture harness for independent typed solver."""
import argparse,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline',type=Path);p.add_argument('output',type=Path);a=p.parse_args();shutil.copytree(a.baseline,a.output,dirs_exist_ok=True)
for entry in [29424,29968,30224,31584]:
 f=a.output/f'guest_region_{entry}.inc';original=f.read_text().replace(f'void region_{entry}(',f'void original_region_{entry}(',1)
 f.write_text(original+f'\nIRISU_GUEST_HD IRISU_GUEST_NOINLINE IRISU_GUEST_REGION_INLINE void region_{entry}(Guest& g,std::uint64_t& budget,unsigned depth) {{::irisu::cuda_native::solver_test::stage<{entry}>(g,budget,depth,original_region_{entry});}}\n')
(a.output/'typed-prefix.hpp').write_text('#include '+json.dumps(str(ROOT/'tools/cuda-native/solver_stage_bridge.hpp'))+'\n')
