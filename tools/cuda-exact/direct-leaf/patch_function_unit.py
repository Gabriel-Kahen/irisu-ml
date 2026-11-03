#!/usr/bin/env python3
"""Patch a copied function-call build's one solver TU after normal generation.

The normal translator will overwrite this experimental patch on regeneration.
Recompile this single TU manually, replace its member in a COPIED region archive,
and relink the pilot. Never apply to the authoritative/default generated tree.
"""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('generated',type=Path);p.add_argument('typed_header',type=Path)
a=p.parse_args();inc=a.generated/'guest_region_30224.inc';unit=inc.with_suffix('.cu')
for file in [inc,unit]:
 backup=file.with_suffix(file.suffix+'.unpatched')
 if not backup.exists():backup.write_bytes(file.read_bytes())
s=inc.with_suffix('.inc.unpatched').read_text();at=s.index('{')+1
s=s[:at]+'''
// Experimental typed leaf. The fallback still handles internal reentry,
// nonempty x87 stacks and budgets too small for a conservatively bounded call.
if(g.pc==30224u && !g.depth && !g.error && g.r[1]<=g.capacity-12) {
  const auto count=g.load<std::int32_t>(g.r[1]+8);
  const auto constraints=g.load<std::uint32_t>(g.r[1]+4);
  const std::uint64_t bound=count>=0?256ull*count+256:0;
  if(count>=0 && count<=4096 && budget>=bound &&
     std::uint64_t(constraints)+120ull*count<=g.capacity) {
    // Native b2ContactConstraint has at most two contact points. Keep the
    // conservative block bound valid even for non-native/corrupt snapshots.
    for(std::int32_t i=0;i<count;++i)
      if(g.load<std::uint32_t>(constraints+120u*i+116u)>2u)goto dispatch;
    const auto before=g.instructions;
    if(::irisu::cuda_direct::velocity(g)) {
      const auto used=g.instructions-before;
      if(used>budget){budget=0;g.fail(::irisu::cuda_guest::budget_exhausted);}
      else budget-=used;return;
    }
  }
}
'''+s[at:]
inc.write_text(s);unit.write_text('#include '+json.dumps(str(a.typed_header.resolve()))+'\n'+unit.with_suffix('.cu.unpatched').read_text())
print(f'Patched only {inc} and {unit}; original .unpatched copies preserved')
