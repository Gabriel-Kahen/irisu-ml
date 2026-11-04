#!/usr/bin/env python3
"""Native x86/x87 differential tests for the instruction lowering layer.

Run with the CUDA development Python (capstone + pyelftools installed):
  .venv-gpu/bin/python tools/cuda-exact/test_translation.py
These compare generated C++ operations with real instruction bytes; they do
not merely compare two copies of the lowering implementation.
"""

from __future__ import annotations

import importlib.util
import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from capstone import Cs, CS_ARCH_X86, CS_MODE_32


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("irisu_cuda_translate", Path(__file__).with_name("translate.py"))
LOWER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOWER)


def decode(code: bytes):
    decoder = Cs(CS_ARCH_X86, CS_MODE_32)
    decoder.detail = True
    return next(decoder.disasm(code, 0x1000, count=1))


def compile_run(source: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="irisu-lowering-") as tmp:
        cpp = Path(tmp) / "test.cpp"
        executable = Path(tmp) / "test"
        cpp.write_text(source)
        compiled = subprocess.run(["g++", "-std=c++20", "-O2", "-ffp-contract=off", "-fno-fast-math",
                                   "-I", str(ROOT), "-I/opt/cuda/include", str(cpp), "-o", str(executable)],
                                  capture_output=True, text=True)
        if compiled.returncode:
            raise AssertionError(compiled.stderr)
        return subprocess.run([str(executable)], capture_output=True, text=True)


class NativeInstructionParity(unittest.TestCase):
    def test_x87_register_and_memory_forms(self):
        cases = []
        arithmetic = ("fadd", "fmul", "fsub", "fsubr", "fdiv", "fdivr")
        for first in (0xd8, 0xdc, 0xde):
            for second in range(0xc0, 0x100):
                if second & 7 > 3:
                    continue
                code = bytes((first, second))
                try:
                    ins = decode(code)
                except StopIteration:
                    continue  # Reserved two-byte x87 encodings.
                if ins.mnemonic.startswith(arithmetic) or ins.mnemonic in ("fcom", "fcomp", "fcompp"):
                    cases.append((code, ins, None))
        for first, values in ((0xd9, range(0xc0, 0xcc)), (0xdd, range(0xd0, 0xdc))):
            for second in values:
                if second & 7 > 3:
                    continue
                code = bytes((first, second))
                ins = decode(code)
                if ins.mnemonic in ("fld", "fxch", "fst", "fstp"):
                    cases.append((code, ins, None))
        for code in (b"\xde\xd9", b"\xda\xe9", b"\xd9\xe0", b"\xd9\xe1", b"\xd9\xfa"):
            cases.append((code, decode(code), None))
        # The same ModRM address means EAX in the guest and RAX in the native
        # x86-64 test, letting the host assembler consume real memory operands.
        for first, value_type in ((0xd8, "float"), (0xdc, "double")):
            for group in range(8):
                code = bytes((first, group << 3))
                ins = decode(code)
                if ins.mnemonic.startswith(arithmetic) or ins.mnemonic in ("fcom", "fcomp"):
                    cases.append((code, ins, value_type))
        chunks = []
        for code, ins, memory_type in cases:
            mnemonic = ins.mnemonic
            pops = 2 if mnemonic.endswith("pp") else int(mnemonic.endswith("p"))
            depth = 4 - pops + int(mnemonic == "fld")
            stores = "".join(f"fstpl %[n{i}]\n" for i in range(depth))
            assembly = "fninit\nfldcw %[cw]\n" + "".join(f"fldl %[i{i}]\n" for i in range(4))
            assembly += ".byte " + ",".join(hex(b) for b in code) + "\nfnstsw %[sw]\n" + stores
            outputs = ",".join([f'[n{i}] "=m"(expected[{i}])' for i in range(depth)] + ['[sw] "=m"(status)'])
            inputs = ','.join(['[cw] "m"(cw)'] + [f'[i{i}] "m"(initial[{i}])' for i in range(4)] + ([ '"a"(&memory)' ] if memory_type else []))
            memory_setup = f"{memory_type or 'float'} memory=3.0;g.r[0]=128;g.store<{memory_type or 'float'}>(128,memory);"
            label = json.dumps(f"{code.hex()} {mnemonic} {ins.op_str}")
            chunks.append(f'''{{
              irisu::cuda_guest::Guest g; unsigned char bytes[1024]{{}};g.memory=bytes;g.capacity=1024;
              for(double x:initial)g.fpush(x); {memory_setup}
              double expected[8]{{}};unsigned short status;
              asm volatile({json.dumps(assembly)} : {outputs} : {inputs} : "st","st(1)","st(2)","st(3)","st(4)");
              {LOWER.Lower(ins).emit()}
              if(g.error||g.depth!={depth}||(g.status&0x4500)!=(status&0x4500)){{
                std::fprintf(stderr,"state mismatch %s depth=%u status=%x/%x error=%u\\n",{label},g.depth,g.status,status,g.error);++failed;
              }}
              for(unsigned i=0;i<{depth};++i)if(std::bit_cast<std::uint64_t>(g.st(i))!=std::bit_cast<std::uint64_t>(expected[i])){{
                std::fprintf(stderr,"value mismatch %s ST%u %.17g/%.17g\\n",{label},i,g.st(i),expected[i]);++failed;
              }}
            }}''')
        source = '''#include "clone/cuda/guest.hpp"
        #include "tools/cuda-exact/x87_numeric.cuh"
        #include <bit>
        #include <cstdio>
        int main(){using namespace irisu::cuda_guest;int failed=0;const unsigned short cw=0x027f;
        const double patterns[2][4]={{3.25,5.0,7.0,2.0},{0.5,-1.25,2.0,7.0}};
        for(const auto& initial:patterns){
        ''' + "\n".join(chunks) + f' }}std::printf("tested {len(cases)} x87 forms, 2 input patterns; failures=%d\\n",failed);return failed?1:0;}}'
        result = compile_run(source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        print(result.stdout.strip())

    def test_carry_borrow_native_flags(self):
        chunks = []
        # Defined flags only: carry, parity, zero, sign, overflow (not AF).
        for code in (bytes.fromhex("15ffffffff"), bytes.fromhex("1dffffffff"),
                     bytes.fromhex("15ffffff7f"), bytes.fromhex("1dffffff7f")):
            ins = decode(code)
            label = json.dumps(code.hex() + " " + ins.mnemonic + " " + ins.op_str)
            for a in (0, 1, 0x7fffffff, 0x80000000, 0xffffffff):
                for carry in (0, 1):
                    assembly = "btl $0,%[carry]\n.byte " + ",".join(hex(b) for b in code) + "\npushfq\npopq %[flags]"
                    chunks.append(f'''{{
                      irisu::cuda_guest::Guest g;g.r[0]={a}u;g.flags={carry};
                      unsigned native={a}u,carry={carry};unsigned long flags;
                      asm volatile({json.dumps(assembly)} : "+a"(native),[flags] "=r"(flags) : [carry] "r"(carry) : "cc");
                      {LOWER.Lower(ins).emit()}
                      if(g.r[0]!=native||(g.flags&0x8c5)!=(flags&0x8c5)){{
                        std::fprintf(stderr,"carry mismatch %s a=%x carry=%u result=%x/%x flags=%x/%lx\\n",{label},{a}u,carry,g.r[0],native,g.flags&0x8c5,flags&0x8c5);++failed;
                      }}
                    }}''')
        source = '#include "clone/cuda/guest.hpp"\n#include <cstdio>\nint main(){int failed=0;\n' + "\n".join(chunks) + 'return failed?1:0;}'
        result = compile_run(source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class GeneratedIdentity(unittest.TestCase):
    def test_pinned_image_identity_and_incremental_build(self):
        source = Path(os.environ.get("IRISU_CUDA_EXACT_HOST", str(
            ROOT / ".local/exact-runtime-20260824/exact-runtime/guest/libirisu_box2d_msvc_exact_multiworld.so")))
        if not source.exists():
            self.skipTest("pinned development physics host is unavailable")
        with tempfile.TemporaryDirectory(prefix="irisu-codegen-") as tmp:
            output = Path(tmp) / "generated"
            LOWER.generate(source, output)
            before = {p.name: p.stat().st_mtime_ns for p in output.iterdir()}
            LOWER.generate(source, output)
            self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in output.iterdir()})
            manifest = json.loads((output / "manifest.json").read_text())
            image_sha = hashlib.sha256((output / "guest_image.bin").read_bytes()).hexdigest()
            self.assertEqual(manifest["image_sha256"], image_sha)
            self.assertIn(image_sha, (output / "guest_image_identity.inc").read_text())
            tampered = Path(tmp) / "tampered.so"
            altered = bytearray(source.read_bytes()); altered[-1] ^= 1
            tampered.write_bytes(altered)
            with self.assertRaises(SystemExit):
                LOWER.generate(tampered, Path(tmp) / "rejected")
            self.assertFalse((Path(tmp) / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
