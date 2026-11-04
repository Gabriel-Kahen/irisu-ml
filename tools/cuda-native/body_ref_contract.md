# Wide body-reference transport qualification

`body_ref_validate.cu` forces `IRISU_CUDA_NATIVE_WIDE_REBIND=1` locally for its
device code. It calls the real `relocate_native_body_refs` helper in both passes:
the host pass retains original `memcpy`, because `__CUDA_ARCH__` is undefined;
the device pass uses aligned CUDA-global u64 loads/stores. The production option
and other targets keep their configured defaults. This qualifier changes no
existing source/header.

The real `vector<Body*>` has eight-byte-aligned storage and a stride divisible
by eight. Input and destination vectors are distinct global allocations. The
qualifier copies every input-vector byte into destination storage before calling
the helper, including inactive slots, count, overflow and padding. Synthetic
`uintptr_t` words are never dereferenced as pointers. Their values exercise all
512 slots and the real `sizeof(Body)` range/modulus rules without requiring an
actual Body allocation at those synthetic addresses.

The corpus includes counts0/1/511/512/513 in both full-capacity and live-only
modes; null, exact first/last Body pointers, upper/lower range boundaries,
eight-byte-aligned but non-Body-aligned words, misaligned words, already-relocated
values and nonbody garbage in active/inactive positions. Every possible failure
position0..511 is tested in both modes with valid words before and after it.
Additional cases cover multiple invalid words, same bases, zero/high old bases
and unsigned wrapping of the new base. Unusual synthetic bases test arithmetic
behavior, not admissibility of a real checkpoint.

Returned bool and **all vector bytes**, including metadata and padding, must
match the host helper exactly. Explicit host assertions verify early return
ordering: preceding valid words have relocated, the failing word and following
words remain unchanged. The complete source-case allocation is read back and
must remain unchanged.

Existing validation is preserved, including its limits: `live_only=true` rejects
count513 immediately without mutation; full-capacity mode scans exactly512
slots and does not add a new count513 rejection. It can return true for such a
header when scanned words otherwise pass. Inactive out-of-range nonzero words
are retained in full mode, but an in-range misaligned word still fails there.
Live-only mode does not inspect inactive slots. This qualifier must not silently
strengthen or weaken those rules.

After build, inspect SASS to confirm the relocation loops emit u64 transport;
the source-copy loop also deliberately uses u64 PTX to preserve padding. Actual
Game/Scene normalized-byte checks, selected-state continuation, immutable cached
prefix gates, native packet parity and memory instrumentation remain separate
required integration gates. No runtime improvement is inferred from this test.
