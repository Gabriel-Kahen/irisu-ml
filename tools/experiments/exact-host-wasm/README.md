# Exact-host WebAssembly translation experiment

This archived experiment packaged the fixed 32-bit MSVC9 Box2D host as a
zero-based guest image and inventoried its ELF boundary and decoded instruction
surface. The isolated packager and smoke interpreter have been removed; their
source remains in Git history. The [findings](FINDINGS.md) preserve the inventory,
host identity, deterministic smoke hash, and proposed translation architecture.

The supported browser implementation runs the exact worker in v86. See the
[production web app](../../../apps/web/README.md) and its
[exact runtime packaging](../../../apps/web/EXACT_RUNTIME.md).

The first executable slice interpreted the extracted bytes of the private
`msvc_b2d_world_test` wrapper in WebAssembly. The path exercises the MSVC
`stdcall` stack, ownership branches, an absolute `.rdata` reference, and x87
`FLD`/`FSTP`. Both the owned and rejected-body branches ran; the smoke test
expected the original raw `3.14159265f` word, `0x40490fdb`.

The historical output contained:

- `manifest.json`: imports, exports, ABI, relocations, and mnemonic counts;
- `exact-host.image.bin`: page-aligned `PT_LOAD` image at guest base zero;
- `exact_host_image.inc`: deterministic C initializer used by the smoke slice;
- `exact-host-smoke.wasm`: freestanding browser-compatible module.

No timestamp or absolute source path entered the package. Rebuilding the same
host with the same binutils/Clang versions produced byte-identical outputs.
The packager pinned the attested host SHA-256 by default.

## Historical translation proposal

Keep the existing mechanics and UI compiled normally to WebAssembly. Translate
only this fixed guest image ahead of time, one recovered basic block per Wasm
function or dispatch-table entry. Implement the six strong imports (`malloc`,
`free`, `memcpy`, `memmove`, `memset`, and `abort`) against Wasm linear memory;
the remaining imports are weak loader hooks.

The smoke interpreter is deliberately tiny, not a proposed full interpreter.
The next slice should expand decoding to the manifest's integer/control-flow
surface, then lower x87 operations through explicit state helpers. The host's
required `0x027f` control word selects 53-bit precision, but mapping arithmetic
to Wasm `f64` is acceptable only after bitwise differential proof. `FSIN`,
`FCOS`, and `FSINCOS` require a reference-matched implementation; JavaScript
or ordinary libm trigonometry is not a parity-preserving substitute.
