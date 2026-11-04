# Fixed-host physics differential and timing gate

`physics_validate.py` compares the instruction-translated physics probe with
the pinned ELF32 native host. It checks every binary32 position, angle, and
velocity word for three bodies at every simulated step. The two fixture modes
include mixed circle/box/triangle collisions, velocity mutation, and body
destruction/recreation. Each GPU lane starts from the same fixture; this gate
does not establish arbitrary-world, full-game, RNG, scoring, or search parity.

After building the probe and CPU oracle, begin with a short bounded gate:

```sh
python3 tools/cuda-exact/physics_validate.py \
  --probe build-gpu-guest/physics-probe-local \
  --library .local/exact-runtime-20260824/exact-runtime/guest/libirisu_box2d_msvc_exact_multiworld.so \
  --steps 10 --lanes 1 32 --cpu-threads 1 8 --repeats 3 \
  --timeout 60 --output build-gpu-guest/physics-tiny.json
```

Use the actual probe/library paths from the local build. The script rejects
an unpinned native library, hashes the probe/image/oracle, and records source
hashes at startup. It also refuses to continue if the probe binary changes.
`--backend translated-host --probe <host-probe>` supports a separate CPU-only
translation smoke test; its results are explicitly labeled translated-host.

After the short gate is fast enough, increase to `--steps 600` with the same
workload for CPU and GPU. The default lane matrix is 1/32/128; larger matrices
can be requested with `--lanes 512 1024 4096 8192`. A one-thread-per-world
kernel with 32-thread blocks needs more than 1,024 worlds even to occupy all
80 V100 SMs, so small matrices cannot establish the card's throughput ceiling.
Keep each invocation bounded with `--timeout`.

Warmups are preserved immediately, including parity and probe-reported CUDA
event timing. Timed samples separately record complete process wall time;
this includes context/module initialization, allocation, uploads, world setup,
simulation, downloads, and stdout capture. Native CPU batches use independent
processes and include their creation cost. These startup-inclusive numbers
must not be described as a saturated native CPU limit. Kernel timing covers
world setup plus simulation and excludes allocation/transfers. Comparisons
run outside the timed interval.

The script samples whole-device memory, GPU activity, power, and temperature
at 100ms. These samples are not process-exclusive. GPU activity at 100% means
that a kernel was active throughout the sample; it does not establish high SM
occupancy or efficient utilization of the device.

The first fast-compile V100 build passed the complete 600-step collision
fixture bit for bit, but took about 54.9 seconds of kernel time for one world,
versus approximately 6ms for the native process including startup. This is a
working parity baseline with poor performance, not an acceleration result.
The 10-step gate also passed at 32 lanes, with similar kernel duration to one
lane, showing the expected parallel execution of identical worlds. Full
performance claims require a normally optimized build, realistic density and
divergence, larger lane counts, and an integrated fixed-budget search test.

See the compact checked-in evidence in
[`cuda-physics-initial-v100-2026-11-03.json`](../../benchmarks/results/cuda-physics-initial-v100-2026-11-03.json).
