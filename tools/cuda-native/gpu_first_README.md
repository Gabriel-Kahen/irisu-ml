# GPU-first search experiments

This is a separate experimental path alongside the preserved
[hybrid campaign](efficiency3_README.md). Existing hybrid commands, arithmetic,
snapshots, and defaults remain available. The matched V100 measurements below
select separate configurations for large frontiers and small batches.

## Measured results — November 4, 2026

Three alternating paired rounds, complete child-process wall time, GPU reset and
100-tick prefix, four-tick horizon, 20 batches, and exact native Result/rank packets:

| Workload | Control median | Selected median | Speedup |
| --- | ---: | ---: | ---: |
| 12,288 candidates, preceding GPU-only implementation versus lean resident scheduler | 1.673061 s | 1.577604 s | 1.0605× |
| 64 spread candidates, all resident, preceding GPU-only implementation versus cooperative physics | 0.746853 s | 0.472029 s | 1.5822× |

The large configuration uses the **lean build**, `--adaptive-shared-prefixes
--reuse-checkpoints --order-by-depth`, 32 threads and the default 16 KiB stack.
Mirror, cooperative service, collision cache and register caps are compiled off.
The small configuration compiles the optional stages and adds
`--cooperative-physics`; all 64 candidates are resident, and mirror/cache remain disabled.
Both configurations use GPU physics exclusively. Reused checkpoints reduce
actual large-case future work from 614,680 to 614,566 ticks across the 20 batches;
all 245,760 candidate suffix evaluations still execute freshly. This benchmark
does not advance an adaptive search frontier or demonstrate a better score.

The preserved hybrid control measured **1.430068 s** over three further runs on
the same large workload. It remains faster there, using its existing CPU prefix
and CPU checkpoint preparation. The new 6% gain is relative to GPU-only search.

Wider inputs up to 26,624 candidates passed native and retained-state checks;
the latter allocated **29.033 GB** with an 8 KiB stack. Larger pools, 192/224-register
caps, the hot mirror, collision memoization, and the persistent queue did not
improve the selected large-workload result. Keep them as explicit experiments.
Compiling optional physics hooks also has a measurable cost even when their
runtime descriptors are inactive; use the lean build for the large configuration.

The [machine-readable receipt](../../benchmarks/results/cuda-native-gpu-first-2026-11-04.json)
records source/binary pins, raw paired runs, rejected experiments and qualification.
Its artifact paths refer to this worktree's ignored `artifacts/gpu-first-20261104/`.

`irisu-cuda-native-gpu-search` executes reset, root prefixes, partial checkpoints,
futures, ranking, and retained-state continuation on GPU. CPU work prepares empty
storage and input/scheduling descriptors. The runner has no host-physics mode.
The native library is used separately to construct independent expected packets.

## Modes and spare VRAM

| Option | Purpose |
| --- | --- |
| `--slots N` | Bound the number of complete execution worlds independently of logical frontier size. Automatic admission uses observed free VRAM and a conservative runtime reserve. |
| `--roots N` | Keep up to 32 independently initialized roots resident. Candidates map round-robin to roots; seed/prefix increments are explicit. |
| `--adaptive-shared-prefixes` | Reuse exact partial trajectories within each root, with actual preparation and suffix ticks reported separately. |
| `--reuse-checkpoints` | Prepare those GPU checkpoints once across repeated batches and guard their complete raw source state after every batch. |
| `--persistent-queue` | Reuse execution slots through a device work queue instead of host-launched waves. Retained states preserve global ranking and original logical IDs. |
| `--order-by-depth` | Schedule longer suffixes first while preserving output IDs and ranking ties. |
| `--solver-mirror` | Allocate an additional tiled hot-state mirror through an island's presolve, iterations, position integration, and postsolve. |
| `--cooperative-physics` | Use a controller warp and dedicated worker warp per world for independent collision, integration, and bounds calculations, followed by ordered canonical commits. |
| `--collision-cache-mib N` | Spend a bounded VRAM budget on exact narrowphase memoization. Full input-key equality is required; published entries are immutable. |
| `--diagnostic-counters` | Report actual mirror/service/cache activity. These counters and their readback add overhead and are not performance instrumentation. |

The logical frontier admits up to 131,072 distinct fixed 40-byte specifications;
the existing bounded reader remains unchanged. Larger frontiers execute in
waves or through the queue when they exceed admitted physical slots. Root count,
seed step, and prefix step change the workload and must match the native reference.
Queue and cooperative service modes cannot be combined.

More VRAM can hold more useful worlds, roots, checkpoint states, or cache entries.
It does not itself increase arithmetic throughput. At capacity 512, canonical
Game+Scene storage is 1,089,512 bytes per slot before counters and other buffers.
The hot mirror adds 86,080 bytes per slot, rounded to 32-lane tiles: approximately
1.058 GB for 12,288 slots and 2.116 GB for 24,576 slots. Cooperative channels and
collision tables add their own separately reported allocations. Automatic slot
admission includes these buffers, fixed allocations, a configurable reserve, and
a conservative stack upper bound; requested slots outside that budget fail.

The [mirror contract](persistent_solver_contract.md) and
[parallel/cache contract](parallel_physics_contract.md) describe their exactness
boundaries. The mirror's allocation persists across waves, but its authoritative
contents currently last one island; it is not a claim of a fully persistent SoA
physics engine across game ticks. Neither extra allocation nor a resource-based
occupancy ceiling proves measured utilization or faster execution.

## Build and run

Use a separate 64-bit CUDA 12.9/GCC 14 build. The large-case lean profile is:

```sh
cmake -S . -B build-cuda-native-gpu-first \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
  -DIRISU_BUILD_CUDA_NATIVE_EXPERIMENTS=ON \
  -DIRISU_BUILD_CUDA_EXPERIMENTS=OFF -DIRISU_BUILD_TESTS=OFF \
  -DIRISU_CUDA_NATIVE_BOUNDED_MATH=ON -DIRISU_CUDA_NATIVE_BOUNDED_STORE=ON \
  -DIRISU_CUDA_NATIVE_LIGHT_SHAPE_SYNC=ON -DIRISU_CUDA_NATIVE_CACHED_ROTATION=ON \
  -DIRISU_CUDA_NATIVE_INLINE_STEP=ON -DIRISU_CUDA_NATIVE_WIDE_REBIND=ON \
  -DIRISU_CUDA_NATIVE_LOCAL_VELOCITY=ON -DIRISU_CUDA_NATIVE_LOCAL_POSITION=ON \
  -DIRISU_CUDA_NATIVE_CONTACT_ANCHORS=ON -DIRISU_CUDA_NATIVE_MAX_CONTACTS=512 \
  -DIRISU_CUDA_NATIVE_GAME_FAST_COMPILE=min \
  -DIRISU_CUDA_NATIVE_PERSISTENT_SOLVER=OFF \
  -DIRISU_CUDA_NATIVE_PARALLEL_PHASES=OFF -DIRISU_CUDA_NATIVE_COLLISION_CACHE=OFF
cmake --build build-cuda-native-gpu-first -j2 --target irisu-cuda-native-gpu-search
```

Use a different build directory for optional physics experiments and enable the
desired default-off options:

```sh
-DIRISU_CUDA_NATIVE_PERSISTENT_SOLVER=ON \
-DIRISU_CUDA_NATIVE_PARALLEL_PHASES=ON \
-DIRISU_CUDA_NATIVE_COLLISION_CACHE=ON
```

Build `irisu-cuda-native-gpu-search`, `irisu-cuda-native-gpu-reference`, and the
relevant qualification targets. Compiled availability does not activate an
experiment: the corresponding runtime option still needs to be selected.
Retain exact trig, original contact order, strict floating-point flags, and the
selected local velocity/position/contact-anchor options from the hybrid profile.

Generate a distinct wide workload, then execute a wave-based control:

```sh
python3 tools/cuda-native/gpu_search_specs.py irisu-high-score-4393101.rpy \
  /tmp/irisu-wide.bin --prefix 100 --horizon 4 --count 24576
env -u CUDA_VISIBLE_DEVICES \
  build-cuda-native-gpu-first/clone/cuda_native/irisu-cuda-native-gpu-search \
  irisu-high-score-4393101.rpy /tmp/irisu-wide.bin 100 4 \
  --adaptive-shared-prefixes --reuse-checkpoints --order-by-depth \
  --final-only --warmups 0 --repeats 3 \
  > /tmp/irisu-gpu-first.bin
```

Add one experiment option at a time. Keep selected-state validation enabled for
qualification. `--skip-compaction-check` skips only the extra independent retained
state replay/continuation; checkpoint guards and executed work remain enabled.

`gpu_search_native_reference.py` constructs a complete expected packet from the
pinned actual-native oracle in bounded chunks. It preserves every Result field
and recomputes global signed objectives, failure ordering, and original-ID ties.
Give it the same root parameters as the GPU run. Native reference generation and
packet comparison are outside the GPU performance timer.

`irisu-cuda-native-gpu-reference` keeps the preceding scalar/adaptive implementation
with only its input admission widened. It is a comparison executable, not the
new bounded-slot scheduler; its world allocation must still fit the device.

## Measurement and qualification

`benchmark_gpu_first.py` compares complete process wall time against a complete
actual-native Result/rank packet. It rejects CPU-prefix flags, samples competing
compute processes, alternates paired order, and includes allocation, GPU prefix
and checkpoint work, futures, retention, guards, output, and teardown. Compare
identical logical candidates, roots, horizons, trace settings, and repeat counts.
Report both candidate futures/second and actual executed GPU ticks. Checkpoint
reuse is a work reduction and must be distinguished from faster identical work.

Required correctness evidence is separate from timing:

- Planner tests and native-reference chunk partition/ranking tests.
- Complete native Result/rank equality for small and wide/multi-root frontiers.
- Retained states independently replayed from the original root, with live Game,
  active physical/SAP state, and poisoned-scratch continuation equality.
- Native-captured solver, collision-cache, and cooperative phase differentials,
  including numeric/structural rejection and exact partial writes.
- Continuous GPU/native game-digest replay with each enabled subsystem actually
  configured. Mirror/service/cache counters must prove that the intended path
  ran; a compiled option with zero activity is insufficient evidence.
- CUDA memory checks of active experimental paths and slot/cache reuse.

The configured mirror/service/cache combination matched all **97,220 native
record frames**, with positive activation counters, zero mirror fallback or
packing rejections, and no CPU physics. Dedicated qualifiers additionally cover
897 solver cases, all three collision families including forced hash collisions,
3,072 native dynamic frames, and rejected/partially written stages. Small tests
retain every leaf across waves, exercise changed seeds/prefixes and late trace
mode, and compare poisoned-scratch continuations. Memcheck, initcheck and
synccheck logs are recorded separately in the receipt.

Whole-game frames are native digest equality, not raw hidden allocator-state
equality. Stage proofs and small searches do not establish arbitrary complete
world behavior. Capacity512 retains explicit overflow restrictions. Repeated
fanout batches are not a deeper adaptive search or a demonstrated score gain.
The existing hybrid result remains the preserved fallback regardless of which
GPU-first experiments qualify or fail to improve throughput.
