find_package(CUDAToolkit 12.9 REQUIRED)
option(IRISU_CUDA_NATIVE_BOUNDED_MATH "Experimental device math for qualified bounded graphs" OFF)
option(IRISU_CUDA_NATIVE_BOUNDED_STORE "Omit redundant double-normal checks within audited bounded graphs" OFF)
if(IRISU_CUDA_NATIVE_BOUNDED_STORE AND NOT IRISU_CUDA_NATIVE_BOUNDED_MATH)
  message(FATAL_ERROR "Bounded stores require the audited bounded-math profile")
endif()
option(IRISU_CUDA_NATIVE_LIGHT_SHAPE_SYNC "Avoid large temporary copies during shape synchronization" OFF)
option(IRISU_CUDA_NATIVE_CACHED_ROTATION "Reuse canonical engine rotation matrices when angle bits are unchanged" OFF)
option(IRISU_CUDA_NATIVE_F32_STORE "Use equivalent RN32 operations for adjacent stores with two float inputs" OFF)
option(IRISU_CUDA_NATIVE_F32_DIV_STORE "Use qualified RN32 division for adjacent float stores" OFF)
set(IRISU_CUDA_NATIVE_DEVICE_TRIG "0" CACHE STRING "Device trig: 0 exact, 1 CUDA library, 2 CUDA intrinsic")
set_property(CACHE IRISU_CUDA_NATIVE_DEVICE_TRIG PROPERTY STRINGS 0 1 2)
if(NOT IRISU_CUDA_NATIVE_DEVICE_TRIG MATCHES "^[012]$")
  message(FATAL_ERROR "Device trig must be 0, 1 or 2")
endif()
option(IRISU_CUDA_NATIVE_SPARSE_RESET "Reset occupied flags instead of unused physical pool payloads" OFF)
option(IRISU_CUDA_NATIVE_INLINE_STEP "Inline the typed game step forwarding wrapper" OFF)
option(IRISU_CUDA_NATIVE_ENTRY_INDEX "Index typed game physics entries by body ID" OFF)
option(IRISU_CUDA_NATIVE_WIDE_REBIND "Use aligned global u64 transport for GPU body-reference relocation" OFF)
option(IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER "Use interleaved hot solver workspace for configured GPU pools" OFF)
option(IRISU_CUDA_NATIVE_CACHED_VELOCITY "Cache exact world-space solver anchors across velocity iterations" OFF)
option(IRISU_CUDA_NATIVE_LOCAL_VELOCITY "Retain each contact's velocity state until the contact boundary" OFF)
option(IRISU_CUDA_NATIVE_LOCAL_POSITION "Retain each contact's position state until the contact boundary" OFF)
option(IRISU_CUDA_NATIVE_CONTACT_ANCHORS "Reuse each velocity contact's anchors between normal and tangent passes" OFF)
option(IRISU_CUDA_NATIVE_SPLIT_CONSTRUCT "Separate rare body construction phases to reduce live registers" OFF)
option(IRISU_CUDA_NATIVE_PERSISTENT_SOLVER "Keep an optional GPU hot-state mirror across island solver phases" OFF)
option(IRISU_CUDA_NATIVE_PARALLEL_PHASES "Enable optional worker-warp physics stages" OFF)
option(IRISU_CUDA_NATIVE_COLLISION_CACHE "Enable optional exact device collision memoization" OFF)
set(IRISU_CUDA_NATIVE_MAX_CONTACTS "4096" CACHE STRING "Typed World/solver contact capacity; native SAP remains 4096")
set_property(CACHE IRISU_CUDA_NATIVE_MAX_CONTACTS PROPERTY STRINGS 512 1024 4096)
set(IRISU_CUDA_NATIVE_ORCHESTRATOR_MAX_CONTACTS "4096" CACHE STRING "Resident search contact capacity; separate from short-horizon benchmark profiles")
set_property(CACHE IRISU_CUDA_NATIVE_ORCHESTRATOR_MAX_CONTACTS PROPERTY STRINGS 512 1024 4096)
if(NOT IRISU_CUDA_NATIVE_ORCHESTRATOR_MAX_CONTACTS MATCHES "^(512|1024|4096)$")
  message(FATAL_ERROR "Resident search contact capacity must be 512, 1024 or 4096")
endif()
if(NOT IRISU_CUDA_NATIVE_MAX_CONTACTS MATCHES "^(512|1024|4096)$")
  message(FATAL_ERROR "Typed contact capacity must be 512, 1024 or 4096")
endif()
set(IRISU_CUDA_NATIVE_MAX_REGISTERS "0" CACHE STRING "Optional register cap for typed game/search kernels (0 uses compiler default)")
if(NOT IRISU_CUDA_NATIVE_MAX_REGISTERS MATCHES "^(0|[1-9][0-9]*)$")
  message(FATAL_ERROR "Invalid typed GPU register cap")
endif()
set(IRISU_CUDA_NATIVE_ARCHITECTURES "70" CACHE STRING "Architectures for separate typed GPU physics")
set(IRISU_CUDA_NATIVE_GAME_FAST_COMPILE "mid" CACHE STRING "Typed game frontend optimization tradeoff: 0, min, mid, max")
set_property(CACHE IRISU_CUDA_NATIVE_GAME_FAST_COMPILE PROPERTY STRINGS 0 min mid max)
if(NOT IRISU_CUDA_NATIVE_GAME_FAST_COMPILE MATCHES "^(0|min|mid|max)$")
  message(FATAL_ERROR "Invalid IRISU_CUDA_NATIVE_GAME_FAST_COMPILE")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13 AND
   "${IRISU_CUDA_NATIVE_ARCHITECTURES}" MATCHES "(^|;)70(;|$)")
  message(FATAL_ERROR "Tesla V100 typed physics requires CUDA 12.x")
endif()

function(irisu_native_cuda target source)
  set(contact_capacity "${IRISU_CUDA_NATIVE_MAX_CONTACTS}")
  if(ARGC GREATER 2)
    set(contact_capacity "${ARGV2}")
  endif()
  add_executable(${target} "${PROJECT_SOURCE_DIR}/${source}")
  set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "${IRISU_CUDA_NATIVE_ARCHITECTURES}"
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--fmad=false;--ftz=false;--prec-div=true;--prec-sqrt=true;-Xcompiler=-ffp-contract=off,-fno-fast-math;-Xptxas=-v>")
  target_link_libraries(${target} PRIVATE CUDA::cudart)
  target_compile_definitions(${target} PRIVATE
    IRISU_CUDA_NATIVE_BOUNDED_MATH=$<BOOL:${IRISU_CUDA_NATIVE_BOUNDED_MATH}>
    IRISU_CUDA_NATIVE_BOUNDED_STORE=$<BOOL:${IRISU_CUDA_NATIVE_BOUNDED_STORE}>
    IRISU_CUDA_NATIVE_LIGHT_SHAPE_SYNC=$<BOOL:${IRISU_CUDA_NATIVE_LIGHT_SHAPE_SYNC}>
    IRISU_CUDA_NATIVE_CACHED_ROTATION=$<BOOL:${IRISU_CUDA_NATIVE_CACHED_ROTATION}>
    IRISU_CUDA_NATIVE_F32_STORE=$<BOOL:${IRISU_CUDA_NATIVE_F32_STORE}>
    IRISU_CUDA_NATIVE_F32_DIV_STORE=$<BOOL:${IRISU_CUDA_NATIVE_F32_DIV_STORE}>
    IRISU_CUDA_NATIVE_DEVICE_TRIG=${IRISU_CUDA_NATIVE_DEVICE_TRIG}
    IRISU_CUDA_NATIVE_SPARSE_RESET=$<BOOL:${IRISU_CUDA_NATIVE_SPARSE_RESET}>
    IRISU_CUDA_NATIVE_INLINE_STEP=$<BOOL:${IRISU_CUDA_NATIVE_INLINE_STEP}>
    IRISU_CUDA_NATIVE_ENTRY_INDEX=$<BOOL:${IRISU_CUDA_NATIVE_ENTRY_INDEX}>
    IRISU_CUDA_NATIVE_WIDE_REBIND=$<BOOL:${IRISU_CUDA_NATIVE_WIDE_REBIND}>
    IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER=$<BOOL:${IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER}>
    IRISU_CUDA_NATIVE_CACHED_VELOCITY=$<BOOL:${IRISU_CUDA_NATIVE_CACHED_VELOCITY}>
    IRISU_CUDA_NATIVE_LOCAL_VELOCITY=$<BOOL:${IRISU_CUDA_NATIVE_LOCAL_VELOCITY}>
    IRISU_CUDA_NATIVE_LOCAL_POSITION=$<BOOL:${IRISU_CUDA_NATIVE_LOCAL_POSITION}>
    IRISU_CUDA_NATIVE_CONTACT_ANCHORS=$<BOOL:${IRISU_CUDA_NATIVE_CONTACT_ANCHORS}>
    IRISU_CUDA_NATIVE_SPLIT_CONSTRUCT=$<BOOL:${IRISU_CUDA_NATIVE_SPLIT_CONSTRUCT}>
    IRISU_CUDA_NATIVE_PERSISTENT_SOLVER=$<BOOL:${IRISU_CUDA_NATIVE_PERSISTENT_SOLVER}>
    IRISU_CUDA_NATIVE_PARALLEL_PHASES=$<BOOL:${IRISU_CUDA_NATIVE_PARALLEL_PHASES}>
    IRISU_CUDA_NATIVE_COLLISION_CACHE=$<BOOL:${IRISU_CUDA_NATIVE_COLLISION_CACHE}>
    IRISU_CUDA_NATIVE_MAX_CONTACTS=${contact_capacity})
endfunction()

irisu_native_cuda(irisu-native-collision tools/cuda-native/collision_runner.cu)
irisu_native_cuda(irisu-native-solver tools/cuda-native/solver_runner.cu)
irisu_native_cuda(irisu-native-interleaved-solver-validate tools/cuda-native/solver_interleaved_validate.cu)
irisu_native_cuda(irisu-native-persistent-solver-validate tools/cuda-native/solver_persistent_validate.cu)
irisu_native_cuda(irisu-native-collision-memo-validate tools/cuda-native/collision_memo_validate.cu)
irisu_native_cuda(irisu-native-parallel-physics-validate tools/cuda-native/parallel_physics_validate.cu)
irisu_native_cuda(irisu-native-cached-velocity-validate tools/cuda-native/solver_velocity_cached_validate.cu)
irisu_native_cuda(irisu-native-local-position-validate tools/cuda-native/solver_position_local_validate.cu)
irisu_native_cuda(irisu-native-bounded-math-validate tools/cuda-native/bounded_math_validate.cu)
irisu_native_cuda(irisu-native-snapshot-validate tools/cuda-native/snapshot_validate.cu)
irisu_native_cuda(irisu-native-search-accel-validate tools/cuda-native/search_accel_validate.cu)
irisu_native_cuda(irisu-native-prefix-guard-validate tools/cuda-native/prefix_guard_validate.cu)
irisu_native_cuda(irisu-native-body-ref-validate tools/cuda-native/body_ref_validate.cu)
irisu_native_cuda(irisu-native-store-math-validate tools/cuda-native/store_math_validate.cu)
irisu_native_cuda(irisu-native-div-store-validate tools/cuda-native/div_store_validate.cu)
irisu_native_cuda(irisu-native-device-trig-validate tools/cuda-native/device_trig_validate.cu)
irisu_native_cuda(irisu-native-bounded-store-validate tools/cuda-native/bounded_store_validate.cu)
irisu_native_cuda(irisu-native-compact-snapshot-validate tools/cuda-native/compact_snapshot_validate.cu)
irisu_native_cuda(irisu-native-compact-game-validate tools/cuda-native/compact_game_snapshot_validate.cu)
irisu_native_cuda(irisu-cuda-native-scene-runner tools/cuda-native/scene_runner.cu)
irisu_native_cuda(irisu-cuda-native-dynamic-scene-runner tools/cuda-native/scene_dynamic_runner.cu)
irisu_native_cuda(irisu-cuda-native-factory-runner tools/cuda-native/factory_runner.cu)
irisu_native_cuda(irisu-native-broadphase tools/cuda-native/broadphase_runner.cu)
irisu_native_cuda(irisu-cuda-native-game-runner tools/cuda-native/game_runner.cu)
irisu_native_cuda(irisu-cuda-native-search-runner tools/cuda-native/search_runner.cu)
irisu_native_cuda(irisu-cuda-native-gpu-reference tools/cuda-native/gpu_search_reference_runner.cu)
irisu_native_cuda(irisu-cuda-native-gpu-search tools/cuda-native/gpu_search_runner.cu)
irisu_native_cuda(irisu-cuda-native-gpu-game-validate tools/cuda-native/gpu_game_validate.cu)
irisu_native_cuda(irisu-cuda-native-batched-search-runner tools/cuda-native/batched_search_runner.cu)
irisu_native_cuda(irisu-cuda-native-instrumented-runner tools/cuda-native/instrumented_runner.cu)
irisu_native_cuda(irisu-cuda-native-instrumented-runner-v2 tools/cuda-native/instrumented_runner_v2.cu)
irisu_native_cuda(irisu-cuda-native-instrumented-runner-v3 tools/cuda-native/instrumented_runner_v3.cu)
foreach(target irisu-cuda-native-game-runner irisu-cuda-native-search-runner irisu-cuda-native-gpu-reference irisu-cuda-native-gpu-search irisu-cuda-native-gpu-game-validate irisu-cuda-native-batched-search-runner irisu-cuda-native-instrumented-runner irisu-cuda-native-instrumented-runner-v2 irisu-cuda-native-instrumented-runner-v3 irisu-native-compact-game-validate)
  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--Ofast-compile=${IRISU_CUDA_NATIVE_GAME_FAST_COMPILE}>")
  if(IRISU_CUDA_NATIVE_MAX_REGISTERS)
    target_compile_options(${target} PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=${IRISU_CUDA_NATIVE_MAX_REGISTERS}>")
  endif()
endforeach()

add_executable(irisu-native-domain "${PROJECT_SOURCE_DIR}/tools/cuda-native/collision_domain.cpp")
target_include_directories(irisu-native-domain PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(irisu-native-domain PRIVATE -ffp-contract=off -fno-fast-math)
foreach(proof cached_rotation compact_padding entry_index shared_future_plan shared_future gpu_search_plan)
  add_executable(irisu-native-${proof}-validate "${PROJECT_SOURCE_DIR}/tools/cuda-native/${proof}_validate.cpp")
  target_include_directories(irisu-native-${proof}-validate PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_options(irisu-native-${proof}-validate PRIVATE -ffp-contract=off -fno-fast-math)
endforeach()
foreach(proof shared_future_plan shared_future gpu_search_plan)
  target_compile_options(irisu-native-${proof}-validate PRIVATE -UNDEBUG)
  target_compile_definitions(irisu-native-${proof}-validate PRIVATE
    IRISU_CUDA_NATIVE_MAX_CONTACTS=${IRISU_CUDA_NATIVE_MAX_CONTACTS}
    IRISU_CUDA_NATIVE_SPARSE_RESET=0 IRISU_CUDA_NATIVE_INLINE_STEP=0
    IRISU_CUDA_NATIVE_ENTRY_INDEX=0)
endforeach()
target_compile_options(irisu-native-shared_future-validate PRIVATE -O1)
if(IRISU_BUILD_TESTS)
  add_test(NAME cuda-native-domain COMMAND irisu-native-domain)
endif()

# Persistent orchestration service, using this build's existing numeric profile.
irisu_native_cuda(irisu-cuda-search-engine tools/orchestrator/cuda_engine.cu ${IRISU_CUDA_NATIVE_ORCHESTRATOR_MAX_CONTACTS})
target_compile_options(irisu-cuda-search-engine PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--Ofast-compile=${IRISU_CUDA_NATIVE_GAME_FAST_COMPILE}>")
if(IRISU_CUDA_NATIVE_MAX_REGISTERS)
  target_compile_options(irisu-cuda-search-engine PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=${IRISU_CUDA_NATIVE_MAX_REGISTERS}>")
endif()
