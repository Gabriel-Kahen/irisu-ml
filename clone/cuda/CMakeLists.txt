set(IRISU_CUDA_ARCHITECTURES "70" CACHE STRING "CUDA architectures; Tesla V100 is 70")
find_package(CUDAToolkit 12.9 REQUIRED)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13 AND
   "${IRISU_CUDA_ARCHITECTURES}" MATCHES "(^|;)70(;|$)")
  message(FATAL_ERROR "Tesla V100 sm_70 builds require CUDA 12.x; use the installed CUDA 12.9 toolkit")
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(OpenMP COMPONENTS CXX REQUIRED)
option(IRISU_CUDA_REGION_LOCAL_GUEST "Experimentally cache guest registers inside each region" OFF)
option(IRISU_CUDA_FUNCTION_CALLS "Experimentally translate original functions with native device calls" OFF)
option(IRISU_CUDA_SEPARATE_REGIONS "Compile translated regions as separate device objects" ON)
option(IRISU_CUDA_FAST_COMPILE "Use lowest-optimization CUDA consumer builds for iteration only" OFF)
set(IRISU_CUDA_GAME_FAST_COMPILE "mid" CACHE STRING "Game consumer device compile tradeoff: 0, min, mid, max")
set_property(CACHE IRISU_CUDA_GAME_FAST_COMPILE PROPERTY STRINGS 0 min mid max)
if(NOT IRISU_CUDA_GAME_FAST_COMPILE MATCHES "^(0|min|mid|max)$")
  message(FATAL_ERROR "IRISU_CUDA_GAME_FAST_COMPILE must be 0, min, mid or max")
endif()
set(IRISU_CUDA_SPLIT_COMPILE "4" CACHE STRING "Parallel CUDA optimizer jobs")
set(IRISU_CUDA_HOST_OPTIMIZATION "2" CACHE STRING "Host-code optimization for translated program 0..3")
set(IRISU_CUDA_PTXAS_OPTIMIZATION "1" CACHE STRING "Translated program assembler optimization 0..3")
foreach(level IN ITEMS IRISU_CUDA_HOST_OPTIMIZATION IRISU_CUDA_PTXAS_OPTIMIZATION)
  if(NOT "${${level}}" MATCHES "^[0-3]$")
    message(FATAL_ERROR "${level} must be 0..3")
  endif()
endforeach()

set(IRISU_CUDA_EXACT_HOST "" CACHE FILEPATH "Pinned ELF32 MSVC host to translate for CUDA experiments")

add_custom_target(irisu-cuda-game-source-check
  COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/cuda-exact/runtime_identity.py" check-game
  VERBATIM)

function(irisu_cuda_target target)
  set_target_properties(${target} PROPERTIES
    CUDA_ARCHITECTURES "${IRISU_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES CXX_STANDARD 20)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false;--prec-div=true;--prec-sqrt=true;--expt-extended-lambda>)
  target_link_libraries(${target} PUBLIC CUDA::cudart)
endfunction()

add_library(irisu_cuda_batch STATIC batch.cu)
irisu_cuda_target(irisu_cuda_batch)
target_include_directories(irisu_cuda_batch PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(irisu-cuda-batch-validate ../../tools/cuda-exact/batch_validate.cu)
irisu_cuda_target(irisu-cuda-batch-validate)
target_compile_options(irisu-cuda-batch-validate PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
target_link_libraries(irisu-cuda-batch-validate PRIVATE irisu_cuda_batch OpenMP::OpenMP_CXX)

add_executable(irisu-cuda-arithmetic-validate ../../tools/cuda-exact/arithmetic_validate.cu)
irisu_cuda_target(irisu-cuda-arithmetic-validate)

if(IRISU_CUDA_EXACT_HOST)
  find_package(OpenSSL REQUIRED COMPONENTS Crypto)
  execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import capstone, elftools"
    RESULT_VARIABLE irisu_cuda_python_modules)
  if(NOT irisu_cuda_python_modules EQUAL 0)
    message(FATAL_ERROR "CUDA translation requires capstone and pyelftools in Python3_EXECUTABLE")
  endif()
  set(generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
  # Discover region sources from the pinned image at configure time. Normal
  # incremental builds regenerate through the same explicit dependency rule.
  set(translator "${PROJECT_SOURCE_DIR}/tools/cuda-exact/translate.py")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${translator}" "${IRISU_CUDA_EXACT_HOST}")
  set(translation_options)
  if(IRISU_CUDA_REGION_LOCAL_GUEST)
    list(APPEND translation_options --region-local-guest)
  endif()
  if(IRISU_CUDA_FUNCTION_CALLS)
    list(APPEND translation_options --function-calls)
  endif()
  execute_process(COMMAND "${Python3_EXECUTABLE}" "${translator}"
    "${IRISU_CUDA_EXACT_HOST}" "${generated}" ${translation_options}
    RESULT_VARIABLE translation_status)
  if(NOT translation_status EQUAL 0)
    message(FATAL_ERROR "Pinned CUDA host translation failed")
  endif()
  file(TOUCH "${generated}/translation.stamp")
  include("${generated}/region_sources.cmake")
  set(outputs "${generated}/guest_blocks.inc" "${generated}/guest_callbacks.inc"
    "${generated}/guest_exports.inc" "${generated}/guest_image.bin" "${generated}/manifest.json"
    "${generated}/guest_region_declarations.inc" "${generated}/guest_image_identity.inc" "${generated}/region_sources.cmake"
    ${IRISU_CUDA_GENERATED_REGION_SOURCES})
  foreach(source IN LISTS IRISU_CUDA_GENERATED_REGION_SOURCES)
    string(REGEX REPLACE "\\.cu$" ".inc" region_include "${source}")
    list(APPEND outputs "${region_include}")
  endforeach()
  # Generation is cheap and writes only changed content. Running this target
  # also repairs individually removed byproducts without rebuilding every region.
  add_custom_target(irisu-cuda-translate
    COMMAND "${Python3_EXECUTABLE}" "${translator}" "${IRISU_CUDA_EXACT_HOST}" "${generated}" ${translation_options}
    COMMAND "${CMAKE_COMMAND}" -E touch "${generated}/translation.stamp"
    BYPRODUCTS ${outputs} "${generated}/translation.stamp"
    DEPENDS "${IRISU_CUDA_EXACT_HOST}" "${translator}" VERBATIM)
  if(IRISU_CUDA_SEPARATE_REGIONS)
    add_library(irisu_cuda_regions STATIC ${IRISU_CUDA_GENERATED_REGION_SOURCES})
    add_dependencies(irisu_cuda_regions irisu-cuda-translate)
    irisu_cuda_target(irisu_cuda_regions)
    set_target_properties(irisu_cuda_regions PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(irisu_cuda_regions PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O${IRISU_CUDA_HOST_OPTIMIZATION};-Xptxas=-O${IRISU_CUDA_PTXAS_OPTIMIZATION};--split-compile=${IRISU_CUDA_SPLIT_COMPILE}>)
  endif()
  add_executable(irisu-cuda-image-identity-validate ../../tools/cuda-exact/image_identity_validate.cpp)
  add_dependencies(irisu-cuda-image-identity-validate irisu-cuda-translate)
  target_include_directories(irisu-cuda-image-identity-validate PRIVATE "${generated}")
  target_link_libraries(irisu-cuda-image-identity-validate PRIVATE OpenSSL::Crypto)
  if(IRISU_BUILD_TESTS)
    add_test(NAME irisu-cuda-image-identity-validate COMMAND irisu-cuda-image-identity-validate
      "${generated}/guest_image.bin")
  endif()
  add_executable(irisu-cuda-physics-probe ../../tools/cuda-exact/physics_probe.cu)
  add_dependencies(irisu-cuda-physics-probe irisu-cuda-translate)
  target_include_directories(irisu-cuda-physics-probe PRIVATE "${generated}")
  irisu_cuda_target(irisu-cuda-physics-probe)
  add_executable(irisu-cuda-game-probe ../../tools/cuda-exact/game_probe.cu)
  add_dependencies(irisu-cuda-game-probe irisu-cuda-translate irisu-cuda-game-source-check)
  target_include_directories(irisu-cuda-game-probe PRIVATE "${generated}")
  irisu_cuda_target(irisu-cuda-game-probe)
  add_executable(irisu-cuda-game-segments ../../tools/cuda-exact/game_segments.cu)
  add_dependencies(irisu-cuda-game-segments irisu-cuda-translate irisu-cuda-game-source-check)
  target_include_directories(irisu-cuda-game-segments PRIVATE "${generated}")
  irisu_cuda_target(irisu-cuda-game-segments)
  add_library(irisu_cuda_search STATIC search.cu)
  add_dependencies(irisu_cuda_search irisu-cuda-translate irisu-cuda-game-source-check)
  target_include_directories(irisu_cuda_search PRIVATE "${generated}")
  irisu_cuda_target(irisu_cuda_search)
  target_link_libraries(irisu_cuda_search PUBLIC irisu_cuda_batch)
  add_executable(irisu-cuda-search ../../tools/cuda-exact/search_runner.cpp)
  target_link_libraries(irisu-cuda-search PRIVATE irisu_cuda_search)
  add_executable(irisu-cuda-search-session ../../tools/cuda-exact/search_session_runner.cpp)
  target_link_libraries(irisu-cuda-search-session PRIVATE irisu_cuda_search)
  # These C++ entry points still perform the final CUDA device link. Inherit
  # the device architecture explicitly instead of CMake's host-wide default.
  set_target_properties(irisu-cuda-search irisu-cuda-search-session PROPERTIES
    CUDA_ARCHITECTURES "${IRISU_CUDA_ARCHITECTURES}")
  foreach(probe IN ITEMS irisu-cuda-physics-probe irisu-cuda-game-probe irisu-cuda-game-segments irisu_cuda_search)
    target_link_libraries(${probe} PRIVATE OpenSSL::Crypto)
    target_compile_options(${probe} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O${IRISU_CUDA_HOST_OPTIMIZATION};-Xptxas=-O${IRISU_CUDA_PTXAS_OPTIMIZATION};--split-compile=${IRISU_CUDA_SPLIT_COMPILE};--expt-relaxed-constexpr;-Xcompiler=-ffp-contract=off,-fno-fast-math>)
    if(IRISU_CUDA_FAST_COMPILE)
      target_compile_options(${probe} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--Ofast-compile=max>)
    elseif(NOT probe STREQUAL "irisu-cuda-physics-probe")
      target_compile_options(${probe} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--Ofast-compile=${IRISU_CUDA_GAME_FAST_COMPILE}>)
    endif()
    if(IRISU_CUDA_SEPARATE_REGIONS)
      target_compile_definitions(${probe} PRIVATE IRISU_GUEST_SEPARATE_REGIONS=1)
      set_target_properties(${probe} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
      target_link_libraries(${probe} PRIVATE irisu_cuda_regions)
    endif()
  endforeach()
endif()

add_executable(irisu-cuda-game-rules-validate ../../tools/cuda-exact/game_validate.cpp)
add_dependencies(irisu-cuda-game-rules-validate irisu-cuda-game-source-check)
target_link_libraries(irisu-cuda-game-rules-validate PRIVATE irisu_core)
if(IRISU_BUILD_TESTS)
  add_test(NAME irisu-cuda-game-rules-validate COMMAND irisu-cuda-game-rules-validate 4096)
  set_tests_properties(irisu-cuda-game-rules-validate PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" TIMEOUT 180)
endif()

add_executable(irisu-search-logic-validate ../../tools/cuda-exact/search_logic_validate.cpp)
if(IRISU_BUILD_TESTS)
  add_test(NAME irisu-search-logic-validate COMMAND irisu-search-logic-validate)
endif()

add_executable(irisu-guest-memory-validate ../../tools/cuda-exact/guest_memory_validate.cpp)
add_executable(irisu-guest-stride-validate ../../tools/cuda-exact/guest_stride_validate.cpp)
if(IRISU_BUILD_TESTS)
  add_test(NAME irisu-guest-memory-validate COMMAND irisu-guest-memory-validate)
  add_test(NAME irisu-guest-stride-validate COMMAND irisu-guest-stride-validate)
endif()

if(IRISU_BUILD_TESTS)
  add_test(NAME irisu-cuda-arithmetic-smoke COMMAND irisu-cuda-arithmetic-validate 100000 7)
  add_test(NAME irisu-cuda-batch-smoke COMMAND irisu-cuda-batch-validate
    --parents 16 --fanout 4 --stride 4096 --depths 2 --repeats 1 --cpu-threads 1)
  set_tests_properties(irisu-cuda-arithmetic-smoke irisu-cuda-batch-smoke PROPERTIES
    LABELS "cuda;component" RESOURCE_LOCK irisu_gpu TIMEOUT 60
    ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
endif()
