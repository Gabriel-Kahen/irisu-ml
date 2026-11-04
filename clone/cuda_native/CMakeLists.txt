find_package(CUDAToolkit 12.9 REQUIRED)
set(IRISU_CUDA_NATIVE_ARCHITECTURES "70" CACHE STRING "Architectures for separate typed GPU physics")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13 AND
   "${IRISU_CUDA_NATIVE_ARCHITECTURES}" MATCHES "(^|;)70(;|$)")
  message(FATAL_ERROR "Tesla V100 typed physics requires CUDA 12.x")
endif()

function(irisu_native_cuda target source)
  add_executable(${target} "${PROJECT_SOURCE_DIR}/${source}")
  set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "${IRISU_CUDA_NATIVE_ARCHITECTURES}"
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false;--prec-div=true;--prec-sqrt=true;-Xcompiler=-ffp-contract=off,-fno-fast-math;-Xptxas=-v>")
  target_link_libraries(${target} PRIVATE CUDA::cudart)
endfunction()

irisu_native_cuda(irisu-native-collision tools/cuda-native/collision_runner.cu)
irisu_native_cuda(irisu-native-solver tools/cuda-native/solver_runner.cu)
irisu_native_cuda(irisu-cuda-native-scene-runner tools/cuda-native/scene_runner.cu)
irisu_native_cuda(irisu-cuda-native-factory-runner tools/cuda-native/factory_runner.cu)
irisu_native_cuda(irisu-native-broadphase tools/cuda-native/broadphase_runner.cu)

add_executable(irisu-native-domain "${PROJECT_SOURCE_DIR}/tools/cuda-native/collision_domain.cpp")
target_include_directories(irisu-native-domain PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(irisu-native-domain PRIVATE -ffp-contract=off -fno-fast-math)
if(IRISU_BUILD_TESTS)
  add_test(NAME cuda-native-domain COMMAND irisu-native-domain)
endif()
