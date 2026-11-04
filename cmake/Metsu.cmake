option(IRISU_BUILD_METSU "Build the separate v2.03 Metsu simulator" OFF)
if(NOT IRISU_BUILD_METSU)
  return()
endif()

if(IRISU_BUILD_TESTS)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME irisu-metsu-state-comparison-tests COMMAND
      "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_metsu_state_comparison.py")
    add_test(NAME irisu-metsu-original-capture-tests COMMAND
      "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_original_replay_runner.py")
  endif()
endif()

add_library(irisu_metsu_core STATIC
  clone/metsu/simulator.cpp clone/metsu/rules.cpp)
set_target_properties(irisu_metsu_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
target_link_libraries(irisu_metsu_core PUBLIC irisu_core)
irisu_enable_warnings(irisu_metsu_core)

add_executable(irisu-metsu-headless clone/metsu/main.cpp)
target_link_libraries(irisu-metsu-headless PRIVATE irisu_metsu_core)
irisu_enable_warnings(irisu-metsu-headless)

add_executable(irisu-metsu-generate-controlled tools/metsu/generate_controlled_replay.cpp)
target_link_libraries(irisu-metsu-generate-controlled PRIVATE irisu_metsu_core)
irisu_enable_warnings(irisu-metsu-generate-controlled)

add_executable(irisu-metsu-trace tools/metsu/replay_trace.cpp)
target_link_libraries(irisu-metsu-trace PRIVATE irisu_metsu_core)
irisu_enable_warnings(irisu-metsu-trace)
if(IRISU_PHYSICS_BACKEND STREQUAL "exact-msvc")
  irisu_verify_exact_runpath(irisu-metsu-generate-controlled)
  target_compile_definitions(irisu-metsu-trace PRIVATE IRISU_METSU_TRACE_PHYSICS="exact-msvc9-r58-metsu-forward")
  irisu_verify_exact_runpath(irisu-metsu-trace)
else()
  target_compile_definitions(irisu-metsu-trace PRIVATE IRISU_METSU_TRACE_PHYSICS="portable-gnu-r58")
endif()

if(IRISU_PHYSICS_BACKEND STREQUAL "exact-msvc")
  irisu_verify_exact_runpath(irisu-metsu-headless)
  add_executable(irisu-metsu-exact-worker tools/exact-physics-prototype/ipc_worker.cpp)
  target_link_libraries(irisu-metsu-exact-worker PRIVATE irisu_metsu_core)
  target_link_options(irisu-metsu-exact-worker PRIVATE "LINKER:-z,now")
  target_compile_definitions(irisu-metsu-exact-worker PRIVATE
    IRISU_METSU_WORKER=1
    IRISU_EXACT_LIBRARY_SHA256="${irisu_exact_box2d_sha256}"
    IRISU_PHYSICS_BACKEND="exact-msvc9-r58-metsu-forward")
  irisu_enable_warnings(irisu-metsu-exact-worker)
  irisu_verify_exact_runpath(irisu-metsu-exact-worker)
  if(IRISU_BUILD_TESTS AND Python3_Interpreter_FOUND)
    add_test(NAME irisu-metsu-client-tests COMMAND "${CMAKE_COMMAND}" -E env
      "IRISU_METSU_WORKER=$<TARGET_FILE:irisu-metsu-exact-worker>"
      "IRISU_EXACT_WORKER=$<TARGET_FILE:irisu-exact-worker>"
      "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_metsu_client.py")
    set_tests_properties(irisu-metsu-client-tests PROPERTIES PROCESSORS 4)
  endif()
endif()

if(IRISU_BUILD_TESTS AND IRISU_PHYSICS_BACKEND STREQUAL "portable")
  add_executable(irisu-metsu-tests tests/native/test_metsu.cpp)
  target_link_libraries(irisu-metsu-tests PRIVATE irisu_metsu_core)
  target_compile_options(irisu-metsu-tests PRIVATE -UNDEBUG)
  add_test(NAME irisu-metsu-tests COMMAND irisu-metsu-tests
    "${CMAKE_CURRENT_SOURCE_DIR}/reference/metsu-level-parameters-pc64.tsv")
  add_executable(irisu-metsu-component-tests tests/native/test_metsu_components.cpp)
  target_link_libraries(irisu-metsu-component-tests PRIVATE irisu_metsu_core)
  target_compile_options(irisu-metsu-component-tests PRIVATE -UNDEBUG)
  add_test(NAME irisu-metsu-component-tests COMMAND irisu-metsu-component-tests
    "${CMAKE_CURRENT_SOURCE_DIR}/reference/metsu-contacts-original-components.json")
endif()
