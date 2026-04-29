cmake_minimum_required(VERSION 3.19)

project(MLIP LANGUAGES CXX Fortran)

# Default to Release build (enables optimizations, disables strict assertions)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(USE_MPI "Build with MPI support" ON)

if(USE_MPI)
    find_package(MPI REQUIRED)
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
endif()

# To force a vendor, use: cmake -DBLA_VENDOR=OpenBLAS ..
if(NOT BLA_VENDOR)
    set(BLA_VENDOR All)
endif()
find_package(LAPACK REQUIRED)
message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")
message(STATUS "LAPACK vendor: ${BLA_VENDOR}")

# Fix for Build Error: Ensure directories exist before targets are built
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# NOTE: GLOB_RECURSE will not re-run automatically when source files are
# added or removed. Re-run cmake manually in that case.
file(GLOB_RECURSE LIB_SOURCES
    "src/common/*.cpp"
    "src/drivers/*.cpp"
    "src/test_suite/*.cpp"
    "src/*.cpp"
)
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*/src/mlp/mlp\\.cpp$")

add_library(mlip_interface STATIC ${LIB_SOURCES})
set_target_properties(mlip_interface PROPERTIES PREFIX "lib_")

target_include_directories(mlip_interface PUBLIC src)
target_compile_features(mlip_interface PUBLIC cxx_std_11)

target_link_libraries(mlip_interface PUBLIC ${LAPACK_LIBRARIES})
if(LAPACK_LINKER_FLAGS)
    target_link_options(mlip_interface PUBLIC ${LAPACK_LINKER_FLAGS})
endif()

if(USE_MPI)
    target_link_libraries(mlip_interface PUBLIC MPI::MPI_CXX)
    target_compile_definitions(mlip_interface PUBLIC MLIP_MPI)
endif()

if(BLA_VENDOR MATCHES "Intel" OR LAPACK_LIBRARIES MATCHES "[Mm][Kk][Ll]")
    target_compile_definitions(mlip_interface PUBLIC MLIP_INTEL_MKL)
endif()

add_custom_command(TARGET mlip_interface POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mlip_interface> ${CMAKE_SOURCE_DIR}/lib/
    COMMAND ${CMAKE_COMMAND} -E echo "-- Library: ${CMAKE_SOURCE_DIR}/lib/lib_mlip_interface.a"
    VERBATIM
)

add_executable(mlp src/mlp/mlp.cpp)

target_link_libraries(mlp PRIVATE mlip_interface)

add_custom_command(TARGET mlp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mlp> ${CMAKE_SOURCE_DIR}/bin/
    COMMAND ${CMAKE_COMMAND} -E echo "-- Binary:  ${CMAKE_SOURCE_DIR}/bin/mlp"
    VERBATIM
)

include(CheckFortranFunctionExists)
include(CMakePushCheckState)
cmake_push_check_state()
set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
set(CMAKE_REQUIRED_FLAGS ${LAPACK_LINKER_FLAGS})
check_fortran_function_exists("dposv" HAVE_DPOSV)
check_fortran_function_exists("dsyev" HAVE_DSYEV)
cmake_pop_check_state()

if(NOT HAVE_DPOSV OR NOT HAVE_DSYEV)
    message(FATAL_ERROR
        "Found LAPACK, but it is missing 'dposv' or 'dsyev'. "
        "You may be linking against a minimal BLAS-only library. "
        "Please install a full LAPACK implementation (e.g., liblapack-dev or OpenBLAS).")
endif()

enable_testing()

set_property(TARGET mlip_interface APPEND PROPERTY
    ADDITIONAL_CLEAN_FILES ${CMAKE_SOURCE_DIR}/lib/lib_mlip_interface.a
)
set_property(TARGET mlp APPEND PROPERTY
    ADDITIONAL_CLEAN_FILES ${CMAKE_SOURCE_DIR}/bin/mlp
)

add_test(NAME LegacyTestSuite
         COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_SOURCE_DIR}/test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

message(STATUS "---------------------------------------------------")
message(STATUS " MLIP Build Summary:")
message(STATUS "  Compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  MPI enabled:   ${USE_MPI}")
if(USE_MPI)
    message(STATUS "  MPI compiler:  ${MPI_CXX_COMPILER}")
endif()
message(STATUS "  LAPACK vendor: ${BLA_VENDOR}")
message(STATUS "  LAPACK Libs:   ${LAPACK_LIBRARIES}")
message(STATUS "  Binary Dir:    ${CMAKE_SOURCE_DIR}/bin")
message(STATUS "  Library Dir:   ${CMAKE_SOURCE_DIR}/lib")
message(STATUS "---------------------------------------------------")