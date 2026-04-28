cmake_minimum_required(VERSION 3.10)

# 1. Project Setup
# We enable Fortran because BLAS/LAPACK are Fortran-based; 
# this allows CMake to automatically find the correct Fortran runtime libs.
project(MLIP LANGUAGES CXX Fortran)

# Compilation Standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Options
option(USE_MPI "Build with MPI support" ON)

# 3. Find MPI
if(USE_MPI)
    find_package(MPI REQUIRED)
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
endif()

# 4. Find LAPACK & BLAS
# This is the "magic" block. It finds MKL, OpenBLAS, FlexiBLAS, or Netlib.
# To force a vendor, use: cmake -DBLA_VENDOR=OpenBLAS ..
if(NOT BLA_VENDOR)
    set(BLA_VENDOR All)
endif()
find_package(LAPACK REQUIRED)
message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")

# 5. Source Discovery
# Collect all .cpp files in src, excluding the main app (mlp.cpp)
file(GLOB_RECURSE LIB_SOURCES 
    "src/common/*.cpp"
    "src/drivers/*.cpp"
    "src/test_suite/*.cpp"
    "src/*.cpp"
)
# Remove the main entry point from the library sources
list(FILTER LIB_SOURCES EXCLUDE REGEX "src/mlp/mlp.cpp")

# 6. Create the Interface Library (mlip_interface)
add_library(mlip_interface STATIC ${LIB_SOURCES})

# Include directories
target_include_directories(mlip_interface PUBLIC src)

# Link LAPACK/BLAS and MPI to the library
target_link_libraries(mlip_interface PUBLIC ${LAPACK_LIBRARIES})

if(USE_MPI)
    target_link_libraries(mlip_interface PUBLIC MPI::MPI_CXX)
    target_compile_definitions(mlip_interface PUBLIC MLIP_MPI)
endif()

# Check for MKL specific flag
if(LAPACK_LIBRARIES MATCHES "mkl")
    target_compile_definitions(mlip_interface PUBLIC MLIP_INTEL_MKL)
endif()

# 7. Create the Main Executable (mlp)
add_executable(mlp src/mlp/mlp.cpp)
target_link_libraries(mlp PRIVATE mlip_interface)

# 8. Sanity Check: Robustness for dposv and dsyev
# This ensures the user's LAPACK version isn't "lite" or missing the solvers.
include(CheckFortranFunctionExists)
set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
check_fortran_function_exists("dposv" HAVE_DPOSV)
check_fortran_function_exists("dsyev" HAVE_DSYEV)

if(NOT HAVE_DPOSV OR NOT HAVE_DSYEV)
    message(FATAL_ERROR "
    ERROR: Found LAPACK, but it is missing 'dposv' or 'dsyev'.
    You may be linking against a minimal BLAS-only library.
    Please install a full LAPACK implementation (e.g., liblapack-dev or OpenBLAS).")
endif()

# 9. Legacy Test Support
enable_testing()

# Create a symlink in the root so the legacy 'test/Makefile' can find the binary
add_custom_command(TARGET mlp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:mlp> ${CMAKE_SOURCE_DIR}/mlp
    COMMENT "Creating symlink to binary for legacy test scripts"
)

# Register the legacy test directory as a ctest target
add_test(NAME LegacyTestSuite 
         COMMAND make -C ${CMAKE_SOURCE_DIR}/test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 10. Summary Verbosity
message(STATUS "---------------------------------------------------")
message(STATUS " MLIP Build Summary:")
message(STATUS "  Compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  MPI enabled:   ${USE_MPI}")
message(STATUS "  LAPACK Libs:   ${LAPACK_LIBRARIES}")
message(STATUS "  Binary Dir:    ${CMAKE_BINARY_DIR}")
message(STATUS "---------------------------------------------------")