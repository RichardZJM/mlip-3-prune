# COMPILATION AND INSTALLATION PROCEDURE

## QUICK OVERVIEW

MLIP has a serial and parallel version. Building the serial version of MLIP
requires a modern C++ compiler (supporting c++11), parallel version requires
an MPI C++ compiler, as well as C and FORTRAN compilers.

This fork of MLIP is **dependent** on BLAS and LAPACK.
This fork **cannot** work without these libraries. You are highly advised
to use a high-performance library (Intel MKL or OpenBLAS).
Other implementations (Netlib) are valid but are much slower.

This fork now uses CMake to build. This is to better detect BLAS/LAPACK.

Create the build folder.

```bash
mkdir build && cd build
cmake ..
```

Set the correct configuration options.

Disable MPI support:

```bash
cmake -D USE_MPI=NO ..
```

Optionally, specify a BLAS vendor. See [options](https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors).

```bash
cmake -DBLA_VENDOR=OpenBLAS ..
```

After configuration, the executables can be built with make command:

```bash
make -j <n>
```

The `build/libmlip_interface.a` can be used with the MLIP-LAMMPS interface as usual.

You can run the test suite with:

```bash
ctest --verbose
```
