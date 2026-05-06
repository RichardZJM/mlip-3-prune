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

Optionally disable MPI support:

```bash
cmake -D USE_MPI=NO ..
```

Optionally, specify a BLAS vendor. See the vendor [options](https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors).

```bash
cmake -DBLA_VENDOR=OpenBLAS ..
```

After configuration, the executable can be compiled with make command:

```bash
make -j <n>
```

The executable will be placed in `build/mlp` and `bin/mlp`. This is similar to the original MLIP-3 package. The `lib/lib_mlip_interface.a` can be used with the MLIP-3-LAMMPS interface.

However, it is **not** recommended to interface this fork to LAMMPS due to the BLAS dependency. If you plan to use this fork with the MLIP-3-LAMMPS interface you need to modify the `LAMMPS/Makefile.lammps.template` in the interface code to properly link BLAS.

You can run the test suite with:

```bash
ctest --verbose
```

## macOS / Apple Silicon

The build works on macOS with one extra include flag and a Homebrew toolchain.
Tested on macOS 26.4 / Apple Silicon with AppleClang 21, Homebrew GCC 15.2
(`gfortran`), CMake 4.1, Homebrew OpenMPI 5.0, and Homebrew OpenBLAS.

### Prerequisites (Homebrew)

```bash
brew install gcc          # provides gfortran in /opt/homebrew/bin
brew install openblas     # provides full F77 LAPACK in /opt/homebrew/opt/openblas
brew install open-mpi     # or use any custom MPI build; pin the compilers below
```

> Apple's Accelerate framework is **not recommended**. Its LAPACK F77 entry
> points have a history of mangled / incomplete symbols on Apple Silicon, and
> the mandatory `dposv` / `dsyev` configure-time checks tend to fail against
> it.

### Configure

The fork's `CMakeLists.txt` links OpenBLAS via `find_package(LAPACK)` but
does not add its include path. On Homebrew installs `cblas.h` lives at
`/opt/homebrew/opt/openblas/include`, which Apple Clang does not search by
default — without an explicit `-I` flag the build fails with
`fatal error: 'cblas.h' file not found` in `src/common/blas.h`. Pass the
include path via `CMAKE_CXX_FLAGS`:

```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_Fortran_COMPILER=/opt/homebrew/bin/gfortran \
  -DCMAKE_CXX_FLAGS="-I/opt/homebrew/opt/openblas/include" \
  -DUSE_MPI=ON \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/openblas
make -j 8
```

Configure should report `LAPACK vendor: OpenBLAS` and `Found MPI` against
the chosen MPI compilers; the F77 `dposv` and `dsyev` checks should both
succeed. Build time on Apple Silicon is around 10 s for `make -j 8`.

If multiple MPI implementations are installed (e.g. Homebrew `open-mpi`
plus a custom build), pin `MPI_C_COMPILER`, `MPI_CXX_COMPILER`, and
`MPI_Fortran_COMPILER` explicitly to the desired toolchain to avoid
silent ABI mismatches with downstream MPI consumers (`mpi4py`, etc.).

### Verify the build

After `make -j 8`, both artefacts land at the top level of the repo
(`bin/mlp`, `lib/libmlip_interface.a`):

```bash
./bin/mlp list                 # should list `prune`, `extract_problem`, `mask_*`
ctest --verbose                # 43 / 44 tests pass on Apple Silicon (see below)
```

### Linking against the MLIP-3-LAMMPS interface (Not recommended for this fork)

Because this fork drops upstream MLIP-3's embedded CBLAS, downstream
linkers (e.g. LAMMPS via `interface-lammps-mlip-3`) must explicitly link
OpenBLAS. Two changes are needed in the LAMMPS interface workflow:

1. Copy this fork's static library into the LAMMPS interface directory,
   renaming to match the upstream filename convention
   (`lib_mlip_interface.a`, with the underscore — note that CMake
   produces `libmlip_interface.a` without the leading underscore):

   ```bash
   cp <mlip-3-prune>/lib/libmlip_interface.a \
      <interface-lammps-mlip-3>/lib_mlip_interface.a
   ```

2. After `interface-lammps-mlip-3/preinstall.sh` has populated
   `<lammps>/lib/mlip/Makefile.lammps`, edit that file to add OpenBLAS
   to the link line:

   ```make
   mlip_SYSLIB  = -std=c++11 -lgfortran -lopenblas
   mlip_SYSPATH = -L/opt/homebrew/opt/openblas/lib    # plus any existing -L flags
   ```

   Then re-run the LAMMPS build (`make -j 8 mode=shared mpi` in
   `<lammps>/src/`). Without this step the `liblammps_mpi.so` link
   fails with `Undefined symbols: _cblas_daxpy`.

`preinstall.sh` overwrites `<lammps>/lib/mlip/Makefile.lammps` from its
bundled template each time it runs, so step 2 must be re-applied (or
the bundled template patched) on subsequent rebuilds.

#### Verify the LAMMPS link

```bash
otool -L lmp_mpi                          # should show libopenblas.0.dylib
                                          # and the chosen libmpi
./lmp_mpi -h | grep -A2 'Pair styles'     # `mlip` should be listed
```

A 500-step run of the bundled Cu surface example
(`interface-lammps-mlip-3/example/in.my`) under `mpirun -n 2` exercises
the full `pair_style mlip → libmlip_interface.a → libopenblas` path and
should report ~98 % of wall time in `pair_style mlip`. Two known
caveats are specific to that example, not to the build:

- The bundled `pot.almtp` is just outside the trained domain for the
  initial config, so `extrapolation_control=true` (set in the example)
  triggers a break at step 0. Set it to `false` for a smoke run.
- The example contains a `reset_timestep` after a run with active
  dumps, which trips a hard error in newer LAMMPS releases. The 500
  steps complete cleanly before that point.

### Known issue: ctest Test 44 (`utest_mw3`)

On Apple Silicon, the bundled test
`MLIP_wrapper test3 (Learning, Recording cfgs)` crashes with
`Trace/BPT trap: 5` (SIGTRAP). Tests 1–43 — including `MLIP_wrapper`
tests 1 and 2, MaxVol selection, and all the basic config-I/O / fitting
tests — pass cleanly. The failing test exercises the online-learning +
cfg-recording path; the rest of the build, including the `prune`,
`extract_problem`, and `mask_*` subcommands, is not affected.
