#!/bin/bash

# Preamble
MLP_EXE=../../../bin/mlp

# Performs pruning using config.json
"$MLP_EXE" prune config.json 


# Example MPI invocation
# export OPENBLAS_NUM_THREADS=1
# export OMP_NUM_THREADS=1
# export OMP_DYNAMIC=FALSE
# export MKL_NUM_THREADS=1
# export MKL_DYNAMIC=FALSE
# export BLIS_NUM_THREADS=1
# export VECLIB_MAXIMUM_THREADS=1
# export GOTO_NUM_THREADS=1
# export FLEXIBLAS_NUM_THREADS=1

# mpirun -np 4 --map-by core --bind-to core "$MLP_EXE" prune config.json 