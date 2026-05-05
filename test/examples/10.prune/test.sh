#!/bin/bash

# Preamble
MLP_EXE=../../../bin/mlp

# Performs pruning using config.json
"$MLP_EXE" prune config.json 
# mpirun -np 4 --map-by core --bind-to core "$MLP_EXE" prune config.json 