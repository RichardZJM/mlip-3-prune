#!/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
OUT_DIR=./out
mkdir -p "$OUT_DIR"

# Both commands will warn that the file will be overwritten. 
# That warning is vestigial — these commands do not overwrite potentials.

# This command extracts the necessary inputs to run the evolutionary algorithm.
# Specifically, it produces the XTWX matrix (binary file), XTWY vector (binary file),
# YTY (scalar), and the average number of neighbours (scalar).
# Scalars are printed to the console.
# extract_problem takes the same options as the mlp train command. 
# This is only relevant if the options affect the loss, 
# such as using different weights for energies and forces.

"$MLP_EXE" extract_problem ni20.almtp train.cfg "$OUT_DIR"/xtwx.bin "$OUT_DIR"/xtwy.bin 

# Calculate loss. This is a convenience function that exposes the objective function 
# to facilitate comparisons between potentials.
# extract_problem takes the same options as the mlp train command. 
# This is only relevant if the options affect the loss, 
# such as using different weights for energies and forces.

"$MLP_EXE" calculate_loss ni20.almtp train.cfg
