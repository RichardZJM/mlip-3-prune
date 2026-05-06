#!/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
OUT_DIR=./out
mkdir -p "$OUT_DIR"

# Write blank potential
"$MLP_EXE" mask_blank ../10.prune/data/28.almtp population.csv 0 out/blank.mtp

# Write inherited potential (use the same config file as during pruning)
"$MLP_EXE" mask_inherited ../10.prune/data/28.almtp config.json population.csv 0 out/inherited.mtp