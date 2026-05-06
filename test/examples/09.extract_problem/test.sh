#!/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
OUT_DIR=./out
mkdir -p "$OUT_DIR"


"$MLP_EXE" extract_problem ni20.almtp train.cfg "$OUT_DIR"/xtwx.bin "$OUT_DIR"/xtwy.bin 

"$MLP_EXE" calculate_loss ni20.almtp train.cfg
