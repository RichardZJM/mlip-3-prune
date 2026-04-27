#!/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
OUT_DIR=./out
mkdir -p "$OUT_DIR"


# Performs pruning using config.json
"$MLP_EXE" write_blank config.json 
