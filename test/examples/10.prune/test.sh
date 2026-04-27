#!/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
OUT_DIR=./out
mkdir -p "$OUT_DIR"


# Performs pruning using config.json
mpirun -np 1 --map-by core --bind-to core "$MLP_EXE" prune config.json 

# Noted that the pruner restarts using successive folders {out_dir}_{n+1}
# Be careful about how to set up your fodler naming schemes

# The config.json looks like this. All elements are mandatory.
# Here are some recommended settings for an example run for MTP 28 SiO with 32 + cores
# {
#   "mtp_file": "data/28.almtp",
#   "pop_size": 1024,
#   "n_gen": 30000,
#   "time": 86000,
#   "save_interval": 1000,
#   "regularization": 5e-7,
#   "neigh_count": 46.017803,

#   "ytwy_train": 4530536105.428369522094727,
#   "xtwx_train_file": "data/xtwx_train.bin",
#   "xtwy_train_file": "data/xtwy_train.bin",

#   "ytwy_val": 1016717595.225109100341797,
#   "xtwx_val_file": "data/xtwx_val.bin",
#   "xtwy_val_file": "data/xtwy_val.bin",

#   "out_dir": "out"
# }
