#!/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
OUT_DIR=./out


# Performs pruning using config.json
"$MLP_EXE" prune config.json 
# mpirun -np 4 --map-by core --bind-to core "$MLP_EXE" prune config.json 


# Noted that the pruner restarts using successive folders {out_dir}_{n+1}
# Be careful about how to set up your fodler naming schemes

# The pruner supports training and validation accuracy heuristics
# The validation arguments are always required but you can use the same values and paths as the training 

# The config.json looks like this. All elements are mandatory.
# Here are some recommended settings for an example run for MTP 28 SiO with 32 + cores
# {
#   "mtp_file": "data/28.almtp",        MTP file path
#   "pop_size": 1024,       Population size (512 or 1024) is usually good
#   "n_gen": 30000,         Usually 30k is fine, minimum of 5-10k for convergence
#   "time": 86000,              Time limit. Just set this to your wall time
#   "save_interval": 1000,              Save interval to write an output. Useful for convergence analysis and restarting.
#   "regularization": 5e-7,             Regularizaton. Adjust this to balance conditioning and heuristic accuracy.
#   "neigh_count": 46.017803,           Expected neighbor count. Choose this based on the training set of desired simulation.

#   "seed": 42      Random Seed for evo algo

#   "ytwy_train": 4530536105.428369522094727,       # Get this from extract_problen
#   "xtwx_train_file": "data/xtwx_train.bin",        # Get this from extract_problen
#   "xtwy_train_file": "data/xtwy_train.bin",        # Get this from extract_problen

#   "ytwy_val": 1016717595.225109100341797,      # Get this from extract_problen
#   "xtwx_val_file": "data/xtwx_val.bin",        # Get this from extract_problen
#   "xtwy_val_file": "data/xtwy_val.bin",        # Get this from extract_problen

#   "out_dir": "out"
# }
