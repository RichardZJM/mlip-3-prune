## Prune

`prune` is used to perform the pruning using NSGA-II.

**Important for parallel performance! Make to bind MPI processes and force BLAS threads to 1!**

The syntax is
`prune config.json`

Where

- `config.json` is the file path to the configuration JSON file.

The configuration file contains exactly these fields. All fields are mandatory. The below shows some recommend values for an MTP of level 28. Some values like the matrices, regularization, and the neighbor count depend on the problem. The values below are for the SiO dataset by Karim et al.

```json
{
  "mtp_file": "data/28.almtp",              # Path to the base potential
  "pop_size": 1024,             # Population Size. Also depends on the number of MPI processes.
  "n_gen": 30000,               # Number of generations. Usually 10k is the minium for convergence.
  "time": 86000,                # Time limit.
  "save_interval": 1000,                     # Interval to save results. Use for restarting and checking convergence.
  "regularization": 1e-9,                   # Tikhonov regularization value. Use to balance conditioning and heuristic accuracy.
  "neigh_count": 46.017803,             # Expected number of average neighbors during inference.

  "seed": 42,               # Random seeding
  "max_fill": 2,                # Aggressiveness of Lamarckian operator. Usually 2 is good. Use at least 1 in most cases.

# Training set values and file paths
  "ytwy_train": 4530536105.428369522094727,
  "xtwx_train_file": "data/xtwx_train.bin",
  "xtwy_train_file": "data/xtwy_train.bin",

# Validation set values and file paths. The training set entries can be used  if no validation set is available.
  "ytwy_val": 1016717595.225109100341797,
  "xtwx_val_file": "data/xtwx_val.bin",
  "xtwy_val_file": "data/xtwy_val.bin",

  "out_dir": "out",             # Output directory. Will add a timestamp.
  "restart_from": ""                # Restart directory.
}
```

## Output

Here is the example script's output.

```
Initializing Pruner...
Condition Number: 8.193250e+11
Base SSE: 6.26928
Base Cost: 3.19038e+06
Saving results to: out_20260505_165622
Evaluating initial population...
Generation 0/3 | Evaluations: 96 | Elapsed: 5.88s
Saving intermediate results at generation 2...
Generation 3/3 | Evaluations: 384 | Elapsed: 18.10s

Optimization finished in 18.10s.
Results saved to out_20260505_165622/pareto_final_*.csv
Evaluation times per process:[18.10s]
Average fitness evaluation time:         99.96%
Communication overhead (Estimated):      0.00%
Load imbalance (Estimated):              0.00%
Pruning complete.
```
