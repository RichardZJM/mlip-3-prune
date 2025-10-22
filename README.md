# MLIP

This is a fork of [MLIP-3](https://gitlab.com/ashapeev/mlip-3), a software for Machine Learning Interatomic Potentials.
MLIP-3 was been developed at Skoltech (Moscow) by
Alexander Shapeev, Evgeny Podryabinkin, Konstantin Gubaev, and Ivan Novikov

The paper for this fork is available [here]().

## Additional Functionality

This fork introduces two additional commands to the `mlp` binary to support [MTP basis optimization](https://github.com/RichardZJM/MTP_basis_optimization). An example is available in `test/examples/09.extract_problem`.

1. `extract_problem path/to/potential path/to/dataset /path/to/write/xtwx /path/to/write/xtwy [options]`

   This command extracts the matrix problem. It accepts:

   - A fitted potential
   - A training dataset

   And outputs:

   - The $\mathbf{X}^\intercal\mathbf{WX}$ matrix (binary file)
   - The $\mathbf{X}^\intercal\mathbf{Wy}$ vector (binary file)
   - The $\mathbf{y}^\intercal\mathbf{Wy}$ value (scalar)
   - The average number of neighbors (scalar)

   The binary files are written to the specified paths, and the scalar values are printed to the command line.

2. `calculate_loss path/to/potential path/to/dataset [options]`

   This is a convenience command that exposes the loss to facilitate comparisons between potentials. It was used in the paper to evaluate the training loss of potentials.

Both these commands can accept the same options as the `train` command. These options are most relevant for parameters that affect the loss functions such as `energy_weight`, `force_weight`, `stress_weight`, `weight_scaling` and `weight_scaling_forces`. Since they use the same trainer constructor as the `train` command, there may be a warning that the potential will be overwritten—this is vestigial, neither of these two commands overwritten the input files.

The codebase is compiled the same as MLIP-3.

## Licence

See [LICENSE](LICENSE)

## Prerequisties

- g++, gcc, gfortran, mpicxx
- Alterlatively: the same set for Intel compilers (tested with the 2017 version)
- make

## Compile

For full instructions see [INSTALL.md](INSTALL.md).

You might also be interested in LAMMPS-MLIP interface distributed here:
[https://gitlab.com/ivannovikov/interface-lammps-mlip-3/](https://gitlab.com/ivannovikov/interface-lammps-mlip-3/)

## Getting Started

Have a look at `doc/paper.pdf` first.

Check the usage examples at `test/examples/`

## Have questions?

Note that we'll not be able to answer all of your questions.
As a rule, we are supporting only the documented functionality of MLIP.
If you think you found a bug or an inconsistency in the documentation or usage examples,
please create a Gitlab.com issue.
