# MLIP

This is a fork of [MLIP-3](https://gitlab.com/ashapeev/mlip-3), a software for Machine Learning Interatomic Potentials.
This fork introduces 5 new commands to prune potentials. **BLAS/LAPACK is now a mandatory dependency** and the project is now built with CMake. Please visit the [installation guide](INSTALL.md) for more information.

These five commands are:

| Command           | Description                                                                                                                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prune`           | Prunes an MTP to jointly optimize cost and accuracy. Produces basis function masks.                                                                                                                                       |
| `extract_problem` | Extracts and saves the matrix data needed for pruning.                                                                                                                                                                    |
| `calculate_loss`  | Convenience command to report the loss for a given MTP and dataset. Useful for comparing potentials and seeds.                                                                                                            |
| `mask_blank`      | Applies a mask to an MTP, yielding a usable but unfitted MTP file. You are then expected to fit this potential with `train` before usage, likely with several random seeds.                                               |
| `mask_inherited`  | Applies a mask to an MTP, yielding a fitted MTP file. This fit uses the same radial parameters as the base MTP and refits the linear parameters. You are then expected to refit this potential with `train` before usage. |

Please visit the following examples for more details.

- [Extracting a Problem](test/examples/09.extract_problem/extract_problem.md)
- [Pruning](test/examples/10.prune/prune.md)
- [Writing New MTP Files](test/examples/11.mask_potential/mask_potential.md)

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
