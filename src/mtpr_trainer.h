/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#ifndef MLIP_MTPR_TRAINER_H
#define MLIP_MTPR_TRAINER_H

#include "common/stdafx.h"
#include "non_linear_regression.h"
#include "mtpr.h"
#include "common/bfgs.h"

extern "C" void dposv_(const char *uplo, const int *n, const int *nrhs,
                       double *a, const int *lda, double *b, const int *ldb, int *info);

extern "C" void dsyrk_(const char *uplo, const char *trans, const int *n, const int *k,
                       const double *alpha, const double *a, const int *lda,
                       const double *beta, double *c, const int *ldc);

extern "C" void dgemv_(const char *trans, const int *m, const int *n,
                       const double *alpha, const double *a, const int *lda,
                       const double *x, const int *incx,
                       const double *beta, double *y, const int *incy);

class MTPR_trainer : public NonLinearRegression
{
private:
    double time_lin = 0.0;
    double time_bfgs = 0.0;

    // LINEAR REGRESSION
    std::vector<double> lin_matrix;     // matrix of the quadratic optimization problem A*A^T = b*A^T
    std::vector<double> lin_vector;     // vector of the quadratic optimization problem A*A^T = b*A^T
    double lin_scalar;                  // scalar of the quadratic optimization problem
    int lin_eqn_count;                  // number of equation in the quadratic optimization problem
    std::vector<double> lin_matrix_mpi; // temporary container to add "lin_matrix" from different cores
    std::vector<double> lin_vector_mpi; // temporary container to add "lin_vector" from different cores

    MLMTPR *p_mlmtpr = nullptr; // Pointer to the instance of MTPR potential to be trained
    BFGS bfgs;                  // Instance of BFGS object for optimization
    double bfgs_f;              // Holder of loss_function value in BFGS
    Array1D bfgs_g;             // Holder of loss_function gradient in BFGS
    bool reg_init = true;       // Whether the reg_vector needs to be rebuilt (based on changing of diagonal elements of SLAE)
    bool reg_absolute = false;  // Cached form of reg_mode (avoids string comparison inside SolveSLAE loops)
    bool reg_logged = false;    // Whether the effective regularization has been reported at least once

public:
    int maxits = 1000;               // Max. number of steps in BFGS
    int random_perturb = 0;          // If >0, than simulatied annealing (random shifting of coefficients) is enabled
    bool skip_preinit = false;       // If false, than 75 BFGS iterations are done before the actual training
    double bfgs_conv_tol = 1e-3;     // Convergence criterion for fucntion decrease in BFGS
    bool init_random = false;        // Random initialization of parameters for an uninitialized potential
    bool no_mindist_update = false;  // Automatically adjust mindist in the potential according to the training set
    bool auto_minmax_magmom = false; // Automatically adjust min_magmom and max_magmom in the potential according to the training set
    double reg_param = -1.0;         // Ridge (Tikhonov) parameter lambda. <0 means "not supplied", resolved in the ctor
    std::string reg_mode = "relative"; // "relative": lambda*max(1,H_ii), i.e. uniform Tikhonov in the Jacobi-scaled
                                       // space, so lambda is dimensionless. "absolute": uniform lambda on every coefficient

    // The species coefficients are per-element energy intercepts: they multiply the constant basis function,
    // contribute nothing to forces or stresses, and absorb the arbitrary DFT energy zero. Penalizing them would
    // make the fit depend on that arbitrary reference and, since they are orders of magnitude larger than the
    // basis coefficients, would let their penalty term dominate the whole loss. They are therefore exempt from
    // the user's lambda and carry only SPECIES_REG_FLOOR (declared in mtpr.h, shared with the pruning tools).

    void InitSettings() // Sets correspondence between variables and setting names in settings file
    {
        MakeSetting(maxits, "iteration_limit");
        MakeSetting(skip_preinit, "skip_preinit");
        MakeSetting(bfgs_conv_tol, "tolerance");
        MakeSetting(init_random, "init_random");
        MakeSetting(no_mindist_update, "no_mindist_update");
        MakeSetting(auto_minmax_magmom, "auto-minmax-magmom");
        MakeSetting(reg_param, "regularization");
        MakeSetting(reg_mode, "regularization_mode");
    };

    MTPR_trainer(MLMTPR *_p_mlmtpr, Settings settings, const bool verbose = true) : NonLinearRegression(_p_mlmtpr, settings, verbose), p_mlmtpr(_p_mlmtpr) // Initialization of the settings
    {
        InitSettings();
        ApplySettings(settings);

        // Resolve the mode and the default lambda before printing, so the log shows what is actually used
        if (reg_mode == "relative")
            reg_absolute = false;
        else if (reg_mode == "absolute")
            reg_absolute = true;
        else
            ERROR("Invalid regularization_mode \"" + reg_mode + "\" (expected \"relative\" or \"absolute\")");

        if (reg_param < 0)
        {
            // No unit-independent default exists for an absolute lambda, so it must be given explicitly
            if (reg_absolute)
                ERROR("regularization_mode=absolute requires an explicit --regularization=<double>");
            reg_param = 1e-10; // historical default, relative mode only
        }

        if (verbose)
            PrintSettings();

        // Placeholder seed: overwritten by the first SolveSLAE(), which needs the SLAE diagonal to compute the
        // real values. It exists so that a loss evaluation preceding the first linear solve (mlp calculate_loss,
        // or the BFGS steps before the first LinOptimize) sees a sane penalty. Filling over the vector's actual
        // size avoids writing past the end when the potential's species have not been extended yet - AddSpecies()
        // resizes it later and seeds the new tail itself.
        for (int i = 0; i < (int)p_mlmtpr->reg_vector.size(); i++)
            p_mlmtpr->reg_vector[i] = RegLambda(i);
    };
    ~MTPR_trainer() {};

    void ClearSLAE();                                      // Setting SLAE Matrix and right part to zero
    void SymmetrizeSLAE();                                 // Symmetrization of the SLAE matrix before solving (only upper right part is filled during adding or removing configuration to regression)
    void SolveSLAE(int TS_size);                           // Find the corresponding linear coefficients
    double RegTarget(int i, int n, int TS_size) const;     // Target per-configuration ridge for coefficient i, given the current SLAE diagonal
    double RegLambda(int i) const                         // Ridge lambda applied to coefficient i: the species intercepts only ever see the floor
    {
        return (i < p_mlmtpr->species_count) ? SPECIES_REG_FLOOR : reg_param;
    }
    bool RegRelative(int i) const                         // Whether coefficient i is scaled by the SLAE diagonal. The species floor is always relative,
    {                                                     // so it stays a pure numerical jitter rather than a unit-dependent absolute shift
        return !reg_absolute || (i < p_mlmtpr->species_count);
    }
    void AddToSLAE(Configuration &cfg, double weight = 1); // Adds configuration to regression SLAE. If weight = -1 removes from regression

    void AddSpecies(std::vector<Configuration> &training_set);                                                                     // Extend the species in the MTPR potential if needed
    void LinOptimize(std::vector<Configuration> &training_set);                                                                    // Solve the linear SLAE to find the linear coefficients
    void Train(std::vector<Configuration> &training_set) override;                                                                 //"Main" training function
    void Rescale(std::vector<Configuration> &training_set);                                                                        // Find the optimal scaling parameter
    void NonLinOptimize(std::vector<Configuration> &training_set, int max_iter);                                                   // launch the nonlinear optimization procedure, with Shapeev BFGS
    double FindLoss(std::vector<Configuration> &training_set);                                                                     // Print the loss to the user
    void ExtractProblem(std::vector<Configuration> &training_set, const std::string &matrix_file, const std::string &vector_file); // Extract matrix problem for pruning
    void ProfileCosts(std::vector<Configuration> &training_set, int sample_count);                                                 // Profile the gradient cost per confg (for MPI load-balancing)
};
#endif // MLIP_MTPR_TRAINER_H
