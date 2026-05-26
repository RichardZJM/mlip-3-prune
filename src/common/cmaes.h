/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   CMA-ES (Covariance Matrix Adaptation Evolution Strategy) engine.
 *   Mirrors the interface style of bfgs.h / bfgs.cpp.
 *
 *   Algorithm reference:
 *     Hansen, N. (2016). "The CMA Evolution Strategy: A Tutorial".
 *     arXiv:1604.00772.
 *
 *   This is a faithful implementation of the (mu/mu_w, lambda)-CMA-ES
 *   with the following standard components:
 *
 *     Sampling:
 *       x_k = m + sigma * B D z_k,   z_k ~ N(0, I)
 *       where C = B D^2 B^T is the eigendecomposition of the covariance matrix.
 *       B   : orthogonal matrix of eigenvectors (n x n)
 *       D   : diagonal matrix of eigenvalue square roots (standard deviations)
 *
 *     Selection & recombination:
 *       mu = floor(lambda / 2)  candidate solutions used as parents
 *       w_i = (ln(mu+0.5) - ln(i)) / sum_j(ln(mu+0.5) - ln(j))  for i=1..mu
 *       mu_eff = 1 / sum(w_i^2)   effective number of parents
 *       m' = sum_i w_i x_{i:lambda}   (weighted mean of top mu candidates)
 *
 *     Step-size control (CSA — Cumulative Step-size Adaptation):
 *       p_sigma += (1-cs) * p_sigma + sqrt(cs*(2-cs)*mu_eff) * B z_w
 *       sigma   *= exp(cs/ds * (||p_sigma||/chiN - 1))
 *       where chiN = sqrt(n)*(1 - 1/(4n) + 1/(21n^2)) is E[||N(0,I)||]
 *
 *     Covariance matrix adaptation (CMA):
 *       Rank-1 update via evolution path p_c:
 *         p_c += (1-cc)*p_c + h_sigma * sqrt(cc*(2-cc)*mu_eff) * B D z_w
 *       Rank-mu update:
 *         C' = (1 - c1 - cmu) C
 *            + c1 (p_c p_c^T + delta_h C)
 *            + cmu sum_i w_i y_{i:lambda} y_{i:lambda}^T
 *         where y_i = (x_i - m_old) / sigma_old
 *
 *     Eigendecomposition:
 *       Recomputed every floor(1/(10*n*(c1+cmu))) generations.
 *       Uses a symmetric Jacobi iteration (no external dependency).
 *
 *   Default hyperparameters are the standard Hansen 2016 defaults:
 *     lambda    = 4 + floor(3 * ln(n))       (population size)
 *     mu        = floor(lambda / 2)           (parents)
 *     sigma0    = 0.3 * (ub - lb)_avg        (initial step size)
 *     cs        = (mu_eff + 2) / (n + mu_eff + 5)
 *     ds        = 1 + 2*max(0, sqrt((mu_eff-1)/(n+1))-1) + cs
 *     cc        = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
 *     c1        = 2 / ((n+1.3)^2 + mu_eff)
 *     cmu       = min(1-c1, 2*(mu_eff-2+1/mu_eff)/((n+2)^2+mu_eff))
 *
 *   Stopping criteria (checked after each generation):
 *     - max_iter reached
 *     - wall-clock timeout (managed externally; CMAES just exposes generation count)
 *     - sigma < sigma_tol  (step size collapsed)
 *     - condition(C) > cond_tol  (covariance ill-conditioned)
 *     - tolfun: range of best values over last 10+floor(30*n/lambda) gens < tolfun
 *
 *   LAPACK-style external declaration required (matches project style):
 *     None — eigendecomposition uses a self-contained symmetric Jacobi solver
 *     to avoid adding a new LAPACK dependency (dsyev).
 *
 *   Note on scaling:
 *     The engine itself knows nothing about MTP scaling.  The caller is
 *     responsible for tracking the best scaling found for each candidate and
 *     reporting it via UpdateAfterEval so it can be stored alongside the
 *     global best position.  See global_best_scaling below.
 */

#ifndef MLIP_CMAES_H
#define MLIP_CMAES_H

#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <string>

// ---------------------------------------------------------------------------
// CMAES
// ---------------------------------------------------------------------------
class CMAES
{
public:
    // -----------------------------------------------------------------------
    // Hyperparameters — set before Init() or leave as defaults.
    // Defaults of -1 mean "compute from n at Init() time" per Hansen 2016.
    // -----------------------------------------------------------------------
    int lambda = -1;          //!< population size (-1 = auto: 4+floor(3*ln(n)))
    int mu = -1;              //!< parents (-1 = auto: floor(lambda/2))
    double sigma0 = -1.0;     //!< initial step size (-1 = auto: 0.3 * init_range)
    double init_range = 2e-2; //!< used for sigma0 auto: sigma0 = 0.3 * init_range
    double sigma_tol = 1e-11; //!< stop if sigma < sigma_tol
    double cond_tol = 1e14;   //!< stop if cond(C) > cond_tol
    double tolfun = 1e-11;    //!< stop if best-value range over history < tolfun
    unsigned int seed = 0;    //!< RNG seed (0 = random device)

    // -----------------------------------------------------------------------
    // State (read-only from outside after Init)
    // -----------------------------------------------------------------------
    int n = 0;          //!< problem dimension
    int generation = 0; //!< current generation index
    double global_best_val = std::numeric_limits<double>::max();

    //! The auxiliary per-candidate scalar that the caller passes in alongside
    //! fitness values.  The engine stores the value corresponding to the
    //! all-time best fitness here so the caller can retrieve it without
    //! having to maintain its own parallel tracking.  For the MTP use-case
    //! this holds the best scaling factor found for the best candidate.
    //!
    //! IMPORTANT: Set this to a sensible initial value BEFORE calling Init().
    //! Init() does NOT reset it to zero, so the value survives across the
    //! Init() call and serves as the fallback if no candidate ever improves
    //! on the initial configuration.
    double global_best_aux = 0.0;

    const std::vector<double> &global_best_pos() const { return best_x_; }

    // Current generation candidates — read by caller to evaluate fitness.
    // candidates[k] is the position vector for candidate k.
    // Size = lambda after Init / PrepareGeneration.
    std::vector<std::vector<double>> candidates;

    // -----------------------------------------------------------------------
    // Init: set up the swarm.
    //   mean0     : initial mean (n-dimensional); centroid of the search.
    //               No bounds are enforced — the search is unconstrained.
    //               init_range is used only for automatic sigma0 scaling.
    // -----------------------------------------------------------------------
    void Init(const std::vector<double> &mean0,
              unsigned int rng_seed = 0);

    // -----------------------------------------------------------------------
    // PrepareGeneration: sample lambda new candidates into `candidates`.
    // Call at the start of each generation before dispatching evaluations.
    // -----------------------------------------------------------------------
    void PrepareGeneration();

    // -----------------------------------------------------------------------
    // UpdateAfterEval: update mean, sigma, and C given fitness values.
    //   fitness[k]  = objective for candidates[k]  (lower is better).
    //   aux[k]      = caller-defined scalar associated with candidates[k].
    //                 The value from the best-ever candidate is stored in
    //                 global_best_aux.  Pass an empty vector to skip.
    // Returns:
    //   0 = continue
    //   1 = stop (sigma collapsed)
    //   2 = stop (covariance ill-conditioned)
    //   3 = stop (tolfun: function value range too small)
    // -----------------------------------------------------------------------
    int UpdateAfterEval(const std::vector<double> &fitness,
                        const std::vector<double> &aux = {});

    // Convenience: did the last UpdateAfterEval improve the global best?
    bool LastGenImproved() const { return last_gen_improved_; }

    // -----------------------------------------------------------------------
    // Summary string for logging
    // -----------------------------------------------------------------------
    std::string StatusString() const;

private:
    // -----------------------------------------------------------------------
    // Internal CMA-ES state
    // -----------------------------------------------------------------------
    int lambda_ = 0;
    int mu_ = 0;

    std::vector<double> weights_; //!< recombination weights w_i (mu)
    double mu_eff_ = 0.0;         //!< effective mu

    // Strategy parameters (computed from n, mu_eff at Init time)
    double cs_ = 0.0;   //!< step-size cumulation rate
    double ds_ = 0.0;   //!< step-size damping
    double cc_ = 0.0;   //!< covariance cumulation rate
    double c1_ = 0.0;   //!< rank-1 learning rate
    double cmu_ = 0.0;  //!< rank-mu learning rate
    double chiN_ = 0.0; //!< E[||N(0,I)||]

    // Distribution parameters
    std::vector<double> mean_;     //!< current mean m (n)
    double sigma_;                 //!< current step size
    std::vector<double> p_sigma_;  //!< step-size evolution path (n)
    std::vector<double> p_c_;      //!< covariance evolution path (n)
    std::vector<double> C_;        //!< covariance matrix, row-major (n x n)
    std::vector<double> B_;        //!< eigenvectors of C, column-major (n x n)
    std::vector<double> D_;        //!< sqrt(eigenvalues) of C (n)
    std::vector<double> invsqrtC_; //!< C^{-1/2} = B D^{-1} B^T (n x n), for CSA

    // Per-generation scratch
    std::vector<std::vector<double>> z_; //!< N(0,I) samples used for candidates (lambda x n)

    // Best solution
    std::vector<double> best_x_;
    bool last_gen_improved_ = false;

    // Eigendecomposition schedule
    int eigen_freq_ = 1;    //!< recompute eigen every this many generations
    int eigen_counter_ = 0; //!< generations since last eigendecomposition

    // History of best values for tolfun stopping criterion
    std::vector<double> best_val_history_;
    int tolfun_window_ = 0;

    // RNG
    std::mt19937 rng_;

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    // Compute eigendecomposition C = B D^2 B^T using symmetric Jacobi.
    // Also updates invsqrtC_ = B D^{-1} B^T.
    void UpdateEigen_();

    // Symmetric Jacobi eigendecomposition (in-place on a copy of C).
    // eigvecs: column-major n x n;  eigvals: n  (may be unsorted on input).
    static void SymmetricJacobi_(int n,
                                 std::vector<double> &A,        // n x n, modified in-place
                                 std::vector<double> &eigvecs,  // n x n col-major output
                                 std::vector<double> &eigvals); // n output

    // Matrix-vector product y = M x (row-major M, n x n)
    static void Matvec_(const std::vector<double> &M,
                        const std::vector<double> &x,
                        std::vector<double> &y,
                        int n_rows, int n_cols);

    // Compute strategy parameters from n and mu_eff
    void ComputeStrategyParams_();
};

#endif // MLIP_CMAES_H