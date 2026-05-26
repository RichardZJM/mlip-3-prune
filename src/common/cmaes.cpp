/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include "cmaes.h"
#include <cstring>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <numeric>

// ===========================================================================
//  Init
// ===========================================================================
void CMAES::Init(const std::vector<double> &mean0,
                 unsigned int rng_seed)
{
    n = (int)mean0.size();
    assert(n > 0);

    // RNG
    if (rng_seed == 0)
    {
        std::random_device rd;
        rng_.seed(rd());
    }
    else
        rng_.seed(rng_seed);

    // Population sizes
    if (lambda <= 0)
        lambda_ = 4 + (int)std::floor(3.0 * std::log((double)n));
    else
        lambda_ = lambda;

    if (mu <= 0)
        mu_ = lambda_ / 2;
    else
        mu_ = mu;

    assert(mu_ > 0 && mu_ < lambda_);

    // Recombination weights  w_i = ln(mu+0.5) - ln(i)  for i = 1..mu, normalised
    weights_.resize(mu_);
    double wsum = 0.0;
    for (int i = 0; i < mu_; ++i)
    {
        weights_[i] = std::log(mu_ + 0.5) - std::log(i + 1.0);
        wsum += weights_[i];
    }
    for (int i = 0; i < mu_; ++i)
        weights_[i] /= wsum;

    // Effective number of parents
    double w2sum = 0.0;
    for (int i = 0; i < mu_; ++i)
        w2sum += weights_[i] * weights_[i];
    mu_eff_ = 1.0 / w2sum;

    ComputeStrategyParams_();

    // Initial step size — use init_range if sigma0 not explicitly set
    if (sigma0 <= 0.0)
        sigma_ = 0.3 * init_range;
    else
        sigma_ = sigma0;

    // Initialise distribution
    mean_ = mean0;
    p_sigma_.assign(n, 0.0);
    p_c_.assign(n, 0.0);

    // C = I, B = I, D = 1
    C_.assign(n * n, 0.0);
    B_.assign(n * n, 0.0);
    D_.assign(n, 1.0);
    invsqrtC_.assign(n * n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        C_[i * n + i] = 1.0;
        B_[i * n + i] = 1.0; // column-major: B_(i,i) = B_[i*n+i]
        invsqrtC_[i * n + i] = 1.0;
    }

    // Eigendecomposition frequency: Hansen recommends every 1/(10*n*(c1+cmu))
    eigen_freq_ = std::max(1, (int)std::floor(1.0 / (10.0 * n * (c1_ + cmu_))));
    eigen_counter_ = 0;

    // Best solution
    // Note: global_best_aux is NOT reset here — the caller sets it before
    // calling Init() to seed the warm-start value (e.g. current MTP scaling).
    // global_best_val is reset unconditionally since we have no prior fitness.
    best_x_ = mean0;
    global_best_val = std::numeric_limits<double>::max();
    last_gen_improved_ = false;
    generation = 0;

    // tolfun history window: 10 + floor(30*n/lambda)
    tolfun_window_ = 10 + (int)std::floor(30.0 * n / lambda_);
    best_val_history_.clear();

    // Allocate candidate and z arrays
    candidates.assign(lambda_, std::vector<double>(n, 0.0));
    z_.assign(lambda_, std::vector<double>(n, 0.0));
}

// ===========================================================================
//  ComputeStrategyParams_
//  All formulae from Hansen 2016 (Table 1, p.28 of arXiv:1604.00772).
// ===========================================================================
void CMAES::ComputeStrategyParams_()
{
    const double nd = (double)n;

    // Step-size control
    cs_ = (mu_eff_ + 2.0) / (nd + mu_eff_ + 5.0);
    ds_ = 1.0 + 2.0 * std::max(0.0, std::sqrt((mu_eff_ - 1.0) / (nd + 1.0)) - 1.0) + cs_;

    // Covariance adaptation
    cc_ = (4.0 + mu_eff_ / nd) / (nd + 4.0 + 2.0 * mu_eff_ / nd);
    c1_ = 2.0 / ((nd + 1.3) * (nd + 1.3) + mu_eff_);
    cmu_ = std::min(1.0 - c1_,
                    2.0 * (mu_eff_ - 2.0 + 1.0 / mu_eff_) / ((nd + 2.0) * (nd + 2.0) + mu_eff_));

    // Expected norm of N(0,I): E[||N(0,I)||] ~ sqrt(n)*(1 - 1/(4n) + 1/(21n^2))
    chiN_ = std::sqrt(nd) * (1.0 - 1.0 / (4.0 * nd) + 1.0 / (21.0 * nd * nd));
}

// ===========================================================================
//  PrepareGeneration
//  Sample lambda candidates:  x_k = m + sigma * B D z_k,  z_k ~ N(0,I)
// ===========================================================================
void CMAES::PrepareGeneration()
{
    std::normal_distribution<double> nd01(0.0, 1.0);

    // Recompute eigendecomposition if scheduled
    if (eigen_counter_ >= eigen_freq_)
    {
        UpdateEigen_();
        eigen_counter_ = 0;
    }

    for (int k = 0; k < lambda_; ++k)
    {
        // Draw z_k ~ N(0, I)
        for (int i = 0; i < n; ++i)
            z_[k][i] = nd01(rng_);

        // y_k = B D z_k  (column-major B: y = B (D z))
        // Step 1: w = D z  (element-wise)
        std::vector<double> w(n);
        for (int i = 0; i < n; ++i)
            w[i] = D_[i] * z_[k][i];

        // Step 2: y = B w  (B column-major: y_i = sum_j B[j*n+i] * w[j])
        // B_ stored column-major: B_[col*n + row]
        std::vector<double> y(n, 0.0);
        for (int col = 0; col < n; ++col)
            for (int row = 0; row < n; ++row)
                y[row] += B_[col * n + row] * w[col];

        // x_k = m + sigma * y  (unconstrained — no bounds clipping)
        for (int i = 0; i < n; ++i)
            candidates[k][i] = mean_[i] + sigma_ * y[i];
    }
}

// ===========================================================================
//  UpdateAfterEval
//  Given fitness[k] for each candidate, perform the CMA-ES update.
//  aux[k] is an optional caller-defined scalar (e.g. best scaling for MTP).
//  The aux value of the all-time best candidate is preserved in global_best_aux.
//  Returns 0 (continue) or a stop code.
// ===========================================================================
int CMAES::UpdateAfterEval(const std::vector<double> &fitness,
                           const std::vector<double> &aux)
{
    assert((int)fitness.size() == lambda_);
    const bool have_aux = ((int)aux.size() == lambda_);

    // --- Sort candidates by fitness (ascending = minimisation) ---
    std::vector<int> order(lambda_);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b)
              { return fitness[a] < fitness[b]; });

    // --- Update global best (including aux scalar) ---
    last_gen_improved_ = false;
    if (fitness[order[0]] < global_best_val)
    {
        global_best_val = fitness[order[0]];
        best_x_ = candidates[order[0]];
        if (have_aux)
            global_best_aux = aux[order[0]];
        last_gen_improved_ = true;
    }
    best_val_history_.push_back(global_best_val);

    // --- Weighted mean update ---
    // m_new = sum_{i=0}^{mu-1} w_i * x_{order[i]}
    // Also keep y_w = (m_new - m_old) / sigma for evolution paths
    std::vector<double> m_new(n, 0.0);
    for (int i = 0; i < mu_; ++i)
        for (int d = 0; d < n; ++d)
            m_new[d] += weights_[i] * candidates[order[i]][d];

    // y_w = (m_new - mean_) / sigma_  (step in normalised space)
    std::vector<double> y_w(n);
    for (int d = 0; d < n; ++d)
        y_w[d] = (m_new[d] - mean_[d]) / sigma_;

    // invsqrtC_ * y_w  (for step-size path update)
    std::vector<double> invsqrtC_yw(n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            invsqrtC_yw[i] += invsqrtC_[i * n + j] * y_w[j];

    // --- Step-size evolution path p_sigma ---
    // p_sigma = (1-cs)*p_sigma + sqrt(cs*(2-cs)*mu_eff) * invsqrtC * y_w
    double coeff_s = std::sqrt(cs_ * (2.0 - cs_) * mu_eff_);
    for (int d = 0; d < n; ++d)
        p_sigma_[d] = (1.0 - cs_) * p_sigma_[d] + coeff_s * invsqrtC_yw[d];

    double p_sigma_norm = 0.0;
    for (int d = 0; d < n; ++d)
        p_sigma_norm += p_sigma_[d] * p_sigma_[d];
    p_sigma_norm = std::sqrt(p_sigma_norm);

    // --- Step-size update ---
    sigma_ *= std::exp((cs_ / ds_) * (p_sigma_norm / chiN_ - 1.0));

    // --- Heaviside h_sigma (stall indicator for rank-1 update) ---
    // h_sigma = 1 if ||p_sigma|| / sqrt(1-(1-cs)^(2*(gen+1))) < (1.4 + 2/(n+1)) * chiN
    double gen_factor = 1.0 - std::pow(1.0 - cs_, 2.0 * (generation + 1));
    double threshold = (1.4 + 2.0 / (n + 1.0)) * chiN_;
    int h_sigma = (p_sigma_norm / std::sqrt(gen_factor) < threshold) ? 1 : 0;

    // --- Covariance evolution path p_c ---
    // p_c = (1-cc)*p_c + h_sigma * sqrt(cc*(2-cc)*mu_eff) * y_w
    double coeff_c = std::sqrt(cc_ * (2.0 - cc_) * mu_eff_);
    for (int d = 0; d < n; ++d)
        p_c_[d] = (1.0 - cc_) * p_c_[d] + h_sigma * coeff_c * y_w[d];

    // --- Covariance matrix update ---
    // Precompute y_i = (x_{order[i]} - mean_) / sigma_  for i = 0..mu-1
    std::vector<std::vector<double>> y_i(mu_, std::vector<double>(n));
    for (int i = 0; i < mu_; ++i)
        for (int d = 0; d < n; ++d)
            y_i[i][d] = (candidates[order[i]][d] - mean_[d]) / sigma_;

    // delta_h = (1 - h_sigma) * cc * (2 - cc)   (correction term)
    double delta_h = (1 - h_sigma) * cc_ * (2.0 - cc_);

    // C = (1 - c1 - cmu) * C
    //   + c1 * (p_c p_c^T + delta_h * C)
    //   + cmu * sum_i w_i * y_i y_i^T
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j) // upper triangle + diagonal
        {
            double val = (1.0 - c1_ - cmu_ + c1_ * delta_h) * C_[i * n + j];
            val += c1_ * p_c_[i] * p_c_[j];
            for (int k = 0; k < mu_; ++k)
                val += cmu_ * weights_[k] * y_i[k][i] * y_i[k][j];
            C_[i * n + j] = val;
            C_[j * n + i] = val; // keep symmetric
        }

    // --- Advance mean ---
    mean_ = m_new;

    // --- Schedule eigendecomposition ---
    ++eigen_counter_;

    ++generation;

    // --- Stopping criteria ---

    // 1. Step size collapsed
    if (sigma_ < sigma_tol)
        return 1;

    // 2. Covariance ill-conditioned: ratio of max to min diagonal of D^2
    {
        double dmax = *std::max_element(D_.begin(), D_.end());
        double dmin = *std::min_element(D_.begin(), D_.end());
        if (dmin > 0.0 && (dmax / dmin) * (dmax / dmin) > cond_tol)
            return 2;
    }

    // 3. tolfun: range of best values in history window < tolfun
    if ((int)best_val_history_.size() >= tolfun_window_)
    {
        int window = tolfun_window_;
        auto first = best_val_history_.end() - window;
        double vmax = *std::max_element(first, best_val_history_.end());
        double vmin = *std::min_element(first, best_val_history_.end());
        if (std::abs(vmax - vmin) < tolfun)
            return 3;
    }

    return 0;
}

// ===========================================================================
//  UpdateEigen_
//  Recompute B_, D_, invsqrtC_ from C_ using symmetric Jacobi.
// ===========================================================================
void CMAES::UpdateEigen_()
{
    // Jacobi works on a copy (modifies in-place)
    std::vector<double> Ccopy(C_);
    std::vector<double> eigvecs(n * n, 0.0);
    std::vector<double> eigvals(n, 0.0);

    SymmetricJacobi_(n, Ccopy, eigvecs, eigvals);

    // Sort eigenvalues ascending; reorder eigvecs accordingly
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b)
              { return eigvals[a] < eigvals[b]; });

    // Clamp eigenvalues to a small positive floor to avoid sqrt of negative
    // (numerical errors can make very small eigenvalues slightly negative)
    for (int i = 0; i < n; ++i)
        if (eigvals[i] < 1e-20)
            eigvals[i] = 1e-20;

    // B_ column-major: B_[col*n + row] = eigvec[col][row]
    // D_[i] = sqrt(eigvals[i])
    B_.assign(n * n, 0.0);
    D_.resize(n);
    invsqrtC_.assign(n * n, 0.0);

    for (int col = 0; col < n; ++col)
    {
        int src_col = idx[col];
        D_[col] = std::sqrt(eigvals[src_col]);
        for (int row = 0; row < n; ++row)
            B_[col * n + row] = eigvecs[src_col * n + row];
    }

    // invsqrtC_ = B D^{-1} B^T  (row-major)
    // invsqrtC_[i,j] = sum_k B[k,i] * (1/D[k]) * B[k,j]
    //                = sum_k B_[k*n+i] * (1/D_[k]) * B_[k*n+j]
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            double val = 0.0;
            for (int k = 0; k < n; ++k)
                val += B_[k * n + i] * (1.0 / D_[k]) * B_[k * n + j];
            invsqrtC_[i * n + j] = val;
        }
}

// ===========================================================================
//  SymmetricJacobi_
//  Classic Jacobi eigenvalue algorithm for symmetric matrices.
//  Input:  A (n x n, row-major, modified in-place — diagonal = eigenvalues on exit)
//  Output: eigvecs (n x n, col-major — eigvecs[:,i] = eigenvector i)
//          eigvals (n)
//  Convergence: sweeps until max off-diagonal < 1e-12 or 100*n sweeps done.
// ===========================================================================
void CMAES::SymmetricJacobi_(int n,
                             std::vector<double> &A,
                             std::vector<double> &eigvecs,
                             std::vector<double> &eigvals)
{
    // Initialise eigvecs to identity (column-major)
    eigvecs.assign(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        eigvecs[i * n + i] = 1.0;

    const int max_sweeps = 100 * n;

    for (int sweep = 0; sweep < max_sweeps; ++sweep)
    {
        // Find maximum off-diagonal element
        double off_max = 0.0;
        for (int p = 0; p < n - 1; ++p)
            for (int q = p + 1; q < n; ++q)
                off_max = std::max(off_max, std::abs(A[p * n + q]));

        if (off_max < 1e-12)
            break;

        // One sweep: zero out all off-diagonal pairs (p,q)
        for (int p = 0; p < n - 1; ++p)
        {
            for (int q = p + 1; q < n; ++q)
            {
                double apq = A[p * n + q];
                if (std::abs(apq) < 1e-20)
                    continue;

                double app = A[p * n + p];
                double aqq = A[q * n + q];

                // Compute rotation angle theta
                double theta = (aqq - app) / (2.0 * apq);
                double t;
                if (theta >= 0.0)
                    t = 1.0 / (theta + std::sqrt(1.0 + theta * theta));
                else
                    t = -1.0 / (-theta + std::sqrt(1.0 + theta * theta));

                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;
                double tau = s / (1.0 + c);

                // Update diagonal
                A[p * n + p] -= t * apq;
                A[q * n + q] += t * apq;
                A[p * n + q] = 0.0;
                A[q * n + p] = 0.0;

                // Update off-diagonal rows/cols r != p, q
                for (int r = 0; r < n; ++r)
                {
                    if (r == p || r == q)
                        continue;

                    double arp = A[r * n + p];
                    double arq = A[r * n + q];

                    A[r * n + p] = arp - s * (arq + tau * arp);
                    A[p * n + r] = A[r * n + p];

                    A[r * n + q] = arq + s * (arp - tau * arq);
                    A[q * n + r] = A[r * n + q];
                }

                // Accumulate rotations into eigvecs (column-major)
                // Rotate columns p and q
                for (int r = 0; r < n; ++r)
                {
                    double vp = eigvecs[p * n + r];
                    double vq = eigvecs[q * n + r];
                    eigvecs[p * n + r] = vp - s * (vq + tau * vp);
                    eigvecs[q * n + r] = vq + s * (vp - tau * vq);
                }
            }
        }
    }

    // Diagonal of A now holds eigenvalues
    for (int i = 0; i < n; ++i)
        eigvals[i] = A[i * n + i];
}

// ===========================================================================
//  StatusString
// ===========================================================================
std::string CMAES::StatusString() const
{
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(6);
    ss << "gen=" << generation
       << " best=" << global_best_val
       << " sigma=" << sigma_;

    double dmax = D_.empty() ? 0.0 : *std::max_element(D_.begin(), D_.end());
    double dmin = D_.empty() ? 0.0 : *std::min_element(D_.begin(), D_.end());
    if (dmin > 0.0)
        ss << " cond=" << (dmax / dmin) * (dmax / dmin);

    ss << " best_aux=" << global_best_aux;

    return ss.str();
}