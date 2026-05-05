#include "evaluator.h"
#include <iostream>
#include <iomanip>
#include <set>
#include <cmath>
#include <algorithm>

// --- SSE Calculator ---
SSECalculator::SSECalculator(const std::vector<double> &xtwx_train_, const std::vector<double> &xtwy_train_, double ytwy_train_,
                             const std::vector<double> &xtwx_val_, const std::vector<double> &xtwy_val_, double ytwy_val_,
                             double reg_, int n_species_, int n_var, int rank)
    : xtwx_train(xtwx_train_), xtwy_train(xtwy_train_), ytwy_train(ytwy_train_),
      xtwx_val(xtwx_val_), xtwy_val(xtwy_val_), ytwy_val(ytwy_val_),
      reg(reg_), n_species(n_species_)
{
    n_features = n_species + n_var;
    scales.resize(n_features);

    // 1. Calculate Jacobi scales based on the pure training matrix diagonal
    for (int i = 0; i < n_features; ++i)
    {
        double diag = xtwx_train[i * n_features + i];
        scales[i] = (diag > 1e-12) ? 1.0 / std::sqrt(diag) : 1.0;
    }

    // 2. Pre-scale all matrices in memory for maximum GA speed
    for (int i = 0; i < n_features; ++i)
    {
        xtwy_train[i] *= scales[i];
        xtwy_val[i] *= scales[i];
        for (int j = 0; j < n_features; ++j)
        {
            xtwx_train[i * n_features + j] *= scales[i] * scales[j];
            xtwx_val[i * n_features + j] *= scales[i] * scales[j];
        }

        // 3. Bake Regularization directly into the scaled training matrix diagonal!
        xtwx_train[i * n_features + i] += reg;
    }

    if (rank == 0)
    {
        // Compute condition number on rank 0
        // (We must make a copy because dsyev_ literally destroys the input array)
        std::vector<double> A_copy = xtwx_train;

        std::vector<double> w(n_features);
        char jobz = 'N';
        char uplo = 'U';
        int n_eq = n_features;
        double lwork_query;
        int info_dsyev = 0;
        int lwork = -1;

        dsyev_(&jobz, &uplo, &n_eq, A_copy.data(), &n_eq, w.data(), &lwork_query, &lwork, &info_dsyev);
        lwork = static_cast<int>(lwork_query);
        std::vector<double> work(lwork);

        dsyev_(&jobz, &uplo, &n_eq, A_copy.data(), &n_eq, w.data(), work.data(), &lwork, &info_dsyev);

        if (info_dsyev == 0)
        {
            double min_eig = w.front();
            double max_eig = w.back();
            double cond = (min_eig > 0) ? (max_eig / min_eig) : INFINITY;

            std::cout << "Condition Number: " << std::scientific << cond << std::defaultfloat << "\n";
            if (cond > 1e12)
            {
                std::cout << "WARNING: Condition number is extremely high. I hope you know what you are doing.\n";
                std::cout << "         Consider increasing the regularization parameter.\n";
            }
        }
    }

    active_buf.reserve(n_features);
    A_buf.resize(n_features * n_features);
    B_buf.resize(n_features);

    std::vector<char> all_ones(n_var, 1);
    base_sse = 1.0;
    base_sse = calculate(all_ones.data());

    if (rank == 0)
    {
        std::cout << "Base SSE: " << base_sse << "\n";
    }
}

bool SSECalculator::solve_theta(int n) const
{
    for (int i = 0; i < n; ++i)
    {
        B_buf[i] = xtwy_train[active_buf[i]];
        for (int j = 0; j < n; j++)
        {
            A_buf[i * n + j] = xtwx_train[active_buf[i] * n_features + active_buf[j]];
        }
    }

    char uplo = 'U';
    int nrhs = 1, info = 0;
    dposv_(&uplo, &n, &nrhs, A_buf.data(), &n, B_buf.data(), &n, &info);

    return (info == 0);
}

double SSECalculator::compute_val_sse(int n) const
{
    double p1 = 0;
    for (int i = 0; i < n; ++i)
    {
        p1 += B_buf[i] * xtwy_val[active_buf[i]];
    }

    double p2 = 0;
    for (int i = 0; i < n; ++i)
    {
        double v_i = 0;
        for (int j = 0; j < n; j++)
        {
            v_i += xtwx_val[active_buf[i] * n_features + active_buf[j]] * B_buf[j];
        }
        p2 += B_buf[i] * v_i;
    }

    return ytwy_val - 2.0 * p1 + p2;
}

double SSECalculator::calculate(const char *genes) const
{
    active_buf.clear();

    for (int i = 0; i < n_species; ++i)
        active_buf.push_back(i);

    int n_var = n_features - n_species;
    for (int i = 0; i < n_var; ++i)
    {
        if (genes[i])
            active_buf.push_back(i + n_species);
    }

    int n = active_buf.size();
    if (n == 0)
        return INFINITY;

    // --- Phase 1: Train ---
    if (!solve_theta(n))
        return INFINITY;

    // --- Phase 2: Validate ---
    double raw_sse = compute_val_sse(n);

    return raw_sse / base_sse;
}

// --- Cost Calculator ---
CostCalculator::CostCalculator(int num_moments_, const std::vector<int> &basic_,
                               const std::vector<int> &times_, const std::vector<int> &scalar_,
                               double neigh, int radial, int rank)
    : num_moments(num_moments_), basic_indices(basic_), scalar_indices(scalar_),
      neigh_count(neigh), radial_basis_size(radial)
{
    std::set<int> mus_set;
    for (size_t i = 0; i < basic_.size() / 4; ++i)
    {
        mus_set.insert(basic_[i * 4]);
    }
    n_mus = mus_set.size();

    std::vector<std::vector<int>> py_parents(num_moments);
    for (size_t i = 0; i < times_.size() / 4; ++i)
    {
        int p1 = times_[i * 4], p2 = times_[i * 4 + 1], child = times_[i * 4 + 3];
        py_parents[child].push_back(p1);
        py_parents[child].push_back(p2);
    }

    parents_idx.push_back(0);
    for (int i = 0; i < num_moments; ++i)
    {
        for (int p : py_parents[i])
            parents_data.push_back(p);
        parents_idx.push_back(parents_data.size());
    }

    mus_flags_buf.resize(n_mus);
    to_preserve_buf.resize(num_moments);

    std::vector<char> all_ones(scalar_indices.size(), 1);
    base_cost = 1.0;
    base_cost = canonicalize_and_calculate(all_ones.data(), scalar_indices.size());

    if (rank == 0)
    {
        std::cout << "Base Cost: " << base_cost << "\n";
    }
}

double CostCalculator::canonicalize_and_calculate(char *genes, int n_var, int max_fill_rounds)
{
    std::fill(mus_flags_buf.begin(), mus_flags_buf.end(), 0);
    std::fill(to_preserve_buf.begin(), to_preserve_buf.end(), 0);

    // BFS to find all moments needed by active scalars
    int head = 0;
    q_buf.clear();
    for (int i = 0; i < n_var; ++i)
    {
        if (genes[i])
        {
            int m = scalar_indices[i];
            if (!to_preserve_buf[m])
            {
                to_preserve_buf[m] = 1;
                q_buf.push_back(m);
            }
        }
    }
    while (head < (int)q_buf.size())
    {
        int child = q_buf[head++];
        for (int j = parents_idx[child]; j < parents_idx[child + 1]; j += 2)
        {
            int p1 = parents_data[j], p2 = parents_data[j + 1];
            if (!to_preserve_buf[p1])
            {
                to_preserve_buf[p1] = 1;
                q_buf.push_back(p1);
            }
            if (!to_preserve_buf[p2])
            {
                to_preserve_buf[p2] = 1;
                q_buf.push_back(p2);
            }
        }
    }

    // Compute raw cost
    int ntimes = 0, nbasic = 0;
    int current_max_active_rank = -1;
    for (int i = 0; i < num_moments; ++i)
    {
        if (to_preserve_buf[i])
        {
            int edges = (parents_idx[i + 1] - parents_idx[i]) / 2;
            ntimes += edges;
            if (edges == 0)
            {
                nbasic++;
                int mu = basic_indices[i * 4];
                int r = std::max({basic_indices[i * 4 + 1], basic_indices[i * 4 + 2], basic_indices[i * 4 + 3]});
                if (mu < n_mus)
                    mus_flags_buf[mu] = 1;
                if (r > current_max_active_rank)
                    current_max_active_rank = r;
            }
        }
    }

    int max_rank_cost = (current_max_active_rank == -1) ? 0 : (current_max_active_rank + 1);
    int mus_count = 0;
    for (char b : mus_flags_buf)
        if (b)
            mus_count++;

    double raw_cost = neigh_count * (24 + 4 * max_rank_cost + 8 * radial_basis_size + 14 +
                                     4 * mus_count * radial_basis_size + 39 * nbasic) +
                      9 * ntimes;

    // 1. FREE-RIDE RULE
    for (int i = 0; i < n_var; ++i)
    {
        if (to_preserve_buf[scalar_indices[i]])
            genes[i] = 1;
    }

    // 2. FAST FILL RULE - BFS levels
    double total_cost = raw_cost;
    double max_incremental_cost = 0.10 * raw_cost;

    for (int round = 0; round < max_fill_rounds; ++round)
    {
        bool any_flipped = false;
        q_buf.clear();

        for (int i = 0; i < n_var; ++i)
        {
            if (!genes[i])
            {
                int m = scalar_indices[i];

                bool deps_met = true;
                for (int j = parents_idx[m]; j < parents_idx[m + 1]; ++j)
                {
                    if (!to_preserve_buf[parents_data[j]])
                    {
                        deps_met = false;
                        break;
                    }
                }

                if (deps_met)
                {
                    int edges = (parents_idx[m + 1] - parents_idx[m]) / 2;
                    if (edges == 0)
                        continue;
                    double incremental_cost = 9 * edges;

                    if (incremental_cost <= max_incremental_cost)
                    {
                        genes[i] = 1; // Immediate update is safe here (only read by `if (!genes[i])`)
                        total_cost += incremental_cost;
                        any_flipped = true;

                        // Defer updating to_preserve_buf to maintain level-synchronous BFS
                        q_buf.push_back(m);
                    }
                }
            }
        }

        // Apply the wave's updates all at once
        for (int m : q_buf)
        {
            to_preserve_buf[m] = 1;
        }

        if (!any_flipped)
            break;
    }

    return total_cost / base_cost;
}