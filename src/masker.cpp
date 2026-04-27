#include "masker.h"
#include "external/json.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <queue>
#include <set>
#include <map>
#include <array>

using json = nlohmann::json;

extern "C" void dposv_(const char *uplo, const int *n, const int *nrhs,
                       double *a, const int *lda, double *b, const int *ldb, int *info);

namespace
{
    template <typename T>
    std::vector<T> read_binary_local(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file)
            throw std::runtime_error("Cannot open binary file: " + filename);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        size_t elements = size / sizeof(T);
        std::vector<T> buffer(elements);
        if (elements > 0)
        {
            file.read(reinterpret_cast<char *>(buffer.data()), elements * sizeof(T));
        }
        return buffer;
    }
}

Masker::Masker(const std::string &base_mtp) : MLMTPR()
{
    Load(base_mtp);
}

void Masker::ApplyMask(const std::vector<char> &mask, const std::vector<double> *theta)
{
    if (static_cast<int>(mask.size()) != alpha_scalar_moments)
    {
        throw std::runtime_error("Mask size does not match alpha_scalar_moments.");
    }

    // 1. Build parent lookup
    std::vector<std::vector<std::pair<int, int>>> parents_lookup(alpha_moments_count);
    for (int i = 0; i < alpha_index_times_count; ++i)
    {
        int p1 = alpha_index_times[i][0];
        int p2 = alpha_index_times[i][1];
        int child = alpha_index_times[i][3];
        parents_lookup[child].push_back({p1, p2});
    }

    // 2. BFS backward to identify strictly required intermediate nodes
    std::vector<char> nodes_to_keep(alpha_moments_count, 0);
    std::queue<int> q;
    for (int i = 0; i < alpha_scalar_moments; ++i)
    {
        if (mask[i])
        {
            int m = alpha_moment_mapping[i];
            if (!nodes_to_keep[m])
            {
                nodes_to_keep[m] = 1;
                q.push(m);
            }
        }
    }

    while (!q.empty())
    {
        int child = q.front();
        q.pop();
        for (auto &p : parents_lookup[child])
        {
            if (!nodes_to_keep[p.first])
            {
                nodes_to_keep[p.first] = 1;
                q.push(p.first);
            }
            if (!nodes_to_keep[p.second])
            {
                nodes_to_keep[p.second] = 1;
                q.push(p.second);
            }
        }
    }

    // 3. Create mapping to compact layout
    std::vector<int> old_to_new_node_map(alpha_moments_count, -1);
    int new_idx_counter = 0;
    for (int i = 0; i < alpha_moments_count; ++i)
    {
        if (nodes_to_keep[i])
        {
            old_to_new_node_map[i] = new_idx_counter++;
        }
    }

    // 4. Prune basic indices and track which radial basis functions (mu) are still used
    std::vector<std::array<int, 4>> intermediate_basic;
    std::set<int> used_mus;
    for (int i = 0; i < alpha_index_basic_count; ++i)
    {
        if (nodes_to_keep[i])
        {
            std::array<int, 4> basic = {alpha_index_basic[i][0], alpha_index_basic[i][1], alpha_index_basic[i][2], alpha_index_basic[i][3]};
            intermediate_basic.push_back(basic);
            used_mus.insert(basic[0]);
        }
    }

    std::vector<int> sorted_used_mus(used_mus.begin(), used_mus.end());
    std::map<int, int> old_mu_to_new_mu;
    for (size_t i = 0; i < sorted_used_mus.size(); ++i)
    {
        old_mu_to_new_mu[sorted_used_mus[i]] = i;
    }

    int new_basic_count = intermediate_basic.size();
    int (*new_alpha_index_basic)[4] = new int[new_basic_count][4];
    for (int i = 0; i < new_basic_count; ++i)
    {
        new_alpha_index_basic[i][0] = old_mu_to_new_mu[intermediate_basic[i][0]];
        new_alpha_index_basic[i][1] = intermediate_basic[i][1];
        new_alpha_index_basic[i][2] = intermediate_basic[i][2];
        new_alpha_index_basic[i][3] = intermediate_basic[i][3];
    }

    // 5. Prune edges
    std::vector<std::array<int, 4>> new_times_vec;
    for (int i = 0; i < alpha_index_times_count; ++i)
    {
        int child_old = alpha_index_times[i][3];
        if (nodes_to_keep[child_old])
        {
            int p1_new = old_to_new_node_map[alpha_index_times[i][0]];
            int p2_new = old_to_new_node_map[alpha_index_times[i][1]];
            int child_new = old_to_new_node_map[child_old];
            new_times_vec.push_back({p1_new, p2_new, alpha_index_times[i][2], child_new});
        }
    }
    int (*new_alpha_index_times)[4] = new int[std::max<size_t>(1, new_times_vec.size())][4];
    for (size_t i = 0; i < new_times_vec.size(); ++i)
    {
        for (int j = 0; j < 4; ++j)
            new_alpha_index_times[i][j] = new_times_vec[i][j];
    }

    // 6. Prune scalar references
    std::vector<int> new_scalar_vec;
    for (int i = 0; i < alpha_scalar_moments; ++i)
    {
        if (mask[i])
        {
            new_scalar_vec.push_back(old_to_new_node_map[alpha_moment_mapping[i]]);
        }
    }
    int *new_alpha_moment_mapping = new int[new_scalar_vec.size()];
    for (size_t i = 0; i < new_scalar_vec.size(); ++i)
    {
        new_alpha_moment_mapping[i] = new_scalar_vec[i];
    }

    // 7. Inject Linear Solution & Prune Radial Matrix Space
    if (theta != nullptr)
    {
        inited = true;
        int R = p_RadialBasis->size;
        int M = mbasis_size;
        int C = species_count;
        int old_K = radial_func_count;
        int new_K = sorted_used_mus.size();

        std::vector<double> new_regression_coeffs(new_K * R * M * M * C * C + C + new_scalar_vec.size());

        for (int s1 = 0; s1 < C; ++s1)
        {
            for (int s2 = 0; s2 < C; ++s2)
            {
                for (int old_mu = 0; old_mu < old_K; ++old_mu)
                {
                    for (int xi = 0; xi < R; ++xi)
                    {
                        for (int a = 0; a < M; ++a)
                        {
                            for (int b = 0; b < M; ++b)
                            {
                                if (old_mu_to_new_mu.count(old_mu))
                                {
                                    int old_idx = (s1 * C + s2) * old_K * R * M * M + old_mu * R * M * M + xi * M * M + a * M + b;
                                    int new_mu = old_mu_to_new_mu[old_mu];
                                    int new_idx = (s1 * C + s2) * new_K * R * M * M + new_mu * R * M * M + xi * M * M + a * M + b;
                                    new_regression_coeffs[new_idx] = regression_coeffs[old_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        int offset = new_K * R * M * M * C * C;
        for (int i = 0; i < C + static_cast<int>(new_scalar_vec.size()); ++i)
        {
            new_regression_coeffs[offset + i] = (*theta)[i];
        }

        regression_coeffs = new_regression_coeffs;
        radial_func_count = new_K;
    }
    else
    {
        inited = false;
        radial_func_count = sorted_used_mus.size();
        regression_coeffs.clear(); // Forces physical blankness out of the save file.
    }

    delete[] alpha_index_basic;
    delete[] alpha_index_times;
    delete[] alpha_moment_mapping;

    alpha_index_basic = new_alpha_index_basic;
    alpha_index_times = new_alpha_index_times;
    alpha_moment_mapping = new_alpha_moment_mapping;

    alpha_moments_count = new_idx_counter;
    alpha_index_basic_count = new_basic_count;
    alpha_index_times_count = new_times_vec.size();
    alpha_scalar_moments = new_scalar_vec.size();
    alpha_count = alpha_scalar_moments + 1;
}

std::vector<char> ReadMask(const std::string &mask_file, int row, int expected_size)
{
    std::ifstream f(mask_file);
    if (!f.is_open())
        throw std::runtime_error("Cannot open mask file: " + mask_file);

    std::string line;
    int current_row = 0;
    while (std::getline(f, line))
    {
        std::vector<char> mask;
        for (char c : line)
        {
            if (c == '0')
                mask.push_back(0);
            else if (c == '1')
                mask.push_back(1);
        }
        if (mask.empty())
            continue;

        if (current_row == row)
        {
            if (static_cast<int>(mask.size()) != expected_size)
            {
                throw std::runtime_error("Mask size mismatch at row " + std::to_string(row) +
                                         ". Expected " + std::to_string(expected_size) +
                                         ", got " + std::to_string(mask.size()));
            }
            return mask;
        }
        current_row++;
    }
    throw std::runtime_error("Row " + std::to_string(row) + " not found in mask file.");
}

std::vector<double> SolveTheta(const std::string &config_file, const std::vector<char> &mask, int species_count)
{
    std::ifstream f(config_file);
    if (!f.is_open())
        throw std::runtime_error("Cannot open config file: " + config_file);
    json config = json::parse(f);

    std::string xtwx_file = config["xtwx_train_file"].get<std::string>();
    std::string xtwy_file = config["xtwy_train_file"].get<std::string>();
    double reg = config.value("regularization", 0.0);

    auto xtwx = read_binary_local<double>(xtwx_file);
    auto xtwy = read_binary_local<double>(xtwy_file);

    int n_var = mask.size();
    int n_features = species_count + n_var;

    int expected_size = n_features * n_features;
    if (static_cast<int>(xtwx.size()) != expected_size)
    {
        throw std::runtime_error("xtwx matrix size mismatch. Expected " + std::to_string(expected_size) + ", got " + std::to_string(xtwx.size()));
    }

    std::vector<int> active_idx;
    for (int i = 0; i < species_count; ++i)
        active_idx.push_back(i);
    for (int i = 0; i < n_var; ++i)
    {
        if (mask[i])
            active_idx.push_back(species_count + i);
    }

    int n_active = active_idx.size();
    std::vector<double> A(n_active * n_active);
    std::vector<double> B(n_active);

    for (int i = 0; i < n_features; ++i)
    {
        xtwx[i * n_features + i] += reg;
    }

    for (int i = 0; i < n_active; ++i)
    {
        B[i] = xtwy[active_idx[i]];
        for (int j = 0; j < n_active; ++j)
        {
            A[i * n_active + j] = xtwx[active_idx[i] * n_features + active_idx[j]];
        }
    }

    char uplo = 'U';
    int nrhs = 1, info = 0;
    dposv_(&uplo, &n_active, &nrhs, A.data(), &n_active, B.data(), &n_active, &info);

    if (info > 0)
    {
        throw std::runtime_error("Cholesky decomposition failed. The active matrix is not positive definite.");
    }

    return B;
}