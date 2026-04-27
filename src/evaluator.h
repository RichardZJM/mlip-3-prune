#pragma once
#include <vector>
#include <queue>

extern "C" void dposv_(const char *uplo, const int *n, const int *nrhs,
                       double *a, const int *lda, double *b, const int *ldb, int *info);

extern "C" void dsyev_(const char *jobz, const char *uplo, const int *n,
                       double *a, const int *lda, double *w,
                       double *work, const int *lwork, int *info);

class SSECalculator
{
    std::vector<double> xtwx_train, xtwy_train;
    std::vector<double> xtwx_val, xtwy_val;
    double ytwy_train, ytwy_val, base_sse;
    bool is_self_validating;
    int n_features, n_species;

    // Pre-allocated buffers to prevent re-allocation
    mutable std::vector<int> active_buf;
    mutable std::vector<double> A_buf;
    mutable std::vector<double> B_buf;

    // Semantic Subroutines
    bool solve_theta(int n) const;
    double compute_train_sse(int n) const;
    double compute_val_sse(int n) const;

public:
    SSECalculator(const std::vector<double> &xtwx_train, const std::vector<double> &xtwy_train, double ytwy_train,
                  const std::vector<double> &xtwx_val, const std::vector<double> &xtwy_val, double ytwy_val,
                  bool is_self_validating, double reg, int n_species, int n_var, int rank);

    double calculate(const char *genes) const;
};

class CostCalculator
{
    int num_moments, n_mus, radial_basis_size;
    std::vector<int> scalar_indices, basic_indices, parents_data, parents_idx;
    double base_cost, neigh_count;

    // Pre-allocated buffers (using char to avoid vector<bool> overhead)
    mutable std::vector<char> mus_flags_buf;
    mutable std::vector<char> to_preserve_buf;
    mutable std::queue<int> q_buf;

public:
    CostCalculator(int num_moments_, const std::vector<int> &basic_, const std::vector<int> &times_,
                   const std::vector<int> &scalar_, double neigh_count, int radial, int rank);
    double canonicalize_and_calculate(char *genes, int n_var) const;
};