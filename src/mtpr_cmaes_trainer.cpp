/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing appropriate references.
 *   See the LICENSE file for details.
 */

#include "mtpr_cmaes_trainer.h"

#include <sstream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

using namespace std;

void MTPR_cmaes_trainer::InitCMAESSettings_()
{
    MakeSetting(subcomm_size, "subcomm_size");
    MakeSetting(cmaes_lambda, "cmaes_lambda");
    MakeSetting(cmaes_mu, "cmaes_mu");
    MakeSetting(cmaes_sigma0, "cmaes_sigma0");
    MakeSetting(cmaes_init_range, "cmaes_init_range");
    MakeSetting(cmaes_seed, "cmaes_seed");
    MakeSetting(cmaes_sigma_tol, "cmaes_sigma_tol");
    MakeSetting(cmaes_cond_tol, "cmaes_cond_tol");
    MakeSetting(cmaes_tolfun, "cmaes_tolfun");
    MakeSetting(cmaes_max_iter, "cmaes_max_iter");
    MakeSetting(cmaes_timeout, "cmaes_timeout");
    MakeSetting(cmaes_save_to, "cmaes_save_to");
    MakeSetting(cmaes_scale_bracket_factor, "cmaes_scale_bracket_factor");
    MakeSetting(cmaes_scale_bracket_steps, "cmaes_scale_bracket_steps");
    MakeSetting(cmaes_scale_tol, "cmaes_scale_tol");
}

void MTPR_cmaes_trainer::BuildSubCommunicators_()
{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        world_rank, MPI_INFO_NULL, &node_comm);

    int node_rank;
    MPI_Comm_rank(node_comm, &node_rank);

    int node_leader_world_rank = world_rank;
    {
        MPI_Comm tmp;
        MPI_Comm_split(MPI_COMM_WORLD,
                       node_rank == 0 ? 0 : MPI_UNDEFINED,
                       world_rank, &tmp);
        MPI_Bcast(&node_leader_world_rank, 1, MPI_INT, 0, node_comm);
        if (tmp != MPI_COMM_NULL)
            MPI_Comm_free(&tmp);
    }

    struct RankInfo
    {
        int node_leader;
        int node_rank;
        int world_rank;
    };

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    RankInfo my_info;
    my_info.node_leader = node_leader_world_rank;
    my_info.node_rank = node_rank;
    my_info.world_rank = world_rank;

    std::vector<RankInfo> all_info(world_size);
    MPI_Allgather(&my_info, sizeof(RankInfo), MPI_BYTE,
                  all_info.data(), sizeof(RankInfo), MPI_BYTE,
                  MPI_COMM_WORLD);

    std::vector<int> sorted(world_size);
    std::iota(sorted.begin(), sorted.end(), 0);
    std::sort(sorted.begin(), sorted.end(), [&](int a, int b)
              {
        if (all_info[a].node_leader != all_info[b].node_leader)
            return all_info[a].node_leader < all_info[b].node_leader;
        return all_info[a].node_rank < all_info[b].node_rank; });

    int num_full = world_size / subcomm_size;
    int leftover = world_size % subcomm_size;
    num_subcomms_ = num_full;

    if (leftover > 0 && world_rank == 0)
    {
        std::stringstream ss;
        ss << "CMA-ES Warning: " << world_size << " ranks with subcomm_size="
           << subcomm_size << " leaves " << leftover
           << " leftover rank(s). They will be folded into sub-communicator 0.\n";
        MLP_LOG("cmaes", ss.str());
    }

    std::vector<int> color_map(world_size);
    for (int i = 0; i < world_size; ++i)
    {
        int sc = i / subcomm_size;
        if (sc >= num_full)
            sc = 0;
        color_map[sorted[i]] = sc;
    }

    subcomm_id_ = color_map[world_rank];

    MPI_Comm_split(MPI_COMM_WORLD, subcomm_id_, world_rank, &subcomm_);
    MPI_Comm_rank(subcomm_, &subcomm_rank_);

    int leader_color = (subcomm_rank_ == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, leader_color, subcomm_id_, &subcomm_leaders_);

    MPI_Comm_free(&node_comm);
}

void MTPR_cmaes_trainer::FreeSubComm_()
{
    if (subcomm_ != MPI_COMM_NULL)
    {
        MPI_Comm_free(&subcomm_);
        subcomm_ = MPI_COMM_NULL;
    }
    if (subcomm_leaders_ != MPI_COMM_NULL)
    {
        MPI_Comm_free(&subcomm_leaders_);
        subcomm_leaders_ = MPI_COMM_NULL;
    }
}

std::vector<Configuration> MTPR_cmaes_trainer::SubCommLoadCfgs_(
    const std::string &filename)
{
    int sc_rank, sc_size;
    MPI_Comm_rank(subcomm_, &sc_rank);
    MPI_Comm_size(subcomm_, &sc_size);

    int total_cfgs = 0;
    bool use_bal = false;
    std::vector<int> amap;
    int max_per_rank = 0;

    if (sc_rank == 0)
    {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open())
            ERROR("CMA-ES: Cannot open training set: " + filename);

        struct Meta
        {
            int idx;
            double cost;
        };
        std::vector<Meta> metas;
        bool all_costs = true;
        Configuration cfg;

        while (cfg.Load(ifs))
        {
            double cost = 0.0;
            if (cfg.features.count("comp_cost"))
                cost = std::stod(cfg.features.at("comp_cost"));
            else
                all_costs = false;
            metas.push_back({total_cfgs, cost});
            ++total_cfgs;
        }

        if (total_cfgs > 0 && all_costs)
        {
            use_bal = true;
            amap.resize(total_cfgs);
            std::sort(metas.begin(), metas.end(),
                      [](const Meta &a, const Meta &b)
                      { return a.cost > b.cost; });

            std::vector<double> loads(sc_size, 0.0);
            std::vector<int> counts(sc_size, 0);
            for (auto &m : metas)
            {
                int best = 0;
                for (int r = 1; r < sc_size; ++r)
                    if (loads[r] < loads[best])
                        best = r;
                amap[m.idx] = best;
                loads[best] += m.cost;
                ++counts[best];
            }
            for (int c : counts)
                if (c > max_per_rank)
                    max_per_rank = c;
        }
        else if (total_cfgs > 0)
            max_per_rank = (total_cfgs + sc_size - 1) / sc_size;
    }

    MPI_Bcast(&use_bal, 1, MPI_C_BOOL, 0, subcomm_);
    MPI_Bcast(&total_cfgs, 1, MPI_INT, 0, subcomm_);
    MPI_Bcast(&max_per_rank, 1, MPI_INT, 0, subcomm_);
    amap.resize(total_cfgs);
    if (use_bal)
        MPI_Bcast(amap.data(), total_cfgs, MPI_INT, 0, subcomm_);

    subcomm_total_cfgs_ = total_cfgs;

    std::vector<Configuration> local;
    {
        std::ifstream ifs(filename, std::ios::binary);
        Configuration cfg;
        int cntr = 0;
        while (cntr < total_cfgs && cfg.Load(ifs))
        {
            bool mine = use_bal ? (amap[cntr] == sc_rank)
                                : (cntr % sc_size == sc_rank);
            if (mine)
                local.emplace_back(cfg);
            ++cntr;
        }
    }

    while ((int)local.size() < max_per_rank)
        local.emplace_back(Configuration());

    return local;
}

int MTPR_cmaes_trainer::RadialCoeffCount_() const
{
    const int C = p_mlmtpr_->species_count;
    const int K = p_mlmtpr_->radial_func_count;
    const int R = p_mlmtpr_->p_RadialBasis->size;
    const int M = p_mlmtpr_->mbasis_size;
    return C * C * K * R * M * M;
}

void MTPR_cmaes_trainer::PosToRadialCoeffs_(const std::vector<double> &pos,
                                            MLMTPR &mtp) const
{
    assert((int)pos.size() == RadialCoeffCount_());
    std::copy(pos.begin(), pos.end(), mtp.regression_coeffs.begin());
}

void MTPR_cmaes_trainer::RadialCoeffsToPos_(const MLMTPR &mtp,
                                            std::vector<double> &pos) const
{
    int n = RadialCoeffCount_();
    pos.resize(n);
    std::copy(mtp.regression_coeffs.begin(),
              mtp.regression_coeffs.begin() + n, pos.begin());
}

void MTPR_cmaes_trainer::BuildInitialMean_(std::vector<double> &mean0) const
{
    RadialCoeffsToPos_(*p_mlmtpr_, mean0);
}

double MTPR_cmaes_trainer::ApplyDScaleAndSolve_(double scaling,
                                                const std::vector<int> &degrees,
                                                std::vector<double> &out_coeffs)
{
    const int n = slae_n_;

    // D[i] = scaling^degrees[i]
    std::vector<double> D(n);
    for (int i = 0; i < n; ++i)
        D[i] = std::pow(scaling, degrees[i]);

    // Construct the scaled Gram matrix and RHS
    std::vector<double> G_scaled(n * n);
    std::vector<double> r_scaled(n);

    for (int i = 0; i < n; ++i)
    {
        r_scaled[i] = D[i] * base_rhs_[i];
        for (int j = 0; j < n; ++j)
        {
            G_scaled[i * n + j] = D[i] * D[j] * base_gram_[i * n + j];
        }
    }

    // Apply diagonal regularization proportional to element size
    for (int i = 0; i < n; ++i)
    {
        double reg_val = reg_param * std::max(1.0, G_scaled[i * n + i]);
        G_scaled[i * n + i] += reg_val;
    }

    // Solve system via LAPACK dposv_
    char uplo = 'U';
    int nrhs = 1;
    int info = 0;
    dposv_(&uplo, &n, &nrhs, G_scaled.data(), &n, r_scaled.data(), &n, &info);

    if (info != 0)
    {
        out_coeffs.assign(n, 0.0);
        return 1e30;
    }

    out_coeffs = r_scaled;

    // Evaluate the condition number proxy of out_coeffs (rms / median)
    std::vector<double> abs_coeffs(n);
    double rms = 0.0;
    for (int i = 0; i < n; ++i)
    {
        abs_coeffs[i] = std::abs(out_coeffs[i]);
        rms += abs_coeffs[i] * abs_coeffs[i];
    }
    rms = std::sqrt(rms);
    std::sort(abs_coeffs.begin(), abs_coeffs.end());
    double median = abs_coeffs[n / 2];

    if (median < 1e-30)
        return 1e30;

    return rms / median;
}

MTPR_cmaes_trainer::ScaleResult MTPR_cmaes_trainer::GoldenSectionScalingSearch_(
    double warm_scaling,
    const std::vector<int> &degrees)
{
    const double log_factor = std::log(cmaes_scale_bracket_factor);

    struct EvalCache
    {
        double u;
        double cond;
        std::vector<double> coeffs;
    };

    auto eval = [&](double u) -> EvalCache
    {
        EvalCache ec;
        ec.u = u;
        ec.cond = ApplyDScaleAndSolve_(std::exp(u), degrees, ec.coeffs);
        return ec;
    };

    if (warm_scaling <= 0.0 || !std::isfinite(warm_scaling))
        warm_scaling = 0.1;
    double u0 = std::log(warm_scaling);

    std::vector<EvalCache> probes;
    probes.reserve(2 * cmaes_scale_bracket_steps + 3);

    auto probe = [&](double u) -> const EvalCache &
    {
        for (const auto &p : probes)
            if (p.u == u)
                return p;
        probes.push_back(eval(u));
        return probes.back();
    };

    probe(u0);
    probe(u0 - log_factor);
    probe(u0 + log_factor);

    for (int step = 2; step <= cmaes_scale_bracket_steps; ++step)
    {
        double u_prev = u0 - (step - 1) * log_factor;
        double u_cur = u0 - step * log_factor;
        if (probe(u_cur).cond >= probe(u_prev).cond)
            break;
    }

    for (int step = 2; step <= cmaes_scale_bracket_steps; ++step)
    {
        double u_prev = u0 + (step - 1) * log_factor;
        double u_cur = u0 + step * log_factor;
        if (probe(u_cur).cond >= probe(u_prev).cond)
            break;
    }

    std::sort(probes.begin(), probes.end(),
              [](const EvalCache &a, const EvalCache &b)
              { return a.u < b.u; });

    EvalCache best_so_far = probes[0];
    for (const auto &p : probes)
        if (p.cond < best_so_far.cond)
            best_so_far = p;

    double a = best_so_far.u, b = best_so_far.u;
    for (const auto &p : probes)
    {
        if (p.u < best_so_far.u && p.cond >= best_so_far.cond)
            a = std::min(a, p.u);
        if (p.u > best_so_far.u && p.cond >= best_so_far.cond)
            b = std::max(b, p.u);
    }

    if (b - a < cmaes_scale_tol)
    {
        ScaleResult sr;
        sr.best_scaling = std::exp(best_so_far.u);
        sr.best_loss = best_so_far.cond;
        sr.best_linear_coeffs = best_so_far.coeffs;
        return sr;
    }

    const double phi = (std::sqrt(5.0) - 1.0) / 2.0;

    double x1 = b - phi * (b - a);
    double x2 = a + phi * (b - a);
    EvalCache f1 = eval(x1), f2 = eval(x2);

    if (f1.cond < best_so_far.cond)
        best_so_far = f1;
    if (f2.cond < best_so_far.cond)
        best_so_far = f2;

    while (b - a > cmaes_scale_tol)
    {
        if (f1.cond < f2.cond)
        {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - phi * (b - a);
            f1 = eval(x1);
            if (f1.cond < best_so_far.cond)
                best_so_far = f1;
        }
        else
        {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + phi * (b - a);
            f2 = eval(x2);
            if (f2.cond < best_so_far.cond)
                best_so_far = f2;
        }
    }

    ScaleResult sr;
    sr.best_scaling = std::exp(best_so_far.u);
    sr.best_loss = best_so_far.cond;
    sr.best_linear_coeffs = best_so_far.coeffs;
    return sr;
}

MTPR_cmaes_trainer::EvalResult MTPR_cmaes_trainer::EvaluateCandidate_(
    const std::vector<double> &position,
    std::vector<Configuration> &local_cfgs,
    double prev_best_scaling)
{
    int sc_rank;
    MPI_Comm_rank(subcomm_, &sc_rank);

    const int rsize = RadialCoeffCount_();
    std::vector<double> saved(p_mlmtpr_->regression_coeffs.begin(),
                              p_mlmtpr_->regression_coeffs.begin() + rsize);

    // Apply raw proposed coefficients without Orthogonalize().
    // The linear solver will happily map onto any non-orthogonal basis functions.
    PosToRadialCoeffs_(position, *p_mlmtpr_);

    const int n = p_mlmtpr_->alpha_count - 1 + p_mlmtpr_->species_count;
    slae_n_ = n;

    ClearSLAE();
    for (auto &cfg : local_cfgs)
        if (cfg.size() > 0)
            AddToSLAE(cfg);
    SymmetrizeSLAE();

    std::vector<double> local_gram, local_rhs;
    double local_scalar;
    int local_N;
    GetSLAEBuffers(local_gram, local_rhs, local_scalar, local_N);

    if (sc_rank == 0)
    {
        base_gram_.assign(n * n, 0.0);
        base_rhs_.assign(n, 0.0);
        base_scalar_ = 0.0;
        base_N_ = 0;
    }

    MPI_Reduce(local_gram.data(),
               sc_rank == 0 ? base_gram_.data() : nullptr,
               n * n, MPI_DOUBLE, MPI_SUM, 0, subcomm_);
    MPI_Reduce(local_rhs.data(),
               sc_rank == 0 ? base_rhs_.data() : nullptr,
               n, MPI_DOUBLE, MPI_SUM, 0, subcomm_);
    MPI_Reduce(&local_scalar,
               sc_rank == 0 ? &base_scalar_ : nullptr,
               1, MPI_DOUBLE, MPI_SUM, 0, subcomm_);
    MPI_Reduce(&local_N,
               sc_rank == 0 ? &base_N_ : nullptr,
               1, MPI_INT, MPI_SUM, 0, subcomm_);

    EvalResult result{1e300, prev_best_scaling};

    if (sc_rank == 0)
    {
        std::vector<int> degrees = p_mlmtpr_->GetScalingDegrees();
        ScaleResult sr = GoldenSectionScalingSearch_(prev_best_scaling, degrees);

        result.scaling = sr.best_scaling;

        // Compute the objective loss of the solved scaled SLAE
        std::vector<double> D(n);
        for (int i = 0; i < n; ++i)
            D[i] = std::pow(sr.best_scaling, degrees[i]);

        double x_G_x = 0.0;
        double x_r = 0.0;
        for (int i = 0; i < n; ++i)
        {
            x_r += sr.best_linear_coeffs[i] * D[i] * base_rhs_[i];
            for (int j = 0; j < n; ++j)
            {
                x_G_x += sr.best_linear_coeffs[i] * sr.best_linear_coeffs[j] * D[i] * D[j] * base_gram_[i * n + j];
            }
        }

        double total_err = base_scalar_ - 2.0 * x_r + x_G_x;

        // L2 weight decay on radial coefficients to prevent scale drift since
        // they are no longer being step-by-step orthogonalized/normalized.
        double l2_norm = 0.0;
        for (double val : position)
        {
            l2_norm += val * val;
        }

        result.loss = std::max(0.0, total_err) + (1e-6 * l2_norm);

        if (!std::isfinite(result.loss))
            result.loss = 1e300;

        const int lin_n = p_mlmtpr_->LinSize();
        for (int i = 0; i < lin_n && i < (int)sr.best_linear_coeffs.size(); ++i)
            p_mlmtpr_->linear_coeffs(i) = sr.best_linear_coeffs[i];

        p_mlmtpr_->scaling = sr.best_scaling;
    }

    std::copy(saved.begin(), saved.end(),
              p_mlmtpr_->regression_coeffs.begin());

    return result;
}

void MTPR_cmaes_trainer::CMAESSearch(std::vector<Configuration> & /*unused*/)
{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::stringstream log;

    BuildSubCommunicators_();

    if (world_rank == 0)
    {
        log << "CMA-ES seed search starting.\n"
            << "  World size:       " << world_size << "\n"
            << "  Sub-comm size:    " << subcomm_size << "\n"
            << "  Num sub-comms:    " << num_subcomms_ << "\n"
            << "  Max generations:  " << cmaes_max_iter << "\n"
            << "  Timeout (s):      " << cmaes_timeout << "\n"
            << "  lambda:           "
            << (cmaes_lambda > 0 ? std::to_string(cmaes_lambda) : "auto") << "\n"
            << "  sigma0:           "
            << (cmaes_sigma0 > 0 ? std::to_string(cmaes_sigma0) : "auto") << "\n"
            << "  init_range:       " << cmaes_init_range << "\n";
        MLP_LOG("cmaes", log.str());
        log.str("");
    }

    std::vector<Configuration> local_cfgs = SubCommLoadCfgs_(train_file_);
    MPI_Barrier(MPI_COMM_WORLD);

    const int dim = RadialCoeffCount_();
    std::vector<double> mean0;
    BuildInitialMean_(mean0);

    if (cmaes_seed < 0 && world_rank == 0)
    {
        log << "CMA-ES Warning: cmaes_seed is negative (" << cmaes_seed
            << "); treating as 0 (random device).\n";
        MLP_LOG("cmaes", log.str());
        log.str("");
    }
    unsigned int rng_seed = (cmaes_seed > 0)
                                ? static_cast<unsigned int>(cmaes_seed)
                                : 0u;

    int effective_lambda = -1;

    if (world_rank == 0)
    {
        cmaes_.lambda = cmaes_lambda;
        cmaes_.mu = cmaes_mu;
        cmaes_.sigma0 = cmaes_sigma0;
        cmaes_.init_range = cmaes_init_range;
        cmaes_.sigma_tol = cmaes_sigma_tol;
        cmaes_.cond_tol = cmaes_cond_tol;
        cmaes_.tolfun = cmaes_tolfun;
        cmaes_.seed = rng_seed;

        cmaes_.global_best_aux = p_mlmtpr_->scaling;
        cmaes_.Init(mean0, rng_seed);

        effective_lambda = (cmaes_lambda > 0)
                               ? cmaes_lambda
                               : 4 + static_cast<int>(std::floor(3.0 * std::log(static_cast<double>(dim))));

        log << "CMA-ES initialised: dim=" << dim
            << " lambda=" << effective_lambda << "\n"
            << "  " << cmaes_.StatusString() << "\n";
        MLP_LOG("cmaes", log.str());
        log.str("");
    }

    MPI_Bcast(&effective_lambda, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (effective_lambda < num_subcomms_ && world_rank == 0)
    {
        log << "CMA-ES Warning: lambda (" << effective_lambda
            << ") < num_subcomms (" << num_subcomms_
            << "). Some sub-communicators will be idle each generation.\n";
        MLP_LOG("cmaes", log.str());
        log.str("");
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    double my_best_scaling = p_mlmtpr_->scaling;

    std::vector<double> gen_times;
    bool done = false;
    int stop_code = 0;

    for (int gen = 0; gen < cmaes_max_iter && !done; ++gen)
    {
        auto t_gen_start = std::chrono::high_resolution_clock::now();
        std::vector<double> candidates_flat(effective_lambda * dim, 0.0);

        if (world_rank == 0)
        {
            cmaes_.PrepareGeneration();
            for (int k = 0; k < effective_lambda; ++k)
                std::copy(cmaes_.candidates[k].begin(),
                          cmaes_.candidates[k].end(),
                          candidates_flat.begin() + k * dim);
        }
        MPI_Bcast(candidates_flat.data(), effective_lambda * dim,
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);

        std::vector<double> fitness_local(effective_lambda, 1e300);
        std::vector<double> scaling_local(effective_lambda, 0.0);

        for (int k = subcomm_id_; k < effective_lambda; k += num_subcomms_)
        {
            std::vector<double> pos(candidates_flat.begin() + k * dim,
                                    candidates_flat.begin() + k * dim + dim);

            MPI_Bcast(pos.data(), dim, MPI_DOUBLE, 0, subcomm_);

            EvalResult er = EvaluateCandidate_(pos, local_cfgs, my_best_scaling);

            MPI_Bcast(&er.loss, 1, MPI_DOUBLE, 0, subcomm_);
            MPI_Bcast(&er.scaling, 1, MPI_DOUBLE, 0, subcomm_);

            fitness_local[k] = er.loss;
            scaling_local[k] = er.scaling;

            if (er.loss < 1e299)
                my_best_scaling = er.scaling;
        }

        std::vector<double> fitness_global(effective_lambda, 1e300);
        std::vector<double> scaling_global(effective_lambda, 0.0);

        MPI_Allreduce(fitness_local.data(), fitness_global.data(),
                      effective_lambda, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(scaling_local.data(), scaling_global.data(),
                      effective_lambda, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (world_rank == 0)
        {
            stop_code = cmaes_.UpdateAfterEval(fitness_global, scaling_global);

            log << "CMA-ES gen " << gen << ": " << cmaes_.StatusString() << "\n";
            if (stop_code != 0)
            {
                static const char *reasons[] = {
                    "", "sigma collapsed",
                    "covariance ill-conditioned",
                    "tolfun: value range too small"};
                log << "CMA-ES internal stop: " << reasons[stop_code] << "\n";
            }
            MLP_LOG("cmaes", log.str());
            log.str("");
        }

        MPI_Bcast(&stop_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (stop_code != 0)
        {
            // If stop condition is 3 ("tolfun") and cmaes_tolfun is set to a negative value (default -1.0),
            // we bypass the exit mechanism. Conditions 1 (sigma) and 2 (covariance condition) are preserved.
            if (stop_code == 3 && cmaes_tolfun < 0)
            {
                if (world_rank == 0)
                {
                    log << "CMA-ES: 'tolfun' convergence detected but ignored (cmaes_tolfun < 0). Continuing search.\n";
                    MLP_LOG("cmaes", log.str());
                    log.str("");
                }
            }
            else
            {
                done = true;
            }
        }

        auto t_gen_end = std::chrono::high_resolution_clock::now();
        double gen_sec = std::chrono::duration<double>(t_gen_end - t_gen_start).count();

        if (world_rank == 0)
        {
            gen_times.push_back(gen_sec);
            double elapsed = std::chrono::duration<double>(t_gen_end - t_start).count();
            double max_gen_sec = *std::max_element(gen_times.begin(), gen_times.end());

            if (elapsed + max_gen_sec > cmaes_timeout)
            {
                log << "CMA-ES: next generation would exceed timeout"
                    << " (elapsed=" << elapsed
                    << "s, max_gen=" << max_gen_sec
                    << "s, timeout=" << cmaes_timeout << "s). Stopping.\n";
                MLP_LOG("cmaes", log.str());
                log.str("");
                done = true;
            }
        }
        MPI_Bcast(&done, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    }

    std::vector<double> best_pos(dim, 0.0);
    double best_scaling = p_mlmtpr_->scaling;

    if (world_rank == 0)
    {
        best_pos = cmaes_.global_best_pos();
        best_scaling = cmaes_.global_best_aux;
    }

    MPI_Bcast(best_pos.data(), dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&best_scaling, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    PosToRadialCoeffs_(best_pos, *p_mlmtpr_);
    p_mlmtpr_->scaling = best_scaling;

    // Orthogonalize the FINAL result so the saved MTP is standard and well-behaved
    p_mlmtpr_->Orthogonalize();

    if (world_rank == 0)
    {
        log << "CMA-ES best candidate installed.\n"
            << "  Global best loss: " << cmaes_.global_best_val << "\n"
            << "  Scaling:          " << best_scaling << "\n";
        MLP_LOG("cmaes", log.str());
        log.str("");
    }

    // Crucial: Retrain linear coefficients one last time on the now-orthogonalized basis
    LinOptimize(local_cfgs);

    if (world_rank == 0)
    {
        if (!cmaes_save_to.empty())
        {
            p_mlmtpr_->inited = true;
            p_mlmtpr_->Save(cmaes_save_to);
            log << "CMA-ES result saved to: " << cmaes_save_to << "\n"
                << "  Final scaling: " << p_mlmtpr_->scaling << "\n";
        }
        else
            log << "CMA-ES Warning: cmaes_save_to not set; result not saved.\n";

        MLP_LOG("cmaes", log.str());
        log.str("");
    }

    FreeSubComm_();
}