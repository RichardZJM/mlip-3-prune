#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "prune.h"
#include "external/json.hpp"
#include "evaluator.h"
#include "nsga2.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <dlfcn.h>
#include <sys/stat.h>
#include <dirent.h>

#ifdef MLIP_MPI
#include <mpi.h>
#endif

using json = nlohmann::json;

namespace
{
    bool path_exists(const std::string &name)
    {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }

    void make_dir(const std::string &path)
    {
        mkdir(path.c_str(), 0777);
    }

    template <typename T>
    std::vector<T> read_binary(const std::string &filename)
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

    std::string find_latest_population(const std::string &dir_path)
    {
        if (!path_exists(dir_path))
            return "";
        std::string final_pop = dir_path + "/pareto_final_population.csv";
        if (path_exists(final_pop))
            return final_pop;

        DIR *dir;
        struct dirent *ent;
        int max_gen = -1;
        std::string latest_pop = "";
        if ((dir = opendir(dir_path.c_str())) != NULL)
        {
            while ((ent = readdir(dir)) != NULL)
            {
                std::string fn = ent->d_name;
                if (fn.find("pareto_") == 0 && fn.find("_population.csv") != std::string::npos)
                {
                    try
                    {
                        int gen = std::stoi(fn.substr(7, fn.find("_population.csv") - 7));
                        if (gen > max_gen)
                        {
                            max_gen = gen;
                            latest_pop = dir_path + "/" + fn;
                        }
                    }
                    catch (...)
                    {
                    }
                }
            }
            closedir(dir);
        }
        return latest_pop;
    }

    void resolve_output_dir(const std::string &base_out_dir, std::string &out_dir, std::string &latest_pop_file)
    {
        std::string base = base_out_dir;
        if (!base.empty() && base.back() == '/')
            base.pop_back();
        out_dir = base;
        int restart_count = 1;
        while (path_exists(out_dir))
        {
            restart_count++;
            out_dir = base + "_" + std::to_string(restart_count);
        }
        latest_pop_file = "";
        for (int r = restart_count - 1; r >= 1; --r)
        {
            std::string check_dir = (r == 1) ? base : base + "_" + std::to_string(r);
            latest_pop_file = find_latest_population(check_dir);
            if (!latest_pop_file.empty())
                break;
        }
        make_dir(out_dir);
    }

    void evaluate_population(int offset, int count, int n_var, int mpi_rank, int mpi_size,
                             CostCalculator &cost_calc, SSECalculator &sse_calc,
                             std::vector<char> &genes, std::vector<double> &cost_sse,
                             std::vector<char> &local_genes, std::vector<double> &local_results,
                             double &eval_time, double &mpi_time, double &gather_time)
    {
        if (count == 0)
            return;
        int local_count = count / mpi_size;
#ifdef MLIP_MPI
        double t0 = MPI_Wtime();
        MPI_Scatter(mpi_rank == 0 ? genes.data() + offset * n_var : nullptr,
                    local_count * n_var, MPI_CHAR, local_genes.data(), local_count * n_var, MPI_CHAR, 0, MPI_COMM_WORLD);
        if (mpi_rank == 0)
            mpi_time += (MPI_Wtime() - t0);
        double start_eval = MPI_Wtime();
        for (int i = 0; i < local_count; ++i)
        {
            local_results[i * 2] = cost_calc.canonicalize_and_calculate(local_genes.data() + i * n_var, n_var);
            local_results[i * 2 + 1] = sse_calc.calculate(local_genes.data() + i * n_var);
        }
        eval_time += (MPI_Wtime() - start_eval);
        double t2 = MPI_Wtime();
        MPI_Gather(local_genes.data(), local_count * n_var, MPI_CHAR, mpi_rank == 0 ? genes.data() + offset * n_var : nullptr, local_count * n_var, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Gather(local_results.data(), local_count * 2, MPI_DOUBLE, mpi_rank == 0 ? cost_sse.data() + offset * 2 : nullptr, local_count * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (mpi_rank == 0)
            gather_time += (MPI_Wtime() - t2);
#else
        auto start_eval = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < count; ++i)
        {
            cost_sse[(offset + i) * 2] = cost_calc.canonicalize_and_calculate(genes.data() + (offset + i) * n_var, n_var);
            cost_sse[(offset + i) * 2 + 1] = sse_calc.calculate(genes.data() + (offset + i) * n_var);
        }
        eval_time += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_eval).count();
#endif
    }
}

Prune::Prune(const std::string &config_path) : config_path_(config_path) {}

void Prune::run()
{
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("MKL_NUM_THREADS", "1", 1);
    setenv("OMP_NUM_THREADS", "1", 1);

    auto set_threads = [](const char *name, int val)
    {
        typedef void (*set_fn)(int);
        set_fn fn = (set_fn)dlsym(RTLD_DEFAULT, name);
        if (fn)
        {
            fn(val);
        }
    };

    set_threads("openblas_set_num_threads", 1);
    set_threads("goto_set_num_threads", 1);
    set_threads("MKL_Set_Num_Threads", 1);
    set_threads("omp_set_num_threads", 1);

    int rank = 0, size = 1;
#ifdef MLIP_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    json config;
    if (rank == 0)
    {
        std::ifstream f(config_path_);
        if (!f.is_open())
            throw std::runtime_error("Config not found: " + config_path_);
        config = json::parse(f);
    }

#ifdef MLIP_MPI
    std::string config_str = (rank == 0) ? config.dump() : "";
    int len = static_cast<int>(config_str.size());
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    config_str.resize(len);
    MPI_Bcast(&config_str[0], len, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0)
        config = json::parse(config_str);
#endif

    this->Load(config["mtp_file"].get<std::string>());

    int r_size = p_RadialBasis->size;
    int a_mom_cnt = alpha_moments_count;
    int a_sca_cnt = alpha_scalar_moments;
    int spec_cnt = species_count;

    std::vector<int> idx_basic, idx_times, map_mom;
    for (int i = 0; i < alpha_index_basic_count; ++i)
        for (int j = 0; j < 4; ++j)
            idx_basic.push_back(alpha_index_basic[i][j]);
    for (int i = 0; i < alpha_index_times_count; ++i)
        for (int j = 0; j < 4; ++j)
            idx_times.push_back(alpha_index_times[i][j]);
    for (int i = 0; i < alpha_scalar_moments; ++i)
        map_mom.push_back(alpha_moment_mapping[i]);

    // Parse variables explicitly
    std::string xtwx_train_file = config["xtwx_train_file"].get<std::string>();
    std::string xtwy_train_file = config["xtwy_train_file"].get<std::string>();
    double ytwy_train = config["ytwy_train"].get<double>();

    std::string xtwx_val_file = config["xtwx_val_file"].get<std::string>();
    std::string xtwy_val_file = config["xtwy_val_file"].get<std::string>();
    double ytwy_val = config["ytwy_val"].get<double>();

    bool is_self_validating = (xtwx_train_file == xtwx_val_file) &&
                              (xtwy_train_file == xtwy_val_file) &&
                              (ytwy_train == ytwy_val);

    std::vector<double> xtwx_train, xtwy_train, xtwx_val, xtwy_val;
    if (rank == 0)
    {
        xtwx_train = read_binary<double>(xtwx_train_file);
        xtwy_train = read_binary<double>(xtwy_train_file);
        if (!is_self_validating)
        {
            xtwx_val = read_binary<double>(xtwx_val_file);
            xtwy_val = read_binary<double>(xtwy_val_file);
        }
    }

#ifdef MLIP_MPI
    auto bcast_vec_d = [&](std::vector<double> &v)
    {
        int sz = (rank == 0) ? (int)v.size() : 0;
        MPI_Bcast(&sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0)
            v.resize(sz);
        if (sz > 0)
            MPI_Bcast(v.data(), sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    };

    bcast_vec_d(xtwx_train);
    bcast_vec_d(xtwy_train);
    if (!is_self_validating)
    {
        bcast_vec_d(xtwx_val);
        bcast_vec_d(xtwy_val);
    }
#endif

    SSECalculator sse_calc(xtwx_train, xtwy_train, ytwy_train,
                           xtwx_val, xtwy_val, ytwy_val,
                           is_self_validating, config.value("regularization", 0.0), spec_cnt, a_sca_cnt, rank);

    CostCalculator cost_calc(a_mom_cnt, idx_basic, idx_times, map_mom, config["neigh_count"], r_size, rank);

    int pop_size = config["pop_size"];
    if (pop_size % size != 0)
        pop_size = ((pop_size + size - 1) / size) * size;
    NSGA2 ga(pop_size, a_sca_cnt, 42 + rank);

    int local_count = pop_size / size;
    std::vector<char> local_genes(local_count * a_sca_cnt);
    std::vector<double> local_results(local_count * 2);

    std::string out_dir, latest_pop_file;
    if (rank == 0)
    {
        resolve_output_dir(config["out_dir"].get<std::string>(), out_dir, latest_pop_file);
        if (!latest_pop_file.empty())
        {
            std::cout << "Restarting from previous results: " << latest_pop_file << "\n";
            std::cout << "Insufficient populations will be randomly generated. The population will be scrambled.\n";
        }
        std::cout << "Saving new results to: " << out_dir << "\n";
        ga.initialize_population(latest_pop_file);
        std::cout << "Evaluating initial population...\n";
    }

    double eval_time = 0.0, mpi_time = 0.0, gather_time = 0.0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto elapsed_s = [&]()
    {
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    };

    evaluate_population(0, pop_size, a_sca_cnt, rank, size, cost_calc, sse_calc, ga.genes, ga.cost_sse, local_genes, local_results, eval_time, mpi_time, gather_time);

    int n_gen = config["n_gen"];
    if (rank == 0)
    {
        ga.survival(pop_size);
        std::cout << "Generation 0/" << n_gen << " | Evaluations: " << pop_size << " | Elapsed: " << std::fixed << std::setprecision(2) << elapsed_s() << "s\n";
    }

    bool time_limit_reached = false;
    double time_limit = config.value("time", -1.0);
    int save_interval = config.value("save_interval", -1);

    for (int gen = 0; gen < n_gen; ++gen)
    {
        if (rank == 0)
            ga.generate_offspring();
        evaluate_population(pop_size, pop_size, a_sca_cnt, rank, size, cost_calc, sse_calc, ga.genes, ga.cost_sse, local_genes, local_results, eval_time, mpi_time, gather_time);

        int stop = 0;
        if (rank == 0)
        {
            ga.survival(2 * pop_size);
            int comp_gen = gen + 1;
            if (comp_gen % 10 == 0 || comp_gen == n_gen)
            {
                std::cout << "Generation " << comp_gen << "/" << n_gen << " | Evaluations: " << (comp_gen + 1) * pop_size << " | Elapsed: " << elapsed_s() << "s\n";
            }
            if (save_interval > 0 && comp_gen % save_interval == 0)
            {
                std::cout << "Saving intermediate results at generation " << comp_gen << "...\n";
                ga.save_pareto(out_dir + "/pareto_" + std::to_string(comp_gen));
            }
            if (time_limit > 0.0 && elapsed_s() >= time_limit)
            {
                std::cout << "Time limit reached. Stopping early.\n";
                stop = 1;
                time_limit_reached = true;
            }
        }
#ifdef MLIP_MPI
        MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
        if (stop)
            break;
    }

    std::vector<double> all_evals(size, 0.0);
#ifdef MLIP_MPI
    MPI_Gather(&eval_time, 1, MPI_DOUBLE, all_evals.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
    all_evals[0] = eval_time;
#endif

    if (rank == 0)
    {
        double total_exec = elapsed_s();
        ga.save_pareto(out_dir + "/pareto_final");
        std::cout << "\nOptimization finished in " << total_exec << "s" << (time_limit_reached ? " (stopped by time limit)" : "") << ".\n";
        std::cout << "Results saved to " << out_dir << "/pareto_final_*.csv\n";
#ifdef MLIP_MPI
        double max_e = all_evals[0], min_e = all_evals[0], sum_e = 0.0;
        for (double e : all_evals)
        {
            max_e = std::max(max_e, e);
            min_e = std::min(min_e, e);
            sum_e += e;
        }
        std::cout << std::fixed << std::setprecision(2) << "Evaluation times per process:[";
        for (int i = 0; i < size; ++i)
            std::cout << all_evals[i] << "s" << (i == size - 1 ? "" : ", ");
        std::cout << "]\n";
        std::cout << "Average fitness evaluation time:         " << (sum_e / size / total_exec * 100.0) << "%\n";
        std::cout << "Communication overhead (Estimated):      " << ((mpi_time + gather_time) / total_exec * 100.0) << "%\n";
        std::cout << "Load imbalance (Estimated):              " << ((max_e - min_e) / total_exec * 100.0) << "%\n";
#endif
    }
#ifdef MLIP_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}