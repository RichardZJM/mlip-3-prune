#include "nsga2.h"
#include <algorithm>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

NSGA2::NSGA2(int pop_size_, int n_var_, int seed) : pop_size(pop_size_), n_var(n_var_), gen(seed)
{
    // Pre-allocate buffers for up to 2 * pop_size to support offspring alongside parents seamlessly
    genes.resize(2 * pop_size * n_var, 0);
    cost_sse.resize(2 * pop_size * 2, 0.0);
    rank.resize(2 * pop_size, 0);
    crowding.resize(2 * pop_size, 0.0);
}

void NSGA2::initialize_population(const std::string &pop_file)
{
    assert(pop_size >= 2 && "Population size must be at least 2.");

    // Clear the first pop_size section
    std::fill(genes.begin(), genes.begin() + pop_size * n_var, 0);

    int loaded_count = 0;
    if (!pop_file.empty())
    {
        std::ifstream f(pop_file);
        if (f.is_open())
        {
            std::string line;
            while (std::getline(f, line) && loaded_count < pop_size)
            {
                std::stringstream ss(line);
                std::string token;
                int j = 0;
                while (std::getline(ss, token, ',') && j < n_var)
                {
                    if (token == "1")
                        genes[loaded_count * n_var + j] = 1;
                    else
                        genes[loaded_count * n_var + j] = 0;
                    ++j;
                }
                loaded_count++;
            }
        }
    }

    if (loaded_count < pop_size)
    {
        int start_idx = loaded_count;
        if (start_idx == 0)
        {
            // Initial individual [0] is all 0s already.
            std::fill(genes.begin() + n_var, genes.begin() + 2 * n_var, 1);
            start_idx = 2;
        }
        else if (start_idx == 1)
        {
            std::fill(genes.begin() + n_var, genes.begin() + 2 * n_var, 1);
            start_idx = 2;
        }

        int remaining = pop_size - start_idx;
        for (int i = start_idx; i < pop_size; ++i)
        {
            // Evenly distribute probabilities for the newly generated batch between 0 and 1
            double prob = (remaining > 1) ? (double)(i - start_idx) / (remaining - 1) : 0.5;

            std::bernoulli_distribution dist(prob);
            for (int j = 0; j < n_var; ++j)
            {
                if (dist(gen))
                    genes[i * n_var + j] = 1;
            }
        }
    }

    // Scramble population using Fisher-Yates across chunk sizes of n_var
    for (int i = pop_size - 1; i > 0; --i)
    {
        std::uniform_int_distribution<int> swap_dist(0, i);
        int j = swap_dist(gen);
        if (i != j)
        {
            std::swap_ranges(genes.begin() + i * n_var,
                             genes.begin() + (i + 1) * n_var,
                             genes.begin() + j * n_var);
        }
    }
}

void NSGA2::generate_offspring()
{
    std::uniform_int_distribution<int> t_dist(0, pop_size - 1);
    std::bernoulli_distribution cross_dist(0.5);
    std::uniform_int_distribution<int> bit_dist(0, n_var - 1);

    int off_offset = pop_size;

    for (int i = 0; i < pop_size; i += 2)
    {
        int p1 = t_dist(gen), p2 = t_dist(gen);
        int p1_idx = (rank[p1] < rank[p2] || (rank[p1] == rank[p2] && crowding[p1] > crowding[p2])) ? p1 : p2;

        int p3 = t_dist(gen), p4 = t_dist(gen);
        int p2_idx = (rank[p3] < rank[p4] || (rank[p3] == rank[p4] && crowding[p3] > crowding[p4])) ? p3 : p4;

        int o1 = off_offset + i;
        int o2 = off_offset + i + 1;

        // Uniform Crossover (gene-by-gene)
        for (int b = 0; b < n_var; ++b)
        {
            bool swap = cross_dist(gen);
            genes[o1 * n_var + b] = swap ? genes[p1_idx * n_var + b] : genes[p2_idx * n_var + b];
            if (o2 < 2 * pop_size)
            {
                genes[o2 * n_var + b] = swap ? genes[p2_idx * n_var + b] : genes[p1_idx * n_var + b];
            }
        }

        // Single Gene Mutation
        int bit1 = bit_dist(gen);
        genes[o1 * n_var + bit1] ^= 1;

        if (o2 < 2 * pop_size)
        {
            int bit2 = bit_dist(gen);
            genes[o2 * n_var + bit2] ^= 1;
        }
    }
}

void NSGA2::survival(int num_inds)
{
    if (sorted_idx.size() < (size_t)num_inds)
    {
        sorted_idx.resize(num_inds);
        ind_front.resize(num_inds);
        min_y.reserve(num_inds);

        front_pool.resize(num_inds);
        front_starts.reserve(num_inds + 1);
        insert_pos.reserve(num_inds + 1);

        next_gen_indices.reserve(pop_size);

        next_genes.resize(genes.size());
        next_cost_sse.resize(cost_sse.size());
        next_rank.resize(rank.size());
        next_crowding.resize(crowding.size());
    }

    // 1. O(N log N) Pre-sort by Obj1 (asc), then Obj2 (asc)
    std::iota(sorted_idx.begin(), sorted_idx.begin() + num_inds, 0);
    std::sort(sorted_idx.begin(), sorted_idx.begin() + num_inds, [&](int a, int b)
              {
        double c_a = cost_sse[a * 2], s_a = cost_sse[a * 2 + 1];
        double c_b = cost_sse[b * 2], s_b = cost_sse[b * 2 + 1];
        if (std::abs(c_a - c_b) > 1e-9) return c_a < c_b;
        return s_a < s_b; });

    // 2. O(N log N) Front Assignment using Patience Sort / LIS Sweep
    min_y.clear();
    for (int i = 0; i < num_inds; ++i)
    {
        int p = sorted_idx[i];
        double cy = cost_sse[p * 2 + 1];

        if (i > 0)
        {
            int prev_p = sorted_idx[i - 1];
            if (std::abs(cost_sse[p * 2] - cost_sse[prev_p * 2]) <= 1e-9 &&
                std::abs(cy - cost_sse[prev_p * 2 + 1]) <= 1e-9)
            {
                ind_front[p] = ind_front[prev_p];
                rank[p] = ind_front[p];
                continue;
            }
        }

        auto it = std::upper_bound(min_y.begin(), min_y.end(), cy);
        int f = std::distance(min_y.begin(), it);
        ind_front[p] = f;
        rank[p] = f;

        if (it == min_y.end())
            min_y.push_back(cy);
        else
            *it = cy;
    }

    // 3. Group by front sequentially (preserves the Obj1 ascending sort perfectly inside each front)
    int num_fronts = min_y.size();
    front_starts.assign(num_fronts + 1, 0);
    for (int i = 0; i < num_inds; ++i)
    {
        front_starts[ind_front[i] + 1]++;
    }
    for (int f = 0; f < num_fronts; ++f)
    {
        front_starts[f + 1] += front_starts[f];
    }

    insert_pos = front_starts;
    for (int i = 0; i < num_inds; ++i)
    {
        int p = sorted_idx[i];
        front_pool[insert_pos[ind_front[p]]++] = p;
    }

    // 4. Survivor Selection + 1D Extracted Crowding Distance (Requires 0 std::sorts!)
    next_gen_indices.clear();
    for (int f = 0; f < num_fronts; ++f)
    {
        int start = front_starts[f];
        int end = front_starts[f + 1];
        int front_size = end - start;

        if (front_size == 0)
            continue;

        if (front_size == 1)
        {
            crowding[front_pool[start]] = INFINITY;
        }
        else if (front_size == 2)
        {
            crowding[front_pool[start]] = INFINITY;
            crowding[front_pool[start + 1]] = INFINITY;
        }
        else
        {
            crowding[front_pool[start]] = INFINITY;
            crowding[front_pool[end - 1]] = INFINITY;

            double min_c = cost_sse[front_pool[start] * 2];
            double max_c = cost_sse[front_pool[end - 1] * 2];

            // Due to Non-Dominated property + Obj1 being natively asc, Obj2 is natively desc
            double max_s = cost_sse[front_pool[start] * 2 + 1];
            double min_s = cost_sse[front_pool[end - 1] * 2 + 1];

            double inv_c = (max_c - min_c > 1e-9) ? 1.0 / (max_c - min_c) : 0.0;
            double inv_s = (max_s - min_s > 1e-9) ? 1.0 / (max_s - min_s) : 0.0;

            for (int j = 1; j < front_size - 1; ++j)
            {
                int p_prev = front_pool[start + j - 1];
                int p_curr = front_pool[start + j];
                int p_next = front_pool[start + j + 1];

                double diff_c = cost_sse[p_next * 2] - cost_sse[p_prev * 2];
                double diff_s = cost_sse[p_prev * 2 + 1] - cost_sse[p_next * 2 + 1]; // Prev is natively larger

                crowding[p_curr] = (diff_c * inv_c) + (diff_s * inv_s);
            }
        }

        if (next_gen_indices.size() + front_size <= (size_t)pop_size)
        {
            next_gen_indices.insert(next_gen_indices.end(),
                                    front_pool.begin() + start,
                                    front_pool.begin() + end);
        }
        else
        {
            auto front_begin = front_pool.begin() + start;
            auto front_end = front_pool.begin() + end;
            std::sort(front_begin, front_end, [&](int a, int b)
                      { return crowding[a] > crowding[b]; });

            int needed = pop_size - next_gen_indices.size();
            next_gen_indices.insert(next_gen_indices.end(),
                                    front_begin,
                                    front_begin + needed);
            break;
        }
    }

    // 5. Ping-pong swap
    for (int k = 0; k < pop_size; ++k)
    {
        int old_idx = next_gen_indices[k];

        std::copy(genes.begin() + old_idx * n_var,
                  genes.begin() + (old_idx + 1) * n_var,
                  next_genes.begin() + k * n_var);

        next_cost_sse[k * 2] = cost_sse[old_idx * 2];
        next_cost_sse[k * 2 + 1] = cost_sse[old_idx * 2 + 1];
        next_rank[k] = rank[old_idx];
        next_crowding[k] = crowding[old_idx];
    }

    std::swap(genes, next_genes);
    std::swap(cost_sse, next_cost_sse);
    std::swap(rank, next_rank);
    std::swap(crowding, next_crowding);
}

void NSGA2::save_pareto(const std::string &prefix)
{
    std::vector<int> pareto_front;
    // We only need to check the first pop_size since these are our survivors.
    for (int i = 0; i < pop_size; ++i)
    {
        if (rank[i] == 0)
        {
            pareto_front.push_back(i);
        }
    }

    std::sort(pareto_front.begin(), pareto_front.end(), [&](int a, int b)
              {
        if (std::abs(cost_sse[a * 2] - cost_sse[b * 2]) > 1e-9) 
            return cost_sse[a * 2] < cost_sse[b * 2];
        return cost_sse[a * 2 + 1] < cost_sse[b * 2 + 1]; });

    std::ofstream f_obj(prefix + "_objectives.csv");
    std::ofstream f_pop(prefix + "_population.csv");

    if (!f_obj.is_open() || !f_pop.is_open())
        return;

    for (int idx : pareto_front)
    {
        f_obj << cost_sse[idx * 2] << "," << cost_sse[idx * 2 + 1] << "\n";

        for (int i = 0; i < n_var; ++i)
        {
            f_pop << (genes[idx * n_var + i] ? "1" : "0") << (i == n_var - 1 ? "" : ",");
        }
        f_pop << "\n";
    }
}