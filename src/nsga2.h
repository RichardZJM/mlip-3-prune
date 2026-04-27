#pragma once
#include <random>
#include <string>
#include <vector>

class NSGA2
{
public:
    int pop_size, n_var;
    std::mt19937 gen;

    // Direct pre-allocated buffers (Capacity for Parent + Offspring)
    std::vector<char> genes;      // length: 2 * pop_size * n_var
    std::vector<double> cost_sse; // length: 2 * pop_size * 2
    std::vector<int> rank;        // length: 2 * pop_size
    std::vector<double> crowding; // length: 2 * pop_size

    NSGA2(int pop_size_, int n_var_, int seed);
    void initialize_population(const std::string &pop_file = "");
    void generate_offspring();
    void survival(int num_inds);
    void save_pareto(const std::string &prefix);

private:
    // O(N log N) 2D sweep-line buffers
    std::vector<int> sorted_idx;
    std::vector<double> min_y;
    std::vector<int> ind_front;
    std::vector<int> insert_pos;

    std::vector<int> front_pool;
    std::vector<int> front_starts;
    std::vector<int> next_gen_indices;

    // Ping-pong buffers
    std::vector<char> next_genes;
    std::vector<double> next_cost_sse;
    std::vector<int> next_rank;
    std::vector<double> next_crowding;
};