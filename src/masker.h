#pragma once
#include "mtpr.h"
#include <string>
#include <vector>

class Masker : public MLMTPR
{
public:
    Masker(const std::string &base_mtp);
    void ApplyMask(const std::vector<char> &mask, const std::vector<double> *theta);
};

std::vector<char> ReadMask(const std::string &mask_file, int row, int expected_size);
std::vector<double> SolveTheta(const std::string &config_file, const std::vector<char> &mask, int species_count);