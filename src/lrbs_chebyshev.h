#ifndef MLIP_LRBS_CHEBYSHEV_H
#define MLIP_LRBS_CHEBYSHEV_H

#include "common/utils.h"
#include "basis.h"
#include "common/multidimensional_arrays.h"
#include <fstream>
#include <string>

class LRBS_Chebyshev : public AnyBasis
{
public:
    LRBS_Chebyshev(std::ifstream &ifs) : AnyBasis(ifs) { ReadBasis(ifs); };

    std::string GetRBTypeString() override { return "LRBSChebyshev"; }

    void ReadBasis(std::ifstream &ifs) override;
    void WriteBasis(std::ofstream &ofs) override;

    void Calc(double val) override {};
    void CalcDers(double val) override {};

    void Calc(double val, int t1, int t2) override;
    void CalcDers(double val, int t1, int t2, int neigh_index) override;

    int GetMinimumNeighbor() override;
    double GetAndResetSpeciesVal() override;
    double GetAndResetSpeciesDerFac(const std::vector<double> &dists) override;
    void WriteEnvelopeData(std::ofstream &ofs, int t1, int t2);

private:
    // Used to smoothly switch the constant term
    int minimum_neigh_index = -1;
    double species_basis_val = 1.0;
    double species_basis_der = 0.0;

    // Physical envelope boundaries
    Array2D mindists;
    Array2D cutoffs;
    Array2D peaks;

    // Precomputed values to enable a 100% division-free evaluation loop
    Array2D inv_left_ranges;  // 1.0 / (peak - mindist)
    Array2D inv_right_ranges; // 1.0 / (cutoff - peak)
    Array2D inv_total_ranges; // 1.0 / (cutoff - mindist)
    Array2D cheb_mults;       // 2.0 / (cutoff - mindist)
    Array2D cheb_offsets;     // (mindist + cutoff) / (cutoff - mindist)

    // Precomputed switching parameters for species coefficient
    Array2D switching_mins;
    Array2D inv_switch_ranges;
};

#endif // MLIP_LRBS_CHEBYSHEV_H