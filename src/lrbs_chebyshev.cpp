#include "lrbs_chebyshev.h"
#include "common/utils.h"
#include <cmath>
#include <map>
#include <algorithm>

using namespace std;

struct EnvelopeParams
{
    double mindist;
    double cutoff;
    double peak;
};

// Faster, branch-free smootherstep evaluation (6t^5 - 15t^4 + 10t^3)
static inline double smootherstep_val(double x, double min_val, double inv_range)
{
    double t = (x - min_val) * inv_range;
    t = std::max(0.0, std::min(1.0, t));
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Branch-free derivative of the smootherstep function
static inline double smootherstep_der(double x, double min_val, double inv_range)
{
    double t = (x - min_val) * inv_range;
    // Multiplication of boolean results eliminates short-circuit jump instructions
    double active = (t > 0.0) * (t < 1.0);
    return active * 30.0 * t * t * (1.0 - t) * (1.0 - t) * inv_range;
}

// Reversed smootherstep evaluation for the right cutoff (decays to 0 at cutoff)
static inline double rev_smootherstep_val(double x, double max_val, double inv_range)
{
    double t = (max_val - x) * inv_range;
    t = std::max(0.0, std::min(1.0, t));
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Branch-free derivative of the reversed smootherstep function
static inline double rev_smootherstep_der(double x, double max_val, double inv_range)
{
    double t = (max_val - x) * inv_range;
    double active = (t > 0.0) * (t < 1.0);
    // Chain rule negative sign because d(max_val - x)/dx = -1
    return -active * 30.0 * t * t * (1.0 - t) * (1.0 - t) * inv_range;
}

// Evaluates the product of the left and right smoothersteps
static inline double envelope(double val,
                              double mindist,
                              double cutoff,
                              double inv_left_range,
                              double inv_right_range)
{
    double s_left = smootherstep_val(val, mindist, inv_left_range);
    double s_right = rev_smootherstep_val(val, cutoff, inv_right_range);
    return s_left * s_right;
}

// Derivative of the product envelope via product rule (ds_left * s_right + s_left * ds_right)
static inline double envelope_der(double val,
                                  double mindist,
                                  double cutoff,
                                  double inv_left_range,
                                  double inv_right_range)
{
    double s_left = smootherstep_val(val, mindist, inv_left_range);
    double ds_left = smootherstep_der(val, mindist, inv_left_range);

    double s_right = rev_smootherstep_val(val, cutoff, inv_right_range);
    double ds_right = rev_smootherstep_der(val, cutoff, inv_right_range);

    return ds_left * s_right + s_left * ds_right;
}

void LRBS_Chebyshev::ReadBasis(std::ifstream &ifs)
{
    if ((!ifs.is_open()) || (ifs.eof()))
        ERROR("LRBS_Chebyshev::ReadBasis: Can't load basis");

    string tmpstr;
    std::map<std::pair<int, int>, EnvelopeParams> envelope_data_map;
    int max_species_index = -1;

    while (ifs >> tmpstr && tmpstr.find('-') != std::string::npos)
    {
        size_t dash_pos = tmpstr.find('-');
        int type1 = std::stoi(tmpstr.substr(0, dash_pos));
        int type2 = std::stoi(tmpstr.substr(dash_pos + 1));
        max_species_index = std::max(max_species_index, std::max(type1, type2));

        EnvelopeParams params;
        char brace, comma;

        // Reads 3-tuple format: {mindist, cutoff, peak}
        if (ifs >> brace && brace == '{' &&
            ifs >> params.mindist >> comma >> params.peak >> comma >> params.cutoff >> brace &&
            brace == '}')
        {
            if (params.mindist >= params.cutoff)
                ERROR("Check your cutoffs! mindist >= cutoff.");
            if (params.peak <= params.mindist || params.peak >= params.cutoff)
                ERROR("Check your peak! It must be strictly between mindist and cutoff.");

            envelope_data_map[{type1, type2}] = params;
        }
        else
        {
            ERROR("LRBS_Chebyshev::ReadBasis: Error parsing radial envelope. Expected {mindist, cutoff, peak}");
        }
    }

    if (max_species_index == -1)
        ERROR("LRBS_Chebyshev::ReadBasis: No envelope data found.");

    int species_count = max_species_index + 1;

    // Resizing boundary arrays
    mindists.resize(species_count, species_count);
    cutoffs.resize(species_count, species_count);
    peaks.resize(species_count, species_count);

    // Resizing evaluation precomputations
    inv_left_ranges.resize(species_count, species_count);
    inv_right_ranges.resize(species_count, species_count);
    inv_total_ranges.resize(species_count, species_count);
    cheb_mults.resize(species_count, species_count);
    cheb_offsets.resize(species_count, species_count);

    // Resizing species switching arrays
    switching_mins.resize(species_count, species_count);
    inv_switch_ranges.resize(species_count, species_count);

    max_val = 0;
    min_val = 999999;

    for (const auto &pair : envelope_data_map)
    {
        int t1 = pair.first.first;
        int t2 = pair.first.second;
        const EnvelopeParams &params = pair.second;

        mindists(t1, t2) = params.mindist;
        cutoffs(t1, t2) = params.cutoff;
        peaks(t1, t2) = params.peak;

        // Division precomputations for the envelope
        inv_left_ranges(t1, t2) = 1.0 / (params.peak - params.mindist);
        inv_right_ranges(t1, t2) = 1.0 / (params.cutoff - params.peak);

        double total_range = params.cutoff - params.mindist;
        inv_total_ranges(t1, t2) = 1.0 / total_range;

        // Precompute scaling values for Chebyshev normalization
        cheb_mults(t1, t2) = 2.0 / total_range;
        cheb_offsets(t1, t2) = (params.mindist + params.cutoff) / total_range;

        // Default the species coefficient transition to the range [mindist, peak]
        switching_mins(t1, t2) = params.mindist;
        inv_switch_ranges(t1, t2) = 1.0 / (params.peak - params.mindist);

        max_val = max(params.cutoff, max_val);  // Update the global cutoff
        min_val = min(params.mindist, min_val); // Update the global mindist
    }

    if (tmpstr != "basis_size" && tmpstr != "radial_basis_size")
        ERROR("Error reading .mtp file");
    ifs.ignore(2);
    ifs >> size;
    if (ifs.fail())
        ERROR("Error reading .mtp file");

    vals.resize(size);
    ders.resize(size);
}

void LRBS_Chebyshev::WriteBasis(std::ofstream &ofs)
{
    if (!ofs.is_open())
        ERROR("LRBS_Chebyshev::WriteBasis: Output stream isn't open");

    ofs.setf(ios::scientific);
    ofs.precision(15);

    ofs << "radial_basis_type = " << GetRBTypeString() << '\n';
    ofs << "\tradial_envelope\n";

    const int species_count = mindists.size1;
    for (int t1 = 0; t1 < species_count; ++t1)
    {
        for (int t2 = 0; t2 < species_count; ++t2)
        {
            ofs << "        " << t1 << "-" << t2 << std::endl;
            // Writes cleaner 3-tuple format to the MTP basis file
            ofs << "            {" << mindists(t1, t2) << ", "
                << cutoffs(t1, t2) << ", "
                << peaks(t1, t2) << "}" << std::endl;
        }
    }

    ofs << "\tradial_basis_size = " << size << '\n';
}

void LRBS_Chebyshev::Calc(double val, int t1, int t2)
{
    double mindist = mindists(t1, t2);
    double cutoff = cutoffs(t1, t2);
    double inv_left = inv_left_ranges(t1, t2);
    double inv_right = inv_right_ranges(t1, t2);

    double mult = cheb_mults(t1, t2);
    double offset = cheb_offsets(t1, t2);

    // Division-free Chebyshev coordinate generation via simple FMA-style mapping
    double ksi = val * mult - offset;

    double env = envelope(val, mindist, cutoff, inv_left, inv_right);

    vals[0] = scaling * env;
    if (size > 1)
        vals[1] = scaling * env * ksi;

    for (int i = 2; i < size; ++i)
        vals[i] = 2.0 * ksi * vals[i - 1] - vals[i - 2];
}

void LRBS_Chebyshev::CalcDers(double val, int t1, int t2, int neigh_index)
{
    Calc(val, t1, t2);

    double mindist = mindists(t1, t2);
    double cutoff = cutoffs(t1, t2);
    double inv_left = inv_left_ranges(t1, t2);
    double inv_right = inv_right_ranges(t1, t2);

    double mult = cheb_mults(t1, t2);
    double offset = cheb_offsets(t1, t2);
    double ksi = val * mult - offset;

    double env = vals[0] / scaling;
    double env_der = envelope_der(val, mindist, cutoff, inv_left, inv_right);

    ders[0] = scaling * env_der;
    if (size > 1)
        ders[1] = scaling * (env_der * ksi + env * mult);

    for (int i = 2; i < size; ++i)
        ders[i] = 2.0 * (mult * vals[i - 1] + ksi * ders[i - 1]) - ders[i - 2];

    // Division-free smootherstep calculation for species switching coefficients
    double switch_min = switching_mins(t1, t2);
    double inv_switch_range = inv_switch_ranges(t1, t2);

    double fc = smootherstep_val(val, switch_min, inv_switch_range);
    double dfc = smootherstep_der(val, switch_min, inv_switch_range);

    // Invert function direction (so it is 1.0 at switch_min, falling to 0.0)
    fc = 1.0 - fc;
    dfc = -dfc;

    if (fc <= species_basis_val)
    {
        minimum_neigh_index = neigh_index;
        species_basis_val = fc;
        species_basis_der = dfc;
    }
}

int LRBS_Chebyshev::GetMinimumNeighbor() { return minimum_neigh_index; }

double LRBS_Chebyshev::GetAndResetSpeciesVal()
{
    double currentValue = species_basis_val;
    species_basis_val = 1.0;
    return 1.0;
}

double LRBS_Chebyshev::GetAndResetSpeciesDerFac(const std::vector<double> &dists)
{
    double der_fac = species_basis_der / dists[minimum_neigh_index];
    species_basis_der = 0.0;
    return 0.0;
}

void LRBS_Chebyshev::WriteEnvelopeData(std::ofstream &ofs, int t1, int t2)
{
    if (!ofs.is_open())
        ERROR("LRBS_Chebyshev::WriteEnvelopeData: Output stream isn't open");

    ofs.setf(ios::scientific);
    ofs.precision(15);

    double mindist = mindists(t1, t2);
    double cutoff = cutoffs(t1, t2);
    double inv_left = inv_left_ranges(t1, t2);
    double inv_right = inv_right_ranges(t1, t2);

    const int num_points = 500;
    double step_size = (cutoff - mindist) / (num_points - 1);

    ofs << "# x_value, envelope_value, envelope_derivative\n";

    for (int i = 0; i < num_points; ++i)
    {
        double x = mindist + i * step_size;

        double env = envelope(x, mindist, cutoff, inv_left, inv_right);
        double der = envelope_der(x, mindist, cutoff, inv_left, inv_right);

        ofs << x << " " << env << " " << der << "\n";
    }
}