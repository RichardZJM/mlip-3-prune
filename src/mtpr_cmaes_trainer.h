#ifndef MLIP_MTPR_CMAES_TRAINER_H
#define MLIP_MTPR_CMAES_TRAINER_H

#ifndef MLIP_MPI
#error "mtpr_cmaes_trainer requires MPI. Compile with MLIP_MPI defined."
#endif

#include "common/stdafx.h"
#include "mtpr_trainer.h"
#include "common/cmaes.h"

#include <vector>
#include <string>
#include <chrono>

// LAPACK declarations (matching project style)
extern "C" void dpotrf_(const char *uplo, const int *n, double *a,
                        const int *lda, int *info);

extern "C" void dpotrs_(const char *uplo, const int *n, const int *nrhs,
                        const double *a, const int *lda,
                        double *b, const int *ldb, int *info);

extern "C" void dposv_(const char *uplo, const int *n, const int *nrhs,
                       double *a, const int *lda,
                       double *b, const int *ldb, int *info);

class MTPR_cmaes_trainer : public MTPR_trainer
{
public:
    int subcomm_size = 1;

    int cmaes_lambda = -1;
    int cmaes_mu = -1;
    double cmaes_sigma0 = -1.0;
    double cmaes_init_range = 1e-2;
    int cmaes_seed = 0;

    double cmaes_sigma_tol = 1e-11;
    double cmaes_cond_tol = 1e14;
    double cmaes_tolfun = -1.0; // Default set to -1.0 to disable tolfun termination

    int cmaes_max_iter = 200;
    double cmaes_timeout = 999999999.0;

    std::string cmaes_save_to = "";

    double cmaes_scale_bracket_factor = 1.5;
    int cmaes_scale_bracket_steps = 8;
    double cmaes_scale_tol = 1e-4;

    MTPR_cmaes_trainer(MLMTPR *p_mtp, Settings settings, bool verbose = true)
        : MTPR_trainer(p_mtp, settings, verbose), p_mlmtpr_(p_mtp)
    {
        InitCMAESSettings_();
        ApplySettings(settings);
        if (verbose)
            PrintSettings();
    }

    ~MTPR_cmaes_trainer() { FreeSubComm_(); }

    void CMAESSearch(std::vector<Configuration> & /*unused*/);
    void SetTrainFile(const std::string &fname) { train_file_ = fname; }

private:
    MLMTPR *p_mlmtpr_ = nullptr;
    std::string train_file_;

    MPI_Comm subcomm_ = MPI_COMM_NULL;
    MPI_Comm subcomm_leaders_ = MPI_COMM_NULL;
    int subcomm_rank_ = 0;
    int subcomm_id_ = -1;
    int num_subcomms_ = 0;
    int subcomm_total_cfgs_ = 0;

    std::vector<double> base_gram_;
    std::vector<double> base_rhs_;
    std::vector<double> chol_L_;
    int slae_n_ = 0;
    double base_scalar_ = 0.0;
    int base_N_ = 0;

    CMAES cmaes_;

    void InitCMAESSettings_();
    void BuildSubCommunicators_();
    void FreeSubComm_();

    std::vector<Configuration> SubCommLoadCfgs_(const std::string &filename);

    struct EvalResult
    {
        double loss;
        double scaling;
    };
    EvalResult EvaluateCandidate_(const std::vector<double> &position,
                                  std::vector<Configuration> &local_cfgs,
                                  double prev_best_scaling);

    struct ScaleResult
    {
        double best_scaling;
        double best_loss;
        std::vector<double> best_linear_coeffs;
    };

    ScaleResult GoldenSectionScalingSearch_(double warm_scaling,
                                            const std::vector<int> &degrees);

    double ApplyDScaleAndSolve_(double scaling,
                                const std::vector<int> &degrees,
                                std::vector<double> &out_coeffs);

    int RadialCoeffCount_() const;
    void PosToRadialCoeffs_(const std::vector<double> &pos, MLMTPR &mtp) const;
    void RadialCoeffsToPos_(const MLMTPR &mtp, std::vector<double> &pos) const;

    void BuildInitialMean_(std::vector<double> &mean0) const;
};

#endif // MLIP_MTPR_CMAES_TRAINER_H