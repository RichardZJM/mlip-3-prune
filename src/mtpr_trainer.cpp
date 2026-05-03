/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include "mtpr_trainer.h"
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <map>

using namespace std;

std::stringstream logstrm1; // stream for output

void MTPR_trainer::ClearSLAE()
{
    const int n = p_mlmtpr->alpha_count - 1 + p_mlmtpr->species_count; // Matrix size

    lin_vector.resize(n);
    FillWithZero(lin_vector);
    lin_matrix.resize(n * n);
    FillWithZero(lin_matrix);

    lin_eqn_count = 0;
    lin_scalar = 0.0;
}

void MTPR_trainer::SymmetrizeSLAE()
{
    const int n = p_mlmtpr->alpha_count + p_mlmtpr->species_count - 1; // Matrix size

    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            lin_matrix[j * n + i] = lin_matrix[i * n + j];
}

void MTPR_trainer::SolveSLAE(int TS_size)
{
    SymmetrizeSLAE();

    int n = p_mlmtpr->alpha_count - 1 + p_mlmtpr->species_count; // Matrix size

    if (!reg_init) // checking the condition of changing the regularization
        for (int i = 0; i < n; i++)
            if ((p_mlmtpr->reg_vector[i] < 1e-2 * reg_param * std::max(1.0, lin_matrix[i * n + i]) / TS_size) || (p_mlmtpr->reg_vector[i] > 1e2 * reg_param * std::max(1.0, lin_matrix[i * n + i]) / TS_size))
            {
                reg_init = true;
                logstrm1 << "Regularization parameters updated. Hessian in BFGS is reset" << endl;
                MLP_LOG("fit", logstrm1.str());
                logstrm1.str("");
                break;
            }

    if (reg_init) // if we need to change the regularization
        for (int i = 0; i < n; i++)
            p_mlmtpr->reg_vector[i] = reg_param * std::max(1.0, lin_matrix[i * n + i]) / TS_size; // assigning a regularization proportional to the matrix element size

    // Regularization
    for (int i = 0; i < n; i++)
        lin_matrix[i * n + i] += p_mlmtpr->reg_vector[i] * TS_size;

    lin_matrix_reg = lin_matrix; // Capture a copy before LAPACK destructively overwrites it

    // Solve SLAE with LAPACK Cholesky Solver (dposv)
    char uplo = 'U';
    int nrhs = 1;
    int info = 0;

    // lin_matrix is symmetric so row-major layout perfectly maps to column-major lapack expectations.
    dposv_(&uplo, &n, &nrhs, lin_matrix.data(), &n, lin_vector.data(), &n, &info);

    // populate linear_coeffs with the solved vector
    for (int i = 0; i < n; i++)
    {
        p_mlmtpr->linear_coeffs(i) = lin_vector[i];
    }
}
// TODO: remove weight?
void MTPR_trainer::AddToSLAE(Configuration &cfg, double weight)
{
    if (cfg.nbh_cutoff != p_mlip->CutOff())
        cfg.InitNbhs(p_mlip->CutOff());

    if (cfg.nbhs.size() == 0)
        return;

    int n = p_mlmtpr->alpha_count - 1 + p_mlmtpr->species_count;

    p_mlmtpr->CalcEFSComponents(cfg);

    int fn = norm_by_forces;
    double d = 0.1;
    double avef = 0;

    if (cfg.has_forces())
        for (int ind = 0; ind < cfg.nbhs.size(); ind++)
            avef += cfg.force(ind).NormSq() / cfg.nbhs.size();

    char uplo_L = 'L';
    char trans_N = 'N';
    double beta_1 = 1.0;
    int inc1 = 1;

    // --- ENERGIES (Unchanged from before) ---
    if (cfg.has_energy())
    {
        double alpha_e = weight * wgt_energy(cfg) * d / (d + fn * avef);
        if (alpha_e != 0.0)
        {
            dsyr_(&uplo_L, &n, &alpha_e, p_mlmtpr->energy_cmpnts.data(), &inc1, lin_matrix.data(), &n);

            double alpha_vec = alpha_e * cfg.energy;
            for (int i = 0; i < n; i++)
                lin_vector[i] += alpha_vec * p_mlmtpr->energy_cmpnts[i];

            lin_scalar += alpha_vec * cfg.energy;
            lin_eqn_count += (weight > 0) ? 1 : ((weight < 0) ? -1 : 0);
        }
    }

    // --- FORCES (Zero-copy DSYRK) ---
    if ((wgt_eqtn_forces > 0) && (cfg.has_forces()))
    {
        double alpha_f = weight * d / (d + fn * avef);
        if (alpha_f != 0.0)
        {
            int K_f = cfg.nbhs.size() * 3;
            if (K_f > 0)
            {
                double *F_ptr = p_mlmtpr->forces_cmpnts.data();
                int offset = 0;

                for (int ind = 0; ind < cfg.nbhs.size(); ind++)
                {
                    int my_ind = cfg.nbhs[ind].my_ind;
                    for (int a = 0; a < 3; a++)
                    {
                        double w = wgt_forces(cfg, ind, a);
                        double f_val = cfg.force(my_ind, a);
                        double alpha_vec = alpha_f * w * f_val;
                        double sqrt_w = std::sqrt(std::max(0.0, w));

                        // We do 2 jobs at once: Add to lin_vector, and scale F_ptr in-place for DSYRK
                        for (int i = 0; i < n; i++)
                        {
                            lin_vector[i] += alpha_vec * F_ptr[offset + i];
                            F_ptr[offset + i] *= sqrt_w;
                        }

                        lin_scalar += alpha_vec * f_val;
                        offset += n;
                    }
                    lin_eqn_count += 3 * ((weight > 0) ? 1 : ((weight < 0) ? -1 : 0));
                }

                // Fire BLAS natively against our contiguous columns!
                dsyrk_(&uplo_L, &trans_N, &n, &K_f, &alpha_f, F_ptr, &n, &beta_1, lin_matrix.data(), &n);
            }
        }
    }

    // --- STRESSES (Zero-copy DSYRK) ---
    if ((wgt_eqtn_stress > 0) && (cfg.has_stresses()))
    {
        double alpha_s = weight;
        if (alpha_s != 0.0)
        {
            int K_s = 9;
            double *S_ptr = p_mlmtpr->stress_cmpnts.data();
            int offset = 0;

            for (int a = 0; a < 3; a++)
            {
                for (int b = 0; b < 3; b++)
                {
                    double w = wgt_stress(cfg, a, b);
                    double s_val = cfg.stresses[a][b];
                    double alpha_vec = alpha_s * w * s_val;
                    double sqrt_w = std::sqrt(std::max(0.0, w));

                    for (int i = 0; i < n; i++)
                    {
                        lin_vector[i] += alpha_vec * S_ptr[offset + i];
                        S_ptr[offset + i] *= sqrt_w;
                    }

                    lin_scalar += alpha_vec * s_val;
                    offset += n;
                }
            }
            lin_eqn_count += 6 * ((weight > 0) ? 1 : ((weight < 0) ? -1 : 0));

            // Fire BLAS natively
            dsyrk_(&uplo_L, &trans_N, &n, &K_s, &alpha_s, S_ptr, &n, &beta_1, lin_matrix.data(), &n);
        }
    }
}

void MTPR_trainer::LinOptimize(vector<Configuration> &training_set)
{
    p_mlmtpr->Orthogonalize(); // orthogonalize the radial functions

    ClearSLAE();

    for (auto &cfg : training_set)
        AddToSLAE(cfg);

    int m = (int)training_set.size(); // train set size on the current core
    int K = m;                        // train set size over all cores

#ifdef MLIP_MPI
    MPI_Allreduce(&m, &K, 1, MPI_INT, MPI_SUM, mpi.comm);
#endif

#ifdef MLIP_MPI
    double scalar = 0;

    int n = p_mlmtpr->alpha_count - 1 + p_mlmtpr->species_count; // Matrix size

    if (mpi.rank == 0)
    {
        // TODO: get rid of new
        lin_matrix_mpi.resize(n * n);
        lin_vector_mpi.resize(n);
        std::fill(lin_matrix_mpi.begin(), lin_matrix_mpi.end(), 0);
        std::fill(lin_vector_mpi.begin(), lin_vector_mpi.end(), 0);
    }

    MPI_Reduce(&lin_matrix[0], &lin_matrix_mpi[0], n * n, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&lin_vector[0], &lin_vector_mpi[0], n, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&lin_scalar, &scalar, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);

    if (mpi.rank == 0) // we solve the SLAE in serial and then broadcast the results
    {
        memcpy(&lin_matrix[0], &lin_matrix_mpi[0], n * n * sizeof(double));
        memcpy(&lin_vector[0], &lin_vector_mpi[0], n * sizeof(double));
        lin_scalar = scalar;

        SolveSLAE(K);
    }

#else
    SolveSLAE(K);
#endif
#ifdef MLIP_MPI // spread the results to all cores
    MPI_Bcast(&p_mlmtpr->Coeff()[0], p_mlmtpr->CoeffCount(), MPI_DOUBLE, 0, mpi.comm);
    MPI_Bcast(&p_mlmtpr->reg_vector[0], (int)p_mlmtpr->reg_vector.size(), MPI_DOUBLE, 0, mpi.comm);
    MPI_Bcast(&reg_init, 1, MPI_C_BOOL, 0, mpi.comm);
#endif
}

void MTPR_trainer::AddSpecies(std::vector<Configuration> &training_set)
{
    MPI_Barrier(mpi.comm);

    // This means no coefficients were present in MTPR potential, and the atomic_numbers should be cleared
    if (!p_mlmtpr->inited)
    {
        p_mlmtpr->species_count = 0;
        p_mlmtpr->atomic_numbers.clear();
        p_mlmtpr->regression_coeffs.resize(p_mlmtpr->alpha_count - 1);
    }

    // species detected in the training set
    vector<int> db_species;
    // Checking the atomic numbers present in the training set
    int ts_size_local = 0;
    int ts_size_global = 0;
    for (Configuration &cfg : training_set)
    {
        for (int j = 0; j < cfg.size(); j++)
            if (std::find(db_species.begin(), db_species.end(), cfg.type(j)) == db_species.end())
                db_species.push_back(cfg.type(j));
        if (cfg.size() > 0)
            ts_size_local++;
    }
    MPI_Allreduce(&ts_size_local, &ts_size_global, 1, MPI_INT, MPI_SUM, mpi.comm);
    Message(to_string(ts_size_global) + " configurations found in the training set");

    double min_dist = 999;
    double min_magmom = 999;
    double max_magmom = -999;
    // getting the lowest min_dist for the training set
    for (int i = 0; i < training_set.size(); i++)
    {
        if (training_set[i].MinDist() < min_dist)
            min_dist = training_set[i].MinDist();
        if (training_set[i].has_magmom())
        {
            if (training_set[i].MinMagMom() < min_magmom)
                min_magmom = training_set[i].MinMagMom();
            if (training_set[i].MaxMagMom() > max_magmom)
                max_magmom = training_set[i].MaxMagMom();
        }
    }

    double total_min_dist = min_dist;
    double total_min_magmom = min_magmom;
    double total_max_magmom = max_magmom;

    MPI_Barrier(mpi.comm);
    int maxlen = 0;
    int this_size = (int)db_species.size();
    MPI_Allreduce(&this_size, &maxlen, 1, MPI_INT, MPI_MAX, mpi.comm);               // finding maximum lengths of db_species among the processes
    MPI_Allreduce(&min_dist, &total_min_dist, 1, MPI_DOUBLE, MPI_MIN, mpi.comm);     // finding minimum distance in configuration among the processes
    MPI_Allreduce(&min_magmom, &total_min_magmom, 1, MPI_DOUBLE, MPI_MIN, mpi.comm); // finding minimum magnetic moment in configuration among the processes
    MPI_Allreduce(&max_magmom, &total_max_magmom, 1, MPI_DOUBLE, MPI_MAX, mpi.comm); // finding maximal magnetic moment in configuration among the processes

    vector<int> sendbuf(maxlen);                                          // constructing a sending buffers of equal sizes for all processes
    std::fill(sendbuf.begin(), sendbuf.end(), -1);                        // fill sendbuf with -1
    memcpy(&sendbuf[0], &db_species[0], db_species.size() * sizeof(int)); // fill some parts of the sendbuf with actual species detected for each process (for a part of the training set)

    vector<int> total_species(mpi.size * maxlen); // constructing the receiving buffer
    std::fill(total_species.begin(), total_species.end(), -1);
    MPI_Barrier(mpi.comm);

    MPI_Allgather(&sendbuf[0], maxlen, MPI_INT, &total_species[0], maxlen, MPI_INT, mpi.comm); // fill the array with species from all processes

    sort(total_species.begin(), total_species.end());

    for (int i = 0; i < mpi.size * maxlen; i++) // adding new species to the MTPR potential
        if (total_species[i] >= 0)
            if (std::find(p_mlmtpr->atomic_numbers.begin(), p_mlmtpr->atomic_numbers.end(), total_species[i]) == p_mlmtpr->atomic_numbers.end())
                p_mlmtpr->atomic_numbers.push_back(total_species[i]);

    int new_spec = (int)p_mlmtpr->atomic_numbers.size(); // new number of species (after adding new atomic types)
    int old_spec = p_mlmtpr->species_count;              // old number of species

    for (int i = 0; i < new_spec; i++)
        if (std::find(total_species.begin(), total_species.end(), p_mlmtpr->atomic_numbers[i]) == total_species.end())
        {
            string errmsg = "Atomic number " + std::to_string(p_mlmtpr->atomic_numbers[i]) + " from the MTP is not present in the training set. You are doing something wrong!";
            ERROR(errmsg);
        }

    if (old_spec < new_spec)
    {
        if (mpi.rank == 0)
        {
            logstrm1 << "Following atomic numbers will be added to the MTP potential: ";
            for (int i = old_spec; i < p_mlmtpr->atomic_numbers.size(); i++)
            {
                logstrm1 << p_mlmtpr->atomic_numbers[i];
                if (i != p_mlmtpr->atomic_numbers.size() - 1)
                    logstrm1 << ", ";
            }
            logstrm1 << "\n";
            MLP_LOG("fit", logstrm1.str());
            logstrm1.str("");
        }
        MLP_LOG("fit", logstrm1.str());
        logstrm1.str("");
        int K = p_mlmtpr->radial_func_count;   // number of radial functions in current potential
        int R = p_mlmtpr->p_RadialBasis->size; // number of Chebyshev polynomials constituting one radial function
        int M = p_mlmtpr->mbasis_size;

        // the code below does the rearrangement of the present coefficients in the memory and adding of the new ones
        vector<double> old_radial(old_spec * old_spec * K * R);
        if (old_radial.size() > 0)
            memcpy(&old_radial[0], &p_mlmtpr->regression_coeffs[0], old_spec * old_spec * K * R * sizeof(double));

        // rearrange and extend the linear coefficients
        int nlin_old = p_mlmtpr->LinSize();
        p_mlmtpr->regression_coeffs.resize(p_mlmtpr->Rsize() + nlin_old + new_spec - old_spec);

        for (int i = nlin_old + new_spec - old_spec - 1; i >= new_spec; i--)
            p_mlmtpr->linear_coeffs(i) = p_mlmtpr->linear_coeffs(i - new_spec + old_spec);

        for (int i = old_spec - 1; i >= 0; i--)
            p_mlmtpr->linear_coeffs(i) = p_mlmtpr->linear_coeffs(i);

        for (int i = 0; i < new_spec - old_spec; i++)
            p_mlmtpr->linear_coeffs(i + old_spec) = 0;

        // rearrange and extend the radial coefficients
        p_mlmtpr->species_count = new_spec;
        p_mlmtpr->regression_coeffs.resize(new_spec * new_spec * K * R * M * M + nlin_old + new_spec - old_spec);

        int pair = 0;
        int old_pair = 0;
        for (int p1 = 0; p1 < new_spec; p1++)
        {
            for (int p2 = 0; p2 < new_spec; p2++)
            {
                if ((p1 < old_spec) && (p2 < old_spec))
                {
                    for (int i = 0; i < K * R * M * M; i++)
                        p_mlmtpr->regression_coeffs[pair * K * R * M * M + i] = old_radial[old_pair * K * R * M * M + i];
                    old_pair++;
                }
                else
                {
                    // setting the new coefficients to default values (in case of random initial guess they will be overwritten)
                    for (int i = 0; i < K; i++)
                    {
                        for (int j = 0; j < R; j++)
                            for (int alpha = 0; alpha < M; alpha++)
                                for (int beta = 0; beta < M; beta++)
                                    p_mlmtpr->regression_coeffs[pair * K * R * M * M + (i * R + j) * M * M + alpha * M + beta] = 1e-6;

                        for (int alpha = 0; alpha < M; alpha++)
                            for (int beta = 0; beta < M; beta++)
                                p_mlmtpr->regression_coeffs[pair * K * R * M * M + (i * R + min(i, R)) * M * M + alpha * M + beta] = 1e-3 * (pair + 1);
                    }
                }
                pair++;
            }
        }
        p_mlmtpr->MemAlloc(); // resize EFS components and basis functions
    }

    p_mlmtpr->reg_vector.resize(p_mlmtpr->alpha_scalar_moments + p_mlmtpr->species_count); // resize the regularization vector

    if (!no_mindist_update)
    {
        if (mpi.rank == 0)
        {
            logstrm1 << "Minimal interatomic distance in the training set is " << total_min_dist << ". MTP's mindist will be updated\n";
            MLP_LOG("fit", logstrm1.str());
            logstrm1.str("");
        }
        p_mlmtpr->p_RadialBasis->min_val = 0.99 * total_min_dist;
    }
    if (auto_minmax_magmom && p_mlmtpr->p_MagneticBasis != nullptr)
    {
        if (total_min_magmom < p_mlmtpr->p_MagneticBasis->min_val)
        {
            if (mpi.rank == 0)
            {
                logstrm1 << "Found configuration with min_magmom=" << total_min_magmom << ", MTP's min_magmom will be decreased\n";
                MLP_LOG("fit", logstrm1.str());
            }
            p_mlmtpr->p_MagneticBasis->min_val = total_min_magmom;
        }
        if (total_max_magmom > p_mlmtpr->p_MagneticBasis->max_val)
        {
            if (mpi.rank == 0)
            {
                logstrm1 << "Found configuration with max_magmom=" << total_max_magmom << ", MTP's max_magmom will be increased\n";
                MLP_LOG("fit", logstrm1.str());
            }
            p_mlmtpr->p_MagneticBasis->max_val = total_max_magmom;
        }
    }
}

void MTPR_trainer::NonLinOptimize(std::vector<Configuration> &training_set, int max_iter) // with Shapeev bfgs
{
    // AddSpecies(training_set);   //add new species to MTPR potential if they are present in the training set

#ifdef MLIP_MPI
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
    {
        logstrm1 << "MTPR training started on " << mpi.size << " core(s)" << endl;
        //      if (GetLogStream()!=nullptr) GetLogStream()->precision(15);
        MLP_LOG("fit", logstrm1.str());
        logstrm1.str("");
    }
#else
    logstrm1 << "MTPR serial(?!?) training started" << endl;
    //      if (GetLogStream()!=nullptr) GetLogStream()->precision(15);
    MLP_LOG("fit", logstrm1.str());
    logstrm1.str("");
#endif

    int n = p_mlmtpr->CoeffCount();
    double *x = p_mlmtpr->Coeff();

    int m = (int)training_set.size(); // train set size on the current core
    int K = m;                        // train set size over all cores

#ifdef MLIP_MPI
    MPI_Allreduce(&m, &K, 1, MPI_INT, MPI_SUM, mpi.comm);
#endif

    bfgs.Set_x(x, n);
    bfgs_g.resize(n);

    for (int i = 0; i < n; i++) // setting the initial hessian equal to identity matrix
        for (int j = 0; j < n; j++)
            if (i == j)
                bfgs.inv_hess(i, j) = 1;
            else
                bfgs.inv_hess(i, j) = 0;

    int num_step = 0;

    double linf = 9e99; // loss function value 100 iterations before. Used for stopping
    double loss_prev = 9e99;
    int max_line_search = 999; // maximum number of iterations in linesearch. Finish when exceeded
    int curr_line_search = 0;  // current iteration of linesearch
    bool converge = false;
    double max_shift = 0.1 * random_perturb; // maximal change in coefficients, for simulated annealing only
    double cooling_rate = 0.2;               // how fast does the random displacement goes down, for simulated annealing only
    bool linesearch = false;                 // is current iteration an iteration of linesearch

    std::random_device random_device; // random stuff, for simulated annealing only
    std::default_random_engine eng(random_device());
    std::uniform_real_distribution<double> distr(-1, 1);

    int iter = 0;

    while (!converge)
    {
        // iter++;
        // cout << iter << endl;
        if (!linesearch)
        {
            curr_line_search = 0;
            for (int i = 0; i < n - p_mlmtpr->LinSize(); i++) // perturb the coefficients, for simulated annealing only
                x[i] += distr(eng) * max_shift;

#ifdef MLIP_MPI
            if (mpi.rank == 0)
                bfgs.Set_x(x, n);

            MPI_Bcast(&x[0], n, MPI_DOUBLE, 0, mpi.comm);
#else
            bfgs.Set_x(x, n);
#endif
            if ((num_step == 25) || (num_step == 70) || (num_step == 100) || (num_step == 150) || (num_step == 250) || (num_step == 400))
            {
                LinOptimize(training_set); // finding the linear coefficients
#ifdef MLIP_MPI
                if (mpi.rank == 0)
#endif
                    bfgs.Set_x(x, n);

                if (reg_init) // if the regularization changes (reg_init flag) the hessian is flushed
                {
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                            if (i == j)
                                bfgs.inv_hess(i, j) = 1;
                            else
                                bfgs.inv_hess(i, j) = 0;

                    reg_init = false;
                }
            }
#ifdef MLIP_MPI
            if (mpi.rank == 0)
#endif
            {
                if (mlip_fitted_fnm != "")
                    p_mlmtpr->Save(mlip_fitted_fnm); // save the current state of the potential
            }
        }

        for (int i = 0; i < n; i++)
            x[i] = bfgs.x(i);

#ifdef MLIP_MPI
        MPI_Bcast(&x[0], n, MPI_DOUBLE, 0, mpi.comm);
#endif
        CalcObjectiveFunctionGrad(training_set);
        // divide the loss function and its gradients on the training set size
        loss_ /= K;
        for (int i = 0; i < n; i++)
            loss_grad_[i] /= K;

#ifdef MLIP_MPI
        MPI_Barrier(mpi.comm);
        MPI_Reduce(&loss_, &bfgs_f, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
        MPI_Reduce(&loss_grad_[0], &bfgs_g[0], n, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
#else
        bfgs_f = loss_;
        memcpy(&bfgs_g[0], &loss_grad_[0], p_mlmtpr->CoeffCount() * sizeof(double));
#endif

#ifdef MLIP_MPI
        if (mpi.rank == 0)
#endif
        {
            if (!converge)
            {

                if (!bfgs.Iterate(bfgs_f, bfgs_g)) // if loss in increasing, BFGS is stopped
                    converge = true;

                while (abs(bfgs.x(0) - x[0]) > 0.5) // prevents too large steps
                    bfgs.ReduceStep(0.25);
            }
        }

        linesearch = bfgs.is_in_linesearch();
        if (linesearch)
            curr_line_search++;
        else
        {
#ifdef MLIP_MPI
            if (mpi.rank == 0)
#endif
            {
                if (loss_prev < bfgs_f) //"cooling"=decreasing the max. perturbation of coefficients, for simulated annealing only
                {
                    max_shift *= (1 - cooling_rate);
                    logstrm1 << "*" << endl;
                    MLP_LOG("fit", logstrm1.str());
                    logstrm1.str("");
                }

                if (abs(loss_prev - bfgs_f) < 1e-16) // hardcoded stopping criterion. Should NOT be removed
                {
                    converge = true;
                    logstrm1 << "BFGS ended due to small decr. for 1 iteration" << endl;
                    MLP_LOG("fit", logstrm1.str());
                    logstrm1.str("");
                }

                loss_prev = bfgs_f;
                logstrm1 << "BFGS iter " << num_step << ": f=" << bfgs_f << endl;
                MLP_LOG("fit", logstrm1.str());
                logstrm1.str("");

                // logstrm1 << num_step << " " << bfgs_f << endl;
                num_step++;

                if (num_step % 50 == 0 && num_step > 100)
                {
                    if ((linf - bfgs_f) / bfgs_f < bfgs_conv_tol) // checking the relative decrease of the loss function in 100 iterations
                    {
                        converge = true;
                        logstrm1 << "BFGS ended due to small decr. in 50 iterations" << endl;
                        MLP_LOG("fit", logstrm1.str());
                        logstrm1.str("");
                    }
                    linf = bfgs_f;
                }

                if (num_step >= max_iter)
                {
                    converge = true;
                    logstrm1 << "step limit reached" << endl;
                    MLP_LOG("fit", logstrm1.str());
                    logstrm1.str("");
                }
            }
        }
        if (curr_line_search > max_line_search)
        {
#ifdef MLIP_MPI
            if (mpi.rank == 0)
#endif
            {
                logstrm1 << "BFGS has exceeded the linesearch limit. Cannot converge further" << endl;
                MLP_LOG("fit", logstrm1.str());
                logstrm1.str("");
            }
            converge = true;
        }

#ifdef MLIP_MPI
        MPI_Barrier(mpi.comm);
        MPI_Bcast(&converge, 1, MPI_C_BOOL, 0, mpi.comm);
        MPI_Bcast(&linesearch, 1, MPI_C_BOOL, 0, mpi.comm);
        MPI_Bcast(&num_step, 1, MPI_INT, 0, mpi.comm);
#endif
    }
    // the potential was initialized
    p_mlmtpr->inited = true;
    // this updates the regularization vector for the next training or rescaling
    reg_init = true;
#ifdef MLIP_MPI
    if (mpi.rank == 0)
#endif
    {
        logstrm1 << "MTPR training ended" << endl;
        MLP_LOG("fit", logstrm1.str());
        logstrm1.str("");
    }
}

void MTPR_trainer::Rescale(std::vector<Configuration> &training_set)
{
    int n = p_mlmtpr->LinSize();
    std::vector<int> deg = p_mlmtpr->GetScalingDegrees();

#ifdef MLIP_MPI
    if (mpi.rank == 0)
    {
        logstrm1 << "Rescaling...\n";
        MLP_LOG("fit", logstrm1.str());
        logstrm1.str("");
    }
#else
    logstrm1 << "Rescaling...\n";
    MLP_LOG("fit", logstrm1.str());
    logstrm1.str("");
#endif

    // Populates lin_matrix (Cholesky U) and lin_matrix_reg (pre-factor A) on rank 0 only
    LinOptimize(training_set);

    double best_r = 1.0;

#ifdef MLIP_MPI
    if (mpi.rank == 0)
#endif
    {
        auto estimate_cond = [&](double r) -> double
        {
            std::vector<double> U_scaled(n * n, 0.0);
            std::vector<double> D(n);
            for (int i = 0; i < n; i++)
                D[i] = std::pow(r, deg[i]);

            double anorm = 0.0;
            for (int c = 0; c < n; c++)
            {
                double col_sum = 0.0;
                for (int row = 0; row < n; row++)
                    col_sum += std::abs(lin_matrix_reg[row * n + c]) * D[row] * D[c];
                if (col_sum > anorm)
                    anorm = col_sum;
            }

            for (int c = 0; c < n; c++)
                for (int row = 0; row <= c; row++)
                    U_scaled[row + c * n] = lin_matrix[row + c * n] * D[c];

            char uplo = 'U';
            double rcond = 0.0;
            std::vector<double> work(3 * n);
            std::vector<int> iwork(n);
            int info = 0;
            dpocon_(&uplo, &n, U_scaled.data(), &n, &anorm, &rcond, work.data(), iwork.data(), &info);

            return (rcond == 0.0) ? 1e99 : 1.0 / rcond;
        };

        std::map<long long, double> cache;
        auto get_cond = [&](double r) -> double
        {
            long long key = std::round(std::log(r) * 1e8);
            auto it = cache.find(key);
            if (it != cache.end())
                return it->second;

            double c = estimate_cond(r);
            cache[key] = c;

            logstrm1 << "   scaling = " << (p_mlmtpr->scaling * r)
                     << ", condition number = " << c << "\n";
            MLP_LOG("fit", logstrm1.str());
            logstrm1.str("");

            return c;
        };

        double best_exp = 0.0;
        double min_cond = get_cond(1.0); // Start at current scaling multiplier (r=1)

        // Coarse grid search in log10 space from roughly 1e-5 to 1e5
        double coarse_step = 0.1;
        for (double exp = -5.0; exp <= 5.0 + 1e-9; exp += coarse_step)
        {
            double r = std::pow(10.0, exp);
            double c = get_cond(r);
            if (c < min_cond)
            {
                min_cond = c;
                best_exp = exp;
            }
        }

        // Golden Section Search around the best coarse exponent bracket
        // Bounding it tight ensures GSS operates only on the valid, unimodal deep valley.
        double a = best_exp - coarse_step;
        double b = best_exp + coarse_step;

        const double invphi = (std::sqrt(5.0) - 1.0) / 2.0;
        const double invphi2 = (3.0 - std::sqrt(5.0)) / 2.0;

        double x1 = a + invphi2 * (b - a);
        double x2 = a + invphi * (b - a);

        double f1 = get_cond(std::pow(10.0, x1));
        double f2 = get_cond(std::pow(10.0, x2));

        while ((b - a) > 1e-4) // Exponent tolerance (translates to roughly a 0.02% error in r itself)
        {
            if (f1 < f2)
            {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = a + invphi2 * (b - a);
                f1 = get_cond(std::pow(10.0, x1));
            }
            else
            {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = a + invphi * (b - a);
                f2 = get_cond(std::pow(10.0, x2));
            }
        }

        double final_exp = 0.5 * (a + b);
        best_r = std::pow(10.0, final_exp);

        // Final evaluation to cache and log the best value safely
        min_cond = get_cond(best_r);

        p_mlmtpr->scaling *= best_r;

        logstrm1 << "Rescaling to " << p_mlmtpr->scaling << "... done\n";
        MLP_LOG("fit", logstrm1.str());
        logstrm1.str("");
    }

#ifdef MLIP_MPI
    // Broadcast the updated scaling to all ranks before the final solve
    MPI_Bcast(&p_mlmtpr->scaling, 1, MPI_DOUBLE, 0, mpi.comm);
    MPI_Bcast(&best_r, 1, MPI_DOUBLE, 0, mpi.comm);
#endif

    if (best_r != 1.0)
        LinOptimize(training_set);
}

void MTPR_trainer::Train(std::vector<Configuration> &training_set)
{
#ifdef MLIP_MPI
    MPI_Barrier(mpi.comm);
#endif

    ClearSLAE();
    bfgs_g.resize(p_mlmtpr->CoeffCount()); // resize the gradient of BFGS

    Configuration cfg;

    // extend the coefficients of the potential to handle the new species from the database
    AddSpecies(training_set);

    // random initialization of radial coefficients
    if (init_random && !p_mlmtpr->inited)
    {
#ifdef MLIP_MPI
        if (mpi.rank == 0)
#endif
        {
            std::random_device rand_device;
            std::default_random_engine generator(rand_device());
            std::uniform_real_distribution<> uniform(-1.0, 1.0);

            logstrm1 << "Random initialization of radial coefficients" << std::endl;
            MLP_LOG("fit", logstrm1.str());
            logstrm1.str("");
            int r_size = p_mlmtpr->p_RadialBasis->size;
            int m_size = p_mlmtpr->mbasis_size;
            for (int k = 0; k < p_mlmtpr->species_count * p_mlmtpr->species_count; k++)
            {
                for (int i = 0; i < p_mlmtpr->radial_func_count; i++)
                {
                    for (int j = 0; j < r_size; j++)
                    {
                        for (int alpha = 0; alpha < m_size; alpha++)
                        {
                            for (int beta = 0; beta < m_size; beta++)
                            {
                                p_mlmtpr->regression_coeffs[k * p_mlmtpr->radial_func_count * r_size * m_size * m_size +
                                                            (i * r_size + j) * m_size * m_size + alpha * m_size + beta] = 5e-7 * uniform(generator);
                            }
                        }
                    }

                    for (int alpha = 0; alpha < m_size; alpha++)
                    {
                        for (int beta = 0; beta < m_size; beta++)
                        {
                            p_mlmtpr->regression_coeffs[k * p_mlmtpr->radial_func_count * r_size * m_size * m_size +
                                                        (i * r_size + min(i, r_size - 1)) * m_size * m_size + alpha * m_size + beta] = 1e-7 * (1 + uniform(generator));
                        }
                    }
                }
            }
        }

#ifdef MLIP_MPI
        MPI_Bcast(&p_mlmtpr->Coeff()[0], p_mlmtpr->CoeffCount(), MPI_DOUBLE, 0, mpi.comm);
#endif
    }

    if (!p_mlmtpr->inited && maxits > 0 && !skip_preinit)
    { // pre-training for initial rescaling
        LinOptimize(training_set);
        Rescale(training_set);

#ifdef MLIP_MPI
        if (mpi.rank == 0)
#endif
            logstrm1 << "Pre-training started" << std::endl;

        MLP_LOG("fit", logstrm1.str());
        logstrm1.str("");

        NonLinOptimize(training_set, 75); // 75 iterations for pre-training

        Rescale(training_set);

#ifdef MLIP_MPI
        if (mpi.rank == 0)
#endif
            logstrm1 << "Pre-training ended" << std::endl;
        MLP_LOG("fit", logstrm1.str());
        logstrm1.str("");
    }

    if (maxits > 0)
    {
#ifdef MLIP_MPI
        if (mpi.rank == 0)
#endif
        {
            logstrm1 << "Iteration limit is " << maxits << std::endl;
            MLP_LOG("fit", logstrm1.str());
            logstrm1.str("");
            logstrm1 << "Convergence tolerance is " << bfgs_conv_tol << std::endl;
            MLP_LOG("fit", logstrm1.str());
            logstrm1.str("");
            if ((wgt_eqtn_energy != 0) || (wgt_eqtn_forces != 0) || (wgt_eqtn_stress != 0))
            {
                logstrm1 << "Energy weight: " << wgt_eqtn_energy << std::endl;
                MLP_LOG("fit", logstrm1.str());
                logstrm1.str("");
                logstrm1 << "Force weight: " << wgt_eqtn_forces << std::endl;
                MLP_LOG("fit", logstrm1.str());
                logstrm1.str("");
                logstrm1 << "Stress weight: " << wgt_eqtn_stress << std::endl;
                MLP_LOG("fit", logstrm1.str());
                logstrm1.str("");
            }
        }

        NonLinOptimize(training_set, maxits);
        LinOptimize(training_set);
        Rescale(training_set);

        if (mpi.rank == 0)
            if (mlip_fitted_fnm != "")
                p_mlmtpr->Save(mlip_fitted_fnm);
    }

    MPI_Barrier(mpi.comm);
}

void MTPR_trainer::ExtractProblem(std::vector<Configuration> &training_set,
                                  const std::string &matrix_file,
                                  const std::string &vector_file)
{
    ClearSLAE();
    bfgs_g.resize(p_mlmtpr->CoeffCount());

    for (auto &cfg : training_set)
        AddToSLAE(cfg);

    int m = static_cast<int>(training_set.size());
    int K = m;

#ifdef MLIP_MPI
    MPI_Allreduce(&m, &K, 1, MPI_INT, MPI_SUM, mpi.comm);
    double scalar = 0.0;
    int n = p_mlmtpr->alpha_count - 1 + p_mlmtpr->species_count;

    if (mpi.rank == 0)
    {
        lin_matrix_mpi.resize(n * n);
        lin_vector_mpi.resize(n);
        std::fill(lin_matrix_mpi.begin(), lin_matrix_mpi.end(), 0);
        std::fill(lin_vector_mpi.begin(), lin_vector_mpi.end(), 0);
    }

    MPI_Reduce(lin_matrix.data(), lin_matrix_mpi.data(), n * n, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(lin_vector.data(), lin_vector_mpi.data(), n, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&lin_scalar, &scalar, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);

    if (mpi.rank == 0)
    {
        std::memcpy(lin_matrix.data(), lin_matrix_mpi.data(), n * n * sizeof(double));
        std::memcpy(lin_vector.data(), lin_vector_mpi.data(), n * sizeof(double));
        lin_scalar = scalar;

        SymmetrizeSLAE();

        int TS_size = K;

        if (!reg_init)
            for (int i = 0; i < n; i++)
                if ((p_mlmtpr->reg_vector[i] < 1e-2 * reg_param * std::max(1.0, lin_matrix[i * n + i]) / TS_size) ||
                    (p_mlmtpr->reg_vector[i] > 1e2 * reg_param * std::max(1.0, lin_matrix[i * n + i]) / TS_size))
                {
                    reg_init = true;
                    break;
                }

        if (reg_init)
            for (int i = 0; i < n; i++)
                p_mlmtpr->reg_vector[i] = reg_param * std::max(1.0, lin_matrix[i * n + i]) / TS_size;

        for (int i = 0; i < n; i++)
            lin_matrix[i * n + i] += p_mlmtpr->reg_vector[i] * TS_size;

        // --- Write matrix and vector ---
        std::ofstream matrix_stream(matrix_file, std::ios::binary);
        if (!matrix_stream)
        {
            std::cerr << "ERROR: Failed to open matrix file for writing: " << matrix_file << std::endl;
            return;
        }
        matrix_stream.write(reinterpret_cast<const char *>(lin_matrix.data()),
                            lin_matrix.size() * sizeof(double));
        matrix_stream.close();
        std::cout << "Successfully wrote " << lin_matrix.size() * sizeof(double)
                  << " bytes to matrix file: " << matrix_file << std::endl;

        std::ofstream vector_stream(vector_file, std::ios::binary);
        if (!vector_stream)
        {
            std::cerr << "ERROR: Failed to open vector file for writing: " << vector_file << std::endl;
            return;
        }
        vector_stream.write(reinterpret_cast<const char *>(lin_vector.data()),
                            lin_vector.size() * sizeof(double));
        vector_stream.close();
        std::cout << "Successfully wrote " << lin_vector.size() * sizeof(double)
                  << " bytes to vector file: " << vector_file << std::endl;

        std::cout << "Sum of Squared Ground Truths (yTWy): "
                  << std::fixed << std::setprecision(15) << lin_scalar << std::endl;
    }

#else
    SymmetrizeSLAE();

    int n = p_mlmtpr->alpha_count - 1 + p_mlmtpr->species_count;
    int TS_size = K;

    if (!reg_init)
        for (int i = 0; i < n; i++)
            if ((p_mlmtpr->reg_vector[i] < 1e-2 * reg_param * std::max(1.0, lin_matrix[i * n + i]) / TS_size) ||
                (p_mlmtpr->reg_vector[i] > 1e2 * reg_param * std::max(1.0, lin_matrix[i * n + i]) / TS_size))
            {
                reg_init = true;
                break;
            }

    if (reg_init)
        for (int i = 0; i < n; i++)
            p_mlmtpr->reg_vector[i] = reg_param * std::max(1.0, lin_matrix[i * n + i]) / TS_size;

    for (int i = 0; i < n; i++)
        lin_matrix[i * n + i] += p_mlmtpr->reg_vector[i] * TS_size;

    // --- Write matrix and vector (merged WriteProblem) ---
    std::ofstream matrix_stream(matrix_file, std::ios::binary);
    if (!matrix_stream)
    {
        std::cerr << "ERROR: Failed to open matrix file for writing: " << matrix_file << std::endl;
        return;
    }
    matrix_stream.write(reinterpret_cast<const char *>(lin_matrix.data()),
                        lin_matrix.size() * sizeof(double));
    matrix_stream.close();
    std::cout << "Successfully wrote " << lin_matrix.size() * sizeof(double)
              << " bytes to matrix file: " << matrix_file << std::endl;

    std::ofstream vector_stream(vector_file, std::ios::binary);
    if (!vector_stream)
    {
        std::cerr << "ERROR: Failed to open vector file for writing: " << vector_file << std::endl;
        return;
    }
    vector_stream.write(reinterpret_cast<const char *>(lin_vector.data()),
                        lin_vector.size() * sizeof(double));
    vector_stream.close();
    std::cout << "Successfully wrote " << lin_vector.size() * sizeof(double)
              << " bytes to vector file: " << vector_file << std::endl;

    std::cout << "Sum of Squared Ground Truths (yTWy): "
              << std::fixed << std::setprecision(15) << lin_scalar << std::endl;
#endif
}

double MTPR_trainer::FindLoss(std::vector<Configuration> &training_set)
{
    bfgs_f = 0;
    ObjectiveFunction(training_set, false);
#ifdef MLIP_MPI
    MPI_Barrier(mpi.comm);
    MPI_Reduce(&loss_, &bfgs_f, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
#else
    bfgs_f = loss_;
#endif
    return bfgs_f;
}