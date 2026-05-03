/*   MLIP is a software for Machine Learning Interatomic Potentials
 *   MLIP is released under the "New BSD License", see the LICENSE file.
 *   Contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */

#include <string>
#include <iostream>
#include "bfgs.h"

//! Input: x was set either before iterations or changed during iterations;
//! f, g are set for the current x before the call.
//! The function sets new x.
//! Internally, it changes the protected members
//! alpha and if outside linesearch then it changes p.
bool BFGS::Iterate(double f, const Array1D &g)
{
    p_dot_g = ScalarProd(&g[0], &p[0]);

    bool loss_decrease = true;

    if (f > f_start + wolfe_c1 * linesearch.x() * p_dot_g_start ||
        fabs(p_dot_g) > wolfe_c2 * fabs(p_dot_g_start))
    {
        is_in_linesearch_ = true;

        loss_decrease = linesearch.Iterate(f, p_dot_g);

        // x = x_start + alpha*p
        const int one = 1;
        dcopy_(&size, &x_start[0], &one, &x_[0], &one);
        const double alpha = linesearch.x();
        daxpy_(&size, &alpha, &p[0], &one, &x_[0], &one);
    }
    else
    {
        is_in_linesearch_ = !is_in_linesearch_;

        if (!no_Hessian_update)
            UpdateInvHess(g);

        // p = -inv_hess * g
        // inv_hess is row-major; row-major upper == col-major lower, so use 'L'.
        const char uplo = 'L';
        const int one = 1;
        const double neg_one = -1.0, zero = 0.0;
        dsymv_(&uplo, &size, &neg_one, inv_hess.data(), &size,
               &g[0], &one, &zero, &p[0], &one);

        p_dot_g = ScalarProd(&g[0], &p[0]);
        linesearch.Reset();
        SetStart(f, g);

        if (p_dot_g > 0)
        {
            std::ofstream ofs("bfgs.error");
            ofs.precision(16);
            ofs << size << '\n';
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                    ofs << inv_hess(i, j) << " ";
                ofs << std::endl;
            }
            ofs.close();
            Warning("BFGS: stepping in accend direction detected.");
            loss_decrease = false;
        }

        loss_decrease = linesearch.Iterate(f, p_dot_g);

        // x = x_start + alpha*p
        dcopy_(&size, &x_start[0], &one, &x_[0], &one);
        const double alpha = linesearch.x();
        daxpy_(&size, &alpha, &p[0], &one, &x_[0], &one);
    }

    return loss_decrease;
}

//! The step x is too far, make a smaller step (as a part of linesearch).
//! Sets new x and alpha
const Array1D &BFGS::ReduceStep(double _coeff)
{
    linesearch.ReduceStep(_coeff);

    const int one = 1;
    dcopy_(&size, &x_start[0], &one, &x_[0], &one);
    const double alpha = linesearch.x();
    daxpy_(&size, &alpha, &p[0], &one, &x_[0], &one);

    return x_;
}

void BFGS::UpdateInvHess(const Array1D &g)
{
    // delta_grad = g - g_start
    const int one = 1;
    dcopy_(&size, &g[0], &one, &delta_grad[0], &one);
    const double neg_one = -1.0;
    daxpy_(&size, &neg_one, &g_start[0], &one, &delta_grad[0], &one);

    // s_ = alpha * p
    const double alpha = linesearch.x();
    dcopy_(&size, &p[0], &one, &s_[0], &one);
    dscal_(&size, &alpha, &s_[0], &one);

    // py = s_^T * delta_grad
    const double py = ddot_(&size, &s_[0], &one, &delta_grad[0], &one);

    if (py == 0)
        return;

    // yC = inv_hess * delta_grad  (row-major upper == col-major lower -> 'L')
    const char uplo = 'L';
    const double one_d = 1.0, zero = 0.0;
    dsymv_(&uplo, &size, &one_d, inv_hess.data(), &size,
           &delta_grad[0], &one, &zero, &yC[0], &one);

    const double yCy = ddot_(&size, &yC[0], &one, &delta_grad[0], &one);

    // inv_hess += (py+yCy)/py^2 * s_*s_^T
    const double foo = (py + yCy) / (py * py);
    dsyr_(&uplo, &size, &foo, &s_[0], &one, inv_hess.data(), &size);

    // inv_hess -= (1/py) * (s_*yC^T + yC*s_^T)
    const double neg_inv_py = -1.0 / py;
    dsyr2_(&uplo, &size, &neg_inv_py, &s_[0], &one, &yC[0], &one,
           inv_hess.data(), &size);

    // Mirror the updated triangle so inv_hess(i,j) remains valid for all callers.
    for (int i = 0; i < size; i++)
        for (int j = i + 1; j < size; j++)
            inv_hess(i, j) = inv_hess(j, i);
}

void BFGS::Set_x(const double *x, int _size)
{
    Resize(_size);
    const int one = 1;
    dcopy_(&_size, x, &one, &x_[0], &one);
    f_start = HUGE_DOUBLE;
    p_dot_g_start = HUGE_DOUBLE;
    linesearch.Reset();
    is_in_linesearch_ = false;
}

void BFGS::Restart()
{
    inv_hess.set(0.0);
    for (int i = 0; i < size; i++)
        inv_hess(i, i) = 1.0;
    f_start = HUGE_DOUBLE;
    p_dot_g_start = HUGE_DOUBLE;
    linesearch.Reset();
    is_in_linesearch_ = false;
}