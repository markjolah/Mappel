/** @file estimator_helpers.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief Estimator helper subroutines
 */

#ifndef MAPPEL_ESTIMATOR_HELPERS_H
#define MAPPEL_ESTIMATOR_HELPERS_H
namespace mappel {
namespace estimator {

/** Common subroutines shared between estimators.
 *
 * These methods are model agnostic.
 *
 */
namespace subroutine {

/** Return a new step that is guaranteed to keep theta in the interior of the feasible region.
 * Uses a relative backtracking technique to step away from the boundary into the interior.
 * @param step proposed step
 * @param theta current theta
 * @param lbound lower bounds
 * @param ubound upper bounds
 * @return bounded step
 */
VecT bound_step(const VecT &step, const VecT &theta, const VecT &lbound, const VecT &ubound);

/** Bounds scaling vector for affine scaling of bounds constrained optimization problems.
 * This v is from Coleman&Li (1996).  It represents a scaling factor for bound constrained
 * problems.  For unconstrained problems v = sgn(grad);
 * @param[in] theta current theta
 * @param[in] g gradient
 * @param[in] lbound lower bound
 * @param[in] ubound upper bound
 * @param[out] v Scaling vector
 * @param[out] Jv Jacobian
 */
void compute_bound_scaling_vec(const VecT &theta, const VecT &g, const VecT &lbound, const VecT &ubound,
                               VecT &v, VecT &Jv);

/** Compute an affine scaling diagonal matrix to scale problem away from boundaries.
 * This works for either minimization or maximization.  sign(grad2) is not important
 * @param oldDscale Last D scaling matrix
 * @param grad2 Diagonal of hessian matrix
 * @return Diagonal scaling matrix as a vector.
 */
VecT compute_D_scale(const VecT &oldDscale, const VecT &grad2);
void compute_scaled_problem(const MatT &H, const VecT &g, const VecT &Dinv, const VecT &Jv, MatT& Hhat, VecT &ghat);



/** Find initial step lengths in profile bounds estimation VM algorithm
 *
 */
VecT solve_profile_initial_step(const MatT &obsI, IdxT fixed_idx, double llh_delta);

double compute_initial_trust_radius(const VecT &ghat);

VecT compute_cauchy_point(const VecT &g, const MatT &H, double delta);
/**
 * @brief Quadratic model value at given step
 * Compute a quadratic model
 */
double compute_quadratic_model_value(const VecT &s, const VecT &g, const MatT &H);

/** @brief Exact solver the TR sub-problem even for non-positive definite H
 *
 * This method is a hybrid technique mixing ideas from
 * Geyer (2013) and the "trust" R-package
 * Nocetal and Wright (2000)
 * More and Sorensen (1981)
 */
VecT solve_TR_subproblem(const VecT &g, const MatT &H, double delta);
VecT solve_restricted_step_length_newton(const VecT &g, const MatT &H, double delta, double lambda_lb, double lambda_ub);


}/* namespace mappel::estimator::subroutine */
}/* namespace mappel::estimator */
} /* namespace mappel */

#endif /* MAPPEL_ESTIMATOR_HELPERS_H */
