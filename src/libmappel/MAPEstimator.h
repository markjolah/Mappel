
/** @file MAPEstimator.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief Class declaration and inline and templated functions for MAPEstimator.
 */

#ifndef _MAPPEL_MLESTIMATOR_H
#define _MAPPEL_MLESTIMATOR_H

#include "PointEmitterModel"

namespace mappel {

/** @brief A Mixin class to configure a for MLE estimation (null prior).
 * 
 * Inheriting from this class modifies the objective function undergoing optimization to use a Null prior, 
 * by simply ignoreing the effect of the prior on the objective.  This which effectively turns the objective
 * function into a pure likelihood function, and the estimator becomes an MLE estimator.
 * 
 */
class MAPEstimator : public virtual PointEmitterModel {
public:
    double prior_log_likelihood(const ParamT &theta) const { return  prior.llh(); }
    double prior_relative_log_likelihood(const ParamT &theta) const { return prior.rllh(); }
    void prior_grad_accumulate(const ParamT &theta, ParamT &grad) const { prior.grad_accumulate(theta,grad);}
    void prior_grad2_accumulate(const ParamT &theta, ParamT &grad2) const { prior.grad2_accumulate(theta,grad2); }
    void prior_hess_accumulate(const ParamT &theta, ParamMatT &hess) const { prior.hess_accumulate(theta,hess); }
    void prior_grad_grad2_accumulate(const ParamT &theta, ParamT &grad, ParamT &grad2) const {  prior.grad_grad2_accumulate(theta,grad,grad2);}
    void prior_grad_hess_accumulate(const ParamT &theta, ParamT &grad, ParamMatT &hess) const {  prior.grad_hess_accumulate(theta,grad,hess);}
};

} /* namespace mappel */

#endif /* _MAPPEL_MLESTIMATOR_H */
