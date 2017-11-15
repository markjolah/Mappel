
/** @file MLEstimator.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief Class declaration and inline and templated functions for MLEstimator.
 */

#ifndef _MAPPEL_MLESTIMATOR_H
#define _MAPPEL_MLESTIMATOR_H

#include "PointEmitterModel.h"

namespace mappel {

/** @brief A Mixin class to configure a for MLE estimation (null prior).
 * 
 * Inheriting from this class modifies the objective function undergoing optimization to use a Null prior, 
 * by simply ignoreing the effect of the prior on the objective.  This which effectively turns the objective
 * function into a pure likelihood function, and the estimator becomes an MLE estimator.
 * 
 */
class MLEstimator : public virtual PointEmitterModel {
public:
    double prior_log_likelihood(const ParamT &theta) const { return 0; }
    double prior_relative_log_likelihood(const ParamT &theta) const { return 0; }
    void prior_grad_accumulate(const ParamT &theta, ParamT &grad) const { }
    void prior_grad2_accumulate(const ParamT &theta, ParamT &grad2) const { }
    void prior_hess_accumulate(const ParamT &theta, MatT &hess) const { }
    void prior_grad_grad2_accumulate(const ParamT &theta, ParamT &grad, ParamT &grad2) const { }
    void prior_grad_hess_accumulate(const ParamT &theta, ParamT &grad, MatT &hess) const { }
 };

} /* namespace mappel */

#endif /* _MAPPEL_MLESTIMATOR_H */
