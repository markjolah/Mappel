
/** @file MLEstimator.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief Class declaration and inline and templated functions for MLEstimator.
 */

#ifndef _MAPPEL_MLESTIMATOR_H
#define _MAPPEL_MLESTIMATOR_H

#include "Mappel/PointEmitterModel.h"
#include "Mappel/MAPEstimator.h"
namespace mappel {

/** @brief A Mixin class to configure a for MLE estimation (null prior).
 * 
 * Inheriting from this class modifies the objective function undergoing optimization to use a Null prior, 
 * by simply ignoring the effect of the prior on the objective.  This which effectively turns the objective
 * function into a pure likelihood function, and the estimator becomes an MLE estimator.
 * 
 */
class MLEstimator : public virtual PointEmitterModel {
protected:
    MLEstimator() = default;
};

namespace methods {
    /* Implements the MLE Estimation by defining the objective computations as identical to the likelihood */
    namespace objective {            

        template<class Model>
        ReturnIfSubclassT<double,Model,MLEstimator>
        llh(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
        {
            return likelihood::llh(model, data_im, s);
        }

        template<class Model>
        ReturnIfSubclassT<double,Model,MLEstimator>
        rllh(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
        {
            return likelihood::rllh(model, data_im, s);
        }

        template<class Model>
        ReturnIfSubclassT<ParamT<Model>, Model,MLEstimator>
        grad(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
        {
            return likelihood::grad(model, data_im, s);
        }

        template<class Model>
        ReturnIfSubclassT<void,Model,MLEstimator>
        grad2(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s, ParamT<Model> &grad, ParamT<Model> &grad2)
        {
            likelihood::grad2(model, data_im, s, grad, grad2);
        }

        template<class Model>
        ReturnIfSubclassT<void,Model,MLEstimator>
        hessian(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s, ParamT<Model> &grad, MatT &hess)
        {
            likelihood::hessian(model, data_im, s, grad, hess);
        }
       
        /* objective per-pixel additive components to the overall model. */
        inline namespace debug {
            template<class Model>
            ReturnIfSubclassT<VecT,Model,MLEstimator>
            llh_components(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
            {
                return likelihood::llh_components(model, data_im, s);
            }

            template<class Model>
            ReturnIfSubclassT<VecT,Model,MLEstimator>
            rllh_components(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
            {
                return likelihood::rllh_components(model, data_im, s);
            }

            template<class Model>
            ReturnIfSubclassT<MatT,Model,MLEstimator>
            grad_components(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
            {
                return likelihood::grad_components(model, data_im, s);
            }
            
            template<class Model>
            ReturnIfSubclassT<CubeT,Model,MLEstimator>
            hessian_components(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
            {
                return likelihood::hessian_components(model, data_im, s);
            }
        } /* namespace mappel::methods::objective::debug */        
    } /* namespace mappel::methods::objective */
} /* namespace mappel::methods */

} /* namespace mappel */

#endif /* _MAPPEL_MLESTIMATOR_H */
