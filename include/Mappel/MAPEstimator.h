
/** @file MAPEstimator.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief Class declaration and inline and templated functions for MAPEstimator.
 */

#ifndef _MAPPEL_MAPESTIMATOR_H
#define _MAPPEL_MAPESTIMATOR_H

#include "Mappel/PointEmitterModel.h"
#include "Mappel/MLEstimator.h"
namespace mappel {

/** @brief A Mixin class to configure a for MLE estimation (null prior).
 * 
 * Inheriting from this class modifies the objective function undergoing optimization to use a Null prior, 
 * by simply ignoreing the effect of the prior on the objective.  This which effectively turns the objective
 * function into a pure likelihood function, and the estimator becomes an MLE estimator.
 * 
 */
class MAPEstimator  : public virtual PointEmitterModel {
protected:
    MAPEstimator() = default;
};

namespace methods{
    /* Implements the MAP Estimation by defining the objective computations as including the prior. */
    namespace objective {            
        template<class Model>
        ReturnIfSubclassT<double,Model,MAPEstimator>
        llh(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
        {
//             std::cout<<"Stencil: "<<s;
//             std::cout<<"Im: "<<data_im.t();
//             std::cout<<"Prior: "<<model.get_prior();
//             std::cout<<"Prior Theta LLH: "<<model.get_prior().llh_components(s.theta)<<std::endl;
//             std::cout<<"Prior Theta RLLH: "<<model.get_prior().rllh_components(s.theta)<<std::endl;
            return likelihood::llh(model, data_im, s) + model.get_prior().llh(s.theta);
        }

        template<class Model>
        ReturnIfSubclassT<double,Model,MAPEstimator>
        rllh(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
        {
            return likelihood::rllh(model, data_im, s) + model.get_prior().rllh(s.theta);
        }

        template<class Model>
        ReturnIfSubclassT<ParamT<Model>,Model,MAPEstimator>
        grad(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
        {
            auto grad = likelihood::grad(model, data_im, s);
            model.get_prior().grad_accumulate(s.theta,grad);
            return grad;
        }

        template<class Model>
        ReturnIfSubclassT<void,Model,MAPEstimator>
        grad2(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s, ParamT<Model> &grad, ParamT<Model> &grad2)
        {
            likelihood::grad2(model, data_im, s, grad, grad2);
            model.get_prior().grad_grad2_accumulate(s.theta, grad, grad2);
        }

        template<class Model>
        ReturnIfSubclassT<void,Model,MAPEstimator>
        hessian(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s, ParamT<Model> &grad, MatT &hess)
        {
            likelihood::hessian(model, data_im, s, grad, hess);
            model.get_prior().grad_hess_accumulate(s.theta, grad, hess);
        }

        /* objective per-pixel additive components to the overall model. */
        inline namespace debug {
            template<class Model>
            ReturnIfSubclassT<VecT,Model,MAPEstimator>
            llh_components(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
            {
                return arma::join_vert(likelihood::debug::llh_components(model, data_im, s),  model.get_prior().llh_components(s.theta));
            }

            template<class Model>
            ReturnIfSubclassT<VecT,Model,MAPEstimator>
            rllh_components(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
            {
                return arma::join_vert(likelihood::debug::rllh_components(model, data_im, s),  model.get_prior().rllh_components(s.theta));
            }

            template<class Model>
            ReturnIfSubclassT<MatT,Model,MAPEstimator>
            grad_components(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
            {
                return arma::join_horiz(likelihood::debug::grad_components(model, data_im, s),  model.get_prior().grad(s.theta));
            }

            template<class Model>
            ReturnIfSubclassT<CubeT,Model,MAPEstimator>
            hessian_components(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
            {
                return arma::join_slices(likelihood::debug::hessian_components(model, data_im, s),  model.get_prior().hess(s.theta));
            }
        } /* namespace mappel::methods::objective::debug */
    } /* namespace mappel::methods::objective */
} /* namespace mappel::methods */

} /* namespace mappel */

#endif /* _MAPPEL_MAPESTIMATOR_H */
