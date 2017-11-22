
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

namespace model {
    template<class Model> using ParamT = typename Model::ParamT;
    template<class Model> using ModelDataT = typename Model::ModelDataT;
    template<class Model> using StencilT = typename Model::Stencil;
    template<class Model> using StencilVecT = typename Model::StencilVecT;
    
    /* model::llh */
    template<class Model>
    double llh(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

    template<class Model>
    double llh(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &stencil);

    /* model::llh - vectorized */
    template<class Model>
    VecT llh(const Model &model, const ModelDataT<Model> &data, const ParamVecT<Model> &thetas);

    template<class Model>
    VecT llh(const Model &model, const ModelDataT<Model> &data, const StencilVecT<Model> &thetas);
    
    template<class Model>
    VecT llh(const Model &model, const ModelDataStackT<Model> &datas, const ParamT<Model> &theta);

    template<class Model>
    VecT llh(const Model &model, const ModelDataStackT<Model> &datas, const StencilT<Model> &theta);

    template<class Model>
    VecT llh(const Model &model, const ModelDataStackT<Model> &datas, const ParamVecT<Model> &thetas);

    template<class Model>
    VecT llh(const Model &model, const ModelDataStackT<Model> &datas, const StencilVecT<Model> &thetas);

    /* model::llh_components - per-pixel and per prior component contributions */
    template<class Model>
    VecT llh_components(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

    template<class Model>
    VecT llh_components(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta);

    
    
    /* model::rllh */
    template<class Model>
    double rllh(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

    template<class Model>
    double rllh(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil);

    template<class Model>
    VecT rllh(const Model &model, const ModelDataT<Model> &data, const ParamVecT<Model> &thetas);
    
    template<class Model>
    VecT rllh(const Model &model, const ModelDataStackT<Model> &datas, const ParamT<Model> &theta);

    template<class Model>
    VecT rllh(const Model &model, const ModelDataStackT<Model> &datas, const Stencil<Model> &theta);

    template<class Model>
    VecT rllh(const Model &model, const ModelDataStackT<Model> &datas, const ParamVecT<Model> &thetas);

    template<class Model>
    VecT rllh_components(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);
    
   
    /* model::grad */
    template<class Model>
    ParamT<Model> grad(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

    template<class Model>
    ParamT<Model> grad(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil);

    template<class Model>
    MatT grad(const Model &model, const ModelDataT<Model> &data, const ParamVecT<Model> &theta);

    template<class Model>
    MatT grad(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil);

    
    
    template<class Model>
    void grad2(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta, ParamT<Model> &grad, ParamT<Model> &grad2);

    template<class Model>
    void grad2(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil, ParamT<Model> &grad, ParamT<Model> &grad2);

    template<class Model>
    void hessian(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta, ParamT<Model> &grad, ParamT<Model> &grad2);

    template<class Model>
    void hessian(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil, ParamT<Model> &grad, ParamT<Model> &grad2);

    
    namespace likelihood_func {
        template<class Model>
        double llh(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

        template<class Model>
        double llh(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil);

        template<class Model>
        VecT llh(const Model &model, const ModelDataT<Model> &data, const ParamVecT<Model> &thetas);
        
        template<class Model>
        VecT llh(const Model &model, const ModelDataStackT<Model> &datas, const ParamT<Model> &theta);

        template<class Model>
        VecT llh(const Model &model, const ModelDataStackT<Model> &datas, const Stencil<Model> &theta);

        template<class Model>
        VecT llh(const Model &model, const ModelDataStackT<Model> &datas, const ParamVecT<Model> &thetas);

        template<class Model>
        double rllh(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

        template<class Model>
        double rllh(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil);

        template<class Model>
        VecT rllh(const Model &model, const ModelDataT<Model> &data, const ParamVecT<Model> &thetas);
        
        template<class Model>
        VecT rllh(const Model &model, const ModelDataStackT<Model> &datas, const ParamT<Model> &theta);

        template<class Model>
        VecT rllh(const Model &model, const ModelDataStackT<Model> &datas, const Stencil<Model> &theta);

        template<class Model>
        VecT rllh(const Model &model, const ModelDataStackT<Model> &datas, const ParamVecT<Model> &thetas);

        template<class Model>
        VecT rllh_components(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);
    }
    namespace prior {
        
    }
}
    
template<class Model>
class Computations {
    using 
llh();  //vectorize
rllh(); //vectorize
grad(); //vectorize
grad2(); 
hessian();
objective();
score();
observed_information();
fisher_information();
crlb();

estimate();
profile_likelihood();


_llh()
likelihoodFunc_rllh()
likelihoodFunc_grad()
likelihoodFunc_grad2()
likelihoodFunc_hess()
likelihoodFunc_objective()

prior_llh()
prior_rllh()
prior_grad()
prior_grad2()
prior_hess()
prior_objective()

model_estimate()
model_fisher_information()
model_profile_likelihood()
}


likelihoodF()

#endif /* _MAPPEL_MLESTIMATOR_H */
