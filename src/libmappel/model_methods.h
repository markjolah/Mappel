
/** @file model_methoods.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief Namespace definitions for the model:: namespace which contains the major methods for computing with PointEmitterModels
 */

#ifndef _MAPPEL_MODEL_METHODS
#define _MAPPEL_MODEL_METHODS

#include "PointEmitterModel.h"

namespace mappel {

/** @brief Templated functions for operating on a PointEmitterModel
 * 
 * Most methods are overloaded to take a ParamT or a StencilT.  The precomputed stencil for a theta value contains
 * the common computational values needed by all methods that compute the likelihood function or its derivatives.  
 * Note that methods in model::prior:: namespace do not take a stencil (or data) a they are independent of the 
 * data and the likelihood function.
 * 
 * Methods with xxx_comonents return a sequence of values representing the results from each pixel in turn.  The
 * sum of these components is the overall model value.  (e.g. sum(llh_components(...))==llh(...) ).  These methods
 * are usefully for detailed inspection of the contributions of each pixel or prior component to the overall result.
 * 
 */
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
    
    /* model::rllh */
    template<class Model>
    double rllh(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

    template<class Model>
    double rllh(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &stencil);

    /* model::grad */
    template<class Model>
    ParamT<Model> grad(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

    template<class Model>
    ParamT<Model> grad(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil);

    /* model::grad2 */
    template<class Model>
    void grad2(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta, ParamT<Model> &grad, ParamT<Model> &grad2);

    template<class Model>
    void grad2(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil, ParamT<Model> &grad, ParamT<Model> &grad2);

    /* model::hessian */
    template<class Model>
    void hessian(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta, ParamT<Model> &grad, ParamT<Model> &grad2);

    template<class Model>
    void hessian(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil, ParamT<Model> &grad, ParamT<Model> &grad2);
    
    /* model::objective*/
    template<class Model>
    void objective(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta, double &rllh, VecT &grad, MatT &hess);

    template<class Model>
    void objective(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta, double &rllh, VecT &grad, MatT &hess);

    /* model::score*/
    template<class Model>
    ParamT<Model> score(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

    template<class Model>
    ParamT<Model> score(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta);

    /* model::observed_information  --- Observed Fisher Information (at posterior mode) */
    template<class Model>
    MatT observed_information(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_mode);

    template<class Model>
    MatT observed_information(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta_mode);

    /* model::expected_information --- Expected Fisher Information */
    template<class Model>
    MatT expected_information(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);
    
    template<class Model>
    MatT expected_information(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta);
    
    /* model::crlb --- Cramer-rao lower bound */
    template<class Model>
    ParamT<Model> crlb(const Model &model, const ParamT<Model> &theta);
    
    template<class Model>
    ParamT<Model> crlb(const Model &model, const ParamT<Model> &theta);

    /* model::estimate */
    
    estimate_map()
    
    

    
    /* model::xxx_components - per-pixel and per-prior-component contributions */
    template<class Model>
    VecT llh_components(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

    template<class Model>
    VecT llh_components(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &stencil);

    template<class Model>
    VecT rllh_components(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

    template<class Model>
    VecT rllh_components(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &stencil);
    
    template<class Model>
    MatT grad_components(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

    template<class Model>
    MatT grad_components(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil);
    
    template<class Model>
    CubeT hess_components(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

    template<class Model>
    CubeT hess_components(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil);
   
    template<class Model>
    void objective_components(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta, VecT &llh_components, MatT &grad_components, CubeT &hess_components);

    template<class Model>
    void objective_components(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil, VecT &llh_components, MatT &grad_components, CubeT &hess_components);
    
    

    
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

    
} /* namespace mappel */

#endif /* _MAPPEL_MODEL_METHODS */
