
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
    
    /** Log-likelihood (with constant terms) of overall objective (log_likelihood + log_prior)
     * Includes all constant terms of the log-likelihood.  Not used for estimation as the constant terms are irrelevent.
     * This may be useful in other numerical routines especially in the Bayesian MAP context as the exponential of the llh 
     * should integrate to unity with constant terms accounted for.
     * 
     * NOTE: the model log-likelihood is the same as the ordinary likelihood for the MLE models, but includes the prior 
     * log-likelihood for MAP models.  
     * 
     * @param[in] model Model object defining the parameters and per
     * @param[in] data The data to evaluate for
     * @param[in] stencil The stencil computed at the parameter of interest, theta 
     *//
    template<class Model>
    double llh(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &stencil);
    
    /** Relative Log-likelihood (without constant terms)  of overall objective (log_likelihood + log_prior)
     * This is the objective value used for estimation as the constant terms are irrelevant.
     * @param[in] model Model object defining the parameters and per
     * @param[in] data The data to evaluate for
     * @param[in] stencil The stencil computed at the parameter of interest, theta 
     */
    template<class Model>
    double rllh(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &stencil);

    /* model::grad */
    template<class Model>
    ParamT<Model> grad(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil);

    /* model::grad2 */
    template<class Model>
    void grad2(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil, ParamT<Model> &grad, ParamT<Model> &grad2);

    /* model::hessian */
    template<class Model>
    void hessian(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil, ParamT<Model> &grad, MatT &hess);
    
    /* model::objective*/
    template<class Model>
    void objective(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta, double &rllh, VecT &grad, MatT &hess);

    /* model::score*/
    template<class Model>
    ParamT<Model> score(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta);

    /* model::observed_information  --- Observed Fisher Information (at posterior mode) */
    template<class Model>
    MatT observed_information(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta_mode);

    /* model::expected_information --- Expected Fisher Information */
    template<class Model>
    MatT expected_information(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta);
    
    /* model::crlb --- Cramer-rao lower bound */
    template<class Model>
    ParamT<Model> crlb(const Model &model, const StencilT<Model> &theta);

    /* model::estimate */
    StencilT estimate_map(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, const std::string &method);
    StencilT estimate_profile_map(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, const ParamT<Model> &fixed_theta, const std::string &method);
    
    void estimate_map_error_expected(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta, const Model::IVecT &est_component, ParamT<Model> &theta_min, ParamT<Model> &theta_max)
    void estimate_map_error_observed(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta, const Model::IVecT &est_component, ParamT<Model> &theta_min, ParamT<Model> &theta_max)
    void estimate_map_error_profile(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta, const Model::IVecT &est_component, ParamT<Model> &theta_min, ParamT<Model> &theta_max, const std::string &method)
    
    MatT estimate_mcmc_sample(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, double burnin=0.0, int thin=1);
    VecT estimate_mcmc_posterior(const Model &model, const MatT &sample, ParamT<Model> &theta_posterior_mean, MatT &theta_posterior_stddev);
    void estimate_mcmc_posterior_credible_interval(const Model &model, const MatT &sample, const Model::IVecT &est_component, ParamT<Model> &theta_min, ParamT<Model> &theta_max);
    

    namespace debug {
        /* model::xxx_components - per-pixel and per-prior-component contributions */
        template<class Model>
        VecT llh_components(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &stencil);

        template<class Model>
        VecT rllh_components(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &stencil);
        
        template<class Model>
        MatT grad_components(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil);
        
        template<class Model>
        CubeT hess_components(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil);
    
        template<class Model>
        void objective_components(const Model &model, const ModelDataT<Model> &data, const Stencil<Model> &stencil, VecT &llh_components, MatT &grad_components, CubeT &hess_components);
        StencilT estimate_map(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, const std::string &method);
        StencilT estimate_profile_map(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, const ParamT<Model> &fixed_theta, const std::string &method);
        void estimate_map_error_profile(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta, const Model::IVecT &est_component, ParamT<Model> &theta_min, ParamT<Model> &theta_max, const std::string &method)
        MatT estimate_mcmc_sample(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, double burnin=0.0, int thin=1);
    };
    
    /**< These methods are simple convenience wrappers that call make_stencil on the theta parameter provided.  The direct use of the stencil functions is encouraged. */
    inline namespace convenience {
            /* model::llh */
        template<class Model>
        double llh(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);
        
        /* model::rllh */
        template<class Model>
        double rllh(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

        /* model::grad */
        template<class Model>
        ParamT<Model> grad(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

        /* model::grad2 */
        template<class Model>
        void grad2(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta, ParamT<Model> &grad, ParamT<Model> &grad2);

        /* model::hessian */
        template<class Model>
        void hessian(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta, ParamT<Model> &grad, MatT &hess);
        
        /* model::objective*/
        template<class Model>
        void objective(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta, double &rllh, VecT &grad, MatT &hess);

        /* model::score*/
        template<class Model>
        ParamT<Model> score(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

        /* model::observed_information  --- Observed Fisher Information (at posterior mode) */
        template<class Model>
        MatT observed_information(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_mode);


        /* model::expected_information --- Expected Fisher Information */
        template<class Model>
        MatT expected_information(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);
        
        /* model::crlb --- Cramer-rao lower bound */
        template<class Model>
        ParamT<Model> crlb(const Model &model, const ParamT<Model> &theta);
    }
    
    namespace prior {
            
    }
    
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

}
    
namespace openmp { 
    /* model::llh - openmp vectorized */
    
} /* namespace openmp */
    
    namespace likelihood_func {
        inline namespace openmp { 
        /* model::rllh */
            template<class Model>
            double rllh(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta);

            template<class Model>
            double rllh(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &stencil);

            /* model::objective*/
            template<class Model>
            void objective(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta, double &rllh, VecT &grad, MatT &hess);

            template<class Model>
            void objective(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta, double &rllh, VecT &grad, MatT &hess);

            StencilT estimate_map(const Model &model, const ParamT<Model> &theta_init, std::string method);
            StencilT estimate_profile_map(const Model &model, const ParamT<Model> &theta_init, const ParamT<Model> &bound_theta, std::string method);    
            void estimate_map_error_expected(const Model &model, const StencilT<Model> &theta, const Model::IVecT &est_component, const ParamT<Model> &theta_min, const ParamT<Model> &theta_max)
            void estimate_map_error_observed(const Model &model, const StencilT<Model> &theta, const Model::IVecT &est_component, const ParamT<Model> &theta_min, const ParamT<Model> &theta_max)
            void estimate_map_error_profile(const Model &model, const StencilT<Model> &theta, const Model::IVecT &est_component, const ParamT<Model> &theta_min, const ParamT<Model> &theta_max)
            
            MatT estimate_mcmc_sample(const Model &model, const ParamT<Model> &theta_init);
            VecT estimate_mcmc_posterior_mean(const Model &model, const MatT &sample)
            MatT estimate_mcmc_posterior_stddev(const Model &model, const MatT &sample)
            void estimate_mcmc_posterior_credible_interval(const Model &model, const MatT &sample, const Model::IVecT &est_component, const ParamT<Model> &theta_min, const ParamT<Model> &theta_max)
 
        } /* namespace openmp */                
    } /* namespace likelihood_func */
    namespace prior {
        inline namespace openmp { 

            
        } /* namespace openmp */                
    } /* namespace prior */
} /* namespace mappel */


} /* namespace mappel */

#endif /* _MAPPEL_MODEL_METHODS */
