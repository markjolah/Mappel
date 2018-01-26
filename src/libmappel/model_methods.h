
/** @file model_methoods.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief Namespace and function definitions for the model:: namespace which contains the major methods for computing with PointEmitterModels
 */

#ifndef _MAPPEL_MODEL_METHODS
#define _MAPPEL_MODEL_METHODS

//#include "PointEmitterModel.h"

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
/** @brief External template based methods for PointEmitterModel's.  
 * These are general or convienience functions
 * that are included in this file.  Those methods specific to other sub-types of Models should be included within that
 * sub-type's .h file, using the enbale_if mechansim to restrict their instantiation to the correct sub-types.
 */
namespace methods {
    /*These types are defined elsewhere in the Model class heierachy we use template types to stand in for them */

    
    /** Expected number of photons at each pixel in image given the emitter model 
     */
    template<class Model>
    typename Model::ImageT model_image(const Model &model, const ParamT<Model> &theta);

    template<class Model, class rng_t>
    ModelDataT<Model> simulate_image(const Model &model, const ParamT<Model> &theta);

    template<class Model, class rng_t>
    ModelDataT<Model> simulate_image(const Model &model, const ParamT<Model> &theta, rng_t &rng);

    template<class Model>
    ModelDataT<Model> simulate_image(const Model &model, const StencilT<Model> &s);

    template<class Model>
    ModelDataT<Model> simulate_image_from_model(const Model &model, const ImageT<Model> &model_im);
    
    namespace objective {            
        template<class Model>
        double llh(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta);
      
        template<class Model>
        double rllh(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta);

        template<class Model>
        ParamT<Model> grad(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta);

        template<class Model>
        ParamT<Model> grad2(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta);

        template<class Model>
        void grad2(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, 
                   ParamT<Model> &grad_val, ParamT<Model> &grad2_val);

        template<class Model>
        MatT hessian(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta);
        
        template<class Model>
        MatT hessian(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s);

        template<class Model>
        void hessian(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, ParamT<Model> &grad, MatT &hess);
        
        template<class Model>
        void hessian(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, MatT &hess);
        
        
        template<class Model>
        MatT negative_definite_hessian(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta);
        
        template<class Model>
        MatT negative_definite_hessian(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s);

        template<class Model>
        void negative_definite_hessian(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, ParamT<Model> &grad, MatT &hess);

        template<class Model>
        void negative_definite_hessian(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s, ParamT<Model> &grad, MatT &hess);
        
        inline namespace debug {
            /* Component methods aid with debugging and understanding the contribution from each pixel */
            template<class Model>
            VecT llh_components(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta);

            template<class Model>
            VecT rllh_components(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta);

            template<class Model>
            MatT grad_components(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta);
            
            template<class Model>
            CubeT hessian_components(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta);
        }
    } /* mappel::methods::objective */

    
    template<class Model>
    void aposteriori_objective(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s, 
                               double &rllh,  ParamT<Model> &grad, MatT &hess);

    template<class Model>
    void aposteriori_objective(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, 
                          double &rllh,  ParamT<Model> &grad, MatT &hess);

    template<class Model>
    void prior_objective(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s, 
                         double &rllh, ParamT<Model> &grad, MatT &hess);

    template<class Model>
    void prior_objective(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, 
                         double &rllh,  ParamT<Model> &grad, MatT &hess);
    
    template<class Model>
    void likelihood_objective(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s, 
                         double &rllh,  ParamT<Model> &grad, MatT &hess);

    template<class Model>
    void likelihood_objective(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, 
                              double &rllh,  ParamT<Model> &grad, MatT &hess);

    /** @brief Calculate the Cramer-Rao lower bound at the given paramters
    * @param[in] theta The parameters to evaluate the CRLB at
    * @param[out] crlb The calculated parameters
    */
    template<class Model>
    ParamT<Model> cr_lower_bound(const Model &model, const typename Model::Stencil &s);

    template<class Model>
    ParamT<Model> cr_lower_bound(const Model &model, const ParamT<Model> &theta);

    /* model::expected_information --- Expected Fisher Information at theta */
    template<class Model>
    MatT expected_information(const Model &model, const ParamT<Model> &theta);
    
    /* model::observed_information  --- Observed Fisher Information (at posterior mode) */
    template<class Model>
    MatT observed_information(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_mode);

    template<class Model>
    MatT observed_information(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta_mode);


    
    /* MAP/MLE Estimation */
    template<class Model>
    StencilT<Model>
    estimate_max(const Model &model, const ModelDataT<Model> &data, const std::string &method);

    template<class Model>
    StencilT<Model>
    estimate_max(const Model &model, const ModelDataT<Model> &data, const std::string &method, const ParamT<Model> &theta_init, 
                 double &rllh);

    template<class Model>
    void estimate_max(const Model &model, const ModelDataT<Model> &data, const std::string &method, 
                      ParamT<Model> &theta_max, double &theta_max_llh, MatT &obsI);

    template<class Model>
    void estimate_max(const Model &model, const ModelDataT<Model> &data, const std::string &method, 
                      ParamT<Model> &theta_max, double &theta_max_llh, MatT &obsI, StatsT &stats);
    
    template<class Model>
    void estimate_max(const Model &model, const ModelDataT<Model> &data, const std::string &method, const ParamT<Model> &theta_init,
                      ParamT<Model> &theta_max, double &theta_max_llh, MatT &obsI);

    template<class Model>
    void estimate_max(const Model &model, const ModelDataT<Model> &data, const std::string &method, const ParamT<Model> &theta_init,
                      ParamT<Model> &theta_max, double &theta_max_llh, MatT &obsI, StatsT &stats);
    
    /* MAP/MLE Profile likelihood computation */    
//     template<class Model>
//     StencilT estimate_profile_max(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &fixed_theta, const std::string &method);
//     
//     template<class Model>
//     StencilT estimate_profile_max(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &fixed_theta, const std::string &method,  const ParamT<Model> &theta_init);
//     
//     /* MCMC posterior sampling likelihood computation */    
    template<class Model>
    MatT estimate_mcmc_sample(const Model &model, const ModelDataT<Model> &data, 
                              IdxT Nsample=1000, IdxT Nburnin=100, IdxT thin=0);
    
    template<class Model>
    MatT estimate_mcmc_sample(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, 
                              IdxT Nsample=1000, IdxT Nburnin=100, IdxT thin=0);
    
    template<class Model>
    void estimate_mcmc_sample(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, 
                              IdxT Nsample, IdxT Nburnin, IdxT thin, MatT &sample, VecT &sample_rllh);

    template<class Model>
    void estimate_mcmc_posterior(const Model &model, const ModelDataT<Model> &data, 
                                 IdxT Nsample, IdxT Nburnin, IdxT thin, 
                                 ParamT<Model> &posterior_mean, MatT &posterior_cov);
    
    template<class Model>
    void estimate_mcmc_posterior(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, 
                                 IdxT Nsample, IdxT Nburnin, IdxT thin, 
                                 ParamT<Model> &posterior_mean, MatT &posterior_cov);


    /* Error bounds computations */    
    template<class Model>
    void error_bounds_expected(const Model &model, const ParamT<Model> &theta_est, double confidence,
                               ParamT<Model> &theta_lb, ParamT<Model> &theta_ub);
    template<class Model>
    void error_bounds_observed(const Model &model, const ParamT<Model> &theta_est, MatT &obsI, double confidence,
                               ParamT<Model> &theta_lb, ParamT<Model> &theta_ub);
//     template<class Model>
//     void error_bounds_profile(const Model &model, const ModelDataT<Model> &data, const std::string &method, const StencilT<Model> &theta_est, 
//                               ParamT<Model> &theta_lb, ParamT<Model> &theta_ub);
    template<class Model>
    void error_bounds_posterior_credible(const Model &model, const MatT &sample, double confidence,
                                          ParamT<Model> &theta_mean, ParamT<Model> &theta_lb, ParamT<Model> &theta_ub);
    

    inline namespace debug {
        /* model::xxx_components - per-pixel and per-prior-component contributions */
        template<class Model>
        void estimate_max_debug(const Model &model, const ModelDataT<Model> &data, const std::string &method,   
                                ParamT<Model> &theta_est, MatT &obsI,  MatT &sequence, VecT &sequence_rllh, StatsT &stats);
        template<class Model>
        void estimate_max_debug(const Model &model, const ModelDataT<Model> &data, const std::string &method, const ParamT<Model> &theta_init,  
                                ParamT<Model> &theta_est, MatT &obsI,  MatT &sequence, VecT &sequence_rllh, StatsT &stats);
        
//         template<class Model>
//         void estimate_profile_max_debug(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, const ParamT<Model> &fixed_theta, const std::string &method, 
//                                         StencilT<Model> &theta_max, double &rllh, StatsT &stats, MatT &sequence, VecT &sequence_rllh);
        template <class Model>
        void estimate_mcmc_sample_debug(const Model &model, const ModelDataT<Model> &data,
                                        IdxT Nsample, 
                                        MatT &sample, VecT &sample_rllh, MatT &candidates, VecT &candidates_rllh);
        template<class Model>
        void estimate_mcmc_sample_debug(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, 
                                        IdxT Nsample, 
                                        MatT &sample, VecT &sample_rllh, MatT &candidates, VecT &candidates_rllh);    
    }; /* namespace mappel::methods::debug */

    
} /* namespace mappel::methods */

} /* namespace mappel */

#include "estimator.h"
#include "mcmc.h"
#include "openmp_methods.h"
#include "model_methods_impl.h" //Implementation of methods
#include "estimator_impl.h"

#endif /* _MAPPEL_MODEL_METHODS */
