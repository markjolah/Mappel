/** @file model_methoods_impl.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2019
 * @brief Methods definitions for the model:: namespace which contains the major methods for computing with PointEmitterModels
 */

#ifndef MAPPEL_MODEL_METHODS_IMPL_H
#define MAPPEL_MODEL_METHODS_IMPL_H

#include "Mappel/numerical.h"

namespace mappel {

namespace methods {
    template<class Model>
    typename Model::ImageT model_image(const Model &model, const ParamT<Model> &theta) 
    {
        return model_image(model, model.make_stencil(theta,false)); //don't compute derivative stencils
    }

    template<class Model>
    ModelDataT<Model> simulate_image(const Model &model, const ParamT<Model> &theta)
    {
        //don't compute stencils derivative 
        return simulate_image(model, model.make_stencil(theta,false), model.get_rng_generator());
    }
    
    template<class Model, class RngT>
    ModelDataT<Model> simulate_image(const Model &model, const ParamT<Model> &theta, RngT &rng)
    {
        return simulate_image(model, model.make_stencil(theta,false), rng); //don't compute derivative stencils
    }

    template<class Model>
    ModelDataT<Model> simulate_image(const Model &model, const StencilT<Model> &s)
    {
        return simulate_image(model,s,model.get_rng_generator()); //Make new generator
    }

    template<class Model>
    ModelDataT<Model> simulate_image_from_model(const Model &model, const ImageT<Model> &model_im)
    {
        return simulate_image_from_model(model,model_im,model.get_rng_generator()); //Make new generator
    }
    
    namespace objective {            
        template<class Model>
        double 
        llh(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
        { 
            if(!model.theta_in_bounds(theta)) return arma::datum::nan;
            return llh(model, data_im, model.make_stencil(theta,false)); //don't compute derivative stencils 
        }
      
        template<class Model>
        double 
        rllh(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
        { 
            if(!model.theta_in_bounds(theta)) return arma::datum::nan;
            return rllh(model, data_im, model.make_stencil(theta,false)); //don't compute derivative stencils
        }

        template<class Model>
        ParamT<Model> 
        grad(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
        {
            if(!model.theta_in_bounds(theta)) {
                auto grad = model.make_param();
                grad.fill(arma::datum::nan);
                return grad;
            }
            return grad(model, data_im, model.make_stencil(theta));
        }

        template<class Model>
        ParamT<Model> 
        grad2(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
        {
            if(!model.theta_in_bounds(theta)) {
                auto grad2 = model.make_param();
                grad2.fill(arma::datum::nan);
                return grad2;
            }
            auto grad_val = model.make_param(); //Ignore un-requested value
            auto grad2_val = model.make_param();
            grad2(model, data_im, model.make_stencil(theta), grad_val, grad2_val);
            return grad2_val;
        }

        template<class Model>
        void
        grad2(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, 
              ParamT<Model> &grad_val, ParamT<Model> &grad2_val)
        {
            grad2(model, data_im, model.make_stencil(theta), grad_val, grad2_val);
        }

        template<class Model>
        MatT 
        hessian(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
        { 
           if(!model.theta_in_bounds(theta)) {
                auto hess = model.make_param_mat();
                hess.fill(arma::datum::nan);
                return hess;
            }
            return hessian(model,data_im, model.make_stencil(theta));
        }

        template<class Model>
        MatT 
        hessian(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
        {
            auto grad = model.make_param(); //Ignore un-requested value
            auto hess = model.make_param_mat();
            hessian(model, data_im, s, grad, hess);
            return hess;
        }
        
        template<class Model>
        void 
        hessian(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, ParamT<Model> &grad, MatT &hess)
        {
            if(!model.theta_in_bounds(theta)) {
                grad.fill(arma::datum::nan);
                hess.fill(arma::datum::nan);
                return;
            }
            hessian(model, data_im, model.make_stencil(theta), grad, hess);
        }
        
        template<class Model>
        void 
        hessian(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, MatT &hess)
        {
            auto grad = model.make_param();
            hessian(model, data_im, model.make_stencil(theta), grad, hess);
        }
        
        template<class Model>
        MatT 
        negative_definite_hessian(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
        { 
            if(!model.theta_in_bounds(theta)) {
                auto hess = model.make_param_mat();
                hess.fill(arma::datum::nan);
                return hess;
            }
            return negative_definite_hessian(model, data_im, model.make_stencil(theta));
        }

        template<class Model>
        MatT 
        negative_definite_hessian(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s)
        {
            auto grad = model.make_param(); //Ignore un-requested value
            auto hess = model.make_param_mat();
            negative_definite_hessian(model, data_im, s, grad, hess);
            return hess;
        }
        
        template<class Model>
        void
        negative_definite_hessian(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta,
                                  ParamT<Model> &grad, MatT &hess)
        {
            if(!model.theta_in_bounds(theta)) {
                grad.fill(arma::datum::nan);
                hess.fill(arma::datum::nan);
                return;
            }
            negative_definite_hessian(model, data_im, model.make_stencil(theta), grad, hess);
        }
        
        template<class Model>
        void
        negative_definite_hessian(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s,
                                  ParamT<Model> &grad, MatT &hess)
        {
            hessian(model, data_im, s, grad, hess);
            cholesky_make_negative_definite(hess);
        }

        inline namespace debug {
            template<class Model>
            VecT 
            llh_components(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
            { 
                return llh_components(model, data_im, model.make_stencil(theta,false)); //don't compute derivative stencils 
            }

            template<class Model>
            VecT 
            rllh_components(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
            { 
                return rllh_components(model, data_im, model.make_stencil(theta,false)); //don't compute derivative stencils
            }

            template<class Model>
            MatT
            grad_components(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
            {
                return grad_components(model, data_im, model.make_stencil(theta));
            }
            
            template<class Model>
            CubeT
            hessian_components(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
            {
                return hessian_components(model, data_im, model.make_stencil(theta));
            }
        } /* mappel::methods::objective::debug */
    } /* mappel::methods::objective */

    
    template<class Model>
    void 
    aposteriori_objective(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s, 
                          double &rllh,  ParamT<Model> &grad, MatT &hess)
    {
        rllh = methods::likelihood::rllh(model, data_im, s);
        methods::likelihood::hessian(model, data_im, s, grad, hess);
        rllh += model.get_prior().rllh(s.theta);
        model.get_prior().grad_hess_accumulate(s.theta,grad,hess);
    }

    template<class Model>
    void 
    prior_objective(const Model &model, const ParamT<Model> &theta, 
                    double &rllh, ParamT<Model> &grad, MatT &hess)
    {
        if(!model.theta_in_bounds(theta)) {
            rllh = arma::datum::nan;
            grad.fill(arma::datum::nan);
            hess.fill(arma::datum::nan);
            return;
        }
        grad.zeros();
        hess.zeros();
        auto &prior = model.get_prior();
        rllh = prior.rllh(theta);
        prior.grad_hess_accumulate(theta,grad,hess);
    }

    template<class Model>
    void 
    likelihood_objective(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s, 
                         double &rllh,  ParamT<Model> &grad, MatT &hess)
    {
        rllh = likelihood::rllh(model, data_im, s);
        methods::likelihood::hessian(model, data_im, s, grad, hess);
    }

    template<class Model>
    void 
    aposteriori_objective(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, 
                          double &rllh,  ParamT<Model> &grad, MatT &hess)
    {
        if(!model.theta_in_bounds(theta)) {
            rllh = arma::datum::nan;
            grad.fill(arma::datum::nan);
            hess.fill(arma::datum::nan);
            return;
        }
        aposteriori_objective(model,data_im,model.make_stencil(theta),rllh,grad,hess);
    }
    
    template<class Model>
    void 
    likelihood_objective(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, 
                          double &rllh,  ParamT<Model> &grad, MatT &hess)
    {
        if(!model.theta_in_bounds(theta)) {
            rllh = arma::datum::nan;
            grad.fill(arma::datum::nan);
            hess.fill(arma::datum::nan);
            return;
        }
        likelihood_objective(model,data_im,model.make_stencil(theta),rllh,grad,hess);
    }

    template<class Model>
    ParamT<Model> cr_lower_bound(const Model &model, const typename Model::Stencil &s)
    {
        auto FI = expected_information(model,s);
        try{
            return arma::pinv(arma::symmatu(FI)).eval().diag();
        } catch ( std::runtime_error E) {
            std::cout<<"Got bad fisher_information!!\n"<<"theta:"<<s.theta.t()<<"\n FI: "<<FI<<'\n';
            return model.make_param(arma::fill::zeros);
        }
    }

    template<class Model>
    ParamT<Model> cr_lower_bound(const Model &model, const ParamT<Model> &theta) 
    {
        return cr_lower_bound(model,model.make_stencil(theta));
    }

    template<class Model>
    MatT expected_information(const Model &model, const ParamT<Model> &theta) 
    {
        return expected_information(model,model.make_stencil(theta));
    }

    template<class Model>
    MatT observed_information(const Model &model, const ModelDataT<Model> &data, const StencilT<Model> &theta_mode)
    {
        MatT obsI = -objective::hessian(model,data,theta_mode); //Observed information is defined for negative llh and so negative hessian should be positive definite
        //if(!is_positive_definite(obsI)) throw NumericalError("Hessian is not positive definite");
        return obsI;
    }

    template<class Model>
    MatT observed_information(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_mode)
    {
        return observed_information(model,data, model.make_stencil(theta_mode));
    }

    /* MAP/MLE Estimation */
    template<class Model>
    void estimate_max(const Model &model, const ModelDataT<Model> &data, const std::string &method, estimator::MLEData &mle)
    {
        auto estimator = make_estimator(model,method);
        auto theta_init = model.make_param();
        theta_init.zeros();
        estimator->estimate_max(data,theta_init,mle);
    }

    template<class Model>
    void estimate_max(const Model &model, const ModelDataT<Model> &data, const std::string &method, const ParamT<Model> &theta_init, estimator::MLEData &mle)
    {
        auto estimator = make_estimator(model,method);
        estimator->estimate_max(data,theta_init,mle);
    }

    template<class Model>
    void estimate_max(const Model &model, const ModelDataT<Model> &data, const std::string &method, estimator::MLEData &mle, StatsT &stats)
    {
        auto estimator = make_estimator(model,method);
        auto theta_init = model.make_param();
        theta_init.zeros();
        estimator->estimate_max(data,theta_init,mle);
        stats = estimator->get_stats();
    }

    template<class Model>
    void estimate_max(const Model &model, const ModelDataT<Model> &data, const std::string &method, const ParamT<Model> &theta_init, estimator::MLEData &mle, StatsT &stats)
    {
        auto estimator = make_estimator(model,method);
        estimator->estimate_max(data,theta_init,mle);
        stats = estimator->get_stats();
    }

    /* MAP/MLE Profile likelihood computation */
    template<class Model>
    double estimate_profile_likelihood(const Model &model, const ModelDataT<Model> &data, const std::string &method, const IdxVecT &fixed_idxs, const ParamT<Model> &fixed_theta_init)
    {
        auto estimator = make_estimator(model,method);
        StencilT<Model> profile_max;
        return estimator->estimate_profile_max(data,fixed_idxs,fixed_theta_init,profile_max);
    }

    template<class Model>
    double estimate_profile_likelihood(const Model &model, const ModelDataT<Model> &data, const std::string &method, const IdxVecT &fixed_idxs,
                                       const ParamT<Model> &fixed_theta_init, StencilT<Model> &profile_max)
    {
        auto estimator = make_estimator(model,method);
        return estimator->estimate_profile_max(data,fixed_idxs,fixed_theta_init,profile_max);
    }

    template<class Model>
    double estimate_profile_likelihood(const Model &model, const ModelDataT<Model> &data, const std::string &method, const IdxVecT &fixed_idxs,
                                       const ParamT<Model> &fixed_theta_init, StencilT<Model> &profile_max, StatsT &stats)
    {
        auto estimator = make_estimator(model,method);
        auto prof_rllh = estimator->estimate_profile_max(data,fixed_idxs,fixed_theta_init,profile_max);
        stats = estimator->get_stats();
        return prof_rllh;
    }

    template<class Model>
    void estimate_posterior(const Model &model, const ModelDataT<Model> &data, mcmc::MCMCData &est)
    {
        auto theta_init = model.make_param();
        theta_init.zeros();
        estimate_posterior(model, data, theta_init, est);
    }

    template<class Model>
    void estimate_posterior(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, mcmc::MCMCData &est)
    {
        if(est.thin == 0) est.thin = model.get_mcmc_num_phases();
        auto Noversample = mcmc::num_oversample(est.Nsample,est.Nburnin,est.thin);
        auto oversample = model.make_param_stack(Noversample);
        VecT oversample_rllh(Noversample);
        mcmc::sample_posterior(model, data, model.initial_theta_estimate(data,theta_init), oversample, oversample_rllh);
        mcmc::thin_sample(oversample, oversample_rllh, est.Nburnin, est.thin, est.sample, est.sample_rllh);
        mcmc::estimate_sample_posterior(est.sample, est.sample_mean, est.sample_cov);
        if(0<est.confidence && est.confidence<1) mcmc::compute_posterior_credible(est.sample,est.confidence,est.credible_lb,est.credible_ub);
    }

    /* Error bounds computations */    
    template<class Model>
    void error_bounds_expected(const Model &model, const ParamT<Model> &theta_est, double confidence,
                               ParamT<Model> &theta_lb, ParamT<Model> &theta_ub)
    {
        auto crlb = cr_lower_bound(model, theta_est);
        double z = normal_quantile_twosided(confidence);
        auto sqrt_crlb = arma::sqrt(crlb);
        theta_lb = arma::max(model.get_lbound(),theta_est - z*sqrt_crlb);
        theta_ub = arma::min(model.get_ubound(),theta_est + z*sqrt_crlb);
    }
    
    template<class Model>
    void error_bounds_observed(const Model &model, const estimator::MLEData &mle, double confidence,
                               ParamT<Model> &theta_lb, ParamT<Model> &theta_ub)
    {
        double z = normal_quantile_twosided(confidence);
        VecT bnd = z*arma::sqrt(arma::pinv(mle.obsI).eval().diag());
        theta_lb = arma::max(model.get_lbound(),mle.theta - bnd);
        theta_ub = arma::min(model.get_ubound(),mle.theta + bnd);
    }

    template<class Model>
    void error_bounds_profile_likelihood(const Model &model, const ModelDataT<Model> &data, estimator::ProfileBoundsData &est)
    {
        estimator::NewtonMaximizer<Model> estimator(model);
        if(!std::isfinite(est.target_rllh_delta)) est.target_rllh_delta =  -.5*chisq_quantile(est.confidence);
        if(est.estimated_idxs.is_empty()) est.estimated_idxs = arma::regspace(0,model.get_num_params()-1);
        estimator.estimate_profile_bounds(data,est);
    }

    template<class Model>
    void error_bounds_profile_likelihood(const Model &model, const ModelDataT<Model> &data, estimator::ProfileBoundsData &est, StatsT &stats)
    {
        estimator::NewtonMaximizer<Model> estimator(model);
        if(!std::isfinite(est.target_rllh_delta)) est.target_rllh_delta =  -.5*chisq_quantile(est.confidence);
        if(est.estimated_idxs.is_empty()) est.estimated_idxs = arma::regspace(0,model.get_num_params()-1);
        stats = estimator.get_stats();
    }

    template<class Model>
    void error_bounds_posterior_credible(const Model &, const MatT &sample, double confidence, ParamT<Model> &theta_lb, ParamT<Model> &theta_ub)
    {
        mcmc::compute_posterior_credible(sample,confidence,theta_lb,theta_ub);
    }
    

    inline namespace debug {
        template<class Model>
        void estimate_max_debug(const Model &model, const ModelDataT<Model> &data, const std::string &method,
                                const ParamT<Model> &theta_init, estimator::MLEDebugData &mle, StatsT &stats)
        {
            auto estimator = make_estimator(model,method);
            estimator->estimate_max_debug(data,theta_init,mle);
            stats = estimator->get_debug_stats();
        }
        
        template<class Model>
        void error_bounds_profile_likelihood_debug(const Model &model, const ModelDataT<Model> &data,
                                                   estimator::ProfileBoundsDebugData &bounds, StatsT &stats)
        {
            estimator::NewtonMaximizer<Model> estimator(model);
            estimator.estimate_profile_bounds_debug(data, bounds);
            stats = estimator.get_stats();
        }

        template <class Model>
        void estimate_posterior_debug(const Model &model, const ModelDataT<Model> &data,
                                      const ParamT<Model> &theta_init, mcmc::MCMCDebugData &mcmc_data)
        {
            if(mcmc_data.Nsample<1) throw ParameterValueError("MCMCDebugData.Nsample must be positive.");
            mcmc_data.initialize_arrays(model.get_num_params());
            mcmc::sample_posterior_debug(model, data, model.initial_theta_estimate(data, theta_init),
                                         mcmc_data.sample, mcmc_data.sample_rllh, mcmc_data.candidate, mcmc_data.candidate_rllh);
        }
    }; /* namespace mappel::methods::debug */

    
} /* namespace mappel::methods */
} /* namespace mappel */

#endif /* MAPPEL_MODEL_METHODS_IMPL_H */
