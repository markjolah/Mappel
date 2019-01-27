
/** @file model_methoods_impl.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
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
    ModelDataT<Model> simulate_image(Model &model, const ParamT<Model> &theta) 
    {
        //don't compute stencils derivative 
        return simulate_image(model, model.make_stencil(theta,false), model.get_rng_generator());
    }
    
    template<class Model, class RngT>
    ModelDataT<Model> simulate_image(Model &model, const ParamT<Model> &theta, RngT &rng) 
    {
        return simulate_image(model, model.make_stencil(theta,false), rng); //don't compute derivative stencils
    }

    template<class Model>
    ModelDataT<Model> simulate_image(Model &model, const StencilT<Model> &s)
    {
        return simulate_image(model,s,model.get_rng_generator()); //Make new generator
    }

    template<class Model>
    ModelDataT<Model> simulate_image_from_model(Model &model, const ImageT<Model> &model_im)
    {
        return simulate_image_from_model(model,model_im,model.get_rng_generator()); //Make new generator
    }
    
    namespace objective {            
        template<class Model>
        double 
        llh(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
        { 
            return llh(model, data_im, model.make_stencil(theta,false)); //don't compute derivative stencils 
        }
      
        template<class Model>
        double 
        rllh(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
        { 
            return rllh(model, data_im, model.make_stencil(theta,false)); //don't compute derivative stencils
        }

        template<class Model>
        ParamT<Model> 
        grad(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
        {
            return grad(model, data_im, model.make_stencil(theta));
        }

        template<class Model>
        ParamT<Model> 
        grad2(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta)
        {
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
            negative_definite_hessian(model, data_im, model.make_stencil(theta), grad, hess);
        }
        
        template<class Model>
        void
        negative_definite_hessian(const Model &model, const ModelDataT<Model> &data_im, const StencilT<Model> &s,
                                  ParamT<Model> &grad, MatT &hess)
        {
            hessian(model, data_im, s, grad, hess);
            hess = -hess;
            modified_cholesky(hess);
            cholesky_convert_full_matrix(hess); //convert from internal format to a full (negative definite) matrix
            hess = -hess;
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
        rllh = likelihood::rllh(model, data_im, s);
        likelihood::hessian(model, data_im, s, grad, hess);
        rllh += model.get_prior().rllh(s.theta);
        model.get_prior().grad_hess_accumulate(s.theta,grad,hess);
    }

    template<class Model>
    void 
    prior_objective(const Model &model, const ParamT<Model> &theta, 
                    double &rllh, ParamT<Model> &grad, MatT &hess)
    {
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
        likelihood::hessian(model, data_im, s, grad, hess);
    }

    template<class Model>
    void 
    aposteriori_objective(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, 
                          double &rllh,  ParamT<Model> &grad, MatT &hess)
    {
        aposteriori_objective(model,data_im,model.make_stencil(theta),rllh,grad,hess);
    }
    
    template<class Model>
    void 
    likelihood_objective(const Model &model, const ModelDataT<Model> &data_im, const ParamT<Model> &theta, 
                          double &rllh,  ParamT<Model> &grad, MatT &hess)
    {
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
        MatT obsI = - objective::hessian(model,data,theta_mode); //Observed information is defined for negative llh and so negative hessian should be positive definite
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
    StencilT<Model> estimate_max(Model &model, const ModelDataT<Model> &data, const std::string &method)
    {
        auto estimator = make_estimator(model,method);
        return estimator->estimate_max(data);
    }
    
    template<class Model>
    StencilT<Model> estimate_max(Model &model, const ModelDataT<Model> &data, const std::string &method, const ParamT<Model> &theta_init, 
                                 double &theta_max_llh)
    {
        auto estimator = make_estimator(model,method);
        return estimator->estimate_max(data,theta_init,theta_max_llh);
    }
    
    template<class Model>
    void estimate_max(Model &model, const ModelDataT<Model> &data, const std::string &method, 
                      ParamT<Model> &theta_max, double &theta_max_llh, MatT &obsI)
    {
        auto estimator = make_estimator(model,method);
        estimator->estimate_max(data,theta_max,theta_max_llh, obsI);
    }

    template<class Model>
    void estimate_max(Model &model, const ModelDataT<Model> &data, const std::string &method, 
                      ParamT<Model> &theta_max, double &theta_max_llh, MatT &obsI, StatsT &stats)
    {
        auto estimator = make_estimator(model,method);
        estimator->estimate_max(data,theta_max,theta_max_llh, obsI);
        stats = estimator.get_stats();
    }

    template<class Model>
    void estimate_max(Model &model, const ModelDataT<Model> &data, const std::string &method, const ParamT<Model> &theta_init,
                      ParamT<Model> &theta_max, double &theta_max_llh, MatT &obsI)
    {
        auto estimator = make_estimator(model,method);
        estimator->estimate_max(data,theta_init, theta_max,theta_max_llh, obsI);
    }

    template<class Model>
    void estimate_max(Model &model, const ModelDataT<Model> &data, const std::string &method, const ParamT<Model> &theta_init,
                      ParamT<Model> &theta_max, double &theta_max_llh, MatT &obsI, StatsT &stats)
    {
        auto estimator = make_estimator(model,method);
        estimator->estimate_max(data,theta_init, theta_max,theta_max_llh, obsI);
        stats = estimator->get_stats();
    }
    
    
//     template<class Model>
//     StencilT<Model>
//     estimate_profile_max(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &fixed_theta, const std::string &method)
//     {
//     }
//     
//     template<class Model>
//     StencilT<Model>
//     estimate_profile_max(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &fixed_theta, const std::string &method,  const ParamT<Model> &theta_init)
//     {
//     }
//     
    /* MCMC posterior sampling likelihood computation */   
    template<class Model>
    MatT estimate_mcmc_sample(Model &model, const ModelDataT<Model> &data, IdxT Nsample, IdxT Nburnin, IdxT thin)
    {
        auto theta_init = model.make_param(arma::fill::zeros);
        return estimate_mcmc_sample(model, data, theta_init, Nsample, Nburnin, thin);
    }

    template<class Model>
    MatT estimate_mcmc_sample(Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, 
                              IdxT Nsample, IdxT Nburnin, IdxT thin)
    {
        if(thin == 0) thin = model.get_mcmc_num_phases();
        IdxT Noversample = mcmc::num_oversample(Nsample,Nburnin,thin);
        auto sample = model.make_param_stack(Noversample);
        VecT sample_rllh(Noversample);
        mcmc::sample_posterior(model, data, model.initial_theta_estimate(data,theta_init), 
                               sample, sample_rllh);
        return mcmc::thin_sample(sample, Nburnin, thin);
    }
    
    template<class Model>
    void estimate_mcmc_sample(Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, 
                              IdxT Nsample, IdxT Nburnin, IdxT thin,
                              MatT &sample, VecT &sample_rllh)
    {
        if(thin == 0) thin = model.get_mcmc_num_phases();
        IdxT Noversample = mcmc::num_oversample(Nsample,Nburnin,thin);
//         std::cout<<"Noversample: "<<Noversample<<" Nsample:"<<Nsample<<" Nburnin:"<<Nburnin<<" thin:"<<thin<<"\n";
        sample.set_size(model.get_num_params(), Nsample);
        sample_rllh.set_size(Nsample);
        auto oversample = model.make_param_stack(Noversample);
        VecT oversample_rllh(Noversample);
        mcmc::sample_posterior(model, data, model.initial_theta_estimate(data,theta_init),
                               oversample, oversample_rllh);
        mcmc::thin_sample(oversample, oversample_rllh, Nburnin, thin, sample, sample_rllh);
    }

    template<class Model>
    void estimate_mcmc_posterior(Model &model, const ModelDataT<Model> &data, 
                                 IdxT Nsample, IdxT Nburnin, IdxT thin, ParamT<Model> &posterior_mean, MatT &posterior_cov)
    {
        auto theta_init = model.make_param(arma::fill::zeros);
        estimate_mcmc_posterior(model,data,theta_init,Nsample, Nburnin, thin, posterior_mean, posterior_cov);
    }

    template<class Model>
    void estimate_mcmc_posterior(Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, 
                                 IdxT Nsample, IdxT Nburnin, IdxT thin, ParamT<Model> &posterior_mean, MatT &posterior_cov)
    {
        auto sample = estimate_mcmc_posterior(model,data,theta_init, Nsample, Nburnin, thin);
        mcmc::estimate_sample_posterior(sample, posterior_mean, posterior_cov); 
    }

    /* Error bounds computations */    
    template<class Model>
    void error_bounds_expected(const Model &model, const ParamT<Model> &theta_est, double confidence,
                               ParamT<Model> &theta_lb, ParamT<Model> &theta_ub)
    {
        auto crlb = cr_lower_bound(model, theta_est);
        double z = normal_quantile_twosided(confidence);
        auto sqrt_crlb = arma::sqrt(crlb);
        theta_lb = theta_est - z*sqrt_crlb;
        theta_ub = theta_est + z*sqrt_crlb;        
    }
    
    template<class Model>
    void error_bounds_observed(const Model &model, const ParamT<Model> &theta_est, MatT &obsI, double confidence,
                               ParamT<Model> &theta_lb, ParamT<Model> &theta_ub)
    {
        auto var = arma::pinv(obsI).eval().diag();
        double z = normal_quantile_twosided(confidence);
        auto sigma = arma::sqrt(var);
        theta_lb = theta_est - z*sigma;
        theta_ub = theta_est + z*sigma;        
    }

//     template<class Model>
//     void error_bounds_profile(const Model &model, const ModelDataT<Model> &data, const std::string &method, const StencilT<Model> &theta_max, 
//                               ParamT<Model> &theta_lb, ParamT<Model> &theta_ub)
//     {
//     }

    template<class Model>
    void error_bounds_posterior_credible(const Model &model, const MatT &sample, double confidence,
                                         ParamT<Model> &theta_mean, ParamT<Model> &theta_lb, ParamT<Model> &theta_ub)
    {
        IdxT Nsample = sample.n_cols;
        double p = (1-confidence)/2.;
        IdxT lb_idx = floor(Nsample*p);
        IdxT ub_idx = ceil(Nsample*(1-p));
        auto sorted_sample = arma::sort(sample,"ascend",1).eval();
        theta_mean = arma::mean(sample,1);
        theta_lb = sorted_sample.col(lb_idx);
        theta_ub = sorted_sample.col(ub_idx);
    }
    

    inline namespace debug {
         template<class Model>
        void estimate_max_debug(Model &model, const ModelDataT<Model> &data, const std::string &method, 
                                ParamT<Model> &theta_max, double &rllh, MatT &obsI, MatT &sequence, VecT &sequence_rllh, StatsT &stats)
        {
            auto theta_init = model.make_param(arma::fill::zeros);
            estimate_max_debug(model, data, method, theta_init, theta_max, rllh, obsI, sequence, sequence_rllh, stats);
        }
        
        template<class Model>
        void estimate_max_debug(Model &model, const ModelDataT<Model> &data, const std::string &method, const ParamT<Model> &theta_init, 
                                ParamT<Model> &theta_max, double &rllh, MatT &obsI, MatT &sequence, VecT &sequence_rllh, StatsT &stats)
        {
            auto estimator = make_estimator(model,method);
            estimator->estimate_max_debug(data,theta_init,theta_max,rllh,obsI,sequence,sequence_rllh);
            stats = estimator->get_debug_stats();
        }
        
//         template<class Model>
//         void estimate_profile_max_debug(const Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, const ParamT<Model> &fixed_theta, const std::string &method, 
//                                         StencilT<Model> &theta_max, double &rllh, StatsT &stats, MatT &sequence, VecT &sequence_rllh);
        
        template <class Model>
        void estimate_mcmc_sample_debug(Model &model, const ModelDataT<Model> &data,
                                        IdxT Nsample, 
                                        MatT &sample, VecT &sample_rllh, MatT &candidates, VecT &candidates_rllh)
        {
            auto theta_init = model.make_param(arma::fill::zeros);
            estimate_mcmc_sample_debug(model, data, Nsample, sample, sample_rllh, candidates, candidates_rllh);
        }
        
        template <class Model>
        void estimate_mcmc_sample_debug(Model &model, const ModelDataT<Model> &data, const ParamT<Model> &theta_init, 
                                        IdxT Nsample, 
                                        MatT &sample, VecT &sample_rllh, MatT &candidates, VecT &candidates_rllh)
        {
            sample.set_size(model.get_num_params(), Nsample);
            sample_rllh.set_size(Nsample);
            candidates.set_size(model.get_num_params(), Nsample);
            candidates_rllh.set_size(Nsample);
            auto stencil = model.initial_theta_estimate(data, theta_init);
            mcmc::sample_posterior_debug(model, data, stencil, sample, sample_rllh, candidates, candidates_rllh);
        }
    }; /* namespace mappel::methods::debug */

    
} /* namespace mappel::methods */
    

} /* namespace mappel */

#endif /* MAPPEL_MODEL_METHODS_IMPL_H */
