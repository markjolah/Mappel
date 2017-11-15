
#ifndef _MCMC_H
#define _MCMC_H
#include<cmath>
#include "stackcomp.h"
#include "display.h"
#include "estimator.h"

namespace mappel {

MatT thin_sample(MatT &sample, IdxT burn_in, IdxT keep_every);

template <class Model>
inline
void evaluate_posterior_stack(Model &model, 
                              const typename Model::ImageStackT &im_stack,
                              IdxT Nsamples, MatT &mean_stack, CubeT &cov_stack)
{
    auto theta_init_stack = model.make_param_vec();
    theta_init_stack.zeros();
    evaluate_posterior_stack(model, im_stack, theta_init_stack, Nsamples,mean_stack, cov_stack);
}

template <class Model>
void evaluate_posterior_stack(Model &model, 
                              const typename Model::ImageStackT &im_stack,
                              const typename Model::ParamVecT &theta_init_stack,
                              IdxT Nsamples, MatT &mean_stack, CubeT &cov_stack)
{
    const IdxT count = model.get_size_image_stack(im_stack);
    #pragma omp parallel
    {
        auto init = model.make_param();
        init.zeros();
        #pragma omp for
        for(IdxT n=0; n<count; n++){
            if(!theta_init_stack.is_empty()) init = theta_init_stack.col(n);
            auto stencil = model.initial_theta_estimate(model.get_image_from_stack(im_stack,n), init);
            auto sample = sample_posterior(model, model.get_image_from_stack(im_stack,n),  Nsamples, stencil);
            mean_stack.col(n) = arma::mean(sample, 1);
            cov_stack.slice(n) = arma::cov(sample.t());
//             auto subsample=sample.cols(burnin,sample.n_cols-1);
//             mean_stack.col(n)=arma::mean(subsample, 1);
//             cov_stack.slice(n)=arma::cov(subsample.t());
        }
    }
}

template <class Model>
inline
void evaluate_posterior(Model &model, const typename Model::ImageT &im,
                        IdxT Nsamples, typename Model::ParamT &mean, typename Model::ParamMatT &cov)
{
    auto theta_init = model.make_param();
    theta_init.zeros();
    evaluate_posterior(model,im,theta_init,Nsamples,mean,cov);
}

template <class Model>
void evaluate_posterior(Model &model, const typename Model::ImageT &im, const typename Model::ParamT &theta_init,
                        IdxT Nsamples, typename Model::ParamT &mean, typename Model::ParamMatT &cov)
{
    auto rng = rng_manager.generator();
    IdxT burnin = Nsamples;
    auto stencil = model.initial_theta_estimate(im, theta_init);
    auto sample = sample_posterior(model, im,  burnin+Nsamples, stencil, rng);
    auto subsample = sample.cols(burnin,sample.n_cols-1);
    mean = arma::mean(subsample, 1);
    cov = arma::cov(subsample.t());
}

template <class Model>
inline
void evaluate_posterior_debug(Model &model, const typename Model::ImageT &im,
                              IdxT Nsamples, VecT &mean, MatT &cov,
                              typename Model::ParamVecT &sample, VecT &sample_llh,
                              typename Model::ParamVecT &candidates, VecT &mcmc_candidate_llh)
{
    auto theta_init = model.make_param();
    theta_init.zeros();
    evaluate_posterior_debug(model,im,theta_init,Nsamples,mean,cov,sample,sample_llh,candidates, mcmc_candidate_llh);
}

template <class Model>
void evaluate_posterior_debug(Model &model, const typename Model::ImageT &im,  const typename Model::ParamT &theta_init,
                              IdxT Nsamples, VecT &mean, MatT &cov,
                              typename Model::ParamVecT &sample, VecT &sample_llh,
                              typename Model::ParamVecT &candidates, VecT &mcmc_candidate_llh)
{
    auto rng = rng_manager.generator();
    auto stencil = model.initial_theta_estimate(im, theta_init);
    sample_posterior_debug(model, im, stencil, sample, candidates, rng);

    #pragma omp parallel for
    for(IdxT n=0; n<Nsamples; n++){
        sample_llh(n) = relative_log_likelihood(model, im, sample.col(n));
        mcmc_candidate_llh(n) = relative_log_likelihood(model, im, candidates.col(n));
    }
    mean = arma::mean(sample, 1);
    cov = arma::cov(sample.t());
}

template <class Model>
typename Model::ParamVecT
sample_posterior(Model &model, const typename Model::ImageT &im, IdxT Nsamples,
                            typename Model::Stencil &theta_init)
{
    auto sample = model.make_param_vec(Nsamples);
    sample.col(0) = theta_init.theta;
    double old_rllh = relative_log_likelihood(model, im, theta_init);
    IdxT phase = 0;
    for(IdxT n=1;n<Nsamples;n++){
        typename Model::ParamT can_theta = sample.col(n-1);
        model.sample_mcmc_candidate_theta(phase, can_theta);
        if(!model.theta_in_bounds(can_theta)) { //OOB so stay put
            sample.col(n) = sample.col(n-1);
            continue;
        }
        double can_rllh=relative_log_likelihood(model, im, can_theta);
        phase++;
        double alpha = std::min(1.,exp(can_rllh-old_rllh));
        if(rng_manager.randu() < alpha) {
            sample.col(n) = can_theta;
            old_rllh = can_rllh;
        } else { //reject: record old point again
            sample.col(n) = sample.col(n-1);
        }
    }
    return sample;
}


template <class Model>
void sample_posterior_debug(Model &model, const typename Model::ImageT &im,
                      typename Model::Stencil &theta_init,
                      typename Model::ParamVecT &sample,
                      typename Model::ParamVecT &candidates)
{
    IdxT Nsamples = sample.n_cols;
    sample.col(0) = theta_init.theta;
    candidates.col(0) = theta_init.theta;
    double old_rllh = relative_log_likelihood(model, im, theta_init);
    IdxT phase = 0;
    for(IdxT n=1; n<Nsamples; n++){
        auto can_theta = sample.col(n-1);
        model.sample_mcmc_candidate_theta(phase, can_theta);
        candidates.col(n) = can_theta;
        if(!model.theta_in_bounds(can_theta)) { //OOB so stay put
            sample.col(n) = sample.col(n-1);
            continue;
        }
        double can_rllh = relative_log_likelihood(model, im, can_theta);
        phase++;
        double alpha = std::min(1.,exp(can_rllh-old_rllh));
        if(rng_manager.randu() < alpha) {
            sample.col(n) = can_theta;
            old_rllh = can_rllh;
        } else {  //reject: record old point again
            sample.col(n) = sample.col(n-1);
        }
    }
}

} /* namespace mappel */

#endif /* _MCMC_H */
