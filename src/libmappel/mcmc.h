/** @file mcmc.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 05-22-2015
 * @brief Templated MCMC methods for posterior estimation
 */

#ifndef _MAPPEL_MCMC_H
#define _MAPPEL_MCMC_H

#include <cmath>
#include "util.h"
#include "rng.h"
#include <trng/uniform01_dist.hpp>

namespace mappel {

namespace mcmc {

IdxT num_oversample(IdxT Nsample, IdxT Nburnin, IdxT thin);
MatT thin_sample(MatT &sample, IdxT Nburnin, IdxT thin);    
void thin_sample(const MatT &sample, const VecT &sample_rllh, IdxT Nburnin, IdxT thin, 
                 MatT &subsample, VecT &subsample_rllh);

inline
void estimate_sample_posterior(const MatT &sample, VecT &theta_posterior_mean, MatT &theta_posterior_cov)
{
    theta_posterior_mean = arma::mean(sample, 1);
    theta_posterior_cov = arma::cov(sample.t());
}


template <class Model>
void sample_posterior(Model &model, const ModelDataT<Model> &im, const StencilT<Model> &theta_init, 
                      MatT &sample, VecT &sample_rllh)
{
    auto &rng = model.get_rng_generator();
    UniformDistT uniform;
    IdxT Nsamples = sample.n_cols;
    sample_rllh.set_size(Nsamples);
    sample.col(0) = theta_init.theta;
    sample_rllh(0) = methods::objective::rllh(model, im, theta_init);
    IdxT phase = 0;
    std::cout<<"Nsamples: "<<Nsamples<<"\n";
    for(IdxT n=1;n<Nsamples;n++){
        ParamT<Model> can_theta = sample.col(n-1);
        model.sample_mcmc_candidate_theta(phase++, can_theta);
        if(!model.theta_in_bounds(can_theta)) { //Reject: out-of-bounds
            sample.col(n) = sample.col(n-1);
            sample_rllh(n) = sample_rllh(n-1);
            continue;
        }
        double can_rllh = methods::objective::rllh(model, im, can_theta);
        double alpha = std::min(1., exp(can_rllh - sample_rllh(n-1)));
        if(uniform(rng) < alpha) {
            //Accept
            sample.col(n) = can_theta;
            sample_rllh(n) = can_rllh;
        } else { 
            //Reject: record old point again
            sample.col(n) = sample.col(n-1);
            sample_rllh(n) = sample_rllh(n-1);
        }
    }
}

template <class Model>
void sample_posterior_debug(Model &model, const ModelDataT<Model> &im, const StencilT<Model> &theta_init, 
                            MatT &sample, VecT &sample_rllh, MatT &candidate, VecT &candidate_rllh)
{
    auto &rng = model.get_rng_generator();
    UniformDistT uniform;
    IdxT Nsamples = sample.n_cols;
    sample_rllh.set_size(Nsamples);
    candidate.set_size(model.get_num_params(), Nsamples);
    candidate_rllh.set_size(Nsamples);
    sample.col(0) = theta_init.theta;
    sample_rllh(0) = methods::objective::rllh(model, im, theta_init);
    candidate.col(0) = sample.col(0);
    candidate_rllh(0) = sample_rllh(0);
    IdxT phase = 0;
    phase++;
    for(IdxT n=1;n<Nsamples;n++){
        ParamT<Model> can_theta = sample.col(n-1);
        model.sample_mcmc_candidate_theta(phase++, can_theta);
        candidate.col(n) = can_theta;
        if(!model.theta_in_bounds(can_theta)) { //Reject: out-of-bounds
            sample.col(n) = sample.col(n-1);
            sample_rllh(n) = sample_rllh(n-1);
            candidate_rllh(n) = -INFINITY;
            continue;
        }
        double can_rllh = methods::objective::rllh(model, im, can_theta);
        candidate_rllh(n) = can_rllh;
        double alpha = std::min(1., exp(can_rllh - sample_rllh(n-1)));
        if(uniform(rng) < alpha) {
            //Accept
            sample.col(n) = can_theta;
            sample_rllh(n) = can_rllh;
        } else { 
            //Reject: record old point again
            sample.col(n) = sample.col(n-1);
            sample_rllh(n) = sample_rllh(n-1);
        }
    }
}

} /* namespace mappel::mcmc */    
} /* namespace mappel */

#endif /* _MAPPEL_MCMC_H */
