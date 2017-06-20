/** @file BlinkModel.h.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class declaration and inline and templated functions for BlinkModel.h.
 */

#ifndef _BLINKMODEL_H
#define _BLINKMODEL_H
#include "rng.h"
#include "estimator.h"

class BlinkModel {
protected:
    /* Hyperparameters */
    double beta_D0=0.00125; /**< The shape parameter for the Beta prior Duty Di  */
    double beta_D1=0.00250; /**< The shape parameter for the Beta prior Duty Di  */

    BetaRNG D_dist;

    double log_prior_D_const; /**< This is -2*lgamma(beta_D)-lgamma(2*beta_D) */
    double candidate_eta_D; /**< The standard deviation for the normally distributed pertebation to theta_Di in the random walk MCMC sampling */
public:
    BlinkModel(double candidate_sample_dist_ratio);
};

template<class Model>
typename Model::ParamVecT
sample_blink_posterior(Model &model, const typename Model::ImageT &im, int max_samples,
                                   typename Model::Stencil &theta_init, RNG &rng)
{
    UnitRNG u;
    auto sample=model.make_param_vec(max_samples);
    sample.col(0)=theta_init.theta;
    double old_rllh=log_likelihood(model, im, theta_init);
//     double old_rllh=relative_log_likelihood(model, im, theta_init);
    typename Model::ModelImage model_image(model, im);
    int phase=0;
    for(int n=1;n<max_samples;n++){
        typename Model::ParamT can_theta=sample.col(n-1);
        model.sample_candidate_theta(phase, rng, can_theta);
        if(!model.theta_in_bounds(can_theta)) { //OOB so stay put
            sample.col(n)=sample.col(n-1);
            continue;
        }
        double can_rllh=model.compute_candidate_rllh(phase, im, can_theta, model_image);
        phase++;
        assert(std::isfinite(can_rllh));
        double alpha=std::min(1.,exp(can_rllh-old_rllh));
        double r=u(rng);
//         std::cout<<"CanLLH:"<<can_rllh<<" OldLLH:"<<old_rllh<<" Alpha:"<<alpha<<" r:"<<r;
        if(r < alpha) {
//             std::cout<<" [ACCEPTED]"<<std::endl;
            sample.col(n)=can_theta;
            old_rllh=can_rllh;
        } else { //reject: record old point again
//             std::cout<<" [REJECTED]"<<std::endl;
            sample.col(n)=sample.col(n-1);
        }
    }
    return sample;
}


template<class Model>
void sample_blink_posterior_debug(Model &model, const typename Model::ImageT &im,
                                typename Model::Stencil &theta_init,
                                typename Model::ParamVecT &sample,
                                typename Model::ParamVecT &candidates,
                                RNG &rng)
{
    UnitRNG u;
    int Nsamples=sample.n_cols;
    sample.col(0)=theta_init.theta;
    candidates.col(0)=theta_init.theta;
    double old_rllh=log_likelihood(model, im, theta_init);
//     double old_rllh=relative_log_likelihood(model, im, theta_init);
    typename Model::ModelImage model_image(model, im);
    int phase=0;
    for(int n=1;n<Nsamples;n++){
        typename Model::ParamT can_theta=sample.col(n-1);
        model.sample_candidate_theta(phase, rng, can_theta);
        candidates.col(n)=can_theta;
        if(!model.theta_in_bounds(can_theta)) { //OOB so stay put
            sample.col(n)=sample.col(n-1);
            continue;
        }
        double can_rllh=model.compute_candidate_rllh(phase, im, can_theta, model_image);
        phase++;
        assert(std::isfinite(can_rllh));
        double alpha=std::min(1.,exp(can_rllh-old_rllh));
        double r=u(rng);
//         std::cout<<"N:"<<n<<" CanLLH:"<<can_rllh<<" OldLLH:"<<old_rllh<<" Alpha:"<<alpha<<" r:"<<r;
        if(r < alpha) {
//             std::cout<<" [ACCEPTED]"<<std::endl;
            sample.col(n)=can_theta;
            old_rllh=can_rllh;
        } else { //reject: record old point again
//             std::cout<<" [REJECTED]"<<std::endl;
            sample.col(n)=sample.col(n-1);
        }
//         std::cout<<"Sample("<<n<<"):"<<sample(1,n)<<","<<sample(2,n)<<","<<sample(3,n)<<","<<sample(4,n)<<std::endl;
    }
}

template<class Model>
typename Model::Stencil
blink_anneal(Model &model,  RNG &rng, const typename Model::ImageT &im,
                  const typename Model::Stencil &theta_init,
                  typename Model::ParamVecT &sequence,
                  int max_iterations, double T, double cooling_rate)
{
    NewtonRaphsonMLE<Model> nr(model);
    UnitRNG u;
    int niters=max_iterations*model.num_candidate_sampling_phases;
    sequence=model.make_param_vec(niters+1);
    sequence.col(0)=theta_init.theta;
    double old_rllh=relative_log_likelihood(model, im, theta_init);
    double max_rllh=old_rllh;
    int max_idx=0;
    typename Model::Stencil max_s;
    int naccepted=1;
    typename Model::ModelImage model_image(model, im);
    for(int n=1; n<niters; n++){
        typename Model::ParamT can_theta=sequence.col(naccepted-1);
        model.sample_candidate_theta(n, rng, can_theta);
//         std::cout<<"N:"<<n;
//         print_vec_row(std::cout,can_theta,"CanTheta:", 20, TERM_DIM_CYAN);
        if(!model.theta_in_bounds(can_theta)) { //reject: OOB
            n--;
            continue;
        }
        double can_rllh=model.compute_candidate_rllh(n, im, can_theta, model_image);
        assert(std::isfinite(can_rllh));
        if(can_rllh < old_rllh && u(rng)>exp((can_rllh-old_rllh)/T) ){//reject
            continue;
        }
        //Accept
        T/=cooling_rate;
//         std::cout<<"T: "<<T<<"\n";
        sequence.col(naccepted)=can_theta;
        old_rllh=can_rllh;
        if(can_rllh>max_rllh){
            max_rllh=can_rllh;
            max_idx=naccepted;
        }
        naccepted++;
    }
    //Run a NR maximization
    nr.local_maximize(im, model.make_stencil(sequence.col(max_idx)), max_s, max_rllh);
    //Fixup sequence to return
    sequence.resize(sequence.n_rows, naccepted+1);
    sequence.col(naccepted)=max_s.theta;
    return max_s;
}

#endif /* _BLINKMODEL_H */
