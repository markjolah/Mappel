
/** @file evaluate.h
 * @author Mark J. Olah (mjo@cs.unm.edu)
 * @date 01-15-2014
 * @brief Templated friend functions for evaulating model estimators
 *
 * The idea is for efficiency we want to avoid using a virtual base class to encapsulate
 * code common to many PSFModels.  Thus Models are unrelated by inheritance, and we use
 * "duck" typing to make each act the same way in the templated evaluations functions.
 * This moves the polymorphism to compile time instead of run-time and will eliminate
 * the runtime overhead associated with virtual functions
 */
#ifndef _EVALUATE_H
#define _EVALUATE_H


#include <iostream>
#include <memory>
#include "stats.h"
#include "estimator.h"
#include "mcmc.h"

using std::cout;
using std::endl;
using namespace mappel;
template<class Model>
void estimate_stack(Estimator<Model> &estimator, int count)
{
    int s=64;
    char str[100];
    Model &model=estimator.model;
    cout<<"Model: "<<model<<endl;
    auto theta=model.make_param_vec(count);
    sample_prior_stack(model,theta);
    auto ims=model.make_image_stack(count);
    simulate_image_stack(model,theta,ims);
    auto theta_est=model.make_param_vec(count);
    auto crlb=model.make_param_vec(count);
    auto llh_est=VecT(count);
    auto llh=VecT(count);
    estimator.estimate_stack(ims,theta_est, crlb,llh_est);
    log_likelihood_stack(model,ims,theta,llh);
    arma::mat error=theta-theta_est;
    cout<<std::setw(64)<<""<<"\033["<<TERM_WHITE<<"m"<<"Model:"<<model.name()
        <<" Estimator:"<<estimator.name()<<"\033[0m"<<endl;
    arma::vec rmserror=arma::sqrt(arma::mean(error%error,1));
    snprintf(str, s, "<LLH>:%g <Theta>:",arma::mean(llh));
    print_vec_row(cout,arma::mean(theta,1),str,s,TERM_CYAN);
    snprintf(str, s, "<LLHEst>:%g <ThetaEst>:",arma::mean(llh_est));
    print_vec_row(cout,arma::mean(theta_est,1),str,s,TERM_YELLOW);
    snprintf(str, s, "<LLHDelta>:%g <Error>:",arma::mean(llh_est-llh));
    print_vec_row(cout,arma::mean(error,1),str,s,TERM_MAGENTA);
    print_vec_row(cout,rmserror,"RMSE:",s,TERM_RED);
    cout<<"Estimator statistics: "<<endl<<estimator<<endl;
}

template<class Model>
void estimate_stack_posterior(Model &model, int count)
{
    int s=64;
    char str[100];
    int iter=1000;
    cout<<"Model: "<<model<<endl;
    auto theta=model.make_param_vec(count);
    sample_prior_stack(model,theta);
    auto ims=model.make_image_stack(count);
    simulate_image_stack(model,theta,ims);
    auto theta_est=model.make_param_vec(count);
    auto llh_est=VecT(count);
    auto llh=VecT(count);
    arma::cube cov(model.num_params, model.num_params, count);
    evaluate_posterior_stack(model, ims, iter, theta_est, cov);
    log_likelihood_stack(model, ims, theta_est, llh_est);
    log_likelihood_stack(model,ims,theta,llh);
    arma::mat error=theta-theta_est;
    cout<<std::setw(64)<<""<<"\033["<<TERM_WHITE<<"m"<<"Model:"<<model.name()
        <<" Estimator:Posterior\033[0m"<<endl;
    arma::vec rmserror=arma::sqrt(arma::mean(error%error,1));
    snprintf(str, s, "<LLH>:%g <Theta>:",arma::mean(llh));
    print_vec_row(cout,arma::mean(theta,1),str,s,TERM_CYAN);
    snprintf(str, s, "<LLHEst>:%g <ThetaEst>:",arma::mean(llh_est));
    print_vec_row(cout,arma::mean(theta_est,1),str,s,TERM_YELLOW);
    snprintf(str, s, "<LLHDelta>:%g <Error>:",arma::mean(llh_est-llh));
    print_vec_row(cout,arma::mean(error,1),str,s,TERM_MAGENTA);
    print_vec_row(cout,rmserror,"RMSE:",s,TERM_RED);
}


template<template<class> class Estimator, class Model>
void evaluate_single(Estimator<Model> &estimator, const typename Model::ParamT &theta)
{
    using std::cout;
    using std::endl;
    int s=64;
    char str[100];
    Model &model=estimator.model;
    auto stencil=model.make_stencil(theta);
    cout<<"Stencil: "<<stencil<<endl;
    cout<<"Model: "<<model<<endl;
    auto im=simulate_image(model, stencil);
    double llh=log_likelihood(model, im, stencil);
    auto crlb=cr_lower_bound(model,stencil);
    cout<<std::setw(2*s)<<std::setfill('=')<<""<<endl<<std::setfill(' ');
//     auto model_im=model_image(model, stencil);
//     print_image(cout, model_im);
    typename Model::ParamVecT sequence;
    VecT sequence_rllh;
    print_image(cout, im);
    snprintf(str, s, "LLH:%g Theta:",llh);
    print_vec_row(cout, theta, str, s, TERM_MAGENTA);
    print_vec_row(cout, arma::sqrt(crlb), "sqrtCRLB:", s, TERM_BLUE);
    auto dummy_theta = model.make_param();
    dummy_theta.zeros();
    auto theta_init=model.initial_theta_estimate(im,dummy_theta);
    assert(model.theta_in_bounds(theta_init.theta));
    assert(theta_init.derivatives_computed);
    snprintf(str, s, "LLH:%g ThetaInit:",log_likelihood(model,im,theta_init));
    print_vec_row(cout, theta_init.theta, str, s, TERM_CYAN);
    auto theta_est=model.make_param();
    VecT theta_bad_init = theta_init.theta;
    theta_bad_init(0)=2;
    theta_bad_init(1)=theta_bad_init(1)/2;
    theta_bad_init(2)=theta_bad_init(2)/5;
    estimator.estimate_debug(im, theta_bad_init, theta_est, crlb, llh, sequence, sequence_rllh);
    snprintf(str, s, "LLH:%.9g ETheta[%s]:",llh, estimator.name().c_str());
    print_vec_row(cout, theta_est, str, s, TERM_YELLOW);
    snprintf(str, s, "Error[%s]:", estimator.name().c_str());
    print_vec_row(cout, theta-theta_est, str, s, TERM_RED);

//     for(unsigned i=0;i<sequence.n_cols;i++){
//         snprintf(str, s, "RLLH:%.9g Seq[%s](%i):",sequence_rllh(i), estimator.name().c_str(),i);
//         print_vec_row(cout, sequence.col(i), str, s, TERM_DIM_MAGENTA);
//     }
    int max_samples=4000;
    auto posterior_mean=model.make_param();
    auto posterior_cov=model.make_param_mat();
    auto post_sequence=model.make_param_vec(max_samples);
    auto post_sequence_rllh=VecT(max_samples);
    auto candidates=model.make_param_vec(max_samples);
    auto candidate_rllh=VecT(max_samples);

    evaluate_posterior_debug(model, im, max_samples, posterior_mean, posterior_cov,
                             post_sequence, post_sequence_rllh,candidates,candidate_rllh);
    auto posterior_mean_llh=log_likelihood(model, im, posterior_mean);
    snprintf(str, s, "LLH:%.9g EstPMean:",posterior_mean_llh);
    print_vec_row(cout, posterior_mean, str, s, TERM_DIM_YELLOW);
    print_vec_row(cout, theta-posterior_mean, "Error[PostMean]:", s, TERM_DIM_RED);
    print_vec_row(cout, arma::sqrt(posterior_cov.diag()).eval(), "PostSE:", s, TERM_RED);

//     for(unsigned i=0;i<post_sequence.n_cols;i++){
//         snprintf(str, s, "RLLH:%.9g Seq[Posterior](%i):",post_sequence_rllh(i), i);
//         print_vec_row(cout, post_sequence.col(i), str, s, TERM_DIM_MAGENTA);
//     }

    cout<<"Model: "<<model<<endl;
    cout<<estimator<<endl;
    
    //TODO: Debug this next line
//     cout<<"Estimator statistics: "<<endl<<estimator<<endl;
    
    
}

/*
template<template<class> class Estimator, class Model>
void evaluate_single(Estimator<Model> &estimator, const typename Model::ParamT &theta, RNG &rng)
{
    using std::cout;
    using std::endl;
    int s=64;
    char str[100];
    Model &model=estimator.model;
    auto stencil=model.make_stencil(theta);
    cout<<"Stencil: "<<stencil<<endl;
    cout<<"Model: "<<model<<endl;
    auto im=simulate_image(model, stencil,rng);
    double llh=log_likelihood(model, im,stencil);
    auto crlb=cr_lower_bound(model,stencil);
    cout<<std::setw(2*s)<<std::setfill('=')<<""<<endl<<std::setfill(' ');
    //     auto model_im=model_image(model, stencil);
    //     print_image(cout, model_im);
    print_image(cout, im);
    snprintf(str, s, "LLH:%g Theta:",llh);
    print_vec_row(cout, theta, str, s, TERM_MAGENTA);
    //     print_vec_row(cout, crlb, "CRLB:", s, TERM_BLUE);
    auto theta_init=model.initial_theta_estimate(im);
    snprintf(str, s, "LLH:%g ThetaInit:",log_likelihood(model,im,theta_init));
    print_vec_row(cout, theta_init.theta, str, s, TERM_CYAN);
    auto theta_est=model.make_param();
    estimator.estimate(im, theta_est, crlb, llh);
    snprintf(str, s, "LLH:%.9g ETheta[%s]:",llh, estimator.name().c_str());
    print_vec_row(cout, theta_est, str, s, TERM_YELLOW);
    snprintf(str, s, "Error[%s]:", estimator.name().c_str());
    print_vec_row(cout, theta-theta_est, str, s, TERM_RED);
    cout<<"Model: "<<model<<endl;
    cout<<"Estimator statistics: "<<endl<<estimator<<endl;
}*/

template <class Model>
void compare_estimators_single(Model &model, const typename Model::ParamT &theta)
{
    using std::cout;
    using std::endl;
    int s=65;
    char str[100];
    auto stencil=model.make_stencil(theta);
    auto im=simulate_image(model,stencil);
    double llh=log_likelihood(model,im,stencil);
    auto crlb=cr_lower_bound(model,stencil);
    cout<<std::setw(2*s)<<std::setfill('=')<<""<<endl<<std::setfill(' ');
    print_image(cout, im);
    snprintf(str, s, "LLH:%g Theta:",llh);
    print_vec_row(cout, theta, str, s, TERM_WHITE);
    print_vec_row(cout, crlb, "CRLB", s, TERM_CYAN);
    std::vector<std::shared_ptr<Estimator<Model>>> estimators;
    for(auto name: model.estimator_names) {
        try {
            auto estimator=make_estimator(model, name);
            if(!estimator){
                cout<<"Bad estimator name: "<<name<<endl;
                continue;
            }
            estimators.push_back(estimator);
//             cout<<"Estimator Name: "<<name<<endl;
//             cout<<"Estimator: "<<*estimator<<endl;
            auto theta_est=model.make_param();
            estimator->estimate(im, theta_est, crlb, llh);
            snprintf(str, s, "LLH:%.9g Theta[%s]:",llh,name.c_str());
            print_vec_row(cout, theta_est, str, s, TERM_YELLOW);
            typename Model::ParamT error=theta-theta_est;
            double se=arma::dot(error,error);
            double pse=error(0)*error(0)+error(1)*error(1);
            snprintf(str, s, "SE_pos:%.9g SE:%.9g |Error[%s]|:",pse, se, name.c_str());
            print_vec_row(cout, abs(error), str, s, TERM_RED);
            snprintf(str, s, "CRLB[%s]:",name.c_str());
            print_vec_row(cout, crlb, str, s, TERM_BLUE);
        } catch (const MaximizerNotImplementedException& e) {
            continue;
        }
    }
    auto mean=model.make_param();
    auto cov=model.make_param_mat();
    evaluate_posterior(model, im, 2000, mean, cov);
    llh=log_likelihood(model,im,mean);
    snprintf(str, s, "LLH:%.9g Est<Posterior>:",llh);
    print_vec_row(cout, mean, str, s, TERM_YELLOW);
    typename Model::ParamT error=theta-mean;
    double se=arma::dot(error,error);
    double pse=error(0)*error(0)+error(1)*error(1);
    snprintf(str, s, "SE_pos:%.9g SE:%.9g |Error[Posterior]|:",pse, se);
    print_vec_row(cout, abs(error), str, s, TERM_RED);
    print_vec_row(cout, cov.diag().eval(), "VAR[Posterior]:", s, TERM_BLUE);
    for (auto estimator: estimators) cout<<*estimator<<endl;
}

template <class Model>
void compare_estimators(Model &model, const typename Model::ParamT &theta, int ntrials)
{
    using std::cout;
    using std::endl;
    int s=65;
    char str[100];
    auto stencil=model.make_stencil(theta);
    auto ims=model.make_image_stack(ntrials);
    simulate_image_stack(model,theta,ims);
    auto llh=VecT(ntrials);
    log_likelihood_stack(model, ims, theta, llh);
    auto crlb=cr_lower_bound(model,stencil);
    cout<<std::setw(2*s)<<std::setfill('=')<<""<<endl<<std::setfill(' ');
    snprintf(str, s, "True <LLH>:%.9g Theta:",arma::mean(llh));
    print_vec_row(cout, theta, str, s, TERM_WHITE);
    print_vec_row(cout, crlb, "CRLB", s, TERM_BLUE);
    std::vector<std::shared_ptr<Estimator<Model>>> estimators;
    for(auto name: model.estimator_names) {
        try {
            auto estimator=make_estimator(model, name);
            estimators.push_back(estimator);
            if(!estimator){
                cout<<"Bad estimator name: "<<name<<endl;
                continue;
            }
            auto theta_est=model.make_param_vec(ntrials);
            auto crlb_est=model.make_param_vec(ntrials);
            auto llh_est=VecT(ntrials);
            estimator->estimate_stack(ims, theta_est, crlb_est, llh_est);
            snprintf(str, s, "<LLH>:%.9g <Theta[%s]>:",arma::mean(llh_est),name.c_str());
            print_vec_row(cout, arma::mean(theta_est,1), str, s, TERM_YELLOW);
            auto error=theta_est;
            error.each_col()-=theta;
            double mse=arma::accu(error%error)/ntrials;
            double pmse=arma::mean(error.row(0)%error.row(0)+error.row(1)%error.row(1));
            snprintf(str, s, "R<SE_pos>:%.9g R<SE>:%.9g <Error[%s]>:",sqrt(pmse), sqrt(mse), name.c_str());
            print_vec_row(cout, arma::mean(error,1), str, s, TERM_RED);
            snprintf(str, s, "RMSE[%s]>:",name.c_str());
            print_vec_row(cout, arma::sqrt(arma::mean(error%error,1)), str, s, TERM_MAGENTA);
            snprintf(str, s, "<CRLB[%s]>:",name.c_str());
            print_vec_row(cout, arma::mean(crlb_est,1), str, s, TERM_BLUE);
        } catch (const MaximizerNotImplementedException& e) {
            continue;
        }
        
    }
    auto mean=model.make_param_vec(ntrials);
    arma::cube cov(model.num_params, model.num_params, ntrials);
    evaluate_posterior_stack(model, ims, 3000, mean, cov);
    auto llh_est=VecT(ntrials);
    log_likelihood_stack(model, ims, mean, llh_est);
    snprintf(str, s, "LLH:%.9g Est<Posterior>:",arma::mean(llh_est));
    print_vec_row(cout, arma::mean(mean,1), str, s, TERM_YELLOW);
    auto error=mean;
    error.each_col()-=theta;
    double mse=arma::accu(error%error)/ntrials;
    double pmse=arma::mean(error.row(0)%error.row(0)+error.row(1)%error.row(1));
    snprintf(str, s, "R<SE_pos>:%.9g R<SE>:%.9g <Error[Posterior]>:",sqrt(pmse), sqrt(mse));
    print_vec_row(cout, arma::mean(error,1), str, s, TERM_RED);
    print_vec_row(cout, arma::sqrt(arma::mean(error%error,1)), "RMSE[Posterior]:", s, TERM_MAGENTA);
    auto var=model.make_param_vec(ntrials);
    for(int n=0; n<ntrials;n++) var.col(n)=cov.slice(n).diag();
    print_vec_row(cout, arma::mean(var,1), "PosteriorVar:", s, TERM_BLUE);
    for (auto estimator: estimators) cout<<*estimator<<endl;
}


template<template<class> class Estimator, class Model>
void point_evaluate_estimator(Estimator<Model> &estimator, int ntrials,
                              const typename Model::ParamT &theta,
                              typename Model::ParamT &efficiency,
                              double &unbiased_p_val)
{
    using std::cout;
    using std::endl;
    int s=65;
    char str[100];
    Model &model=estimator.model;
    auto stencil=model.make_stencil(theta);
    auto ims=model.make_image_stack(ntrials);
    simulate_image_stack(model,theta,ims);
    auto crlb=cr_lower_bound(model,stencil);
    cout<<std::setw(2*s)<<std::setfill('=')<<""<<endl<<std::setfill(' ');
    print_vec_row(cout, crlb, "CRLB", s, "1;34");

    auto theta_est=model.make_param_vec(ntrials);
    auto crlb_est=model.make_param_vec(ntrials);
    auto llh_est=VecT(ntrials);
    estimator.estimate_stack(ims, theta_est, crlb_est, llh_est);
    for(int i=0;i<ntrials;i++) print_vec_row(cout, theta_est.col(i), "Theta_est:",15,TERM_BLUE);
   
    auto error=theta_est;
    error.each_col()-=theta;
    typename Model::ParamT error_var=arma::var(error,0,1);
    efficiency=crlb/error_var;
    
    cout<<"Number of trials: "<<ntrials<<endl;
    cout<<"Estimator: "<<estimator.name()<<endl;
    auto llhs_true=VecT(ntrials);
    log_likelihood_stack(model, ims, theta, llhs_true);
    snprintf(str, s, "True <LLH:%.9g> Theta:",arma::mean(llhs_true));
    print_vec_row(cout, theta, str, s, TERM_WHITE);
    snprintf(str, s, "<LLH>:%.9g <Theta[%s]>:",arma::mean(llh_est),estimator.name().c_str());
    print_vec_row(cout, arma::mean(theta_est,1), str, s, TERM_RED);
    print_vec_row(cout, arma::mean(error,1), "Mean(Error):", s,TERM_YELLOW);
    print_vec_row(cout, error_var, "Var(Error):", s, TERM_MAGENTA);
    print_vec_row(cout, crlb, "CRLB(Theta):", s, TERM_BLUE);
    print_vec_row(cout, efficiency, "Efficiency(Theta):", s,TERM_CYAN);
    cout<<"Cov matrix: "<< arma::cov(theta_est.t())<<endl;
    unbiased_p_val=test_mean_nD_one_sided(ntrials, arma::mean(theta_est,1),
                                          arma::cov(theta_est.t()), theta);
    cout<<"Unbiased Estimator p-val (chance we would be this far off): "<<unbiased_p_val<<endl; //small is good
    cout<<"---------------------------------------------------------------"<<endl<<endl;
}

template<template<class> class Estimator, class Model>
void evaluate_estimator(Estimator<Model> &estimator, int num_trials, int num_points,
                        typename Model::ParamVecT &efficiencies,
                        VecT &p_vals)
{
    using std::cout;
    using std::endl;
    Model &model=estimator.model;
    int i,j, idx;
    std::string pname;
    auto thetas=model.make_param_vec(num_points);
    sample_prior_stack(model, thetas);

    auto eff=model.make_param();
    for(int i=0; i<num_points; i++){
        point_evaluate_estimator(estimator, num_trials, thetas.col(i), eff, p_vals(i));
        efficiencies.col(i)=eff;
    }
    arma::uvec p_vals_sidx=sort_index(p_vals);
    arma::umat efficiency_sidx(model.num_params, num_points);
    for(i=0; i<model.num_params; i++){
        efficiency_sidx.row(i) = sort_index(efficiencies.row(i)).t();
    }

    cout<<"=========================== Evaluating Estimator ==============================="<<endl;
    cout<<"Model: "<<model<<endl;
    cout<<"Estimator: "<<estimator<<endl;
    cout<<"Number of point estimates: "<<num_points<<endl;
    cout<<"Number of trials per estimate: "<<num_trials<<endl<<endl;
    cout<<"+++ Is the estimator unbiased? +++"<<endl;
    idx=p_vals_sidx(0);
    cout<<"  [p_val] Min:"<<p_vals(idx)<<" @Theta:"<<thetas.col(idx).t();
    idx=p_vals_sidx(num_points-1);
    cout<<"  [p_val] Max:"<<p_vals(idx)<<" @Theta:"<<thetas.col(idx).t();
    cout<<"  [p_val] Mean:"<<arma::mean(p_vals)<<" Var:"<<arma::var(p_vals,0,1)<<endl;
    cout<<"  [p_val] Frac<0.01: "<<arma::sum(p_vals<0.01)/double(num_points)
        <<" Frac<0.05: "<<arma::sum(p_vals<0.05)/(double)num_points<<endl;
    cout<<"  [p_val] 10 Smallest "<<endl;
    for(i=0;i<10;i++){
        idx=p_vals_sidx(i);
        cout<<"        #"<<i+1<<": "<<p_vals(idx)<<" @Theta:"<<thetas.col(idx).t();
    }
    cout<<endl;
    cout<<"+++ Is the estimator efficient? +++"<<endl;
    for(j=0; j<model.num_params; j++){
        pname=model.param_names[j];
        idx=efficiency_sidx(j,0);
        cout<<"  [eff-"<<pname<<"] Min:"<<efficiencies(j,idx)<<" @Theta:"<<thetas.col(idx).t();
        idx=efficiency_sidx(j,num_points-1);
        cout<<"  [eff-"<<pname<<"] Max:"<<efficiencies(j,idx)<<" @Theta:"<<thetas.col(idx).t();
        cout<<"  [eff-"<<pname<<"] Mean:"<<arma::mean(efficiencies.row(j))<<" Var:"<<arma::var(efficiencies.row(j),0,1)<<endl;
        cout<<"  [eff-"<<pname<<"] Frac<0.90: "<<arma::sum(efficiencies.row(j)<0.90)/(double)num_points
        <<" Frac<0.95: "<<arma::sum(efficiencies.row(j)<0.95)/double(num_points)<<endl;
        cout<<"  [eff-"<<pname<<"] 10 Smallest "<<endl;
        for(i=0;i<10;i++){
            idx=efficiency_sidx(j,i);
            cout<<"        #"<<i+1<<": "<<efficiencies(j,idx)<<" @Theta:"<<thetas.col(idx).t();
        }
    }
    cout<<"=================================================================================="<<endl<<endl;
}

#endif /* _EVALUATE_H */
