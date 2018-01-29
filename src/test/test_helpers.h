
/** @file test_helpers.h
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
#ifndef _MAPPEL_TEST_HELPERS_H
#define _MAPPEL_TEST_HELPERS_H


#include <iostream>
#include <memory>

using std::cout;
using std::endl;


namespace mappel {

namespace test {

template<class Model>
typename std::enable_if<std::is_base_of<Gauss1DModel,Model>::value, typename Model::ParamT>::type
read_theta(Model &model, int argc, const char *argv[])
{
    // args: x I bg
    int n=3;
    double bg    = argc>=n-- ? strtod(argv[n],NULL) : -1;
    int I        = argc>=n-- ? atoi(argv[n])   : -1;
    double x     = argc>=n-- ? strtod(argv[n],NULL) : -1;
    auto theta = model.sample_prior();
    if(x>=0) theta(0)=x;
    if(I>=0) theta(1)=I;
    if(bg>=0) theta(2)=bg;
    return theta;
}


template<class Model>
typename std::enable_if<std::is_base_of<Gauss1DsModel,Model>::value, typename Model::ParamT>::type
read_theta(Model &model, int argc, const char *argv[])
{
    // args: x I bg sigma
    int n=4;
    double sigma = argc>=n-- ? strtod(argv[n],NULL) : -1;
    double bg    = argc>=n-- ? strtod(argv[n],NULL) : -1;
    int I        = argc>=n-- ? atoi(argv[n])   : -1;
    double x     = argc>=n-- ? strtod(argv[n],NULL) : -1;
    auto theta = model.sample_prior();
    if(x>=0) theta(0)=x;
    if(I>=0) theta(1)=I;
    if(bg>=0) theta(2)=bg;
    if(sigma>=0) theta(3)=sigma;
    return theta;
}

// template<class Model, typename=std::enable_if<std::is_base_of<Gauss2DModel>::value>::type >
// typename Model::ParamT read_theta(Model &model, int argc, const char *argv[])
// {
//     // args: x y I bg sigma
//     int n=5;
//     double sigma = argc>=n-- ? strtod(argv[n],NULL) : -1;
//     double bg    = argc>=n-- ? strtod(argv[n],NULL) : -1;
//     int I        = argc>=n-- ? atoi(argv[n])   : -1;
//     double y     = argc>=n-- ? strtod(argv[n],NULL) : -1;
//     double x     = argc>=n-- ? strtod(argv[n],NULL) : -1;
//     auto theta=model.sample_prior();
//     if(x>=0) theta(0)=x;
//     if(y>=0) theta(1)=y;
//     if(I>=0) theta(2)=I;
//     if(bg>=0) theta(3)=bg;
//     if(model.get_num_params()>4 && sigma>=0) theta(4)=sigma;
//     return theta;
// }

// template<class Model>
// typename Model::ParamT read_HS_theta(Model &model, int argc, const char *argv[])
// {
//     // args: x y L I bg sigma sigmaL
//     int n=7;
//     double sigmaL= argc>=n-- ? strtod(argv[n],NULL) : -1;
//     double sigma = argc>=n-- ? strtod(argv[n],NULL) : -1;
//     double bg    = argc>=n-- ? strtod(argv[n],NULL) : -1;
//     int I        = argc>=n-- ? atoi(argv[n])   : -1;
//     double L     = argc>=n-- ? strtod(argv[n],NULL) : -1;
//     double y     = argc>=n-- ? strtod(argv[n],NULL) : -1;
//     double x     = argc>=n-- ? strtod(argv[n],NULL) : -1;
//     auto theta=model.sample_prior();
//     if(x>=0) theta(0)=x;
//     if(y>=0) theta(1)=y;
//     if(L>=0) theta(2)=L;
//     if(I>=0) theta(3)=I;
//     if(bg>=0) theta(4)=bg;
//     if(sigma>=0) theta(5)=sigma;
//     if(sigmaL>=0) theta(6)=sigmaL;
// //     if(model.num_params>5 && sigma>=0) theta(4)=sigma;
//     return theta;
// }

template<class Model>
void estimate_stack(Model &model, std::string method, int count)
{
    int s=64;
    char str[100];
    
    cout<<"Model: "<<model<<endl;
    
    auto theta_stack = model.make_param_stack(count);
    methods::sample_prior_stack(model,theta_stack);
    auto im_stack = model.make_image_stack(count);
    methods::simulate_image_stack(model,theta_stack,im_stack);
    auto theta_est_stack = model.make_param_stack(count);
    auto obsI_stack = model.make_param_mat_stack(count);
    VecT rllh_stack(count);
    StatsT stats;
    methods::estimate_max_stack(model, im_stack,method,theta_est_stack,rllh_stack,obsI_stack,stats);
    VecT true_rllh_stack(count);
    methods::objective::rllh_stack(model, im_stack,theta_stack, true_rllh_stack);
    
    double mean_rllh_delta = arma::mean(rllh_stack - true_rllh_stack);
    
    MatT error = theta_est_stack - theta_stack;
    VecT rmserror = arma::sqrt(arma::mean(error%error,1));
    
    cout<<std::setw(64)<<""<<"\033["<<TERM_WHITE<<"m"<<"Model:"<<model.name
        <<" Estimator:"<<method<<"\033[0m"<<endl;
    
    snprintf(str, s, "<LLHDelta>:%g RMSE:",mean_rllh_delta);
    print_vec_row(cout,rmserror,str,s,TERM_MAGENTA);
    
}

template<class Model>
void estimate_stack_posterior(Model &model, int count, int Nsample)
{
    int s=64;
    char str[100];
    
    cout<<"Nsample: "<<Nsample<<"\n";
    cout<<"Model: "<<model<<endl;
    
    auto theta_stack = model.make_param_stack(count);
    methods::sample_prior_stack(model,theta_stack);
    auto im_stack = model.make_image_stack(count);
    methods::simulate_image_stack(model,theta_stack,im_stack);
    CubeT sample_stack(model.get_num_params(),Nsample, count);
    MatT sample_rllh_stack(Nsample, count);
    
    auto rllh_theta_mean = VecT(count);
    auto rllh_theta_true = VecT(count);
    IdxT Nburnin = 100;
    IdxT thin = 0;
    double confidence = 0.95;
    methods::estimate_mcmc_sample_stack(model, im_stack, Nsample, Nburnin, thin, sample_stack, sample_rllh_stack);
    auto theta_mean_stack = model.make_param_stack(count);
    auto theta_lb_stack = model.make_param_stack(count);
    auto theta_ub_stack = model.make_param_stack(count);
    methods::error_bounds_posterior_credible_stack(model, sample_stack, confidence, theta_mean_stack, theta_lb_stack, theta_ub_stack);
    methods::objective::rllh_stack(model,im_stack, theta_mean_stack, rllh_theta_mean);
    methods::objective::rllh_stack(model,im_stack, theta_stack, rllh_theta_true);
    MatT error = theta_stack-theta_mean_stack;
    VecT rmserror = arma::sqrt(arma::mean(error%error,1));
    double mean_rllh_delta = arma::mean(rllh_theta_mean - rllh_theta_true);
    
    cout<<std::setw(64)<<""<<"\033["<<TERM_WHITE<<"m"<<"Model:"<<model.name
        <<" Estimator:Posterior Nsample:"<<Nsample<<" Nburnin:"<<Nburnin<<" thin:"<<thin<<"\033[0m"<<endl;

    snprintf(str, s, "<LLHDelta>:%g RMSE:",mean_rllh_delta);
    print_vec_row(cout,rmserror,str,s,TERM_MAGENTA);
}


template<class Model>
void evaluate_single(Model &model, std::string method, const ParamT<Model> &theta)
{
    int s=64;
    char str[100];

    auto stencil = model.make_stencil(theta);
    cout<<"Model: "<<model<<endl;
    cout<<"Stencil: "<<stencil<<endl;

    auto im = methods::simulate_image(model, stencil);
    
    cout<<std::setw(2*s)<<std::setfill('=')<<""<<endl<<std::setfill(' ');
    print_image(std::cout, im);
    
    double theta_llh = methods::objective::llh(model, im, stencil);

    auto theta_max = model.make_param();
    auto obsI = model.make_param_mat();
    MatT sequence;
    VecT sequence_rllh;
    StatsT stats;
    auto init_stencil = model.initial_theta_estimate(im);
    methods::estimate_max_debug(model, im, method, init_stencil.theta, theta_max, obsI, sequence, sequence_rllh, stats);
    auto theta_lb = model.make_param();
    auto theta_ub = model.make_param();
    double confidence = 0.95;
    methods::error_bounds_observed(model, theta_max, obsI, confidence, theta_lb, theta_ub);
    std::cout<<"Estimator Stats: "<<stats<<std::endl;
    
    snprintf(str, s, "RLLH:%g TrueTheta:",theta_llh);
    print_vec_row(cout, theta, str, s, TERM_MAGENTA);
    
    double theta_init_rllh = methods::objective::rllh(model,im,init_stencil.theta);
    snprintf(str, s, "RLLH:%g ThetaInit:",theta_init_rllh );
    print_vec_row(cout, init_stencil.theta, str, s, TERM_GREEN);
    
    double theta_max_rllh = methods::objective::rllh(model,im,theta_max);
    snprintf(str, s, "RLLH:%.9g ETheta[%s]:",theta_max_rllh , method.c_str());
    print_vec_row(cout, theta_max, str, s, TERM_YELLOW);
    snprintf(str, s, "Error[%s]:", method.c_str());
    print_vec_row(cout, theta-theta_max, str, s, TERM_RED);
    print_vec_row(cout, theta_lb, "ThetaLB:", s, TERM_CYAN);
    print_vec_row(cout, theta_ub, "ThetaUB:", s, TERM_CYAN);
    
    auto crlb = methods::cr_lower_bound(model, theta);
    print_vec_row(cout, arma::sqrt(crlb), "sqrtCRLB:", s, TERM_BLUE);
    

    IdxT Nsamples=4000;
   
    auto posterior_mean=model.make_param();
    auto posterior_cov=model.make_param_mat();
    auto sample = model.make_param_stack(Nsamples);
    VecT sample_rllh(Nsamples);
    auto candidates = model.make_param_stack(Nsamples);
    VecT candidates_rllh(Nsamples);

    methods::estimate_mcmc_sample_debug(model, im, init_stencil.theta, Nsamples,
                                        sample, sample_rllh, candidates, candidates_rllh);
    auto post_mean = model.make_param();
    auto post_lb = model.make_param();
    auto post_ub = model.make_param();
    methods::error_bounds_posterior_credible(model, sample, confidence, post_mean, post_lb, post_ub);
    auto post_cov = model.make_param_mat();
    mcmc::estimate_sample_posterior(sample, post_mean, post_cov);
    
    auto post_mean_llh = methods::objective::llh(model, im, post_mean);
    VecT post_se = arma::sqrt(post_cov.diag());
    snprintf(str, s, "RLLH:%.9g EstPMean:",post_mean_llh);
    print_vec_row(cout, post_mean, str, s, TERM_DIM_YELLOW);
    print_vec_row(cout, theta-posterior_mean, "Error[PostMean]:", s, TERM_DIM_RED);
    print_vec_row(cout, post_se, "PostSE:", s, TERM_DIM_WHITE);
    print_vec_row(cout, post_lb, "PostLB:", s, TERM_DIM_CYAN);
    print_vec_row(cout, post_ub, "PostUB:", s, TERM_DIM_CYAN);

    IdxT Nseq = sequence.n_cols;
    for(IdxT i=0; i<Nseq; i++){
        snprintf(str, s, "RLLH:%.9g Seq[%s](%llu):",sequence_rllh(i),method.c_str(), i);
        print_vec_row(cout, sequence.col(i), str, s, TERM_DIM_MAGENTA);
    }    
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
    double llh=methods::objective::llh(model, im,stencil);
    auto crlb=cr_lower_bound(model,stencil);
    cout<<std::setw(2*s)<<std::setfill('=')<<""<<endl<<std::setfill(' ');
    //     auto model_im=model_image(model, stencil);
    //     print_image(cout, model_im);
    print_image(cout, im);
    snprintf(str, s, "LLH:%g Theta:",llh);
    print_vec_row(cout, theta, str, s, TERM_MAGENTA);
    //     print_vec_row(cout, crlb, "CRLB:", s, TERM_BLUE);
    auto theta_init=model.initial_theta_estimate(im);
    snprintf(str, s, "LLH:%g ThetaInit:",methods::objective::llh(model,im,theta_init));
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
    auto im=methods::simulate_image(model,stencil);
    double llh=methods::objective::llh(model,im,stencil);
    auto crlb=methods::cr_lower_bound(model,stencil);
    cout<<std::setw(2*s)<<std::setfill('=')<<""<<endl<<std::setfill(' ');
    print_image(cout, im);
    snprintf(str, s, "LLH:%g Theta:",llh);
    print_vec_row(cout, theta, str, s, TERM_WHITE);
    print_vec_row(cout, crlb, "CRLB", s, TERM_CYAN);
    std::vector<std::unique_ptr<Estimator<Model>>> estimators;
    for(auto name: model.estimator_names) {
        try {
            estimators.emplace_back(methods::make_estimator(model, name));
            auto theta_est=model.make_param();
            estimators.back()->estimate(im, theta_est, crlb, llh);
            snprintf(str, s, "LLH:%.9g Theta[%s]:",llh,name.c_str());
            print_vec_row(cout, theta_est, str, s, TERM_YELLOW);
            typename Model::ParamT error=theta-theta_est;
            double se=arma::dot(error,error);
            double pse=error(0)*error(0)+error(1)*error(1);
            snprintf(str, s, "SE_pos:%.9g SE:%.9g |Error[%s]|:",pse, se, name.c_str());
            print_vec_row(cout, abs(error), str, s, TERM_RED);
            snprintf(str, s, "CRLB[%s]:",name.c_str());
            print_vec_row(cout, crlb, str, s, TERM_BLUE);
        } catch (const NotImplementedError& e) {
            continue;
        }
    }
    auto mean=model.make_param();
    auto cov=model.make_param_mat();
    evaluate_posterior(model, im, 2000, mean, cov);
    llh=methods::objective::llh(model,im,mean);
    snprintf(str, s, "LLH:%.9g Est<Posterior>:",llh);
    print_vec_row(cout, mean, str, s, TERM_YELLOW);
    typename Model::ParamT error=theta-mean;
    double se=arma::dot(error,error);
    double pse=error(0)*error(0)+error(1)*error(1);
    snprintf(str, s, "SE_pos:%.9g SE:%.9g |Error[Posterior]|:",pse, se);
    print_vec_row(cout, abs(error), str, s, TERM_RED);
    print_vec_row(cout, cov.diag().eval(), "VAR[Posterior]:", s, TERM_BLUE);
}

// template <class Model>
// void compare_estimators(Model &model, const typename Model::ParamT &theta, IdxT ntrials)
// {
//     using std::cout;
//     using std::endl;
//     int s=70;
//     char str[200];
//     auto stencil=model.make_stencil(theta);
//     auto ims=model.make_image_stack(ntrials);
//     methods::simulate_image_stack(model,theta,ims);
//     auto llh=VecT(ntrials);
//     methods::objective::llh_stack(model, ims, theta, llh);
//     auto crlb=methods::cr_lower_bound(model,stencil);
//     cout<<std::setw(2*s)<<std::setfill('=')<<""<<endl<<std::setfill(' ');
//     snprintf(str, s, "True <LLH>:%.9g Theta:",arma::mean(llh));
//     print_vec_row(cout, theta, str, s, TERM_WHITE);
//     print_vec_row(cout, crlb, "CRLB", s, TERM_BLUE);
//     std::vector<std::unique_ptr<Estimator<Model>>> estimators;
//     for(auto name: model.estimator_names) {
//         try {
//             estimators.emplace_back(methods::make_estimator(model, name));
//             auto theta_est=model.make_param_stack(ntrials);
//             auto crlb_est=model.make_param_stack(ntrials);
//             auto llh_est=VecT(ntrials);
//             estimators.back()->estimate_stack(ims, theta_est, crlb_est, llh_est);
//             snprintf(str, s, "<LLH>:%.9g <Theta[%s]>:",arma::mean(llh_est),name.c_str());
//             print_vec_row(cout, arma::mean(theta_est,1), str, s, TERM_YELLOW);
//             auto error=theta_est;
//             error.each_col()-=theta;
//             double mse=arma::accu(error%error)/ntrials;
//             double pmse=arma::mean(error.row(0)%error.row(0)+error.row(1)%error.row(1));
//             snprintf(str, s, "R<SE_pos>:%.9g R<SE>:%.9g <Error[%s]>:",sqrt(pmse), sqrt(mse), name.c_str());
//             print_vec_row(cout, arma::mean(error,1), str, s, TERM_RED);
//             snprintf(str, s, "RMSE[%s]>:",name.c_str());
//             print_vec_row(cout, arma::sqrt(arma::mean(error%error,1)), str, s, TERM_MAGENTA);
//             snprintf(str, s, "<CRLB[%s]>:",name.c_str());
//             print_vec_row(cout, arma::mean(crlb_est,1), str, s, TERM_BLUE);
//         } catch (const NotImplementedError& e) {
//             continue;
//         }
//         
//     }
//     auto mean=model.make_param_stack(ntrials);
//     arma::cube cov(model.get_num_params(), model.get_num_params(), ntrials);
//     evaluate_posterior_stack(model, ims, 3000, mean, cov);
//     auto llh_est=VecT(ntrials);
//     methods::objective::llh_stack(model, ims, mean, llh_est);
//     snprintf(str, s, "LLH:%.9g Est<Posterior>:",arma::mean(llh_est));
//     print_vec_row(cout, arma::mean(mean,1), str, s, TERM_YELLOW);
//     auto error=mean;
//     error.each_col()-=theta;
//     double mse=arma::accu(error%error)/ntrials;
//     double pmse=arma::mean(error.row(0)%error.row(0)+error.row(1)%error.row(1));
//     snprintf(str, s, "R<SE_pos>:%.9g R<SE>:%.9g <Error[Posterior]>:",sqrt(pmse), sqrt(mse));
//     print_vec_row(cout, arma::mean(error,1), str, s, TERM_RED);
//     print_vec_row(cout, arma::sqrt(arma::mean(error%error,1)), "RMSE[Posterior]:", s, TERM_MAGENTA);
//     auto var=model.make_param_stack(ntrials);
//     for(IdxT n=0; n<ntrials;n++) var.col(n)=cov.slice(n).diag();
//     print_vec_row(cout, arma::mean(var,1), "PosteriorVar:", s, TERM_BLUE);
// }
// 

// template<template<class> class Estimator, class Model>
// void point_evaluate_estimator(Estimator<Model> &estimator, IdxT ntrials,
//                               const typename Model::ParamT &theta,
//                               typename Model::ParamT &efficiency,
//                               double &unbiased_p_val)
// {
//     using std::cout;
//     using std::endl;
//     int s=65;
//     char str[100];
//     auto &model=estimator.model;
//     auto stencil=model.make_stencil(theta);
//     auto ims=model.make_image_stack(ntrials);
//     methods::simulate_image_stack(model,theta,ims);
//     auto crlb=methods::cr_lower_bound(model,stencil);
//     cout<<std::setw(2*s)<<std::setfill('=')<<""<<endl<<std::setfill(' ');
//     print_vec_row(cout, crlb, "CRLB", s, "1;34");
// 
//     auto theta_est=model.make_param_stack(ntrials);
//     auto crlb_est=model.make_param_stack(ntrials);
//     auto llh_est=VecT(ntrials);
//     estimator.estimate_max_stack(ims, theta_est, crlb_est, llh_est);
//     for(IdxT i=0;i<ntrials;i++) print_vec_row(cout, theta_est.col(i), "Theta_est:",15,TERM_BLUE);
//    
//     auto error=theta_est;
//     error.each_col()-=theta;
//     typename Model::ParamT error_var=arma::var(error,0,1);
//     efficiency=crlb/error_var;
//     
//     cout<<"Number of trials: "<<ntrials<<endl;
//     cout<<"Estimator: "<<estimator.name()<<endl;
//     auto llhs_true=VecT(ntrials);
//     methods::objective::llh_stack(model, ims, theta, llhs_true);
//     snprintf(str, s, "True <LLH:%.9g> Theta:",arma::mean(llhs_true));
//     print_vec_row(cout, theta, str, s, TERM_WHITE);
//     snprintf(str, s, "<LLH>:%.9g <Theta[%s]>:",arma::mean(llh_est),estimator.name().c_str());
//     print_vec_row(cout, arma::mean(theta_est,1), str, s, TERM_RED);
//     print_vec_row(cout, arma::mean(error,1), "Mean(Error):", s,TERM_YELLOW);
//     print_vec_row(cout, error_var, "Var(Error):", s, TERM_MAGENTA);
//     print_vec_row(cout, crlb, "CRLB(Theta):", s, TERM_BLUE);
//     print_vec_row(cout, efficiency, "Efficiency(Theta):", s,TERM_CYAN);
//     cout<<"Cov matrix: "<< arma::cov(theta_est.t())<<endl;
//     cout<<"---------------------------------------------------------------"<<endl<<endl;
// }

} /* namespace mappel::test */
} /* namespace mappel */

#endif /* _MAPPEL_TEST_HELPERS_H */
