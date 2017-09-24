/** @file Gauss2DModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class definition and template Specializations for Gauss2DModel
 */

#include "Gauss2DModel.h"
#include "stencil.h"

namespace mappel {

const std::vector<std::string> Gauss2DModel::param_names({ "x", "y", "I", "bg" });

const std::vector<std::string> Gauss2DModel::hyperparameter_names(
    { "beta_pos", "mean_I", "kappa_I", "mean_bg", "kappa_bg"});


Gauss2DModel::Gauss2DModel(const IVecT &_size, const VecT &_psf_sigma)
    : ImageFormat2DBase(_size),
      PointEmitterModel(4),
      pos_dist(BetaRNG(beta_pos,beta_pos)),
      I_dist(GammaRNG(kappa_I,mean_I/kappa_I)),
      bg_dist(GammaRNG(kappa_bg,mean_bg/kappa_bg)),
      log_prior_pos_const(log_prior_beta_const(beta_pos)),
      log_prior_I_const(log_prior_gamma_const(kappa_I,mean_I)),
      log_prior_bg_const(log_prior_gamma_const(kappa_bg,mean_bg)),
      psf_sigma(_psf_sigma),
      x_model(size(0),psf_sigma(0)),
      y_model(size(1),psf_sigma(1))
{
    /* Initialize MCMC step sizes */
    mcmc_num_candidate_sampling_phases=2;
    mcmc_candidate_eta_x = size(0)*mcmc_candidate_sample_dist_ratio;
    mcmc_candidate_eta_y = size(1)*mcmc_candidate_sample_dist_ratio;
    mcmc_candidate_eta_I = mean_I*mcmc_candidate_sample_dist_ratio;
    mcmc_candidate_eta_bg = mean_bg*mcmc_candidate_sample_dist_ratio;

    /* Initialization stencils */
    gaussian_Xstencil = make_gaussian_stencil(size(0),psf_sigma(0));
    gaussian_Ystencil = make_gaussian_stencil(size(1),psf_sigma(1));
    x_model.set_hyperparameters(get_hyperparameters());
    y_model.set_hyperparameters(get_hyperparameters());
    
    
    ParamT lb = {0,0,0,0};
    ParamT ub = {static_cast<double>(size(0)),static_cast<double>(size(1)),INFINITY,INFINITY};
    set_bounds(lb,ub);
}

Gauss2DModel::Stencil::Stencil(const Gauss2DModel &model_,
                               const Gauss2DModel::ParamT &theta,
                               bool _compute_derivatives)
: model(&model_),theta(theta)
{
    int szX=model->size(0);
    int szY=model->size(1);
    dx=make_d_stencil(szX, x());
    dy=make_d_stencil(szY, y());
    X=make_X_stencil(szX, dx,model->psf_sigma(0));
    Y=make_X_stencil(szY, dy,model->psf_sigma(1));
    if(_compute_derivatives) compute_derivatives();
}

void Gauss2DModel::Stencil::compute_derivatives()
{
    if(derivatives_computed) return;
    derivatives_computed=true;
    int szX=model->size(0);
    int szY=model->size(1);
    double sigmaX=model->psf_sigma(0);
    double sigmaY=model->psf_sigma(1);
    Gx=make_G_stencil(szX, dx,sigmaX);
    Gy=make_G_stencil(szY, dy,sigmaY);
    DX=make_DX_stencil(szX, Gx,sigmaX);
    DY=make_DX_stencil(szY, Gy,sigmaY);
    DXS=make_DXS_stencil(szX, dx, Gx,sigmaX);
    DYS=make_DXS_stencil(szY, dy, Gy,sigmaY);
}

StatsT Gauss2DModel::get_stats() const
{
    StatsT stats = ImageFormat2DBase::get_stats();
    stats["numParams"] = num_params;
    stats["hyperparameters.Beta_pos"]=beta_pos;
    stats["hyperparameters.Mean_I"]=mean_I;
    stats["hyperparameters.Kappa_I"]=kappa_I;
    stats["hyperparameters.Mean_bg"]=mean_bg;
    stats["hyperparameters.Kappa_bg"]=kappa_bg;
    stats["mcmcparams.num_phases"]=mcmc_num_candidate_sampling_phases;
    stats["mcmcparams.etaX"]=mcmc_candidate_eta_x;
    stats["mcmcparams.etaY"]=mcmc_candidate_eta_y;
    stats["mcmcparams.etaI"]=mcmc_candidate_eta_I;
    stats["mcmcparams.etabg"]=mcmc_candidate_eta_bg;
    for(int n=0;n<num_params;n++) {
        std::ostringstream outl,outu;
        outl<<"lbound."<<n+1;
        stats[outl.str()]= lbound(n);
        outu<<"ubound."<<n+1;
        stats[outu.str()]= ubound(n);
    }
    return stats;
}


std::ostream& operator<<(std::ostream &out, const Gauss2DModel::Stencil &s)
{
    int w=8;
    print_vec_row(out,s.theta,"Theta:",w,TERM_WHITE);
    print_vec_row(out,s.dx,"dx:",w,TERM_CYAN);
    print_vec_row(out,s.dy,"dy:",w,TERM_CYAN);
    print_vec_row(out,s.X,"X:",w,TERM_CYAN);
    print_vec_row(out,s.Y,"Y:",w,TERM_CYAN);
    if(s.derivatives_computed) {
        print_vec_row(out,s.Gx,"Gx:",w,TERM_BLUE);
        print_vec_row(out,s.Gy,"Gy:",w,TERM_BLUE);
        print_vec_row(out,s.DX,"DX:",w,TERM_BLUE);
        print_vec_row(out,s.DY,"DY:",w,TERM_BLUE);
        print_vec_row(out,s.DXS,"DXS:",w,TERM_BLUE);
        print_vec_row(out,s.DYS,"DYS:",w,TERM_BLUE);
    }
    return out;
}


void Gauss2DModel::set_hyperparameters(const VecT &hyperparameters)
{
    // Params are {beta_pos, mean_I, kappa_I, mean_bg, kappa_bg}
    beta_pos=check_lower_bound_hyperparameter("beta position",hyperparameters(0),1);
    mean_I=check_positive_hyperparameter("mean I",hyperparameters(1));
    kappa_I=check_positive_hyperparameter("kappa I",hyperparameters(2));
    mean_bg=check_positive_hyperparameter("mean bg",hyperparameters(3));
    kappa_bg=check_positive_hyperparameter("kappa bg",hyperparameters(4));
    log_prior_pos_const=log_prior_beta_const(beta_pos);
    log_prior_I_const=log_prior_gamma_const(kappa_I,mean_I);
    log_prior_bg_const=log_prior_gamma_const(kappa_bg,mean_bg);
    //Reset distributions
    pos_dist.set_params(beta_pos, beta_pos);
    I_dist.kappa(kappa_I);
    I_dist.theta(mean_I/kappa_I);
    bg_dist.kappa(mean_bg);
    bg_dist.theta(mean_bg/kappa_bg);
    x_model.set_hyperparameters(get_hyperparameters());
    y_model.set_hyperparameters(get_hyperparameters());
}

Gauss2DModel::VecT Gauss2DModel::get_hyperparameters() const
{
    return VecT({beta_pos,mean_I, kappa_I, mean_bg, kappa_bg});
}


void
Gauss2DModel::pixel_hess_update(int i, int j, const Stencil &s, double dm_ratio_m1, double dmm_ratio, ParamT &grad, ParamMatT &hess) const
{
    /* Caclulate pixel derivative */
    auto pgrad=make_param();
    pixel_grad(i,j,s,pgrad);
    double I=s.I();
    /* Update grad */
    grad+=dm_ratio_m1*pgrad;
    /* Update hess */
    hess(0,0)+=dm_ratio_m1 * I/psf_sigma(0) * s.DXS(i) * s.Y(j);
    hess(0,1)+=dm_ratio_m1 * I * s.DX(i) * s.DY(j);
    hess(1,1)+=dm_ratio_m1 * I/psf_sigma(1) * s.DYS(j) * s.X(i);
    hess(0,2)+=dm_ratio_m1 * pgrad(0) / I; 
    hess(1,2)+=dm_ratio_m1 * pgrad(1) / I; 
    //This is the pixel-gradient dependent part of the hessian
    for(int c=0; c<(int)hess.n_cols; c++) for(int r=0; r<=c; r++)
        hess(r,c) -= dmm_ratio * pgrad(r) * pgrad(c);
}



Gauss2DModel::Stencil
Gauss2DModel::heuristic_initial_theta_estimate(const ImageT &im, const ParamT &theta_init) const
{
    double x_pos=0, y_pos=0, I=0, bg=0;
    double min_bg=1; //default minimum background.  Will be updated only if estimate_gaussian_2Dmax is called.
//     std::cout<<"Theta_init: "<<theta_init.t()<<" -->";
    if (!theta_init.is_empty()) {
        x_pos = theta_init(0);
        y_pos = theta_init(1);
        I = theta_init(2);
        bg = theta_init(3);
    }
    if(x_pos<=0 || x_pos>size(0) || y_pos<=0 || y_pos>size(1)){ //Invlaid positions, estimate them
//         std::cout<<"Full init\n";
        int px_pos[2];
        estimate_gaussian_2Dmax(im, gaussian_Xstencil, gaussian_Ystencil, px_pos, min_bg);
        refine_gaussian_2Dmax(im, gaussian_Xstencil, gaussian_Ystencil, px_pos);
        x_pos = static_cast<double>(px_pos[0])+0.5;
        y_pos = static_cast<double>(px_pos[1])+0.5;
        auto unit_im = unit_model_image(size,px_pos,psf_sigma);
        bg = estimate_background(im, unit_im, min_bg);
        I = estimate_intensity(im, unit_im, bg);
    } else if(I<=0 || bg<=0) {
//         std::cout<<"Intenisty init\n";
        int px_pos[2];
        px_pos[0] = static_cast<int>(floor(x_pos));
        px_pos[1] = static_cast<int>(floor(y_pos));
        auto unit_im = unit_model_image(size,px_pos,psf_sigma);
        bg = estimate_background(im, unit_im, min_bg);
        I = estimate_intensity(im, unit_im, bg);
    } /*else {
        std::cout<<"Null init\n";
    }*/
    auto theta= make_stencil(x_pos, y_pos, I, bg);
//     std::cout<<"ThetaFinal: "<<theta.theta.t()<<"\n";
    return theta;
}

Gauss2DModel::Stencil
Gauss2DModel::seperable_initial_theta_estimate(const ImageT &im, const ParamT &theta_init, 
                                               const std::string &estimator) const
{
    double x_pos=0, y_pos=0, I=0, bg=0;
    if (!theta_init.is_empty()) {
        x_pos = theta_init(0);
        y_pos = theta_init(1);
        I = theta_init(2);
        bg = theta_init(3);
    }
    if(x_pos<=0 || x_pos>size(0) || y_pos<=0 || y_pos>size(1) || I<=0 || bg<=0){ 
        //Invlaid theta init.  Run sub-estimators
        auto x_estimator = make_estimator(x_model, estimator);
        auto y_estimator = make_estimator(y_model, estimator);
        Gauss1DModel::ImageT x_im = arma::sum(im,0).t();
        Gauss1DModel::ImageT y_im = arma::sum(im,1);
        auto x_est = x_estimator->estimate(x_im);
        auto y_est = y_estimator->estimate(y_im);
        
        if(x_pos<=0 || x_pos>size(0)) x_pos = x_est.theta(0);
        if(y_pos<=0 || y_pos>size(1)) y_pos = y_est.theta(0);
        if(I<=0) I = std::max(x_est.theta(1), y_est.theta(1)); //max of X and Y est of I
        if(bg<=0) bg = .5*(x_est.theta(2)/size(1) + y_est.theta(2)/size(0)); //mean of X and Y est of bg corrected for 1D vs 2D interpretation of bg
    }
    return make_stencil(x_pos, y_pos, I, bg);
}



void Gauss2DModel::sample_mcmc_candidate_theta(int sample_index, RNG &rng, ParamT &mcmc_candidate_theta, double scale) const
{
    int phase=sample_index%mcmc_num_candidate_sampling_phases;
    switch(phase) {
        case 0:  //change x,y
            mcmc_candidate_theta(0)+=generate_normal(rng,0.0,mcmc_candidate_eta_x*scale);
            mcmc_candidate_theta(1)+=generate_normal(rng,0.0,mcmc_candidate_eta_y*scale);
            break;
        case 1: //change I, bg
            mcmc_candidate_theta(2)+=generate_normal(rng,0.0,mcmc_candidate_eta_I*scale);
            mcmc_candidate_theta(3)+=generate_normal(rng,0.0,mcmc_candidate_eta_bg*scale);
    }
}

} /* namespace mappel */
