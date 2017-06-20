/** @file Gauss1DModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class definition and template Specializations for Gauss1DModel
 */

#include "Gauss1DModel.h"
#include "stencil.h"

namespace mappel {

const std::vector<std::string> Gauss1DModel::param_names({ "x", "y", "I", "bg" });

const std::vector<std::string> Gauss1DModel::hyperparameter_names(
    { "beta_pos", "mean_I", "kappa_I", "mean_bg", "kappa_bg"});


Gauss1DModel::Gauss1DModel(int size_, double psf_sigma_)
    : ImageFormat1DBase(size_),
      PointEmitterModel(3),
      pos_dist(BetaRNG(beta_pos,beta_pos)),
      I_dist(GammaRNG(kappa_I,mean_I/kappa_I)),
      bg_dist(GammaRNG(kappa_bg,mean_bg/kappa_bg)),
      log_prior_pos_const(log_prior_beta_const(beta_pos)),
      log_prior_I_const(log_prior_gamma_const(kappa_I,mean_I)),
      log_prior_bg_const(log_prior_gamma_const(kappa_bg,mean_bg)),
      psf_sigma(psf_sigma_)
{
    /* Initialize MCMC step sizes */
    mcmc_num_candidate_sampling_phases=2;
    mcmc_candidate_eta_x = size*mcmc_candidate_sample_dist_ratio;
    mcmc_candidate_eta_I = mean_I*mcmc_candidate_sample_dist_ratio;
    mcmc_candidate_eta_bg = mean_bg*mcmc_candidate_sample_dist_ratio;

    /* Initialization stencils */
    ParamT lb = {0,0,0};
    ParamT ub = {static_cast<double>(size),INFINITY,INFINITY};
    set_bounds(lb,ub);
}

Gauss1DModel::Stencil::Stencil(const Gauss1DModel &model_,
                               const Gauss1DModel::ParamT &theta,
                               bool _compute_derivatives)
: model(&model_),theta(theta)
{
    int szX = model->size;
    dx = make_d_stencil(szX, x());
    X = make_X_stencil(szX, dx,model->psf_sigma);
    if(_compute_derivatives) compute_derivatives();
}

void Gauss1DModel::Stencil::compute_derivatives()
{
    if(derivatives_computed) return;
    derivatives_computed=true;
    int szX = model->size;
    double sigmaX = model->psf_sigma;
    Gx = make_G_stencil(szX, dx,sigmaX);
    DX = make_DX_stencil(szX, Gx,sigmaX);
    DXS = make_DXS_stencil(szX, dx, Gx,sigmaX);
}

StatsT Gauss1DModel::get_stats() const
{
    StatsT stats = ImageFormat1DBase::get_stats();
    stats["numParams"] = num_params;
    stats["hyperparameters.Beta_pos"]=beta_pos;
    stats["hyperparameters.Mean_I"]=mean_I;
    stats["hyperparameters.Kappa_I"]=kappa_I;
    stats["hyperparameters.Mean_bg"]=mean_bg;
    stats["hyperparameters.Kappa_bg"]=kappa_bg;
    stats["mcmcparams.num_phases"]=mcmc_num_candidate_sampling_phases;
    stats["mcmcparams.etaX"]=mcmc_candidate_eta_x;
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


std::ostream& operator<<(std::ostream &out, const Gauss1DModel::Stencil &s)
{
    int w=8;
    print_vec_row(out,s.theta,"Theta:",w,TERM_WHITE);
    print_vec_row(out,s.dx,"dx:",w,TERM_CYAN);
    print_vec_row(out,s.X,"X:",w,TERM_CYAN);
    if(s.derivatives_computed) {
        print_vec_row(out,s.Gx,"Gx:",w,TERM_BLUE);
        print_vec_row(out,s.DX,"DX:",w,TERM_BLUE);
        print_vec_row(out,s.DXS,"DXS:",w,TERM_BLUE);
    }
    return out;
}


void Gauss1DModel::set_hyperparameters(const VecT &hyperparameters)
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
}

Gauss1DModel::VecT Gauss1DModel::get_hyperparameters() const
{
    return VecT({beta_pos,mean_I, kappa_I, mean_bg, kappa_bg});
}


void
Gauss1DModel::pixel_hess_update(int i, const Stencil &s, double dm_ratio_m1, double dmm_ratio, ParamT &grad, ParamMatT &hess) const
{
    /* Caclulate pixel derivative */
    auto pgrad=make_param();
    pixel_grad(i,s,pgrad);
    double I = s.I();
    /* Update grad */
    grad += dm_ratio_m1*pgrad;
    /* Update hess */
    hess(0,0) += dm_ratio_m1 * I/psf_sigma * s.DXS(i);
    hess(0,1) += dm_ratio_m1 * pgrad(0) / I; 
    //This is the pixel-gradient dependent part of the hessian
    for(int c=0; c<(int)hess.n_cols; c++) for(int r=0; r<=c; r++)
        hess(r,c) -= dmm_ratio * pgrad(r) * pgrad(c);
}



Gauss1DModel::Stencil
Gauss1DModel::initial_theta_estimate(const ImageT &im, const ParamT &theta_init) const
{
    double x_pos=0, I=0, bg=0;
    if (!theta_init.is_empty()) {
        x_pos = theta_init(0);
        I = theta_init(1);
        bg = theta_init(2);
    }
    if(x_pos<=0 || x_pos>size){ //Invlaid position, estimate it
        x_pos = im.index_max()+0.5;
    } 
    if (I<=0 || bg<=0) {
        bg = 0.75*im.min();
        I = arma::sum(im)-std::min(0.3,bg*size);
    }
//     std::cout<<"[1D]Theta_init: ["<<x_pos<<","<<I<<","<<bg<<"]";
    return make_stencil(x_pos,  I, bg);
}


void Gauss1DModel::sample_mcmc_candidate_theta(int sample_index, RNG &rng, ParamT &mcmc_candidate_theta, double scale) const
{
    int phase=sample_index%mcmc_num_candidate_sampling_phases;
    switch(phase) {
        case 0:  //change pos
            mcmc_candidate_theta(0)+=generate_normal(rng,0.0,mcmc_candidate_eta_x*scale);
            break;
        case 1: //change I, bg
            mcmc_candidate_theta(1)+=generate_normal(rng,0.0,mcmc_candidate_eta_I*scale);
            mcmc_candidate_theta(2)+=generate_normal(rng,0.0,mcmc_candidate_eta_bg*scale);
    }
}



} /* namespace mappel */
