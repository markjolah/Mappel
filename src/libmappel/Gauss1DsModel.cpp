/** @file Gauss1DsModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief The class definition and template Specializations for Gauss1DsModel
 */

#include "Gauss1DsModel.h"
#include "stencil.h"

namespace mappel {

const std::vector<std::string> Gauss1DsModel::param_names({ "x", "I", "bg", "sigma" });

const std::vector<std::string> Gauss1DsModel::hyperparameter_names(
    { "beta_pos", "mean_I", "kappa_I", "mean_bg", "kappa_bg", "alpha_sigma", "min_sigma", "max_sigma"});


Gauss1DsModel::Gauss1DsModel(int size_, double min_sigma_, double max_sigma_)
    : ImageFormat1DBase(size_),
      PointEmitterModel(4),
      min_sigma(min_sigma_),
      max_sigma(max_sigma_),
      pos_dist(BetaRNG(beta_pos,beta_pos)),
      I_dist(GammaRNG(kappa_I,mean_I/kappa_I)),
      bg_dist(GammaRNG(kappa_bg,mean_bg/kappa_bg)),
      sigma_dist(ParetoRNG(min_sigma,alpha_sigma)),
      log_prior_pos_const(log_prior_beta_const(beta_pos)),
      log_prior_I_const(log_prior_gamma_const(kappa_I,mean_I)),
      log_prior_bg_const(log_prior_gamma_const(kappa_bg,mean_bg)),
      log_prior_sigma_const(log_prior_pareto_const(alpha_sigma,min_sigma))
{
    /* Initialize MCMC step sizes */
    mcmc_num_candidate_sampling_phases = 3;
    mcmc_candidate_eta_x = size*mcmc_candidate_sample_dist_ratio;
    mcmc_candidate_eta_I = mean_I*mcmc_candidate_sample_dist_ratio;
    mcmc_candidate_eta_bg = mean_bg*mcmc_candidate_sample_dist_ratio;
    mcmc_candidate_eta_sigma = 1.0*mcmc_candidate_sample_dist_ratio;
    
    /* Initialization stencils */
    ParamT lb = {0,0,0,min_sigma};
    ParamT ub = {static_cast<double>(size),INFINITY,INFINITY,max_sigma};
    set_bounds(lb,ub);
}

Gauss1DsModel::Stencil::Stencil(const Gauss1DsModel &model_,
                               const Gauss1DsModel::ParamT &theta,
                               bool _compute_derivatives)
: model(&model_),theta(theta)
{
    int szX = model->size;
    dx = make_d_stencil(szX, x());
    X = make_X_stencil(szX, dx,sigma());
    if(_compute_derivatives) compute_derivatives();
}

void Gauss1DsModel::Stencil::compute_derivatives()
{
    if(derivatives_computed) return;
    derivatives_computed=true;
    int szX = model->size;
    Gx = make_G_stencil(szX, dx, sigma());
    DX = make_DX_stencil(szX, Gx, sigma());
    DXS = make_DXS_stencil(szX, dx, Gx, sigma());
    DXS2 = make_DXS2_stencil(szX, dx, Gx, DXS, sigma());
    DXSX = make_DXSX_stencil(szX, dx, Gx, DX, sigma());
}

std::ostream& operator<<(std::ostream &out, const Gauss1DsModel::Stencil &s)
{
    int w=8;
    print_vec_row(out,s.theta,"Theta:",w,TERM_WHITE);
    print_vec_row(out,s.dx,"dx:",w,TERM_CYAN);
    print_vec_row(out,s.X,"X:",w,TERM_CYAN);
    if(s.derivatives_computed) {
        print_vec_row(out,s.Gx,"Gx:",w,TERM_BLUE);
        print_vec_row(out,s.DX,"DX:",w,TERM_BLUE);
        print_vec_row(out,s.DXS,"DXS:",w,TERM_BLUE);
        print_vec_row(out,s.DXS2,"DXS2:",w,TERM_BLUE);
        print_vec_row(out,s.DXSX,"DXSX:",w,TERM_BLUE);
    }
    return out;
}

StatsT Gauss1DsModel::get_stats() const
{
    StatsT stats = ImageFormat1DBase::get_stats();
    stats["numParams"] = num_params;
    stats["hyperparameters.Beta_pos"] = beta_pos;
    stats["hyperparameters.Mean_I"] = mean_I;
    stats["hyperparameters.Kappa_I"] = kappa_I;
    stats["hyperparameters.Mean_bg"] = mean_bg;
    stats["hyperparameters.Kappa_bg"] = kappa_bg;
    stats["hyperparameters.alpha_sigma"] = alpha_sigma;
    stats["hyperparameters.min_sigma"] = min_sigma;
    stats["hyperparameters.max_sigma"] = max_sigma;
    stats["mcmcparams.num_phases"] = mcmc_num_candidate_sampling_phases;
    stats["mcmcparams.eta_X"] = mcmc_candidate_eta_x;
    stats["mcmcparams.eta_I"] = mcmc_candidate_eta_I;
    stats["mcmcparams.eta_bg"] = mcmc_candidate_eta_bg;
    stats["mcmcparams.eta_sigma"] = mcmc_candidate_eta_bg;
    for(int n=0;n<num_params;n++) {
        std::ostringstream outl,outu;
        outl<<"lbound."<<n+1;
        stats[outl.str()] = lbound(n);
        outu<<"ubound."<<n+1;
        stats[outu.str()] = ubound(n);
    }
    return stats;
}

void Gauss1DsModel::set_hyperparameters(const VecT &hyperparameters)
{
    // Params are {beta_pos, mean_I, kappa_I, mean_bg, kappa_bg, alpha_sigma, min_sigma, max_sigma}
    beta_pos=check_lower_bound_hyperparameter("beta position",hyperparameters(0),1);
    mean_I=check_positive_hyperparameter("mean I",hyperparameters(1));
    kappa_I=check_positive_hyperparameter("kappa I",hyperparameters(2));
    mean_bg=check_positive_hyperparameter("mean bg",hyperparameters(3));
    kappa_bg=check_positive_hyperparameter("kappa bg",hyperparameters(4));
    alpha_sigma = check_lower_bound_hyperparameter("alpha sigma",hyperparameters(5),1);
    
    //Reset bounds on sigma as part of hyperparameters
    min_sigma = check_positive_hyperparameter("min sigma",hyperparameters(6));
    max_sigma = check_lower_bound_hyperparameter("max sigma",hyperparameters(7),min_sigma);

    //Reset constants
    log_prior_pos_const=log_prior_beta_const(beta_pos);
    log_prior_I_const=log_prior_gamma_const(kappa_I,mean_I);
    log_prior_bg_const=log_prior_gamma_const(kappa_bg,mean_bg);
    log_prior_sigma_const = log_prior_pareto_const(alpha_sigma, min_sigma);
    
    //Reset distributions
    pos_dist.set_params(beta_pos, beta_pos);
    I_dist.kappa(kappa_I);
    I_dist.theta(mean_I/kappa_I);
    bg_dist.kappa(mean_bg);
    bg_dist.theta(mean_bg/kappa_bg);
    sigma_dist.theta(min_sigma);
    sigma_dist.gamma(alpha_sigma);
}

Gauss1DsModel::VecT Gauss1DsModel::get_hyperparameters() const
{
    return VecT({beta_pos,mean_I, kappa_I, mean_bg, kappa_bg, alpha_sigma, min_sigma, max_sigma});
}


void
Gauss1DsModel::pixel_hess_update(int i, const Stencil &s, double dm_ratio_m1, double dmm_ratio, ParamT &grad, MatT &hess) const
{
    /* Caclulate pixel derivative */
    auto pgrad=make_param();
    pixel_grad(i,s,pgrad);
    double I = s.I();
    /* Update grad */
    grad += dm_ratio_m1*pgrad;
    /* Update hess */
    hess(0,0) += dm_ratio_m1 * I/s.sigma() * s.DXS(i);
    hess(0,1) += dm_ratio_m1 * s.DX(i);
    hess(0,3) += dm_ratio_m1 * I * s.DXSX(i);
    hess(1,3) += dm_ratio_m1 * s.DXS(i);
    hess(3,3) += dm_ratio_m1 * I * s.DXS2(i);
    //This is the pixel-gradient dependent part of the hessian
    for(int c=0; c<(int)hess.n_cols; c++) for(int r=0; r<=c; r++)
        hess(r,c) -= dmm_ratio * pgrad(r) * pgrad(c);
}



Gauss1DsModel::Stencil
Gauss1DsModel::initial_theta_estimate(const ImageT &im, const ParamT &theta_init) const
{
    //TODO: Propose a more robust initialization
    double x_pos=0, I=0, bg=0, sigma=0;
    if (!theta_init.is_empty()) {
        x_pos = theta_init(0);
        I = theta_init(1);
        bg = theta_init(2);
        sigma = theta_init(3);
    }
    if(x_pos<=0 || x_pos>size){
        //Estimate position as the brightest pixel
        x_pos = im.index_max()+0.5;
    }
    if(sigma<min_sigma || sigma>max_sigma){
        //Pick an initial sigma in-between min and max for sigma
        //This is a rough approximation
        double eta=0.8; //Weight applies to min sigma in weighted average
        sigma = eta*min_sigma + (1-eta)*max_sigma;
    }
    if (I<=0 || bg<=0) {
        bg = 0.75*im.min();
        I = arma::sum(im)-std::min(0.3,bg*size);
    }
    return make_stencil(x_pos,  I, bg, sigma);
}


void Gauss1DsModel::sample_mcmc_candidate_theta(int sample_index, RNG &rng, ParamT &mcmc_candidate_theta, double scale) const
{
    int phase = sample_index%mcmc_num_candidate_sampling_phases;
    switch(phase) {
        case 0:  //change pos
            mcmc_candidate_theta(0) += generate_normal(rng,0.0,mcmc_candidate_eta_x*scale);
            break;
        case 1: //change I, sigma
            mcmc_candidate_theta(1) += generate_normal(rng,0.0,mcmc_candidate_eta_I*scale);
            mcmc_candidate_theta(3) += generate_normal(rng,0.0,mcmc_candidate_eta_sigma*scale);
            break;
        case 2: //change I, bg
            mcmc_candidate_theta(1) += generate_normal(rng,0.0,mcmc_candidate_eta_I*scale);
            mcmc_candidate_theta(2) += generate_normal(rng,0.0,mcmc_candidate_eta_bg*scale);
    }
}



} /* namespace mappel */
