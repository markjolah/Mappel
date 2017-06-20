/** @file Gauss2DsMAP.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-25-2014
 * @brief The class definition and template Specializations for Gauss2DsMAP
 */
#include <algorithm>

#include "Gauss2DsMAP.h"
#include "cGaussMLE/cGaussMLE.h"
#include "cGaussMLE/GaussLib.h"

const std::vector<std::string> Gauss2DsMAP::hyperparameter_names(
    { "beta_pos", "mean_I", "kappa_I", "mean_bg", "kappa_bg", "alpha_sigma" });

/** @brief Create a new Model for 2D Gaussian Point Emitters with known PSF under uniform priors.
 * @param[in] size The width and hegiht of the image in pixels
 * @param[in] PSFSigma The standard deviation of the Gaussian PSF
 * Also initializes internal precomputed computational stencils and seed rng.
 */
Gauss2DsMAP::Gauss2DsMAP(const IVecT &size, const VecT &psf_sigma)
    : Gauss2DsModel(size,psf_sigma),
      pos_dist(BetaRNG(beta_pos,beta_pos)),
      I_dist(GammaRNG(kappa_I,mean_I/kappa_I)),
      bg_dist(GammaRNG(kappa_bg,mean_bg/kappa_bg)),
      sigma_dist(ParetoRNG(alpha_sigma, 1.0)),
      log_prior_pos_const(log_prior_beta_const(beta_pos)),
      log_prior_I_const(log_prior_gamma_const(kappa_I,mean_I)),
      log_prior_bg_const(log_prior_gamma_const(kappa_bg,mean_bg)),
      log_prior_sigma_const(log_prior_pareto_const(alpha_sigma, 1.0))
{
    candidate_eta_I=mean_I*candidate_sample_dist_ratio;
    candidate_eta_bg=mean_bg*candidate_sample_dist_ratio;
}

Gauss2DsMAP::StatsT Gauss2DsMAP::get_stats() const
{
    StatsT stats=Gauss2DsModel::get_stats();
    stats["hyperparameter.Beta_pos"]=beta_pos;
    stats["hyperparameter.Mean_I"]=mean_I;
    stats["hyperparameter.Kappa_I"]=kappa_I;
    stats["hyperparameter.Mean_bg"]=mean_bg;
    stats["hyperparameter.Kappa_bg"]=kappa_bg;
    stats["hyperparameter.Alpha_sigma"]=alpha_sigma;
    stats["candidate.etaSigma"]=candidate_eta_sigma;
    return stats;
}


void Gauss2DsMAP::set_hyperparameters(const VecT &hyperparameters)
{
    // Params are {beta_pos, mean_I, kappa_I, mean_bg, kappa_bg}
    beta_pos=hyperparameters(0);
    mean_I=hyperparameters(1);
    kappa_I=hyperparameters(2);
    mean_bg=hyperparameters(3);
    kappa_bg=hyperparameters(4);
    alpha_sigma=hyperparameters(5);
    log_prior_pos_const=log_prior_beta_const(beta_pos);
    log_prior_I_const=log_prior_gamma_const(kappa_I,mean_I);
    log_prior_bg_const=log_prior_gamma_const(kappa_bg,mean_bg);
    log_prior_sigma_const=log_prior_pareto_const(alpha_sigma, 1.0);
    //Reset distributions
    pos_dist.set_params(beta_pos,beta_pos);
    I_dist.kappa(kappa_I);
    I_dist.theta(mean_I/kappa_I);
    bg_dist.kappa(mean_bg);
    bg_dist.theta(mean_bg/kappa_bg);
    sigma_dist.gamma(alpha_sigma); //gamma=alpha for trng's powerlaw dist
}

bool Gauss2DsMAP::theta_in_bounds(const ParamT &theta) const
{
    bool xOK = (theta(0)>=prior_epsilon) && (theta(0)<=size(0)-prior_epsilon);
    bool yOK = (theta(1)>=prior_epsilon) && (theta(1)<=size(1)-prior_epsilon);
    bool IOK = (theta(2)>=prior_epsilon);
    bool bgOK = (theta(3)>=prior_epsilon);
    bool sigmaOK = (theta(4)>=sigma_min);
    return xOK && yOK && IOK && bgOK && sigmaOK;
}

void Gauss2DsMAP::bound_theta(ParamT &theta) const
{
    theta(0)=restrict_value_range(theta(0), prior_epsilon, size(0)-prior_epsilon); // Prior: Support on [0,size]
    theta(1)=restrict_value_range(theta(1), prior_epsilon, size(1)-prior_epsilon); // Prior: Support on [0,size]
    theta(2)=std::max(prior_epsilon,theta(2));// Prior: Support on [0, inf)
    theta(3)=std::max(prior_epsilon,theta(3));// Prior: Support on [0, inf)
    theta(4)=std::max(sigma_min,theta(4));// Prior: Support on [0, inf)
}

double Gauss2DsMAP::prior_log_likelihood(const Stencil &s) const
{
    double rllh=prior_relative_log_likelihood(s);
    return rllh+ 2*log_prior_pos_const + log_prior_I_const + log_prior_bg_const+ log_prior_sigma_const;
}

double Gauss2DsMAP::prior_relative_log_likelihood(const Stencil &s) const
{
    double xrllh=rllh_beta_prior(beta_pos, s.x(), size(0));
    double yrllh=rllh_beta_prior(beta_pos, s.y(), size(1));
    double Irllh=rllh_gamma_prior(kappa_I, mean_I, s.I());
    double bgrllh=rllh_gamma_prior(kappa_bg, mean_bg, s.bg());
    double sigmarllh=rllh_gamma_prior(kappa_sigma, mean_sigma, s.sigma());
    return xrllh+yrllh+Irllh+bgrllh+sigmarllh;
}

Gauss2DsMAP::ParamT
Gauss2DsMAP::prior_grad(const Stencil &s) const
{
    ParamT grad=make_param();
    grad(0)=beta_prior_grad(beta_pos, s.x(), size(0));
    grad(1)=beta_prior_grad(beta_pos, s.y(), size(1));
    grad(2)=gamma_prior_grad(kappa_I, mean_I, s.I());
    grad(3)=gamma_prior_grad(kappa_bg, mean_bg, s.bg());
    grad(4)=gamma_prior_grad(kappa_sigma, mean_sigma, s.bg());
    return grad;
}

Gauss2DsMAP::ParamT
Gauss2DsMAP::prior_grad2(const Stencil &s) const
{
    ParamT grad2=make_param();
    grad2(0)= beta_prior_grad2(beta_pos, s.x(), size(0));
    grad2(1)= beta_prior_grad2(beta_pos, s.y(), size(1));
    grad2(2)= gamma_prior_grad2(kappa_I, s.I());
    grad2(3)= gamma_prior_grad2(kappa_bg, s.bg());
    grad2(4)= gamma_prior_grad2(kappa_sigma, s.bg());
    return grad2;
}

Gauss2DsMAP::ParamT
Gauss2DsMAP::prior_cr_lower_bound(const Stencil &s) const
{
    //TODO complete these calculations
    ParamT pcrlb=make_param();
    pcrlb.zeros();
    return pcrlb;
}

/* Template Specializations */
template<>
Gauss2DsMAP::Stencil
CGaussHeuristicMLE<Gauss2DsMAP>::compute_estimate(const ImageT &im, const ParamT &theta_init)
{
    Gauss2DsMAP::ParamT theta_est(arma::fill::zeros);
    if(model.size(0)==model.size(1) && model.psf_sigma(0)==model.psf_sigma(1)){ //only works for square images and iso-tropic psf
        float Nmax;
        arma::fvec5 ftheta_est;
        //Convert from double
        arma::fmat fim=arma::conv_to<arma::fmat>::from(im);
        //Compute
        CenterofMass2D(model.size(0), fim.memptr(), &ftheta_est(0), &ftheta_est(1));
        GaussFMaxMin2D(model.size(0), model.psf_sigma(0), fim.memptr(), &Nmax, &ftheta_est(3));
        ftheta_est(2)=std::max(0., (Nmax-ftheta_est(3))*2*arma::datum::pi*model.psf_sigma(0)*model.psf_sigma(0));
        ftheta_est(4)=model.psf_sigma(0);
        //Back to double
        theta_est=arma::conv_to<arma::mat>::from(ftheta_est);
        //Swap x/y and add .5 tp convert from CGauss to mappel coordinates
        float temp=theta_est(0)+.5;
        theta_est(0)=theta_est(1)+.5;
        theta_est(1)=temp;
    }
    return model.make_stencil(theta_est);
}

template<>
void
CGaussMLE<Gauss2DsMAP>::compute_estimate(const ImageT &im, const ParamT &theta_init, ParamT &theta, ParamT &crlb, double &llh)
{
    if(model.size(0)==model.size(1) && model.psf_sigma(0)==model.psf_sigma(1)){//only works for square images and iso-tropic psf
        float fllh;
        arma::fvec5 fcrlb, ftheta;
        //Convert from double
        arma::fmat fim=arma::conv_to<arma::fmat>::from(im);
        //Compute
        MLEFit_sigma(fim.memptr(), model.psf_sigma(0), model.size(0), max_iterations,
            ftheta.memptr(), fcrlb.memptr(), &fllh);
        //Back to double
        theta=arma::conv_to<arma::vec>::from(ftheta);
        crlb=arma::conv_to<arma::vec>::from(fcrlb);
        //Swap x/y and add .5 tp convert from CGauss to mappel coordinates
        float temp=theta(0)+.5;
        theta(0)=theta(1)+.5;
        theta(1)=temp;
        llh=log_likelihood(model, im,model.make_stencil(theta));
    } else {
        theta.zeros();
        crlb.zeros();
        llh=0.0;
    }
}

