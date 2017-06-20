/** @file Gauss2DsMLE.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-25-2014
 * @brief The class definition and template Specializations for Gauss2DsMLE
 */
#include <algorithm>
#include <memory>

#include "Gauss2DsMLE.h"

const std::vector<std::string> Gauss2DsMLE::hyperparameter_names(
    { "I_min", "I_max", "bg_min", "bg_max", "sigma_max"});

/** @brief Create a new Model for 2D Gaussian Point Emitters with known PSF under uniform priors.
 * @param[in] size The width and hegiht of the image in pixels
 * @param[in] PSFSigma The standard deviation of the Gaussian PSF
 * Also initializes internal precomputed computational stencils and seed rng.
 */
Gauss2DsMLE::Gauss2DsMLE(const IVecT &size, const VecT &psf_sigma)
    : Gauss2DsModel(size,psf_sigma),
    pos_dist(UniformRNG(0,1)),
    I_dist(UniformRNG(I_min,I_max)),
    bg_dist(UniformRNG(bg_min,bg_max)),
    sigma_dist(UniformRNG(1.0,sigma_max))
{
    candidate_eta_I=(0.5*(I_max-I_min))*candidate_sample_dist_ratio;
    candidate_eta_bg=(0.5*(bg_max-bg_min))*candidate_sample_dist_ratio;
}

Gauss2DsMLE::StatsT Gauss2DsMLE::get_stats() const
{
    StatsT stats=Gauss2DsModel::get_stats();
    stats["hyperparameter.I_min"]=I_min;
    stats["hyperparameter.I_max"]=I_max;
    stats["hyperparameter.BG_min"]=bg_min;
    stats["hyperparameter.BG_max"]=bg_max;
    stats["hyperparameter.Sigma_max"]=sigma_max;
    stats["candidate.etaSigma"]=candidate_eta_sigma;
    return stats;
}

void Gauss2DsMLE::set_hyperparameters(const VecT &hyperparameters)
{
    I_min=hyperparameters(0);
    I_max=hyperparameters(1);
    bg_min=hyperparameters(2);
    bg_max=hyperparameters(3);
    sigma_max=hyperparameters(4);
    //Reset distributions
    I_dist.a(I_min);
    I_dist.b(I_max);
    bg_dist.a(bg_min);
    bg_dist.b(bg_max);
    sigma_dist.b(sigma_max);
}

bool Gauss2DsMLE::theta_in_bounds(const ParamT &theta) const
{
    bool xOK =     (theta(0)>=0)         && (theta(0)<=size(0));
    bool yOK =     (theta(1)>=0)         && (theta(1)<=size(1));
    bool IOK =     (theta(2)>=I_min)     && (theta(2)<=I_max);
    bool bgOK =    (theta(3)>=bg_min)    && (theta(3)<=bg_max);
    bool sigmaOK = (theta(4)>=1.0) && (theta(4)<=sigma_max);
    return xOK && yOK && IOK && bgOK && sigmaOK;
}

void Gauss2DsMLE::bound_theta(ParamT &theta) const
{
    theta(0)=restrict_value_range(theta(0), 0, size(0));       // Prior: Uniform on [0,size(0)]
    theta(1)=restrict_value_range(theta(1), 0, size(1));       // Prior: Uniform on [0,size(1)]
    theta(2)=restrict_value_range(theta(2), I_min, I_max);  // Prior: Uniform on [I_min,I_max]
    theta(3)=restrict_value_range(theta(3), bg_min, bg_max);// Prior: Uniform on [bg_min,bg_max]
    theta(4)=restrict_value_range(theta(4), 1.0, sigma_max);// Prior: Uniform on [bg_min,bg_max]
}

double Gauss2DsMLE::prior_log_likelihood(const Stencil &s) const
{
    return 0;
}

double Gauss2DsMLE::prior_relative_log_likelihood(const Stencil &s) const
{
    return 0;
}

Gauss2DsMLE::ParamT
Gauss2DsMLE::prior_grad(const Stencil &s) const
{
    return ParamT(arma::fill::zeros);
}

Gauss2DsMLE::ParamT
Gauss2DsMLE::prior_grad2(const Stencil &s) const
{
    return ParamT(arma::fill::zeros);
}

Gauss2DsMLE::ParamT
Gauss2DsMLE::prior_cr_lower_bound(const Stencil &s) const
{
    return ParamT(arma::fill::zeros);
}

/* Template Specializations */
template<>
Gauss2DsMLE::Stencil
CGaussHeuristicMLE<Gauss2DsMLE>::compute_estimate(const ImageT &im, const ParamT &theta_init)
{
    Gauss2DsMLE::ParamT theta_est(arma::fill::zeros);
    if(model.size(0)==model.size(1) && model.psf_sigma(0)==model.psf_sigma(1)){ //only works for square images and iso-tropic psf
        float Nmax;
        arma::fvec5 ftheta_est;
        //Convert from double
        arma::fmat fim=arma::conv_to<arma::fmat>::from(im);
        //Compute
        CenterofMass2D(model.size(0), fim.memptr(), &ftheta_est(0), &ftheta_est(1));
        GaussFMaxMin2D(model.size(0), model.psf_sigma(0), fim.memptr(), &Nmax, &ftheta_est(3));
        ftheta_est(2)=std::max(0., (Nmax-ftheta_est(3)) * 2 * arma::datum::pi * model.psf_sigma(0) * model.psf_sigma(0));
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
CGaussMLE<Gauss2DsMLE>::compute_estimate(const ImageT &im, const ParamT &theta_init, ParamT &theta, ParamT &crlb, double &llh)
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
