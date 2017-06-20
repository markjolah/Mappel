
/** @file Gauss2DsMAP.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-18-2014
 * @brief The class declaration and inline and templated functions for Gauss2DsMAP.
 */

#ifndef _GAUSS2DSMAP_H
#define _GAUSS2DSMAP_H

#include "Gauss2DsModel.h"


/** @brief A 2D Likelyhood model for a point emitter localization with
 * Symmetric Gaussian PSF and Poisson Noise, using a
 *   * Beta prior over positions with \alpha=\beta>1.0;
 *   * Gamma prior over I and bg with k~1.5-4.
 *
 */
class Gauss2DsMAP : public Gauss2DsModel {
private:
    /* Hyperparameters */
    double beta_pos=1.5; /**< The shape parameter for the Beta prior on the x and y positions. 0=Uniform, 1=Peaked  */
    double mean_I=1000.; /**< The mean of the intensity gamma prior */
    double kappa_I=1.1;  /**< The shape parameter for the I prior gamma distribution 1=exponential 2-5=skewed large=normal */
    double mean_bg=3.; /**< The mean of the background gamma prior */
    double kappa_bg=1.1;  /**< The shape parameter for the bg prior gamma distribution 1=exponential 2-5=skewed large=normal */
    double alpha_sigma=3.;  /**< The shape parameter for the bg prior gamma distribution 1=exponential 2-5=skewed large=normal */
    double sigma_min=0.5;
    double kappa_sigma=4.;
    double mean_sigma=1.4;
    
    BetaRNG pos_dist;
    GammaRNG I_dist;
    GammaRNG bg_dist;
    ParetoRNG sigma_dist;

    double log_prior_pos_const; /**< This is -2*lgamma(beta_x)-lgamma(2*beta_x) */
    double log_prior_I_const; /**< This is kappa_I*(log(kappa_I)-1/mean_I-log(mean_I))-lgamma(kappa_I) */
    double log_prior_bg_const; /**< This is kappa_bg*(log(kappa_bg)-1/mean_bg-log(mean_bg))-lgamma(kappa_bg) */
    double log_prior_sigma_const;
public:
    static const std::vector<std::string> hyperparameter_names;

    /* Constructor/Destructor */
    Gauss2DsMAP(const IVecT &size, const VecT &psf_sigma);

    /* Model values setting and information */
    std::string name() const {return "Gauss2DsMAP";}
    StatsT get_stats() const;

    /* Sample from Theta Prior */
    ParamT sample_prior(RNG &rng);
    void set_hyperparameters(const VecT &hyperparameters);
    void bound_theta(ParamT &theta) const;
    bool theta_in_bounds(const ParamT &theta) const;

    double prior_log_likelihood(const Stencil &s) const;
    double prior_relative_log_likelihood(const Stencil &s) const;
    ParamT prior_grad(const Stencil &s) const;
    ParamT prior_grad2(const Stencil &s) const;
    ParamT prior_cr_lower_bound(const Stencil &s) const;
};

/* Template Specialization Declarations */

/* Inlined Methods */
inline
Gauss2DsMAP::ParamT
Gauss2DsMAP::sample_prior(RNG &rng)
{
    ParamT theta=make_param();
    theta(0)=size(0)*pos_dist(rng);
    theta(1)=size(1)*pos_dist(rng);
    theta(2)=I_dist(rng);
    theta(3)=bg_dist(rng);
    theta(4)=sigma_dist(rng);
    bound_theta(theta);
    return theta;
}

#endif /* _GAUSS2DSMAP_H */
