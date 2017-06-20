
/** @file Gauss2DsMLE.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-22-2014
 * @brief The class declaration and inline and templated functions for Gauss2DsMLE.
 */

#ifndef _GAUSS2DSMLE_H
#define _GAUSS2DSMLE_H

#include "Gauss2DsModel.h"


/** @brief A 2D Likelyhood model for a point emitter localization with 
 * Symmetric Gaussian PSF and Poisson Noise, using a  Uniform Prior over the 
 * parameter vectors
 *
 * This model matches the model used in cGaussMLE.  
 * So we can use this as comparison.
 * 
 */
class Gauss2DsMLE : public Gauss2DsModel {
private:
    /* Theta prior parameters */
    double I_min=1e1; /**< The minimum intensity for our Uniform prior */
    double I_max=1e5; /**< The maximum intensity for our Uniform prior */
    double bg_min=1.0e-6; /**< The minimum bg for our Uniform prior (estimating bg=0 is bad for the numerics) */
    double bg_max=1e2; /**< The maximum bg for our Uniform prior */
    double sigma_min=0.5;
    double sigma_max=3.0; /**< The maximum bg for our Uniform prior */
    UniformRNG pos_dist;
    UniformRNG I_dist;
    UniformRNG bg_dist;
    UniformRNG sigma_dist;
public:
    static const std::vector<std::string> hyperparameter_names;

    /* Constructor/Destructor */
    Gauss2DsMLE(const IVecT &size, const VecT &psf_sigma);

    /* Model values setting and information */
    std::string name() const {return "Gauss2DsMLE";}
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
Gauss2DsMLE::ParamT
Gauss2DsMLE::sample_prior(RNG &rng)
{
    ParamT theta;
    theta(0)=pos_dist(rng); // Prior: Uniform on [0,size]
    theta(1)=pos_dist(rng); // Prior: Uniform on [0,size]
    theta(2)=I_dist(rng);   // Prior: Uniform on [I_min,I_max]
    theta(3)=bg_dist(rng);  // Prior: Uniform on [bg_min,bg_max]
    theta(4)=sigma_dist(rng);  // Prior: Uniform on [sigma_min,sigma_max]
    return theta;
}

#endif /* _GAUSS2DSMLE_H */
