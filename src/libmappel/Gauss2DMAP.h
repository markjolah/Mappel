
/** @file Gauss2DMAP.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 04-2017
 * @brief The class declaration and inline and templated functions for Gauss2DMAP.
 */

#ifndef _GAUSS2DMAP_H
#define _GAUSS2DMAP_H

#include "Gauss2DModel.h"
#include "PoissonNoise2DObjective.h"
namespace mappel {

/** @brief A 2D Gaussian with fixed PSF under an Poisson Read Noise assumption and MAP Objective
 * 
 *   Model: Gauss2DModel a 2D gaussian PSF with fixed psf_sigma
 *   Objective: PoissonNoise2DMAPObjective an MLE objective for Poisson noise
 * 
 * 
 */
class Gauss2DMAP : public Gauss2DModel, public PoissonNoise2DObjective {
public:
    /* Constructor/Destructor */
    Gauss2DMAP(const IVecT &size, const VecT &psf_sigma): ImageFormat2DBase(size), Gauss2DModel(size,psf_sigma), PoissonNoise2DObjective(size) {};

    /* Model values setting and information */
    std::string name() const {return "Gauss2DMAP";}
    
    double prior_log_likelihood(const ParamT &theta) const;
    double prior_relative_log_likelihood(const ParamT &theta) const;
    void prior_grad_update(const ParamT &theta, ParamVecT &grad) const;
    void prior_grad2_update(const ParamT &theta, ParamVecT &grad2) const;
    void prior_hess_update(const ParamT &theta, MatT &hess) const;
};

/* Inline Method Definitions */
inline
double Gauss2DMAP::prior_log_likelihood(const ParamT &theta) const
{
    double rllh = prior_relative_log_likelihood(theta);
    return rllh + 2*log_prior_pos_const + log_prior_I_const + log_prior_bg_const;
}

inline
double Gauss2DMAP::prior_relative_log_likelihood(const ParamT &theta) const
{
    double xrllh = rllh_beta_prior(beta_pos, theta(0), size(0));
    double yrllh = rllh_beta_prior(beta_pos, theta(1), size(1));
    double Irllh = rllh_gamma_prior(kappa_I, mean_I, theta(2));
    double bgrllh = rllh_gamma_prior(kappa_bg, mean_bg, theta(3));
    return xrllh+yrllh+Irllh+bgrllh;
}

inline
void Gauss2DMAP::prior_grad_update(const ParamT &theta, ParamVecT &grad) const
{
    grad(0) += beta_prior_grad(beta_pos, theta(0), size(0));
    grad(1) += beta_prior_grad(beta_pos, theta(1), size(1));
    grad(2) += gamma_prior_grad(kappa_I, theta(2), theta(2));
    grad(3) += gamma_prior_grad(kappa_bg, theta(3), theta(3));
}

inline
void Gauss2DMAP::prior_grad2_update(const ParamT &theta, ParamVecT &grad2) const
{
    grad2(0) += beta_prior_grad2(beta_pos, theta(0), size(0));
    grad2(1) += beta_prior_grad2(beta_pos, theta(1), size(1));
    grad2(2) += gamma_prior_grad2(kappa_I, theta(2));
    grad2(3) += gamma_prior_grad2(kappa_bg, theta(3));
}

inline
void Gauss2DMAP::prior_hess_update(const ParamT &theta, MatT &hess) const
{
    hess(0,0) += beta_prior_grad2(beta_pos, theta(0), size(0));
    hess(1,1) += beta_prior_grad2(beta_pos, theta(1), size(1));
    hess(2,2) += gamma_prior_grad2(kappa_I, theta(2));
    hess(3,3) += gamma_prior_grad2(kappa_bg, theta(3));
}

} /* namespace mappel */

#endif /* _GAUSS2DMAP_H */
