
/** @file Gauss1DMAP.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 04-2017
 * @brief The class declaration and inline and templated functions for Gauss1DMAP.
 */

#ifndef _GAUSS1DMAP_H
#define _GAUSS1DMAP_H

#include "Gauss1DModel.h"
#include "PoissonNoise1DObjective.h"

namespace mappel {

/** @brief A 1D Gaussian with fixed PSF under an Poisson Read Noise assumption and MAP Objective
 * 
 *   Model: Gauss1DModel a 1D gaussian PSF with fixed psf_sigma
 *   Objective: PoissonNoise1DMAPObjective an MLE objective for Poisson noise
 * 
 * 
 */

class Gauss1DMAP : public Gauss1DModel, public PoissonNoise1DObjective {
public:
    /* Constructor/Destructor */
    Gauss1DMAP(int size, double psf_sigma): ImageFormat1DBase(size), Gauss1DModel(size,psf_sigma), PoissonNoise1DObjective(size) {};

    /* Model values setting and information */
    std::string name() const {return "Gauss1DMAP";}
    
    double prior_log_likelihood(const ParamT &theta) const;
    double prior_relative_log_likelihood(const ParamT &theta) const;
    void prior_grad_update(const ParamT &theta, ParamVecT &grad) const;
    void prior_grad2_update(const ParamT &theta, ParamVecT &grad2) const;
    void prior_hess_update(const ParamT &theta, ParamMatT &hess) const;
};

/* Inline Method Definitions */
inline
double Gauss1DMAP::prior_log_likelihood(const ParamT &theta) const
{
    double rllh = prior_relative_log_likelihood(theta);
    return rllh + log_prior_pos_const + log_prior_I_const + log_prior_bg_const;
}

inline
double Gauss1DMAP::prior_relative_log_likelihood(const ParamT &theta) const
{
    double xrllh = rllh_beta_prior(beta_pos, theta(0), size);
    double Irllh = rllh_gamma_prior(kappa_I, mean_I, theta(1));
    double bgrllh = rllh_gamma_prior(kappa_bg, mean_bg, theta(2));
    return xrllh+Irllh+bgrllh;
}

inline
void Gauss1DMAP::prior_grad_update(const ParamT &theta, ParamVecT &grad) const
{
    grad(0) += beta_prior_grad(beta_pos, theta(0), size);
    grad(1) += gamma_prior_grad(kappa_I, theta(1), theta(1));
    grad(2) += gamma_prior_grad(kappa_bg, theta(2), theta(2));
}

inline
void Gauss1DMAP::prior_grad2_update(const ParamT &theta, ParamVecT &grad2) const
{
    grad2(0) += beta_prior_grad2(beta_pos, theta(0));
    grad2(1) += gamma_prior_grad2(kappa_I, theta(1));
    grad2(2) += gamma_prior_grad2(kappa_bg, theta(2));
}

inline
void Gauss1DMAP::prior_hess_update(const ParamT &theta, ParamMatT &hess) const
{
    hess(0,0) += beta_prior_grad2(beta_pos, theta(0), size);
    hess(1,1) += gamma_prior_grad2(kappa_I, theta(1));
    hess(2,2) += gamma_prior_grad2(kappa_bg, theta(2));
}

} /* namespace mappel */

#endif /* _GAUSS1DMAP_H */
