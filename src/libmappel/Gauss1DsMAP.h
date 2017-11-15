
/** @file Gauss1DsMAP.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief The class declaration and inline and templated functions for Gauss1DsMAP.
 */

#ifndef _GAUSS1DSMAP_H
#define _GAUSS1DSMAP_H

#include "Gauss1DsModel.h"
#include "PoissonNoise1DObjective.h"

namespace mappel {

/** @brief A 1D Gaussian with variable sigma under an Poisson Read Noise assumption and a MAP Objective
 * 
 *   Model: Gauss1DsModel a 1D Gaussian PSF with variable Gaussian sigma
 *   Objective: PoissonNoise1DMAPObjective an MLE objective for Poisson noise
 * 
 * 
 */

class Gauss1DsMAP : public Gauss1DsModel, public PoissonNoise1DObjective {
public:
    /* Constructor/Destructor */
    Gauss1DsMAP(int size, double min_sigma, double max_sigma) : 
        ImageFormat1DBase(size), 
        Gauss1DsModel(size,min_sigma,max_sigma), 
        PoissonNoise1DObjective(size) {};

    /* Model values setting and information */
    std::string name() const {return "Gauss1DsMAP";}
    
    double prior_log_likelihood(const ParamT &theta) const;
    double prior_relative_log_likelihood(const ParamT &theta) const;
    void prior_grad_update(const ParamT &theta, ParamVecT &grad) const;
    void prior_grad2_update(const ParamT &theta, ParamVecT &grad2) const;
    void prior_hess_update(const ParamT &theta, ParamMatT &hess) const;
};

/* Inline Method Definitions */
inline
double Gauss1DsMAP::prior_log_likelihood(const ParamT &theta) const
{
    double rllh = prior_relative_log_likelihood(theta);
    return rllh + log_prior_pos_const + log_prior_I_const + log_prior_bg_const + log_prior_sigma_const;
}

inline
double Gauss1DsMAP::prior_relative_log_likelihood(const ParamT &theta) const
{
    double xrllh = rllh_beta_prior(beta_pos, theta(0), size);
    double Irllh = rllh_gamma_prior(kappa_I, mean_I, theta(1));
    double bgrllh = rllh_gamma_prior(kappa_bg, mean_bg, theta(2));
    double sigmarllh = rllh_pareto_prior(alpha_sigma, theta(3));
    return xrllh+Irllh+bgrllh+sigmarllh;
}

inline
void Gauss1DsMAP::prior_grad_update(const ParamT &theta, ParamVecT &grad) const
{
    grad(0) += beta_prior_grad(beta_pos, theta(0), size);
    grad(1) += gamma_prior_grad(kappa_I, mean_I, theta(1));
    grad(2) += gamma_prior_grad(kappa_bg, mean_bg, theta(2));
    grad(3) += pareto_prior_grad(alpha_sigma, theta(3));
}

inline
void Gauss1DsMAP::prior_grad2_update(const ParamT &theta, ParamVecT &grad2) const
{
    grad2(0) += beta_prior_grad2(beta_pos, theta(0));
    grad2(1) += gamma_prior_grad2(kappa_I, theta(1));
    grad2(2) += gamma_prior_grad2(kappa_bg, theta(2));
    grad2(3) += pareto_prior_grad2(alpha_sigma, theta(3));
}

inline
void Gauss1DsMAP::prior_hess_update(const ParamT &theta, ParamMatT &hess) const
{
    hess(0,0) += beta_prior_grad2(beta_pos, theta(0), size);
    hess(1,1) += gamma_prior_grad2(kappa_I, theta(1));
    hess(2,2) += gamma_prior_grad2(kappa_bg, theta(2));
    hess(3,3) += pareto_prior_grad2(alpha_sigma, theta(3));
}

} /* namespace mappel */

#endif /* _GAUSS1DSMAP_H */
