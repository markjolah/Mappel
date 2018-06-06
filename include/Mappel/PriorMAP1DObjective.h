
/** @file PriorMAP1DObjective.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-22-2014
 * @brief The class declaration and inline and templated functions for PriorMAP1DObjective.
 */

#ifndef _PRIORMAP1DOBJECTIVE_H
#define _PRIORMAP1DOBJECTIVE_H

#include <armadillo>
#include "Mappel/stencil.h"

namespace mappel {

/** @brief A Mixin class to configure a Gauss1DModel for MAP estimation (default 1D prior).
 */
class PriorMAP1DObjective {
protected:
    /* Datatypes for prior objective calculations */
    using ParamT = arma::vec;
    using ParamMatT = arma::mat;
    
    
    
    
    /* Constants for prior calculations */
    double log_prior_pos_const; /**< This is -2*lgamma(beta_x)-lgamma(2*beta_x) */
    double log_prior_I_const; /**< This is kappa_I*(log(kappa_I)-1/mean_I-log(mean_I))-lgamma(kappa_I) */
    double log_prior_bg_const; /**< This is kappa_bg*(log(kappa_bg)-1/mean_bg-log(mean_bg))-lgamma(kappa_bg) */
    void set_hyperparameters(double beta_x, double mean_I, double kappa_I, double mean_bg, double kappa_bg)

public:
    PriorMAP1DObjective(double beta_x, double mean_I, double kappa_I, double mean_bg, double kappa_bg):
        log_prior_pos_const(log_prior_beta_const(beta_pos)),
        log_prior_I_const(log_prior_gamma_const(kappa_I,mean_I)),
        log_prior_bg_const(log_prior_gamma_const(kappa_bg,mean_bg)) {}

    double prior_log_likelihood(const ParamT &theta) const;
    double prior_relative_log_likelihood(const ParamT &theta) const;
    void prior_grad_update(const ParamT &theta, ParamT &grad) const;
    void prior_grad2_update(const ParamT &theta, ParamT &grad2) const;
    void prior_hess_update(const ParamT &theta, ParamMatT &hess) const;
};

/* Inline Method Definitions */
inline
double PriorMAP1DObjective::prior_log_likelihood(const ParamT &theta) const
{
    double rllh = prior_relative_log_likelihood(theta);
    return rllh + log_prior_pos_const + log_prior_I_const + log_prior_bg_const;
}

inline
double PriorMAP1DObjective::prior_relative_log_likelihood(const ParamT &theta) const
{
    double xrllh = rllh_beta_prior(beta_pos, theta(0), size);
    double Irllh = rllh_gamma_prior(kappa_I, mean_I, theta(1));
    double bgrllh = rllh_gamma_prior(kappa_bg, mean_bg, theta(2));
    return xrllh+Irllh+bgrllh;
}

inline
void PriorMAP1DObjective::prior_grad_update(const ParamT &theta, ParamT &grad) const
{
    grad(0) += beta_prior_grad(beta_pos, theta(0), size);
    grad(1) += gamma_prior_grad(kappa_I, mean_I, theta(1));
    grad(2) += gamma_prior_grad(kappa_bg, mean_bg, theta(2));
}

inline
void PriorMAP1DObjective::prior_grad2_update(const ParamT &theta, ParamT &grad2) const
{
    grad2(0) += beta_prior_grad2(beta_pos, theta(0));
    grad2(1) += gamma_prior_grad2(kappa_I, theta(1));
    grad2(2) += gamma_prior_grad2(kappa_bg, theta(2));
}

inline
void PriorMAP1DObjective::prior_hess_update(const ParamT &theta, ParamMatT &hess) const
{
    hess(0,0) += beta_prior_grad2(beta_pos, theta(0), size);
    hess(1,1) += gamma_prior_grad2(kappa_I, theta(1));
    hess(2,2) += gamma_prior_grad2(kappa_bg, theta(2));
}


} /* namespace mappel */

#endif /* _PRIORMAP1DOBJECTIVE_H */
