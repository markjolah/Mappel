
/** @file Gauss1DsMLE.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief The class declaration and inline and templated functions for Gauss1DsMLE.
 */

#ifndef _GAUSS1DSMLE_H
#define _GAUSS1DSMLE_H

#include "Gauss1DsModel.h"
#include "PoissonNoise1DObjective.h"

namespace mappel {

/** @brief A 1D Gaussian with variable sigma under an Poisson Read Noise assumption and a MLE Objective
 * 
 *   Model: Gauss1DsModel a 1D Gaussian PSF with variable Gaussian sigma
 *   Objective: PoissonNoise1DMAPObjective an MLE objective for Poisson noise
 * 
 * 
 */

class Gauss1DsMLE : public Gauss1DsModel, public PoissonNoise1DObjective {
public:
    /* Constructor/Destructor */
    Gauss1DsMLE(int size, double min_sigma, double max_sigma) : 
        ImageFormat1DBase(size), 
        Gauss1DsModel(size,min_sigma,max_sigma), 
        PoissonNoise1DObjective(size) {};

    /* Model values setting and information */
    std::string name() const {return "Gauss1DsMLE";}
    
    double prior_log_likelihood(const ParamT &theta) const;
    double prior_relative_log_likelihood(const ParamT &theta) const;
    void prior_grad_update(const ParamT &theta, ParamVecT &grad) const;
    void prior_grad2_update(const ParamT &theta, ParamVecT &grad2) const;
    void prior_hess_update(const ParamT &theta, MatT &hess) const;
};

/* Inline Method Definitions */
inline
double Gauss1DsMLE::prior_log_likelihood(const ParamT &theta) const
{
    return 0;
}

inline
double Gauss1DsMLE::prior_relative_log_likelihood(const ParamT &theta) const
{
    return 0;
}

inline
void Gauss1DsMLE::prior_grad_update(const ParamT &theta, ParamVecT &grad) const
{
    return;
}

inline
void Gauss1DsMLE::prior_grad2_update(const ParamT &theta, ParamVecT &grad2) const
{
    return;
}

inline
void Gauss1DsMLE::prior_hess_update(const ParamT &theta, MatT &hess) const
{
    return;
}

} /* namespace mappel */

#endif /* _GAUSS1DSMLE_H */
