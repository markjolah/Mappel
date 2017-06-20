
/** @file Gauss2DMLE.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-22-2014
 * @brief The class declaration and inline and templated functions for Gauss2DMLE.
 */

#ifndef _GAUSS2DMLE_H
#define _GAUSS2DMLE_H

#include "Gauss2DModel.h"
#include "PoissonNoise2DObjective.h"

namespace mappel {

/** @brief A 2D Gaussian with fixed PSF under an Poisson Read Noise assumption and MLE Objective
 * 
 *   Model: Gauss2DModel a 2D gaussian PSF with fixed psf_sigma
 *   Objective: PoissonNoise2DMLEObjective an MLE objective for Poisson noise
 * 
 * 
 */
class Gauss2DMLE : public Gauss2DModel, public PoissonNoise2DObjective {
public:
    /* Constructor/Destructor */
    Gauss2DMLE(const IVecT &_size, const VecT &_psf_sigma): ImageFormat2DBase(_size), Gauss2DModel(_size,_psf_sigma), PoissonNoise2DObjective(_size) {};
    
    /* Model values setting and information */
    std::string name() const {return "Gauss2DMLE";}
    
    double prior_log_likelihood(const ParamT &theta) const;
    double prior_relative_log_likelihood(const ParamT &theta) const;
    void prior_grad_update(const ParamT &theta, ParamVecT &grad) const;
    void prior_grad2_update(const ParamT &theta, ParamVecT &grad2) const;
    void prior_hess_update(const ParamT &theta, ParamMatT &hess) const;
};

/* Inline Method Definitions */
inline
double Gauss2DMLE::prior_log_likelihood(const ParamT &theta) const
{
    return 0;
}

inline
double Gauss2DMLE::prior_relative_log_likelihood(const ParamT &theta) const
{
    return 0;
}

inline
void Gauss2DMLE::prior_grad_update(const ParamT &theta, ParamVecT &grad) const
{
    return;
}

inline
void Gauss2DMLE::prior_grad2_update(const ParamT &theta, ParamVecT &grad2) const
{
    return;
}

inline
void Gauss2DMLE::prior_hess_update(const ParamT &theta, ParamMatT &hess) const
{
    return;
}


} /* namespace mappel */

#endif /* _GAUSS2DMLE_H */
