
/** @file Gauss1DMLE.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2013-2017
 * @brief The class declaration and inline and templated functions for Gauss1DMLE.
 */

#ifndef _MAPPEL_GAUSS1DMLE_H
#define _MAPPEL_GAUSS1DMLE_H

#include "MLEstimator.h"
#include "Gauss1DModel.h"
#include "PoissonNoise1DObjective.h"

namespace mappel {

/** @brief A 1D Gaussian with fixed PSF under an Poisson noise assumption and maximum-likelihood estimator
 * 
 *   Model: Gauss1DModel - 1D Gaussian PSF with fixed sigma
 *   Objective: PoissonNoise1DObjective - Poisson noise model for 1D
 *   Estimator: MLEstimator - Pure-likelihood estimator
 */
class Gauss1DMLE : public Gauss1DModel, public PoissonNoise1DObjective, public MLEstimator 
{
public:
    Gauss1DMLE(ImageSizeVecT size, VecT psf_sigma) : 
            PointEmitterModel(make_prior(size(0))), 
            ImageFormat1DBase(size(0)),
            Gauss1DModel(size(0), psf_sigma(0))
    { }

    Gauss1DMLE(int size, double psf_sigma) : 
            PointEmitterModel(make_prior(size)), 
            ImageFormat1DBase(size),
            Gauss1DModel(size, psf_sigma)
    { }
    
    Gauss1DMLE(int size, double psf_sigma, CompositeDist&& prior) : 
            PointEmitterModel(std::move(prior)), 
            ImageFormat1DBase(size),
            Gauss1DModel(size, psf_sigma)
    { }
     
    
    static std::string name() {return "Gauss1DMLE";}
};
 
} /* namespace mappel */

#endif /* _MAPPEL_GAUSS1DMLE_H */
