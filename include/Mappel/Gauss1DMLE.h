/** @file Gauss1DMLE.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2018
 * @brief The class declaration and inline and templated functions for Gauss1DMLE.
 */

#ifndef _MAPPEL_GAUSS1DMLE_H
#define _MAPPEL_GAUSS1DMLE_H

#include "Mappel/Gauss1DModel.h"
#include "Mappel/PoissonNoise1DObjective.h"
#include "Mappel/MLEstimator.h"
#include "Mappel/model_methods.h"

namespace mappel {

/** @brief A 1D Gaussian with fixed PSF under an Poisson noise assumption and maximum-likelihood objective
 * 
 *   Model: Gauss1DModel - 1D Gaussian PSF with fixed PSF sigma
 *   Objective: PoissonNoise1DObjective - Poisson noise model for 1D
 *   Estimator: MLEstimator - Pure-likelihood estimator
 */
class Gauss1DMLE : public Gauss1DModel, public PoissonNoise1DObjective, public MLEstimator 
{
public:
    Gauss1DMLE(arma::Col<ImageCoordT> size, VecT psf_sigma);
    Gauss1DMLE(ImageSizeT size, double psf_sigma);
    Gauss1DMLE(ImageSizeT size, double psf_sigma, CompositeDist&& prior);   
    Gauss1DMLE(ImageSizeT size, double psf_sigma, const CompositeDist& prior);   
    Gauss1DMLE(const Gauss1DMLE &o);
    Gauss1DMLE& operator=(const Gauss1DMLE &o);
    Gauss1DMLE(Gauss1DMLE &&o);
    Gauss1DMLE& operator=(Gauss1DMLE &&o);    
    static const std::string name;
};
 
} /* namespace mappel */

#endif /* _MAPPEL_GAUSS1DMLE_H */
