
/** @file Gauss1DMLE.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2013-2017
 * @brief The class declaration and inline and templated functions for Gauss1DMLE.
 */

#ifndef _MAPPEL_GAUSS1DMLE_H
#define _MAPPEL_GAUSS1DMLE_H

#include "Gauss1DModel.h"
#include "PoissonNoise1DObjective.h"
#include "MLEstimator.h"

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
    Gauss1DMLE(ImageSizeVecT size, VecT psf_sigma);
    Gauss1DMLE(ImageSizeT size, double psf_sigma);
    template<class PriorDistT>
    Gauss1DMLE(ImageSizeT size, double psf_sigma, PriorDistT&& prior);     
    
    static std::string name() {return "Gauss1DMLE";}

    friend std::ostream& operator<<(std::ostream &out, Gauss1DMLE &model);
};
 
} /* namespace mappel */

#endif /* _MAPPEL_GAUSS1DMLE_H */
