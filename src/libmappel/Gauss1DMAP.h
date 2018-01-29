
/** @file Gauss1DMAP.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief The class declaration and inline and templated functions for Gauss1DMAP.
 */

#ifndef _MAPPEL_GAUSS1DMAP_H
#define _MAPPEL_GAUSS1DMAP_H

#include "PoissonNoise1DObjective.h"
#include "MAPEstimator.h"
#include "Gauss1DModel.h"
#include "model_methods.h" //Declaration of methods

namespace mappel {

/** @brief A 1D Gaussian with fixed PSF under an Poisson Read Noise assumption and MAP Objective
 * 
 *   Model: Gauss1DModel a 1D gaussian PSF with fixed psf_sigma
 *   Objective Statistical Noise Model: PoissonNoise1DMAPObjective an MLE objective for Poisson noise
 *   ImageFormat: ImageFormat1DBase - Data format
 * 
 */
class Gauss1DMAP : public Gauss1DModel, public PoissonNoise1DObjective, public MAPEstimator {
public:    
    Gauss1DMAP(arma::Col<ImageCoordT> size, VecT psf_sigma);
    
    Gauss1DMAP(ImageSizeT size, double psf_sigma);
    
    template<class PriorDistT>
    Gauss1DMAP(ImageSizeT size, double psf_sigma, PriorDistT&& prior);     
    
    static const std::string name;
};

} /* namespace mappel */

#endif /* _MAPPEL_GAUSS1DMAP_H */
