
/** @file Gauss2DsxyMAP.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief The class declaration and inline and templated functions for Gauss2DsxyMAP.
 */

#ifndef _MAPPEL_GAUSS2DSXYMAP_H
#define _MAPPEL_GAUSS2DSXYMAP_H

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
class Gauss2DsxyMAP : public Gauss1DModel, public PoissonNoise1DObjective, public MAPEstimator {
public:    
    Gauss2DsxyMAP(const ImageSizeT &size, const VecT &min_sigma, const VecT &max_sigma);
    Gauss2DsxyMAP(const ImageSizeT &size, const VecT &min_sigma, const VecT &max_sigma, CompositeDist&& prior);     
    
    static const std::string name;
};

} /* namespace mappel */

#endif /* _MAPPEL_GAUSS2DSXYMAP_H */
