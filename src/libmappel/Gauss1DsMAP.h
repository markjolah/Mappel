
/** @file Gauss1DsMAP.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief The class declaration and inline and templated functions for Gauss1DsMAP.
 */

#ifndef _MAPPEL_GAUSS1DSMAP_H
#define _MAPPEL_GAUSS1DSMAP_H

#include "Gauss1DsModel.h"
#include "PoissonNoise1DObjective.h"
#include "MAPEstimator.h"
#include "model_methods.h" //Declaration of methods

namespace mappel {

/** @brief A 1D Gaussian with variable PSF sigma under an Poisson read noise assumption and MAP Objective
 * 
 *   Model: Gauss1DsModel a 1D gaussian PSF with variable psf_sigma
 *   Objective Statistical Noise Model: PoissonNoise1DMAPObjective an MLE objective for Poisson noise
 *   ImageFormat: ImageFormat1DBase - Data format
 */
class Gauss1DsMAP : public Gauss1DsModel, public PoissonNoise1DObjective, public MAPEstimator 
{
public:    
    Gauss1DsMAP(arma::Col<ImageCoordT> size, VecT min_sigma, VecT max_sigma);
    Gauss1DsMAP(ImageSizeT size, double min_sigma, double max_sigma);
    Gauss1DsMAP(ImageSizeT size, CompositeDist&& prior);     
    
    static const std::string name;
};

} /* namespace mappel */

#endif /* _MAPPEL_GAUSS1DSMAP_H */
