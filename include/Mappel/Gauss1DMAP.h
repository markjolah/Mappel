/** @file Gauss1DMAP.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class declaration and inline and templated functions for Gauss1DMAP.
 */

#ifndef MAPPEL_GAUSS1DMAP_H
#define MAPPEL_GAUSS1DMAP_H

#include "Mappel/Gauss1DModel.h"
#include "Mappel/PoissonNoise1DObjective.h"
#include "Mappel/MAPEstimator.h"
#include "Mappel/model_methods.h"

namespace mappel {

/** @brief A 1D Gaussian with fixed PSF under an Poisson Read Noise assumption and MAP Objective
 * 
 *   Model: Gauss1DModel - 1D Gaussian PSF with fixed PSF sigma
 *   Objective: PoissonNoise1DObjective - Poisson noise model for 1D
 *   Estimator: MAPstimator - Maximum a-posteriori estimator
 */
class Gauss1DMAP : public Gauss1DModel, public PoissonNoise1DObjective, public MAPEstimator 
{
public:    
    Gauss1DMAP(arma::Col<ImageCoordT> size, VecT psf_sigma, const std::string &prior_type = DefaultPriorType);
    Gauss1DMAP(ImageSizeT size, double psf_sigma, const std::string &prior_type = DefaultPriorType);
    Gauss1DMAP(ImageSizeT size, double psf_sigma, CompositeDist&& prior);     
    Gauss1DMAP(ImageSizeT size, double psf_sigma, const CompositeDist& prior);     
    Gauss1DMAP(const Gauss1DMAP &o);
    Gauss1DMAP& operator=(const Gauss1DMAP &o);
    Gauss1DMAP(Gauss1DMAP &&o);
    Gauss1DMAP& operator=(Gauss1DMAP &&o);
    static const std::string name;
};

} /* namespace mappel */

#endif /* MAPPEL_GAUSS1DMAP_H */
