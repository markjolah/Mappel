/** @file Gauss2DMAP.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2018
 * @brief The class declaration and inline and templated functions for Gauss2DMAP.
 */

#ifndef _MAPPEL_GAUSS2DMAP_H
#define _MAPPEL_GAUSS2DMAP_H

#include "Mappel/Gauss2DModel.h"
#include "Mappel/PoissonNoise2DObjective.h"
#include "Mappel/MAPEstimator.h"
#include "Mappel/model_methods.h"

namespace mappel {

/** @brief A 2D Gaussian with fixed PSF under an Poisson Read Noise assumption and MAP Objective
 * 
 *   Model: Gauss2DModel a 2D gaussian PSF with fixed psf_sigma
 *   Objective: PoissonNoise2DObjective - Poisson noise model for 2D
 *   Estimator: MAPEstimator - Maximum a-posteriori estimator
 */
class Gauss2DMAP : public Gauss2DModel, public PoissonNoise2DObjective, public MAPEstimator 
{
public:
    Gauss2DMAP(ImageCoordT size, double psf_sigma);
    Gauss2DMAP(const ImageSizeT &size, double psf_sigma);
    Gauss2DMAP(const ImageSizeT &size, const VecT &psf_sigma);
    Gauss2DMAP(const ImageSizeT &size, const VecT &psf_sigma, CompositeDist&& prior);     
    Gauss2DMAP(const ImageSizeT &size, const VecT &psf_sigma, const CompositeDist& prior);
    Gauss2DMAP(const Gauss2DMAP &o);
    Gauss2DMAP& operator=(const Gauss2DMAP &o);
    Gauss2DMAP(Gauss2DMAP &&o);
    Gauss2DMAP& operator=(Gauss2DMAP &&o);
    static const std::string name;
};

} /* namespace mappel */

#endif /* _MAPPEL_GAUSS2DMAP_H */