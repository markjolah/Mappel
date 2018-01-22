
/** @file Gauss1DsMLE.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief The class declaration and inline and templated functions for Gauss1DsMLE.
 */

#ifndef _MAPPEL_GAUSS1DSMLE_H
#define _MAPPEL_GAUSS1DSMLE_H

#include "Gauss1DsModel.h"
#include "PoissonNoise1DObjective.h"
#include "MLEstimator.h"
#include "model_methods.h" //Declaration of methods

namespace mappel {

/** @brief A 1D Gaussian with variable PSF under an Poisson noise assumption and maximum-likelihood estimator
 * 
 *   Model: Gauss1DsModel - 1D Gaussian PSF with variable PSF sigma
 *   Objective: PoissonNoise1DObjective - Poisson noise model for 1D
 *   Estimator: MLEstimator - Pure-likelihood estimator
 */
class Gauss1DsMLE : public Gauss1DsModel, public PoissonNoise1DObjective, public MLEstimator 
{
public:
    Gauss1DsMLE(arma::Col<ImageCoordT> size, VecT min_sigma, VecT max_sigma);

    Gauss1DsMLE(ImageSizeT size, double min_sigma, double max_sigma);

    template<class PriorDistT>
    Gauss1DsMLE(ImageSizeT size, PriorDistT&& prior);     
    
    static std::string name() {return "Gauss1DsMLE";}
};
 
} /* namespace mappel */

#endif /* _MAPPEL_GAUSS1DSMLE_H */
    
