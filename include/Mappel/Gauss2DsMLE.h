
/** @file Gauss2DsMLE.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class declaration and inline and templated functions for Gauss2DsMLE.
 */

#ifndef MAPPEL_GAUSS2DSMLE_H
#define MAPPEL_GAUSS2DSMLE_H

#include "Mappel/Gauss2DsModel.h"
#include "Mappel/PoissonNoise2DObjective.h"
#include "Mappel/MLEstimator.h"
#include "Mappel/model_methods.h"

namespace mappel {

/** @brief A 2D Gaussian with a variable scalar PSF sigma under a Poisson noise assumption using a maximum-likelihood objective
 * 
 *   Model: Gauss2DsModel - 2D Gaussian variable scalar PSF sigma
 *   Objective: PoissonNoise2DObjective - Poisson noise model for 2D
 *   Estimator: MLEstimator - Pure-likelihood estimator
 */
class Gauss2DsMLE : public Gauss2DsModel, public PoissonNoise2DObjective, public MLEstimator 
{
public:
    /* Constructor/Destructor */
    Gauss2DsMLE(const ImageSizeT &size, const VecT &min_sigma, double max_sigma_ratio, const std::string &prior_type = DefaultPriorType);
    Gauss2DsMLE(const ImageSizeT &size, const VecT &min_sigma, const VecT &max_sigma, const std::string &prior_type = DefaultPriorType);
    Gauss2DsMLE(const ImageSizeT &size, const VecT &min_sigma, CompositeDist&& prior);     
    Gauss2DsMLE(const ImageSizeT &size, const VecT &min_sigma, const CompositeDist& prior);     
    Gauss2DsMLE(const Gauss2DsMLE &o);
    Gauss2DsMLE& operator=(const Gauss2DsMLE &o);
    Gauss2DsMLE(Gauss2DsMLE &&o);
    Gauss2DsMLE& operator=(Gauss2DsMLE &&o);
    static const std::string name;
};

} /* namespace mappel */

#endif /* MAPPEL_GAUSS2DSMLE_H */
