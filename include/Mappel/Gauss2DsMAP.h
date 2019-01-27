
/** @file Gauss2DsMAP.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class declaration and inline and templated functions for Gauss2DsMAP.
 */

#ifndef MAPPEL_GAUSS2DSMAP_H
#define MAPPEL_GAUSS2DSMAP_H

#include "Mappel/Gauss2DsModel.h"
#include "Mappel/PoissonNoise2DObjective.h"
#include "Mappel/MLEstimator.h"
#include "Mappel/model_methods.h"

namespace mappel {

/** @brief A 2D Gaussian with a variable scalar PSF sigma under a Poisson noise assumption using a maximum a-posteriori objective
 * 
 *   Model: Gauss2DsModel - 2D Gaussian variable scalar PSF sigma
 *   Objective: PoissonNoise2DObjective - Poisson noise model for 2D
 *   Estimator: MAPEstimator - Maximum a-posteriori estimator
 */
class Gauss2DsMAP : public Gauss2DsModel, public PoissonNoise2DObjective, public MAPEstimator 
{
public:
    /* Constructor/Destructor */
    Gauss2DsMAP(const ImageSizeT &size, const VecT &min_sigma, double max_sigma_ratio, const std::string &prior_type = DefaultPriorType);
    Gauss2DsMAP(const ImageSizeT &size, const VecT &min_sigma, const VecT &max_sigma, const std::string &prior_type = DefaultPriorType);
    Gauss2DsMAP(const ImageSizeT &size, const VecT &min_sigma, CompositeDist&& prior);     
    Gauss2DsMAP(const ImageSizeT &size, const VecT &min_sigma, const CompositeDist& prior);     
    Gauss2DsMAP(const Gauss2DsMAP &o);
    Gauss2DsMAP& operator=(const Gauss2DsMAP &o);
    Gauss2DsMAP(Gauss2DsMAP &&o);
    Gauss2DsMAP& operator=(Gauss2DsMAP &&o);    
    static const std::string name;
};

} /* namespace mappel */

#endif /* MAPPEL_GAUSS2DSMAP_H */
