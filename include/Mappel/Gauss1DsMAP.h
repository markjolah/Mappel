
/** @file Gauss1DsMAP.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class declaration and inline and templated functions for Gauss1DsMAP.
 */

#ifndef MAPPEL_GAUSS1DSMAP_H
#define MAPPEL_GAUSS1DSMAP_H

#include "Mappel/Gauss1DsModel.h"
#include "Mappel/PoissonNoise1DObjective.h"
#include "Mappel/MAPEstimator.h"
#include "Mappel/model_methods.h"

namespace mappel {

/** @brief A 1D Gaussian with variable PSF sigma under an Poisson read noise assumption and MAP Objective
 * 
 *   Model: Gauss1DsModel a 1D gaussian PSF with variable psf_sigma
 *   Objective: PoissonNoise1DObjective - Poisson noise model for 1D
 *   Estimator: MAPstimator - Maximum a-posteriori estimator
 */
class Gauss1DsMAP : public Gauss1DsModel, public PoissonNoise1DObjective, public MAPEstimator 
{
public:    
    Gauss1DsMAP(arma::Col<ImageCoordT> size, VecT min_sigma, VecT max_sigma, const std::string &prior_type = DefaultPriorType);
    Gauss1DsMAP(ImageSizeT size, double min_sigma, double max_sigma, const std::string &prior_type = DefaultPriorType);
    Gauss1DsMAP(ImageSizeT size, CompositeDist&& prior);
    Gauss1DsMAP(ImageSizeT size, const CompositeDist& prior);
    Gauss1DsMAP(const Gauss1DsMAP &o);
    Gauss1DsMAP& operator=(const Gauss1DsMAP &o);
    Gauss1DsMAP(Gauss1DsMAP &&o);
    Gauss1DsMAP& operator=(Gauss1DsMAP &&o);
    static const std::string name;
};

} /* namespace mappel */

#endif /* MAPPEL_GAUSS1DSMAP_H */
