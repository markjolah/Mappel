/** @file PointEmitter2DModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-26-2014
 * @brief The class definition and template Specializations for PointEmitter2DModel
 */

#include "PoissonGaussianNoise2DObjective.h"

namespace mappel {
    
    const std::vector<std::string> PoissonGaussianNoise2DObjective::estimator_names(
        { "HeuristicEstimator", "CGaussHeuristicEstimator", "CGaussMLE", 
            "NewtonMaximizer", "NewtonDiagonalMaximizer", "QuasiNewtonMaximizer", "TrustRegionMaximizer"
            "SimulatedAnnealingMaximizer"});
    
} /* namespace mappel */
