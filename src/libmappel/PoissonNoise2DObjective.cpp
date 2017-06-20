/** @file PoissonNoise2DObjective.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-26-2014
 * @brief The class definition and template Specializations for PoissonNoise2DObjective
 */

#include "PoissonNoise2DObjective.h"

namespace mappel {
    
const std::vector<std::string> PoissonNoise2DObjective::estimator_names(
    { "HeuristicEstimator", "SeperableHeuristicEstimator", "CGaussHeuristicEstimator", "CGaussMLE", 
      "NewtonMaximizer", "NewtonDiagonalMaximizer", "QuasiNewtonMaximizer", "TrustRegionMaximizer"
      "SimulatedAnnealingMaximizer"});

} /* namespace mappel */
