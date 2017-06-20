/** @file PoissonNoise1DObjective.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-26-2014
 * @brief The class definition and template Specializations for PoissonNoise1DObjective
 */

#include "PoissonNoise1DObjective.h"

namespace mappel {
    
const std::vector<std::string> PoissonNoise1DObjective::estimator_names(
    { "HeuristicEstimator", "CGaussHeuristicEstimator", "CGaussMLE", 
      "NewtonMaximizer", "NewtonDiagonalMaximizer", "QuasiNewtonMaximizer", "TrustRegionMaximizer"
      "SimulatedAnnealingMaximizer"});

} /* namespace mappel */
