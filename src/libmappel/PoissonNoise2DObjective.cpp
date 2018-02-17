/** @file PoissonNoise2DObjective.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2014-2018
 * @brief The class definition and template Specializations for PoissonNoise2DObjective
 */

#include "PoissonNoise2DObjective.h"

namespace mappel {

const std::vector<std::string> PoissonNoise2DObjective::estimator_names(
    { "Heuristic",  "CGaussHeuristic", "CGauss",  "Newton", "NewtonDiagonal",
       "QuasiNewton", "TrustRegion", "SimulatedAnnealing"});

} /* namespace mappel */
