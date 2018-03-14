/** @file PoissonNoise1DObjective.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2018
 * @brief The class definition and template Specializations for PoissonNoise1DObjective
 */

#include "PoissonNoise1DObjective.h"

namespace mappel {
    
const std::vector<std::string> PoissonNoise1DObjective::estimator_names(
    { "Heuristic", "Newton", "NewtonDiagonal", "QuasiNewton", "TrustRegion", "SimulatedAnnealing"});

} /* namespace mappel */
