/** @file PoissonNoise1DObjective.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class definition and template Specializations for PoissonNoise1DObjective
 */

#include "Mappel/PoissonNoise1DObjective.h"

namespace mappel {
    
const std::vector<std::string> PoissonNoise1DObjective::estimator_names(
    { "Heuristic", "Newton", "NewtonDiagonal", "QuasiNewton", "TrustRegion", "SimulatedAnnealing"});

PoissonNoise1DObjective::PoissonNoise1DObjective()
    : ImageFormat1DBase{}
{ }

PoissonNoise1DObjective::PoissonNoise1DObjective(const PoissonNoise1DObjective &o)
    : ImageFormat1DBase{o}
{ }

PoissonNoise1DObjective::PoissonNoise1DObjective(PoissonNoise1DObjective &&o)
    : ImageFormat1DBase{std::move(o)}
{ }

PoissonNoise1DObjective& PoissonNoise1DObjective::operator=(const PoissonNoise1DObjective &o)
{
   return *this;
}

PoissonNoise1DObjective& PoissonNoise1DObjective::operator=(PoissonNoise1DObjective &&o)
{
    return *this;
}

} /* namespace mappel */
