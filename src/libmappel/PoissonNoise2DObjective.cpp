/** @file PoissonNoise2DObjective.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class definition and template Specializations for PoissonNoise2DObjective
 */

#include "Mappel/PoissonNoise2DObjective.h"

namespace mappel {

const std::vector<std::string> PoissonNoise2DObjective::estimator_names(
    { "Heuristic",  "CGaussHeuristic", "CGauss",  "Newton", "NewtonDiagonal",
       "QuasiNewton", "TrustRegion", "SimulatedAnnealing"});

PoissonNoise2DObjective::PoissonNoise2DObjective()
    : ImageFormat2DBase()
{ }

PoissonNoise2DObjective::PoissonNoise2DObjective(const PoissonNoise2DObjective &o)
    : ImageFormat2DBase{o}
{ }

PoissonNoise2DObjective::PoissonNoise2DObjective(PoissonNoise2DObjective &&o)
    : ImageFormat2DBase{std::move(o)}
{ }

PoissonNoise2DObjective& PoissonNoise2DObjective::operator=(const PoissonNoise2DObjective &o)
{
    return *this;
}

PoissonNoise2DObjective& PoissonNoise2DObjective::operator=(PoissonNoise2DObjective &&o)
{
    return *this;
}

} /* namespace mappel */
