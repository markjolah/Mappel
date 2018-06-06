/** @file PointEmitter2DModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 03-26-2014
 * @brief The class definition and template Specializations for PointEmitter2DModel
 */

#include "Mappel/PoissonGaussianNoise2DObjective.h"

namespace mappel {
    
    const std::vector<std::string> PoissonGaussianNoise2DObjective::estimator_names(
        { "Heuristic", "CGaussHeuristic", "CGaussMLE", 
            "Newton", "NewtonDiagonal", "QuasiNewton", "TrustRegion"
            "SimulatedAnnealing"});
    
} /* namespace mappel */
