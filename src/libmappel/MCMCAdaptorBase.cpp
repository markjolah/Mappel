/** @file MCMCAdaptorBase.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2018
 * @brief The class definition and template Specializations for MCMCAdaptorBase
 */

#include "Mappel/MCMCAdaptorBase.h"

namespace mappel {
    /* Static data members */
const double MCMCAdaptorBase::global_default_mcmc_sigma_scale = 0.05;
const double MCMCAdaptorBase::global_max_mcmc_sigma_scale = 0.5;
    
MCMCAdaptorBase::MCMCAdaptorBase(IdxT num_phases)
    : MCMCAdaptorBase(num_phases, global_default_mcmc_sigma_scale)
{ }

MCMCAdaptorBase::MCMCAdaptorBase(IdxT num_phases, double sigma_scale)
    : num_phases(num_phases),
      sigma_scale(sigma_scale)
{
    if(num_phases < 1) {
        std::ostringstream os;
        os<<"num_phases: "<<num_phases<<" invalid.";
        throw ParameterValueError(os.str());
    }
    if(sigma_scale > global_max_mcmc_sigma_scale) {
        std::ostringstream os;
        os<<"sigma_scale: "<<sigma_scale<<" above global max: "<<global_max_mcmc_sigma_scale;
        throw ParameterValueError(os.str());
    }
    if(sigma_scale <=0 || !std::isfinite(sigma_scale)) {
        std::ostringstream os;
        os<<"sigma_scale: "<<sigma_scale<<" invalid.";
        throw ParameterValueError(os.str());
    }
}

void MCMCAdaptorBase::set_mcmc_sigma_scale(double scale) {
    if(scale > global_max_mcmc_sigma_scale) {
        std::ostringstream os;
        os<<"sigma_scale: "<<scale<<" above global max: "<<global_max_mcmc_sigma_scale;
        throw ParameterValueError(os.str());
    }
    if(scale <=0 || !std::isfinite(scale)) {
        std::ostringstream os;
        os<<"sigma_scale: "<<scale<<" invalid.";
        throw ParameterValueError(os.str());
    }
    sigma_scale = scale; //safe to assign
}

double MCMCAdaptorBase::get_mcmc_sigma_scale() const
{ return sigma_scale; }
    
IdxT MCMCAdaptorBase::get_mcmc_num_phases() const
{ return num_phases; }

StatsT MCMCAdaptorBase::get_stats() const
{
    StatsT stats{};
    stats["mcmc_num_phases"] = num_phases;
    stats["mcmc_sigma_scale"] = sigma_scale;
    return stats;
}


} /* namespace mappel */
