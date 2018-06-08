/** @file MCMCAdaptor2Ds.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The class definition and template Specializations for MCMCAdaptor2Ds
 */

#include "Mappel/MCMCAdaptor2Ds.h"

namespace mappel {

MCMCAdaptor2Ds::MCMCAdaptor2Ds()
    : PointEmitterModel{}, //VB never called here.
      MCMCAdaptorBase{3}
{ initialize(); }

MCMCAdaptor2Ds::MCMCAdaptor2Ds(double sigma_scale)
    : PointEmitterModel{}, //VB never called here.
      MCMCAdaptorBase{3,sigma_scale}
{ initialize(); }

MCMCAdaptor2Ds::MCMCAdaptor2Ds(const MCMCAdaptor2Ds &o)
    : PointEmitterModel{o}, //VB never called here.
      MCMCAdaptorBase{o}
{ initialize(); }

MCMCAdaptor2Ds::MCMCAdaptor2Ds(MCMCAdaptor2Ds &&o)
    : PointEmitterModel{std::move(o)}, //VB never called here.
      MCMCAdaptorBase{std::move(o)}
{ initialize(); }

MCMCAdaptor2Ds& MCMCAdaptor2Ds::operator=(const MCMCAdaptor2Ds &o)
{
    if(this == &o) return *this; //No self copy
    //Ignore virtual base copy, someone else will do that
    MCMCAdaptorBase::operator=(o);
    initialize();
    return *this;
}

MCMCAdaptor2Ds& MCMCAdaptor2Ds::operator=(MCMCAdaptor2Ds &&o)   
{
    if(this == &o) return *this; //No self copy
    //Ignore virtual base copy, someone else will do that
    MCMCAdaptorBase::operator=(std::move(o));
    initialize();
    return *this;
}

/* Initialize MCMC step sizes */
void MCMCAdaptor2Ds::initialize()
{
    double xsize = get_ubound()(0) - get_lbound()(0);
    double ysize = get_ubound()(1) - get_lbound()(1);
    eta_x = xsize*sigma_scale;
    eta_y = ysize*sigma_scale;
    eta_I = find_hyperparam("mean_I",default_mean_I)*sigma_scale;
    eta_bg = find_hyperparam("mean_bg",default_pixel_mean_bg)*sigma_scale;
    eta_sigma = 1.0*sigma_scale;
}

StatsT MCMCAdaptor2Ds::get_stats() const
{
    auto stats=MCMCAdaptorBase::get_stats();
    stats["mcmc_eta_x"] = eta_x;
    stats["mcmc_eta_y"] = eta_y;
    stats["mcmc_eta_I"] = eta_I;
    stats["mcmc_eta_bg"] = eta_bg;
    stats["mcmc_eta_sigma"] = eta_sigma;
    return stats;
}

void 
MCMCAdaptor2Ds::sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, double step_scale)
{
    IdxT phase = sample_index % num_phases;
    switch(phase) {
        case 0:  //change pos
            candidate(0) += rng_manager.randn()*eta_x*step_scale;
            candidate(1) += rng_manager.randn()*eta_y*step_scale;
            break;
        case 1: //change I, sigma
            candidate(2) += rng_manager.randn()*eta_I*step_scale;
            candidate(4) += rng_manager.randn()*eta_sigma*step_scale;
            break;
        case 2: //change I, bg
            candidate(2) += rng_manager.randn()*eta_I*step_scale;
            candidate(3) += rng_manager.randn()*eta_bg*step_scale;
    }
}

} /* namespace mappel */
