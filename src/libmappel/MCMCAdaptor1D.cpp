/** @file MCMCAdaptor1D.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The class definition and template Specializations for MCMCAdaptor1D
 */

#include "Mappel/MCMCAdaptor1D.h"

namespace mappel {

MCMCAdaptor1D::MCMCAdaptor1D()
    : PointEmitterModel{}, //VB never called here.
      MCMCAdaptorBase{2}
{ initialize(); }

MCMCAdaptor1D::MCMCAdaptor1D(double sigma_scale)
    : PointEmitterModel{}, //VB never called here.
      MCMCAdaptorBase{2,sigma_scale}
{ initialize(); }

MCMCAdaptor1D::MCMCAdaptor1D(const MCMCAdaptor1D &o)
    : PointEmitterModel{o}, //VB never called here.
      MCMCAdaptorBase{o}
{ initialize(); }

MCMCAdaptor1D::MCMCAdaptor1D(MCMCAdaptor1D &&o)
    : PointEmitterModel{std::move(o)}, //VB never called here.
      MCMCAdaptorBase{std::move(o)}
{ initialize(); }

MCMCAdaptor1D& MCMCAdaptor1D::operator=(const MCMCAdaptor1D &o)
{
    if(this == &o) return *this; //No self copy
    //Ignore virtual base copy, someone else will do that
    MCMCAdaptorBase::operator=(o);
    initialize();
    return *this;
}

MCMCAdaptor1D& MCMCAdaptor1D::operator=(MCMCAdaptor1D &&o)   
{
    if(this == &o) return *this; //No self copy
    //Ignore virtual base copy, someone else will do that
    MCMCAdaptorBase::operator=(std::move(o));
    initialize();
    return *this;
}

/* Initialize MCMC step sizes */
void MCMCAdaptor1D::initialize()
{
    double xsize = get_ubound()(0) - get_lbound()(0);
    eta_x = xsize*sigma_scale;
    eta_I = find_hyperparam("mean_I",default_mean_I)*sigma_scale;
    eta_bg = find_hyperparam("mean_bg",default_pixel_mean_bg)*sigma_scale;    
}

StatsT MCMCAdaptor1D::get_stats() const
{
    auto stats=MCMCAdaptorBase::get_stats();
    stats["mcmc_eta_x"] = eta_x;
    stats["mcmc_eta_I"] = eta_I;
    stats["mcmc_eta_bg"] = eta_bg;
    return stats;
}

void 
MCMCAdaptor1D::sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, double step_scale)
{
    IdxT phase = sample_index % num_phases;
    switch(phase) {
        case 0:  //change pos
            candidate(0) += rng_manager.randn()*eta_x*step_scale;
            break;
        case 1: //change I, bg
            candidate(1) += rng_manager.randn()*eta_I*step_scale;
            candidate(2) += rng_manager.randn()*eta_bg*step_scale;
    }
}

} /* namespace mappel */
