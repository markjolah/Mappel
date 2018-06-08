/** @file MCMCAdaptor2D.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The class definition and template Specializations for MCMCAdaptor2D
 */

#include "Mappel/MCMCAdaptor2D.h"

namespace mappel {

MCMCAdaptor2D::MCMCAdaptor2D()
    : PointEmitterModel{}, //VB never called here.
      MCMCAdaptorBase{2}
{ initialize(); }

MCMCAdaptor2D::MCMCAdaptor2D(double sigma_scale)
    : PointEmitterModel{}, //VB never called here.
      MCMCAdaptorBase{2,sigma_scale}
{ initialize(); }

MCMCAdaptor2D::MCMCAdaptor2D(const MCMCAdaptor2D &o)
    : PointEmitterModel{o}, //VB never called here.
      MCMCAdaptorBase{o}
{ initialize(); }

MCMCAdaptor2D::MCMCAdaptor2D(MCMCAdaptor2D &&o)
    : PointEmitterModel{std::move(o)}, //VB never called here.
      MCMCAdaptorBase{std::move(o)}
{ initialize(); }

MCMCAdaptor2D& MCMCAdaptor2D::operator=(const MCMCAdaptor2D &o)
{
    if(this == &o) return *this; //No self copy
    //Ignore virtual base copy, someone else will do that
    MCMCAdaptorBase::operator=(o);
    initialize();
    return *this;
}

MCMCAdaptor2D& MCMCAdaptor2D::operator=(MCMCAdaptor2D &&o)   
{
    if(this == &o) return *this; //No self copy
    //Ignore virtual base copy, someone else will do that
    MCMCAdaptorBase::operator=(std::move(o));
    initialize();
    return *this;
}

/* Initialize MCMC step sizes */
void MCMCAdaptor2D::initialize()
{
    double xsize = get_ubound()(0) - get_lbound()(0);
    double ysize = get_ubound()(1) - get_lbound()(1);
    eta_x = xsize*sigma_scale;
    eta_y = ysize*sigma_scale;
    eta_I = find_hyperparam("mean_I",default_mean_I)*sigma_scale;
    eta_bg = find_hyperparam("mean_bg",default_pixel_mean_bg)*sigma_scale;    
}

StatsT MCMCAdaptor2D::get_stats() const
{
    auto stats=MCMCAdaptorBase::get_stats();
    stats["mcmc_eta_x"] = eta_x;
    stats["mcmc_eta_y"] = eta_y;
    stats["mcmc_eta_I"] = eta_I;
    stats["mcmc_eta_bg"] = eta_bg;
    return stats;
}

void 
MCMCAdaptor2D::sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, double step_scale)
{
    IdxT phase = sample_index % num_phases;
    switch(phase) {
        case 0:  //change pos
            candidate(0) += rng_manager.randn()*eta_x*step_scale;
            candidate(1) += rng_manager.randn()*eta_y*step_scale;
            break;
        case 1: //change I, bg
            candidate(2) += rng_manager.randn()*eta_I*step_scale;
            candidate(3) += rng_manager.randn()*eta_bg*step_scale;
    }
}

} /* namespace mappel */
