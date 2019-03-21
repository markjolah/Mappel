/** @file MCMCAdaptor2D.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The class definition and template Specializations for MCMCAdaptor2D
 */

#include "Mappel/MCMCAdaptor2D.h"

namespace mappel {

MCMCAdaptor2D::MCMCAdaptor2D() : MCMCAdaptor2D{global_default_mcmc_sigma_scale}
{ }

MCMCAdaptor2D::MCMCAdaptor2D(double sigma_scale)
    : PointEmitterModel{}, //VB never called here.
      MCMCAdaptor1D{sigma_scale}
{
    double ysize = get_ubound()(1) - get_lbound()(1);
    eta_y = ysize*sigma_scale;
}

MCMCAdaptor2D::MCMCAdaptor2D(const MCMCAdaptor2D &o)
    : PointEmitterModel{o}, //VB never called here.
      MCMCAdaptor1D{o}
{ eta_y = o.eta_y; }

MCMCAdaptor2D::MCMCAdaptor2D(MCMCAdaptor2D &&o)
    : PointEmitterModel{std::move(o)}, //VB never called here.
      MCMCAdaptor1D{std::move(o)}
{ eta_y = o.eta_y; }

MCMCAdaptor2D& MCMCAdaptor2D::operator=(const MCMCAdaptor2D &o)
{
    if(this == &o) return *this; //No self copy
    //Ignore virtual base copy, someone else will do that
    MCMCAdaptor1D::operator=(o);
    eta_y = o.eta_y;
    return *this;
}

MCMCAdaptor2D& MCMCAdaptor2D::operator=(MCMCAdaptor2D &&o)   
{
    if(this == &o) return *this; //No self copy
    //Ignore virtual base copy, someone else will do that
    MCMCAdaptor1D::operator=(std::move(o));
    eta_y = o.eta_y;
    return *this;
}


StatsT MCMCAdaptor2D::get_stats() const
{
    auto stats=MCMCAdaptor1D::get_stats();
    stats["mcmc_eta_y"] = eta_y;
    return stats;
}

void 
MCMCAdaptor2D::sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, double step_scale) const
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

void
MCMCAdaptor2D::sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, const IdxVecT &fixed_mask, double step_scale) const
{
    IdxT phase = sample_index % num_phases;
    bool step_taken=false;
    while(!step_taken){
        switch(phase) {
            case 0:  //change pos
                if(!fixed_mask(0)) {
                    candidate(0) += rng_manager.randn()*eta_x*step_scale;
                    step_taken=true;
                }
                if(!fixed_mask(1)) {
                    candidate(1) += rng_manager.randn()*eta_y*step_scale;
                    step_taken=true;
                }
                break;
            case 1: //change I, bg
                if(!fixed_mask(2)) {
                    candidate(2) += rng_manager.randn()*eta_I*step_scale;
                    step_taken=true;
                }
                if(!fixed_mask(3)) {
                    candidate(3) += rng_manager.randn()*eta_bg*step_scale;
                    step_taken=true;
                }
        }
        phase = (phase+1) % num_phases;
    }
}


} /* namespace mappel */
