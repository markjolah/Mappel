/** @file MCMCAdaptor2Ds.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The class definition and template Specializations for MCMCAdaptor2Ds
 */

#include "Mappel/MCMCAdaptor2Ds.h"

namespace mappel {

MCMCAdaptor2Ds::MCMCAdaptor2Ds() : MCMCAdaptor2Ds{global_default_mcmc_sigma_scale}
{ }

MCMCAdaptor2Ds::MCMCAdaptor2Ds(double sigma_scale)
    : PointEmitterModel{}, //VB never called here.
      MCMCAdaptor2D{sigma_scale}
{
    set_mcmc_num_phases(3);
    eta_sigma = 0.5*sigma_scale;
}

MCMCAdaptor2Ds::MCMCAdaptor2Ds(const MCMCAdaptor2Ds &o)
    : PointEmitterModel{o}, //VB never called here.
      MCMCAdaptor2D{o}
{ eta_sigma = o.eta_sigma; }

MCMCAdaptor2Ds::MCMCAdaptor2Ds(MCMCAdaptor2Ds &&o)
    : PointEmitterModel{std::move(o)}, //VB never called here.
      MCMCAdaptor2D{std::move(o)}
{ eta_sigma = o.eta_sigma; }

MCMCAdaptor2Ds& MCMCAdaptor2Ds::operator=(const MCMCAdaptor2Ds &o)
{
    if(this == &o) return *this; //No self copy
    MCMCAdaptor2D::operator=(o);
    eta_sigma = o.eta_sigma;
    return *this;
}

MCMCAdaptor2Ds& MCMCAdaptor2Ds::operator=(MCMCAdaptor2Ds &&o)   
{
    if(this == &o) return *this; //No self copy
    MCMCAdaptor2D::operator=(std::move(o));
    eta_sigma = o.eta_sigma;
    return *this;
}

StatsT MCMCAdaptor2Ds::get_stats() const
{
    auto stats=MCMCAdaptor2D::get_stats();
    stats["mcmc_eta_sigma"] = eta_sigma;
    return stats;
}

void 
MCMCAdaptor2Ds::sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, double step_scale) const
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

void
MCMCAdaptor2Ds::sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, const IdxVecT &fixed_mask, double step_scale) const
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
            case 1: //change I, sigma
                if(!fixed_mask(2)) {
                    candidate(2) += rng_manager.randn()*eta_I*step_scale;
                    step_taken=true;
                }
                if(!fixed_mask(4)) {
                    candidate(4) += rng_manager.randn()*eta_sigma*step_scale;
                    step_taken=true;
                }
                break;
            case 2: //change I, bg
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
