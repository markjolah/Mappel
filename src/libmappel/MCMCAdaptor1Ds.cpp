/** @file MCMCAdaptor1Ds.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The class definition and template Specializations for MCMCAdaptor1Ds
 */

#include "Mappel/MCMCAdaptor1Ds.h"

namespace mappel {

MCMCAdaptor1Ds::MCMCAdaptor1Ds() : MCMCAdaptor1D{global_default_mcmc_sigma_scale}
{ }

MCMCAdaptor1Ds::MCMCAdaptor1Ds(double sigma_scale)
    : MCMCAdaptor1D{sigma_scale}
{
    set_mcmc_num_phases(3);
    eta_sigma = 0.5*sigma_scale;
}

MCMCAdaptor1Ds::MCMCAdaptor1Ds(const MCMCAdaptor1Ds &o)
    : PointEmitterModel{},
      MCMCAdaptor1D{o}
{
    eta_sigma = o.eta_sigma;
}

MCMCAdaptor1Ds::MCMCAdaptor1Ds(MCMCAdaptor1Ds &&o)
    : PointEmitterModel{std::move(o)},
      MCMCAdaptor1D{std::move(o)}
{
    eta_sigma = o.eta_sigma;
}

MCMCAdaptor1Ds& MCMCAdaptor1Ds::operator=(const MCMCAdaptor1Ds &o)
{
    if(this == &o) return *this; //No self copy
    MCMCAdaptor1D::operator=(o);
    eta_sigma = o.eta_sigma;
    return *this;
}

MCMCAdaptor1Ds& MCMCAdaptor1Ds::operator=(MCMCAdaptor1Ds &&o)   
{
    if(this == &o) return *this; //No self copy
    MCMCAdaptor1D::operator=(std::move(o));
    eta_sigma = o.eta_sigma;
    return *this;
}

StatsT MCMCAdaptor1Ds::get_stats() const
{
    auto stats=MCMCAdaptor1D::get_stats();
    stats["mcmc_eta_sigma"] = eta_sigma;
    return stats;
}

void 
MCMCAdaptor1Ds::sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, double step_scale) const
{
    IdxT phase = sample_index % num_phases;
    switch(phase) {
        case 0:  //change pos
            candidate(0) += rng_manager.randn()*eta_x*step_scale;
            break;
        case 1: //change I, sigma
            candidate(1) += rng_manager.randn()*eta_I*step_scale;
            candidate(3) += rng_manager.randn()*eta_sigma*step_scale;
            break;
        case 2: //change I, bg
            candidate(1) += rng_manager.randn()*eta_I*step_scale;
            candidate(2) += rng_manager.randn()*eta_bg*step_scale;
    }
}

void
MCMCAdaptor1Ds::sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, const IdxVecT &fixed_mask, double step_scale) const
{
    IdxT phase = sample_index % num_phases;
    bool step_taken=false;
    while(!step_taken){
        switch(phase) {
            case 0:  //change pos
                if(!fixed_mask(0)) {
                    step_taken=true;
                    candidate(0) += rng_manager.randn()*eta_x*step_scale;
                }
                break;
            case 1: //change I, sigma
                if(!fixed_mask(1)) {
                    candidate(1) += rng_manager.randn()*eta_I*step_scale;
                    step_taken=true;
                }
                if(!fixed_mask(3)) {
                    candidate(3) += rng_manager.randn()*eta_sigma*step_scale;
                    step_taken=true;
                }
                break;
            case 2: //change I, bg
                if(!fixed_mask(1)) {
                    candidate(1) += rng_manager.randn()*eta_I*step_scale;
                    step_taken=true;
                }
                if(!fixed_mask(2)) {
                    candidate(2) += rng_manager.randn()*eta_bg*step_scale;
                    step_taken=true;
                }
        }
        phase = (phase+1) % num_phases;
    }
}

} /* namespace mappel */
