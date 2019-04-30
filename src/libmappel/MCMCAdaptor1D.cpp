/** @file MCMCAdaptor1D.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The class definition and template Specializations for MCMCAdaptor1D
 */

#include "Mappel/MCMCAdaptor1D.h"

namespace mappel {

MCMCAdaptor1D::MCMCAdaptor1D() : MCMCAdaptor1D(global_default_mcmc_sigma_scale)
{ }

MCMCAdaptor1D::MCMCAdaptor1D(double sigma_scale)
    : PointEmitterModel{}, //VB never called here.
      MCMCAdaptorBase{2,sigma_scale}
{
    double xsize = get_ubound()(0) - get_lbound()(0);
    eta_x = xsize*sigma_scale;
    set_intensity_mcmc_sampling();
    set_background_mcmc_sampling();
}

MCMCAdaptor1D::MCMCAdaptor1D(const MCMCAdaptor1D &o)
    : PointEmitterModel{o}, //VB never called here.
      MCMCAdaptorBase{o}
{
    eta_x = o.eta_x;
    eta_I = o.eta_I;
    eta_bg = o.eta_bg;
}

MCMCAdaptor1D::MCMCAdaptor1D(MCMCAdaptor1D &&o)
    : PointEmitterModel{std::move(o)}, //VB never called here.
      MCMCAdaptorBase{std::move(o)}
{
    eta_x = o.eta_x;
    eta_I = o.eta_I;
    eta_bg = o.eta_bg;
}

MCMCAdaptor1D& MCMCAdaptor1D::operator=(const MCMCAdaptor1D &o)
{
    if(this == &o) return *this; //No self copy
    //Ignore virtual base copy, someone else will do that
    MCMCAdaptorBase::operator=(o);
    eta_x = o.eta_x;
    eta_I = o.eta_I;
    eta_bg = o.eta_bg;
    return *this;
}

MCMCAdaptor1D& MCMCAdaptor1D::operator=(MCMCAdaptor1D &&o)   
{
    if(this == &o) return *this; //No self copy
    //Ignore virtual base copy, someone else will do that
    MCMCAdaptorBase::operator=(std::move(o));
    eta_x = o.eta_x;
    eta_I = o.eta_I;
    eta_bg = o.eta_bg;
    return *this;
}

/* Initialize MCMC step sizes */
void MCMCAdaptor1D::set_intensity_mcmc_sampling(double new_eta_I)
{
    if(new_eta_I<=0) {
        double mean_I;
        try {
            mean_I = get_hyperparam_value("intensity_scale")*get_hyperparam_value("intensity_shape");
        } catch (ParameterValueError&) {
            //use defaults
            mean_I = DefaultPriorMeanI;
        }
        eta_I = mean_I*sigma_scale;
    } else {
        eta_I = new_eta_I;
    }
}

void MCMCAdaptor1D::set_background_mcmc_sampling(double new_eta_bg)
{
    if(new_eta_bg<=0) {
        double mean_bg;
        try {
            mean_bg = get_hyperparam_value("background_scale")*get_hyperparam_value("background_shape");
        } catch (ParameterValueError&) {
            //use defaults
            double xsize = get_ubound()(0) - get_lbound()(0);
            mean_bg = DefaultPriorPixelMeanBG * xsize; //Adjust for 1D summation over pixels
        }
        eta_bg = mean_bg*sigma_scale;
    } else {
        eta_bg = new_eta_bg;
    }
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
MCMCAdaptor1D::sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, double step_scale) const
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

void
MCMCAdaptor1D::sample_mcmc_candidate(IdxT sample_index, ParamT &candidate, const IdxVecT &fixed_mask, double step_scale) const
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
            case 1: //change I, bg
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
