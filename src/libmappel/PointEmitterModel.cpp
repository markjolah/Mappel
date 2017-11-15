/** @file PointEmitterModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class definition and template Specializations for PointEmitterModel
 */

#include "PointEmitterModel.h"
// #include "util.h"
// #include <omp.h>

namespace mappel {


PointEmitterModel::PointEmitterModel(CompositeDist&& prior_)
    : prior(std::move(prior_)),
      num_params(prior.num_dim()),
      num_hyperparams(prior.num_params()),
      lbound(prior.lbound()),
      ubound(prior.ubound())
{
}


StatsT PointEmitterModel::get_stats() const
{
    StatsT stats;
    stats["num_params"] = num_params;
    stats["num_hyperparams"] = num_hyperparams;
    auto hyperparams = prior.params();
    auto hyperparams_desc = prior.params_desc();
    std::string hp_str("hyperparameters.");
    for(IdxT i=0; i<num_params; i++) stats[hp_str+hyperparams_desc[i]] = hyperparams[i];
    stats["mcmcparams.num_phases"]=mcmc_num_candidate_sampling_phases;
    stats["mcmcparams.etaX"]=mcmc_candidate_eta_x;
    stats["mcmcparams.etaI"]=mcmc_candidate_eta_I;
    stats["mcmcparams.etabg"]=mcmc_candidate_eta_bg;
    for(IdxT n=0;n<num_params;n++) {
        std::ostringstream outl,outu;
        outl<<"lbound."<<n+1;
        stats[outl.str()]= lbound(n);
        outu<<"ubound."<<n+1;
        stats[outu.str()]= ubound(n);
    }
    return stats;
}




void PointEmitterModel::set_prior(CompositeDist&& prior_)
{
    prior = std::move(prior_);
    num_params = prior.num_dim();
    num_hyperparams = prior.num_dim();
    lbound = prior.lbound();
    ubound = prior.ubound();
}


/**
 * 
 * Ensures (prior.lbound <= lbound) && (prior.ubound >= ubound) so that the prior constraints are not violated
 */
void PointEmitterModel::set_bounds(const ParamT &lbound_, const ParamT &ubound_)
{
    if(lbound_.n_elem != num_params) throw BoundsException("Invalid lower bound size");
    if(ubound_.n_elem != num_params) throw BoundsException("Invalid upper bound size");
    auto p_lbound = prior.lbound();
    auto p_ubound = prior.ubound();
    for(IdxT n=0; n<num_params; n++) {
        if(lbound(n)>ubound(n)) throw BoundsException("Bounds inverted.");
        if(std::fabs(lbound(n)-ubound(n))<10*bounds_epsilon) throw BoundsException("Bounds too close.");
        if(lbound(n) < p_lbound(n)) throw BoundsException("Lower bound below prior lower bound");
        if(ubound(n) > p_ubound(n)) throw BoundsException("Upper bound above prior lower bound");
    }
    lbound = lbound_;
    ubound = ubound_;
}


void PointEmitterModel::bound_theta(ParamT &theta, double epsilon) const
{
    if(epsilon<0) epsilon = bounds_epsilon;
    for(IdxT n=0;n<num_params;n++) {
        if(theta(n) < lbound(n)+epsilon) theta(n)=lbound(n)+epsilon;
        if(theta(n) > ubound(n)-epsilon) theta(n)=ubound(n)-epsilon;
    }
}

bool PointEmitterModel::theta_in_bounds(const ParamT &theta, double epsilon) const
{
    for(IdxT n=0; n<num_params; n++) 
        if(lbound(n)>=theta(n) || theta(n)>=ubound(n)) return false;
    return true;
}

PointEmitterModel::ParamT PointEmitterModel::bounded_theta(const ParamT &theta, double epsilon) const
{
    if(epsilon<0) epsilon=bounds_epsilon;
    ParamT btheta = theta;
    for(IdxT n=0;n<num_params;n++) {
        if(theta(n) < lbound(n)+epsilon) btheta(n)=lbound(n)+epsilon;
        if(theta(n) > ubound(n)-epsilon) btheta(n)=ubound(n)-epsilon;
    }
    return btheta;
}

PointEmitterModel::ParamT PointEmitterModel::reflected_theta(const ParamT &theta, double epsilon) const
{
    if(epsilon<0) epsilon=bounds_epsilon;
    ParamT btheta = theta;
    for(IdxT n=0;n<num_params;n++) {
        if(std::isfinite(lbound(n))) {
            if(std::isfinite(ubound(n))){//both valid bounds.  Do reflection
                double d = 2*(ubound(n)-lbound(n));
                double w = std::fmod(std::fabs(theta(n)-lbound(n)), d);
                btheta(n) = std::min(w,d-w)+lbound(n);
            } else if (theta(n)<lbound(n)) {
                btheta(n)=2*lbound(n)-theta(n); //valid lower bound only
            }
        } else if(theta(n)>ubound(n)) {
            btheta(n)=2*ubound(n)-theta(n); //valid upper bound only
        }
    }
    return btheta;
}



// std::ostream& operator<<(std::ostream &out, PointEmitterModel &model)
// {
//     auto stats=model.get_stats();
//     out<<"["<<model.name()<<":";
//     for(auto it=stats.cbegin(); it!=stats.cend(); ++it) out<<" "<<it->first<<"="<<it->second;
//     out<<"]";
//     return out;
// }

} /* namespace mappel */
