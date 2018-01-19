/** @file PointEmitterModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class definition and template Specializations for PointEmitterModel
 */
#include <algorithm>

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

prior_hessian::NormalDist        
PointEmitterModel::make_prior_component_position_normal(std::string var, IdxT size, double pos_sigma)
{
    std::vector<std::string> param_names = {var+"pos_mean", var+"pos_sigma"};
    double pos_mean = size/2;
    return prior_hessian::NormalDist(pos_mean,pos_sigma,0,size,var,std::move(param_names));
}
    
prior_hessian::SymmetricBetaDist 
PointEmitterModel::make_prior_component_position_beta(std::string var, IdxT size, double pos_beta)
{
    std::vector<std::string> param_names = {var+"pos_beta"};
    return prior_hessian::SymmetricBetaDist(pos_beta,0,size,var,std::move(param_names));
}

prior_hessian::GammaDist         
PointEmitterModel::make_prior_component_intensity(std::string var, double mean, double kappa)
{
    return prior_hessian::GammaDist(mean,kappa,var);
}

prior_hessian::ParetoDist        
PointEmitterModel::make_prior_component_sigma(std::string var, double min_sigma, double max_sigma, double alpha)
{
    return prior_hessian::ParetoDist(alpha,min_sigma,max_sigma,var);
}


StatsT PointEmitterModel::get_stats() const
{
    StatsT stats;
    stats["num_params"] = num_params;
    stats["num_hyperparams"] = num_hyperparams;
    auto hyperparams = prior.params();
    auto hyperparams_desc = prior.params_desc();
    std::string hp_str("hyperparameters.");
    for(IdxT i=0; i<num_hyperparams; i++) stats[hp_str+hyperparams_desc[i]] = hyperparams[i];
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
 * @param param_name Exact name to search for
 * @param default_val Optional.  defaults to NaN
 * @return Hyperparam value. default_val if not found. NaN if not found and no default given.
 */
double PointEmitterModel::find_hyperparam(std::string param_name, double default_val=std::numeric_limits<double>::quiet_NaN()) const
{
    auto hp = get_hyperparams(); 
    auto hp_desc = get_hyperparams_desc(); 
    auto desc_idx = std::find(hp_desc.begin(), hp_desc.end(), param_name);
    if(desc_idx != hp_desc.end()) {
        return hp(std::distance(hp_desc.begin(),desc_idx));   
    } else {
        return default_val;
    }
}

/**
 *
 * Modifies the prior bounds to prevent sampling outside the valid box-constraints.
 */
void PointEmitterModel::set_bounds(const ParamT &lbound_, const ParamT &ubound_)
{
    if(lbound_.n_elem != num_params) throw BoundsError("Invalid lower bound size");
    if(ubound_.n_elem != num_params) throw BoundsError("Invalid upper bound size");
    for(IdxT n=0; n<num_params; n++) {
        if(lbound_(n)>ubound_(n)) throw BoundsError("Bounds inverted.");
        if(std::fabs(lbound_(n)-ubound_(n))<10*bounds_epsilon) throw BoundsError("Bounds too close.");
    }
    prior.set_bounds(lbound_,ubound_);
    lbound = prior.lbound();
    ubound = prior.ubound();
}


void PointEmitterModel::bound_theta(ParamT &theta, double epsilon) const
{
    for(IdxT n=0;n<num_params;n++) {
        if(theta(n) < lbound(n)+epsilon) theta(n)=lbound(n)+epsilon;
        if(theta(n) > ubound(n)-epsilon) theta(n)=ubound(n)-epsilon;
    }
}

bool PointEmitterModel::theta_in_bounds(const ParamT &theta, double epsilon) const
{
    for(IdxT n=0; n<num_params; n++) 
        if(lbound(n)+epsilon >= theta(n) || theta(n) >= ubound(n)-epsilon) return false;
    return true;
}

PointEmitterModel::ParamT PointEmitterModel::bounded_theta(const ParamT &theta, double epsilon) const
{
    ParamT btheta = theta;
    for(IdxT n=0;n<num_params;n++) {
        if(theta(n) < lbound(n)+epsilon) btheta(n)=lbound(n)+epsilon;
        if(theta(n) > ubound(n)-epsilon) btheta(n)=ubound(n)-epsilon;
    }
    return btheta;
}

PointEmitterModel::ParamT PointEmitterModel::reflected_theta(const ParamT &theta, double epsilon) const
{
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

} /* namespace mappel */
