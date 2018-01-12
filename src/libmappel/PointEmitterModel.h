/** @file PointEmitterModel.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class declaration and inline and templated functions for PointEmitterModel.
 *
 * The base class for all point emitter localization models
 */

#ifndef _POINTEMITTERMODEL_H
#define _POINTEMITTERMODEL_H

#include <iostream>
#include <string>

#include <armadillo>

#include <PriorHessian/CompositeDist.h>

#include "util.h"
#include "stencil.h"
#include "display.h"

namespace mappel {

/** @brief A virtual Base type for point emitter localization models.
 * 
 * Initialized with a prior as a PriorHessian::CompositeDist object, this sets the dimensionality (num_params) and num_hyperparams,
 * and the associated descriptions.
 * 
 * Box-type bounding constraints are controlled with the set_bounds() method.
 * 
 * 
 * 
 * Of note some of the common MCMC variables are rooted here in the inheritance tree.
 */
class PointEmitterModel {
protected:
    PointEmitterModel() {}; //Never called by actual top-level classes.  Allows virtual inheritance non-concrete base classes to ignore creating a constructor if necessary.
public:
    /* Static data members */
    constexpr static double default_pos_beta = 1.5; /**< Default position parameter in symmetric beta-distributions */
    constexpr static double default_mean_I = 300; /**< Default emitter intensity mean*/
    constexpr static double default_kappa_I = 2;  /**< Default emitter intensity gamma distribution shape */
    constexpr static double default_pixel_mean_bg = 4; /**< Default per-pixel mean background counts */
    constexpr static double default_kappa_bg = 2; /**< Default per-pixel background gamma distribution shape */

    /* Internal Types */
    
    /** @brief Priors are represented by the CompositeDist container class from the PriorHessian library */
    template<class T> using IsPointEmitterModelT = typename std::enable_if<std::is_base_of<PointEmitterModel,T>::value>::type;
    using CompositeDist = prior_hessian::CompositeDist<ParallelRngT>;
    
    using ParamT = arma::vec; /**< Parameter vector */
    using ParamVecT = arma::mat; /**< Vector of parameter vectors */
    using StringVecT = prior_hessian::StringVecT;
    PointEmitterModel(CompositeDist&& prior_);    
    
    StatsT get_stats() const;
    
    IdxT get_num_params() const;
    ParamT make_param() const;
    ParamVecT make_param_vec(IdxT n) const;
    MatT make_param_mat() const;
    
    CompositeDist& get_prior();
    const CompositeDist& get_prior() const;
    void set_prior(CompositeDist&& prior_);
    
    IdxT get_num_hyperparams() const;
    void set_hyperparams(const VecT &hyperparams);
    VecT get_hyperparams() const;

    StringVecT get_params_desc() const;
    void set_params_desc(const StringVecT &desc);
    StringVecT get_hyperparams_desc() const;
    void set_hyperparams_desc(const StringVecT &desc);
    
    template<class RngT> ParamT sample_prior(RngT &rng);
    ParamT sample_prior();
    
    /** Box-type parameter bounds */
    void set_bounds(const ParamT &lbound, const ParamT &ubound);
    const ParamT& get_lbound() const;
    const ParamT& get_ubound() const;
    void bound_theta(ParamT &theta,double epsilon=-1) const;
    bool theta_in_bounds(const ParamT &theta,double epsilon=-1) const;
    /* aids for bound-constrained optimization routines */
    ParamT bounded_theta(const ParamT &theta,double epsilon=-1) const;
    ParamT reflected_theta(const ParamT &theta,double epsilon=-1) const;

    /* MCMC related */
    /**< The number of different sampling phases for candidate selection MCMC.  Each phase changes a different subset of variables.*/
    IdxT get_mcmc_num_candidate_sampling_phases() const;
    
protected:
    /* Constant model parameter information */
    CompositeDist prior; /* Prior distribution represented using libPriorHessian */
    IdxT num_params;
    IdxT num_hyperparams;
    ParamT lbound,ubound; /* Vectors of lower and upper bounds. (lbound>=prior.lbound && ubound<=prior.ubound) */
    
    /* Data members */
    IdxT mcmc_num_candidate_sampling_phases=0; /**< The number of different sampling phases for candidate selection MCMC.  Each phase changes a different subset of variables.*/
    double mcmc_candidate_sample_dist_ratio=1./30.; /**< Controls the candidate distribution spread for MCMC stuff */
    double mcmc_candidate_eta_x; /**< The standard deviation for the normally distributed pertebation to theta_I in the random walk MCMC sampling */
    double mcmc_candidate_eta_I; /**< The standard deviation for the normally distributed pertebation to theta_I in the random walk MCMC sampling */
    double mcmc_candidate_eta_bg; /**< The standard deviation for the normally distributed pertebation to theta_bg in the random walk MCMC sampling */
    double bounds_epsilon=1E-6; /**< The amount to keep away parameter values from the singularities of the prior distribtions */


};



template<class Model, typename=PointEmitterModel::IsPointEmitterModelT<Model>>
std::ostream& operator<<(std::ostream &out,const Model &model);

/* Inline member function definitions */

inline
IdxT PointEmitterModel::get_num_params() const 
{ return num_params; }

inline
PointEmitterModel::ParamT PointEmitterModel::make_param() const
{ return ParamT(num_params); }

inline
PointEmitterModel::ParamVecT PointEmitterModel::make_param_vec(IdxT n) const
{ return ParamVecT(num_params, n); }

inline
MatT PointEmitterModel::make_param_mat() const
{ return MatT(num_params, num_params); }

inline
PointEmitterModel::CompositeDist& PointEmitterModel::get_prior() 
{ return prior; }

inline 
const PointEmitterModel::CompositeDist& PointEmitterModel::get_prior() const 
{ return prior; }

inline 
IdxT PointEmitterModel::get_num_hyperparams() const 
{ return num_hyperparams; }

inline 
const PointEmitterModel::ParamT& PointEmitterModel::get_lbound() const 
{ return lbound; }

inline 
const PointEmitterModel::ParamT& PointEmitterModel::get_ubound() const 
{ return ubound; }    

inline 
IdxT PointEmitterModel::get_mcmc_num_candidate_sampling_phases() const 
{ return mcmc_num_candidate_sampling_phases; }


inline
void PointEmitterModel::set_hyperparams(const VecT &hyperparams)
{ prior.set_params(hyperparams); }

inline
PointEmitterModel::ParamT PointEmitterModel::get_hyperparams() const
{ return prior.params(); }


inline
PointEmitterModel::StringVecT PointEmitterModel::get_params_desc() const
{ return prior.dim_variables(); }

inline
void PointEmitterModel::set_params_desc(const StringVecT &desc)
{ prior.set_dim_variables(desc); }

inline
PointEmitterModel::StringVecT PointEmitterModel::get_hyperparams_desc() const
{ return prior.params_desc(); }

inline
void PointEmitterModel::set_hyperparams_desc(const StringVecT &desc)
{ prior.set_params_desc(desc); }


template<class RngT>
PointEmitterModel::ParamT PointEmitterModel::sample_prior(RngT &rng)
{ return prior.sample(rng); }

inline
PointEmitterModel::ParamT PointEmitterModel::sample_prior()
{ return prior.sample(rng_manager.generator()); }


/* Template function definitions */

template<class Model, typename>
std::ostream& operator<<(std::ostream &out, const Model &model)
{
    auto stats = model.get_stats();
    out<<"["<<model.name()<<"]:\n";
    for(auto& stat: stats) out<<"\t"<<stat.first<<" = "<<stat.second<<"\n";
    return out;
}


} /* namespace mappel */


#endif /* _POINTEMITTERMODEL_H */
