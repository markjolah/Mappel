/** @file PointEmitterModel.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class declaration and inline and templated functions for PointEmitterModel.
 *
 * The base class for all point emitter localization models
 */

#ifndef MAPPEL_POINTEMITTERMODEL_H
#define MAPPEL_POINTEMITTERMODEL_H

#include <iostream>
#include <string>

#include <armadillo>

#include <PriorHessian/CompositeDist.h>
#include <PriorHessian/TruncatedNormalDist.h>
#include <PriorHessian/ScaledSymmetricBetaDist.h>
#include <PriorHessian/TruncatedGammaDist.h>
#include <PriorHessian/TruncatedParetoDist.h>

#include "Mappel/util.h"
#include "Mappel/stencil.h"
#include "Mappel/display.h"
#include "Mappel/rng.h"

namespace mappel {

using prior_hessian::CompositeDist; /**<Composite distribution from prior_hessian:: for representing priors */

    
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
public:
    /* Internal Types */    
    using ParamT = arma::vec; /**< Parameter vector */
    using ParamVecT = arma::mat; /**< Vector of parameter vectors */

    /* Static data members */
    static const std::string DefaultEstimatorMethod; ///<Default optimization method for MLE/MAP estimation.
    static const std::string DefaultProfileBoundsEstimatorMethod; ///<Default optimization method for profile bounds optimizations.
    static const std::string DefaultSeperableInitEstimator; /**< Estimator name to use in 1D separable initializations */
    static const IdxT DefaultMCMCNumSamples; ///< Number of final samples to use in estimation of posterior properties (mean, credible interval, cov, etc.)
    static const IdxT DefaultMCMCBurnin; ///< Number of samples to throw away (burn-in) on initialization
    static const IdxT DefaultMCMCThin; ///< Keep every # samples. [Value of 0 indicates use the model default. This is suggested.]
    static const double DefaultConfidenceLevel; ///< Default level at which to estimate confidence intervals must be in range (0,1).

    static const double DefaultPriorBetaPos;// = 3; /**< Default position parameter in symmetric beta-distributions */
    static const double DefaultPriorSigmaPos;// = 1; /**< Default position parameter in symmetric beta-distributions */
    static const double DefaultPriorMeanI;// = 300; /**< Default emitter intensity mean*/
    static const double DefaultPriorMaxI;// = infinity; /**< Default emitter intensity mean*/
    static const double DefaultPriorIntensityKappa;// = 2;  /**< Default shape for intensity gamma distributions */
    static const double DefaultPriorPixelMeanBG;// = 4; /**< Default per-pixel mean background counts */
    static const double DefaultPriorPSFSigmaAlpha;// = 2; /**< Default per-pixel background gamma distribution shape */

    static const double bounds_epsilon;// = 1.0E-6; /**< Distance from the boundary to constrain in bound_theta and bounded_theta methods */
    static const double global_min_psf_sigma;// = 1E-1; /**< Global minimum for any psf_sigma.  Sizes below this value are invalid, and nowhere near useful for practical point emitter localization */ 
    static const double global_max_psf_sigma;// = 1E2; /**< Global maximum for any psf_sigma.  Sizes above this value are invalid, and nowhere near useful for practical point emitter localization */

    /* prior building functions.  These generate individual prior elements and can be used by subclasses to easily make a prior  */
    static prior_hessian::TruncatedNormalDist make_prior_component_position_normal(IdxT size, double pos_sigma=DefaultPriorSigmaPos);
    static prior_hessian::ScaledSymmetricBetaDist make_prior_component_position_beta(IdxT size, double pos_beta=DefaultPriorBetaPos);
    static prior_hessian::TruncatedGammaDist make_prior_component_intensity(double mean=DefaultPriorMeanI, double kappa=DefaultPriorIntensityKappa);
    static prior_hessian::TruncatedParetoDist make_prior_component_sigma(double min_sigma, double max_sigma, double alpha=DefaultPriorPSFSigmaAlpha);
    
    static void set_rng_seed(RngSeedT seed);
    static ParallelRngManagerT& get_rng_manager();
    static ParallelRngGeneratorT& get_rng_generator();
    
    StatsT get_stats() const;
    
    IdxT get_num_params() const;
    void assert_valid_param_shape(const ParamT &theta) const;
    void assert_valid_param_shape(const ParamVecT &theta) const;
    void check_psf_sigma(double psf_sigma) const;
    void check_psf_sigma(const VecT &psf_sigma) const;
    
    ParamT make_param() const;
    ParamVecT make_param_stack(IdxT n) const;
    MatT make_param_mat() const;
    CubeT make_param_mat_stack(IdxT n) const;
    
    template<class FillT>
    ParamT make_param(FillT fill) const;
    template<class FillT>
    ParamVecT make_param_stack(IdxT n, FillT fill) const;
    template<class FillT>
    MatT make_param_mat(FillT fill) const;
    template<class FillT>
    CubeT make_param_mat_stack(IdxT n, FillT fill) const;
        
    CompositeDist& get_prior();
    const CompositeDist& get_prior() const;
    void set_prior(CompositeDist&& prior_);
    void set_prior(const CompositeDist& prior_);
    
    IdxT get_num_hyperparams() const;
    void set_hyperparams(const VecT &hyperparams);
    VecT get_hyperparams() const;
    
    bool has_hyperparam(const std::string &name) const;
    double get_hyperparam_value(const std::string &name) const; 
    int get_hyperparam_index(const std::string &name) const; 
    void set_hyperparam_value(const std::string &name, double value);
    void rename_hyperparam(const std::string &old_name, const std::string &new_name);
    
    StringVecT get_param_names() const;
    void set_param_names(const StringVecT &desc);
    StringVecT get_hyperparam_names() const;
    void set_hyperparam_names(const StringVecT &desc);
    
    template<class RngT> ParamT sample_prior(RngT &rng) const;
    ParamT sample_prior() const; //Use global ParallelRngManager

    /** Box-type parameter bounds */
    void set_bounds(const ParamT &lbound, const ParamT &ubound);
    void set_lbound(const ParamT &lbound);
    void set_ubound(const ParamT &ubound);
    const ParamT& get_lbound() const;
    const ParamT& get_ubound() const;
    
    bool theta_in_bounds(const ParamT &theta) const;
    void bound_theta(ParamT &theta,double epsilon=bounds_epsilon) const;
    ParamT bounded_theta(const ParamT &theta,double epsilon=bounds_epsilon) const;
    ParamT reflected_theta(const ParamT &theta) const;
    BoolVecT theta_stack_in_bounds(const ParamVecT &theta) const;
    ParamVecT bounded_theta_stack(const ParamVecT &theta,double epsilon=bounds_epsilon) const;
    ParamVecT reflected_theta_stack(const ParamVecT &theta) const;
    
protected:
    /* Constructor */
    PointEmitterModel();
    explicit PointEmitterModel(const CompositeDist& prior_);    
    explicit PointEmitterModel(CompositeDist&& prior_);    
    PointEmitterModel(const PointEmitterModel &);
    PointEmitterModel(PointEmitterModel &&);
    PointEmitterModel& operator=(const PointEmitterModel &);
    PointEmitterModel& operator=(PointEmitterModel &&);

    /* Data members */
    CompositeDist prior; /* Prior distribution represented using libPriorHessian */
    IdxT num_params;
    IdxT num_hyperparams;
    ParamT lbound,ubound; /* Vectors of lower and upper bounds. (lbound>=prior.lbound && ubound<=prior.ubound) */
    
private:
    void update_cached_prior_values();    
};

template<class Model, typename=EnableIfSubclassT<Model,PointEmitterModel>>
std::ostream& operator<<(std::ostream &out,const Model &model);

/* Inline member function definitions */

inline
IdxT PointEmitterModel::get_num_params() const 
{ return num_params; }

inline
PointEmitterModel::ParamT PointEmitterModel::make_param() const
{ return ParamT(num_params); }

inline
PointEmitterModel::ParamVecT PointEmitterModel::make_param_stack(IdxT n) const
{ return ParamVecT(num_params, n); }

inline
MatT PointEmitterModel::make_param_mat() const
{ return MatT(num_params, num_params); }

inline
CubeT PointEmitterModel::make_param_mat_stack(IdxT n) const
{ return CubeT(num_params, num_params, n); }

template<class FillT>
PointEmitterModel::ParamT 
PointEmitterModel::make_param(FillT fill) const
{ return ParamT(num_params, fill); }

template<class FillT>
PointEmitterModel::ParamVecT
PointEmitterModel::make_param_stack(IdxT n, FillT fill) const
{ return ParamVecT(num_params, n, fill); }

template<class FillT>
MatT 
PointEmitterModel::make_param_mat(FillT fill) const
{ return MatT(num_params, num_params, fill); }

template<class FillT>
CubeT 
PointEmitterModel::make_param_mat_stack(IdxT n, FillT fill) const
{ return CubeT(num_params, num_params, n, fill); }

inline
CompositeDist& PointEmitterModel::get_prior() 
{ return prior; }

inline 
const CompositeDist& PointEmitterModel::get_prior() const 
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
void PointEmitterModel::set_hyperparams(const VecT &hyperparams)
{ prior.set_params(hyperparams); }

inline
PointEmitterModel::ParamT PointEmitterModel::get_hyperparams() const
{ return prior.params(); }

inline
bool PointEmitterModel::has_hyperparam(const std::string &name) const
{ return prior.has_param(name); }

inline
double PointEmitterModel::get_hyperparam_value(const std::string &name) const
{ return prior.get_param_value(name); }

inline
int PointEmitterModel::get_hyperparam_index(const std::string &name) const
{ return prior.get_param_index(name); }

inline
void PointEmitterModel::set_hyperparam_value(const std::string &name, double value)
{ prior.set_param_value(name,value); }

inline
void PointEmitterModel::rename_hyperparam(const std::string &old_name, const std::string &new_name)
{ prior.rename_param(old_name,new_name); }

inline
StringVecT PointEmitterModel::get_param_names() const
{ return prior.dim_variables(); }

inline
void PointEmitterModel::set_param_names(const StringVecT &desc)
{ prior.set_dim_variables(desc); }

inline
StringVecT PointEmitterModel::get_hyperparam_names() const
{ return prior.param_names(); }

inline
void PointEmitterModel::set_hyperparam_names(const StringVecT &desc)
{ prior.set_param_names(desc); }

template<class RngT>
PointEmitterModel::ParamT PointEmitterModel::sample_prior(RngT &rng) const
{ return prior.sample(rng); }

inline PointEmitterModel::ParamT
PointEmitterModel::sample_prior() const
{
    return prior.sample(rng_manager.generic_generator());
}


/* Template function definitions */
template<class Model, typename>
std::ostream& operator<<(std::ostream &out, const Model &model)
{
    auto stats = model.get_stats();
    out<<"["<<model.name<<"]:\n";
    for(auto& stat: stats) out<<"\t"<<stat.first<<" = "<<stat.second<<"\n";
    return out;
}

} /* namespace mappel */

#endif /* MAPPEL_POINTEMITTERMODEL_H */
