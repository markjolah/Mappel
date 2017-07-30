/** @file PointEmitterModel.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class declaration and inline and templated functions for PointEmitterModel.
 *
 * The base class for all point emitter localization models
 */

#ifndef _POINTEMITTERMODEL_H
#define _POINTEMITTERMODEL_H

#include <fstream>
#include <string>

#include <armadillo>

#include "util.h"
#include "numerical.h"
#include "stencil.h"
#include "display.h"
#include "stackcomp.h"
#include "estimator.h"
#include "mcmc.h"


namespace mappel {

/** @brief A virtual Base type for point emitter localization models.
 * We don't assume much here, so that it is possible to have a wide range of 2D and 3D models.
 * 
 * Of note some of the common MCMC variables are rooted here in the inheritance tree.
 */
class PointEmitterModel {
public:
    /* Internal Types */
    using UVecT = arma::uvec;
    using VecT = arma::vec;
    using MatT = arma::mat;
    using ParamT = arma::vec; /**< A type for the set of parameters estimated by the model */
    using ParamVecT = arma::mat; /**< A Vector of parameter values */
    using ParamMatT = arma::mat; /**< A matrix type for the Hessian */
    using ParamMatStackT = arma::cube; /**< A type for a stack of hessian matrices */
    
    /* Data members */
    int mcmc_num_candidate_sampling_phases=0; /**< The number of different sampling phases for candidate selection MCMC.  Each phase changes a different subset of variables.*/

    /* Constant model parameter information */
    const int num_params;

    ParamT lbound,ubound; /* Vectors of lower and upper bounds */
    
    PointEmitterModel(int num_params);
    virtual ~PointEmitterModel() {}

    virtual std::string name() const =0;
    virtual StatsT get_stats() const =0;

    ParamVecT make_param_vec(int n) const;
    ParamT make_param() const;
    ParamMatT make_param_mat() const;
    
    void set_bounds(const ParamT &lbound, const ParamT &ubound);
    void bound_theta(ParamT &theta,double epsilon=-1) const;
    bool theta_in_bounds(const ParamT &theta,double epsilon=-1) const;
    ParamT bounded_theta(const ParamT &theta,double epsilon=-1) const;
    ParamT reflected_theta(const ParamT &theta,double epsilon=-1) const;
    
    friend std::ostream& operator<<(std::ostream &out, PointEmitterModel &model);
protected:
    double prior_epsilon=1E-6; /**< The amount to keep away parameter values from the singularities of the prior distribtions */
    double mcmc_candidate_sample_dist_ratio=1./30.; /**< Controls the candidate distribution spread for MCMC stuff */
    double mcmc_candidate_eta_x; /**< The standard deviation for the normally distributed pertebation to theta_x in the random walk MCMC sampling */
    double mcmc_candidate_eta_y; /**< The standard deviation for the normally distributed pertebation to theta_y in the random walk MCMC sampling */
    double mcmc_candidate_eta_I; /**< The standard deviation for the normally distributed pertebation to theta_I in the random walk MCMC sampling */
    double mcmc_candidate_eta_bg; /**< The standard deviation for the normally distributed pertebation to theta_bg in the random walk MCMC sampling */

    UVecT lbound_valid, ubound_valid;
    
};

/* Inline member function definitions */

inline
PointEmitterModel::ParamT
PointEmitterModel::make_param() const
{
    return ParamT(num_params);
}


inline
PointEmitterModel::ParamVecT PointEmitterModel::make_param_vec(int n=0) const
{
    return ParamVecT(num_params, n);
}


inline
PointEmitterModel::ParamMatT 
PointEmitterModel::make_param_mat() const
{
    return ParamMatT(num_params, num_params);
}


inline
bool PointEmitterModel::theta_in_bounds(const ParamT &theta, double epsilon) const
{
    if(epsilon<0) epsilon=prior_epsilon;
    for(int n=0;n<num_params;n++) {
        if(lbound_valid(n) && theta(n)<lbound(n)+epsilon) return false;
        if(ubound_valid(n) && theta(n)>ubound(n)-epsilon) return false;
    }
    return true;
}

inline
void PointEmitterModel::bound_theta(ParamT &theta, double epsilon) const
{
    if(epsilon<0) epsilon=prior_epsilon;
    for(int n=0;n<num_params;n++) {
        if(lbound_valid(n) && theta(n)<lbound(n)+epsilon) theta(n)=lbound(n)+epsilon;
        if(ubound_valid(n) && theta(n)>ubound(n)-epsilon) theta(n)=ubound(n)-epsilon;
    }
}

inline
PointEmitterModel::ParamT PointEmitterModel::bounded_theta(const ParamT &theta, double epsilon) const
{
    if(epsilon<0) epsilon=prior_epsilon;
    ParamT btheta = theta;
    for(int n=0;n<num_params;n++) {
        if(lbound_valid(n) && theta(n)<lbound(n)+epsilon) btheta(n)=lbound(n)+epsilon;
        if(ubound_valid(n) && theta(n)>ubound(n)-epsilon) btheta(n)=ubound(n)-epsilon;
    }
    return btheta;
}

inline
PointEmitterModel::ParamT PointEmitterModel::reflected_theta(const ParamT &theta, double epsilon) const
{
    if(epsilon<0) epsilon=prior_epsilon;
    ParamT btheta = theta;
    for(int n=0;n<num_params;n++) {
        if(lbound_valid(n)) {
            if(ubound_valid(n)){//both valid bounds.  Do reflection
                double d = 2*(ubound(n)-lbound(n));
                double w = std::fmod(std::fabs(theta(n)-lbound(n)), d);
                btheta(n) = std::min(w,d-w)+lbound(n);
            } else if (theta(n)<lbound(n)) btheta(n)=2*lbound(n)-theta(n); //valid lower bound only
        } else if(ubound_valid(n) && theta(n)>ubound(n)) btheta(n)=2*ubound(n)-theta(n); //valid upper bound only
    }
    return btheta;
}


/* Template function definitions */

/* These are convenience functions for other templated functions that allow calling without stencils,
 * by converting param vector into stencil with make_stencil
 */
template<class Model>
inline
typename Model::ImageT
model_image(const Model &model, const typename Model::ParamT &theta) 
{
    return model_image(model, model.make_stencil(theta,false));
}

template<class Model, class rng_t>
inline
typename Model::ImageT
simulate_image(const Model &model, const typename Model::ParamT &theta, rng_t &rng) 
{
    return simulate_image(model, model.make_stencil(theta,false), rng);
}

template<class Model>
inline
double
log_likelihood(const Model &model, const typename Model::ImageT &data_im, 
               const typename Model::ParamT &theta)
{
    return log_likelihood(model, data_im, model.make_stencil(theta,false));
}

template<class Model>
inline
double
relative_log_likelihood(const Model &model, const typename Model::ImageT &data_im, 
               const typename Model::ParamT &theta)
{
    return relative_log_likelihood(model, data_im, model.make_stencil(theta,false));
}

template<class Model>
inline
typename Model::ParamT
model_grad(const Model &model, const typename Model::ImageT &data_im, 
               const typename Model::ParamT &theta)
{
    auto grad = model.make_param();
    model_grad(model, data_im, model.make_stencil(theta), grad);
    return grad;
}

template<class Model>
inline
typename Model::ParamMatT
model_hessian(const Model &model, const typename Model::ImageT &data_im, 
               const typename Model::ParamT &theta)
{
    auto grad = model.make_param();
    auto hess = model.make_param_mat();
    model_hessian(model, data_im, model.make_stencil(theta), grad, hess);
    copy_Usym_mat(hess);
    return hess;
}

template<class Model>
inline
typename Model::ParamMatT
model_positive_hessian(const Model &model, const typename Model::ImageT &data_im, 
              const typename Model::ParamT &theta)
{
    auto grad = model.make_param();
    auto hess = model.make_param_mat();
    model_hessian(model, data_im, model.make_stencil(theta), grad, hess);
    hess = -hess;
    copy_Usym_mat(hess);
    modified_cholesky(hess);
    cholesky_convert_full_matrix(hess); //convert from internal format to a full (poitive definite) matrix
    return hess;
}


/** @brief Calculate the Cramer-Rao lower bound at the given paramters
 * @param[in] theta The parameters to evaluate the CRLB at
 * @param[out] crlb The calculated parameters
 */
template<class Model>
typename Model::ParamT
cr_lower_bound(const Model &model, const typename Model::Stencil &s)
{
    auto FI = fisher_information(model,s);
    try{
        return arma::pinv(FI).eval().diag();
    } catch ( std::runtime_error E) {
        std::cout<<"Got bad fisher_information!!\n"<<"theta:"<<s.theta.t()<<"\n FI: "<<FI<<"\n";
        auto z = model.make_param();
        z.zeros();
        return z;
    }
}

template<class Model>
inline
typename Model::ParamT
cr_lower_bound(const Model &model, const typename Model::ParamT &theta) 
{
    return cr_lower_bound(model,model.make_stencil(theta));
}

template<class Model>
inline
typename Model::ParamMatT
fisher_information(const Model &model, const typename Model::ParamT &theta) 
{
    return fisher_information(model,model.make_stencil(theta));
}

} /* namespace mappel */


#endif /* _POINTEMITTERMODEL_H */
