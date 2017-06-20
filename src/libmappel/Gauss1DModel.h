/** @file Gauss1DModel.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class declaration and inline and templated functions for Gauss1DModel.
 */

#ifndef _GAUSS1DMODEL_H
#define _GAUSS1DMODEL_H

#include "PointEmitterModel.h"
#include "ImageFormat1DBase.h"

namespace mappel {

/** @brief A base class for 2D Gaussian PSF with fixed but possibly asymmetric sigma.
 *
 */
class Gauss1DModel : public PointEmitterModel, public virtual ImageFormat1DBase {
protected:
    /* Hyperparameters */
    double beta_pos=1.5; /**< The shape parameter for the Beta prior on the x and y positions. 0=Uniform, 1=Peaked  */
    double mean_I=300.; /**< The mean of the intensity gamma prior */
    double kappa_I=2.;  /**< The shape parameter for the I prior gamma distribution 1=exponential 2-5=skewed large=normal */
    double mean_bg=3.; /**< The mean of the background gamma prior */
    double kappa_bg=2.;  /**< The shape parameter for the bg prior gamma distribution 1=exponential 2-5=skewed large=normal */
    
    /* RNG distributions for prior */
    BetaRNG pos_dist;
    GammaRNG I_dist;
    GammaRNG bg_dist;

    /* Constants for prior calculations */
    double log_prior_pos_const; /**< This is -2*lgamma(beta_x)-lgamma(2*beta_x) */
    double log_prior_I_const; /**< This is kappa_I*(log(kappa_I)-1/mean_I-log(mean_I))-lgamma(kappa_I) */
    double log_prior_bg_const; /**< This is kappa_bg*(log(kappa_bg)-1/mean_bg-log(mean_bg))-lgamma(kappa_bg) */
    
public:
    /* Internal Types */
    class Stencil {
    public:
        bool derivatives_computed=false;
        using ParamT = Gauss1DModel::ParamT;
        Gauss1DModel const *model;

        ParamT theta; 
        VecT dx;
        VecT Gx;
        VecT X;
        VecT DX;
        VecT DXS;
        Stencil() {}
        Stencil(const Gauss1DModel &model, const ParamT &theta, bool compute_derivatives=true);
        void compute_derivatives();
        inline double x() const {return theta(0);}
        inline double I() const {return theta(1);}
        inline double bg() const {return theta(2);}
        friend std::ostream& operator<<(std::ostream &out, const Gauss1DModel::Stencil &s);
    };

    /* Static Data members */
    static const std::vector<std::string> param_names;
    static const std::vector<std::string> hyperparameter_names;
    
    /* Data Members */
    double psf_sigma; /**< The standard deviation of the stymmetric gaussian PSF in units of pixels for X and Y */
    
    Gauss1DModel(int size_, double psf_sigma);

    StatsT get_stats() const;

    /* Make arrays for working with model data */
    using PointEmitterModel::make_param;
    ParamT make_param(double x, double I, double bg) const;
    ParamT make_param(const ParamT &theta) const;
    Stencil make_stencil(const ParamT &theta, bool compute_derivatives=true) const;
    Stencil make_stencil(double x, double I=1.0, double bg=0.0, bool compute_derivatives=true) const;


    /* Model Pixel Value And Derivatives */
    double pixel_model_value(int i,  const Stencil &s) const;
    void pixel_grad(int i, const Stencil &s, ParamT &pgrad) const;
    void pixel_grad2(int i, const Stencil &s, ParamT &pgrad2) const;
    void pixel_hess(int i, const Stencil &s, ParamMatT &hess) const;
    void pixel_hess_update(int i, const Stencil &s, double dm_ratio_m1, 
                           double dmm_ratio, ParamT &grad, ParamMatT &hess) const;

    /* Prior values and derivatives */
    void set_hyperparameters(const VecT &hyperparameters);
    VecT get_hyperparameters() const;
    ParamT sample_prior(RNG &rng);
    double prior_log_likelihood(const Stencil &s) const;
    double prior_relative_log_likelihood(const Stencil &s) const;
    ParamT prior_grad(const Stencil &s) const;
    ParamT prior_grad2(const Stencil &s) const;
    ParamT prior_hess(const Stencil &s) const;
                           
    /* Compute the Log likelihood of an image at theta */
    Stencil initial_theta_estimate(const ImageT &im, const ParamT &theta_init) const;

    /* Posterior Sampling */
    void sample_mcmc_candidate_theta(int sample_index, RNG &rng, ParamT &canidate_theta, double scale=1.0) const;
};

/* Function Declarations */



/* Inline Method Definitions */

inline
Gauss1DModel::ParamT
Gauss1DModel::sample_prior(RNG &rng)
{
    ParamT theta = make_param();
    theta(0) = size*pos_dist(rng);
    theta(1) = I_dist(rng);
    theta(2) = bg_dist(rng);
    bound_theta(theta);
    return theta;
}


inline
Gauss1DModel::ParamT
Gauss1DModel::make_param(double x, double I, double bg) const
{
    ParamT theta = {x,I,bg};
    bound_theta(theta);
    return theta;
}

inline
Gauss1DModel::ParamT
Gauss1DModel::make_param(const ParamT &theta) const
{
    ParamT ntheta(theta);
    bound_theta(ntheta);
    return ntheta;
}

inline
Gauss1DModel::Stencil
Gauss1DModel::make_stencil(const ParamT &theta, bool compute_derivatives) const
{
//    return Stencil(*this,make_param(theta),compute_derivatives);
    //Remove implicit bounding.  This allows for computations outside of the limited region
    //And prevents false impression that LLH and grad and hessian do not change outside of boundaries
    return Stencil(*this,theta,compute_derivatives);
}

inline
Gauss1DModel::Stencil
Gauss1DModel::make_stencil(double x, double I, double bg, bool compute_derivatives) const
{
    return Stencil(*this,make_param(x,I,bg),compute_derivatives);
}


inline
double Gauss1DModel::pixel_model_value(int i,  const Stencil &s) const
{
    return s.bg()+s.I()*s.X(i);
}

inline
void
Gauss1DModel::pixel_grad(int i, const Stencil &s, ParamT &pgrad) const
{
    double I = s.I();
    pgrad(0) = I * s.DX(i);
    pgrad(1) = s.X(i);
    pgrad(2) = 1.;
}

inline
void
Gauss1DModel::pixel_grad2(int i, const Stencil &s, ParamT &pgrad2) const
{
    double I = s.I();
    pgrad2(0) = I/psf_sigma * s.DXS(i);
    pgrad2(1) = 0.;
    pgrad2(2) = 0.;
}

inline
void
Gauss1DModel::pixel_hess(int i, const Stencil &s, ParamMatT &hess) const
{
    hess.zeros();
    double I = s.I();
    hess(0,0) = I/psf_sigma * s.DXS(i);
    hess(0,1) = s.DX(i); 
}

} /* namespace mappel */

#endif /* _GAUSS1DMODEL_H */
