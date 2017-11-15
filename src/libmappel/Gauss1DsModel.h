/** @file Gauss1DsModel.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief The class declaration and inline and templated functions for Gauss1DsModel.
 */

#ifndef _Gauss1DSMODEL_H
#define _Gauss1DSMODEL_H

#include "PointEmitterModel.h"
#include "ImageFormat1DBase.h"

namespace mappel {

/** @brief A base class for 2D Gaussian PSF with fixed but possibly asymmetric sigma.
 *
 */
class Gauss1DsModel : public PointEmitterModel, public virtual ImageFormat1DBase {
protected:
    /* Hyperparameters */
    double beta_pos=1.5; /**< The shape parameter for the Beta prior on the x and y positions. 0=Uniform, 1=Peaked  */
    double mean_I=300.; /**< The mean of the intensity gamma prior */
    double kappa_I=2.;  /**< The shape parameter for the I prior gamma distribution 1=exponential 2-5=skewed large=normal */
    double mean_bg=3.; /**< The mean of the background gamma prior */
    double kappa_bg=2.;  /**< The shape parameter for the bg prior gamma distribution 1=exponential 2-5=skewed large=normal */
    double alpha_sigma=3.0; /**<Shape parameter for the pareto dist. over sigma (larger=more concentrated about min) */
    double min_sigma; /**< minimum sigma for the symmetric Gaussian PSF in pixels.  This is a constraint on the sigma parameter in maximization.  */
    double max_sigma; /**< maximum sigma for the symmetric Gaussian PSF in pixels.  This is a constraint on the sigma parameter in maximization.*/
    
    /* RNG distributions for prior */
    BetaRNG pos_dist;
    GammaRNG I_dist;
    GammaRNG bg_dist;
    ParetoRNG sigma_dist;

    /* Constants for prior calculations */
    double log_prior_pos_const; /**< This is -2*lgamma(beta_x)-lgamma(2*beta_x) */
    double log_prior_I_const; /**< This is kappa_I*(log(kappa_I)-1/mean_I-log(mean_I))-lgamma(kappa_I) */
    double log_prior_bg_const; /**< This is kappa_bg*(log(kappa_bg)-1/mean_bg-log(mean_bg))-lgamma(kappa_bg) */
    double log_prior_sigma_const;

    double mcmc_candidate_eta_sigma; /**< The standard deviation for the normally distributed pertebation to theta_sigma in the random walk MCMC sampling */
public:
    /* Internal Types */
    class Stencil {
    public:
        bool derivatives_computed=false;
        using ParamT = Gauss1DsModel::ParamT;
        Gauss1DsModel const *model;

        ParamT theta; 
        VecT dx;
        VecT Gx;
        VecT X;
        VecT DX;
        VecT DXS;
        VecT DXS2;
        VecT DXSX;
        Stencil() {}
        Stencil(const Gauss1DsModel &model, const ParamT &theta, bool _compute_derivatives=true);
        void compute_derivatives();
        inline double x() const {return theta(0);}
        inline double I() const {return theta(1);}
        inline double bg() const {return theta(2);}
        inline double sigma() const {return theta(3);}
        friend std::ostream& operator<<(std::ostream &out, const Gauss1DsModel::Stencil &s);
    };

    /* Static Data members */
    static const std::vector<std::string> param_names;
    static const std::vector<std::string> hyperparameter_names;
    
        Gauss1DsModel(int size, double min_sigma, double max_sigma);

    StatsT get_stats() const;

    /* Make arrays for working with model data */
    using PointEmitterModel::make_param;
    ParamT make_param(double x, double I, double bg, double sigma) const;
    ParamT make_param(const ParamT &theta) const;
    Stencil make_stencil(const ParamT &theta, bool compute_derivatives=true) const;
    Stencil make_stencil(double x, double I, double bg,double sigma, bool compute_derivatives=true) const;


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
Gauss1DsModel::ParamT
Gauss1DsModel::sample_prior(RNG &rng)
{
    ParamT theta = make_param();
    theta(0) = size*pos_dist(rng);
    theta(1) = I_dist(rng);
    theta(2) = bg_dist(rng);
    theta(3) = sigma_dist(rng);
    bound_theta(theta);
    return theta;
}

inline
Gauss1DsModel::ParamT
Gauss1DsModel::make_param(double x, double I, double bg, double sigma) const
{
    ParamT theta = {x,I,bg,sigma};
    bound_theta(theta);
    return theta;
}

inline
Gauss1DsModel::ParamT
Gauss1DsModel::make_param(const ParamT &theta) const
{
    ParamT ntheta(theta);
    bound_theta(ntheta);
    return ntheta;
}

inline
Gauss1DsModel::Stencil
Gauss1DsModel::make_stencil(const ParamT &theta, bool compute_derivatives) const
{
//    return Stencil(*this,make_param(theta),compute_derivatives);
    //Remove implicit bounding.  This allows for computations outside of the limited region
    //And prevents false impression that LLH and grad and hessian do not change outside of boundaries
    return Stencil(*this,theta,compute_derivatives);
}

inline
Gauss1DsModel::Stencil
Gauss1DsModel::make_stencil(double x, double I, double bg, double sigma, bool compute_derivatives) const
{
    return Stencil(*this,make_param(x,I,bg,sigma),compute_derivatives);
}


inline
double Gauss1DsModel::pixel_model_value(int i,  const Stencil &s) const
{
    return s.bg()+s.I()*s.X(i);
}

inline
void
Gauss1DsModel::pixel_grad(int i, const Stencil &s, ParamT &pgrad) const
{
    double I = s.I();
    pgrad(0) = I * s.DX(i);
    pgrad(1) = s.X(i);
    pgrad(2) = 1.;
    pgrad(3) = I * s.DXS(i);
}

inline
void
Gauss1DsModel::pixel_grad2(int i, const Stencil &s, ParamT &pgrad2) const
{
    double I = s.I();
    pgrad2(0) = I/s.sigma() * s.DXS(i);
    pgrad2(1) = 0.;
    pgrad2(2) = 0.;
    pgrad2(3) = I * s.DXS2(i);
}

inline
void
Gauss1DsModel::pixel_hess(int i, const Stencil &s, ParamMatT &hess) const
{
    hess.zeros();
    double I = s.I();
    hess(0,0) = I/s.sigma() * s.DXS(i);
    hess(0,1) = s.DX(i); 
    hess(0,3) = I * s.DXSX(i);
    hess(1,3) = s.DXS(i);
    hess(3,3) = I * s.DXS2(i);
}

} /* namespace mappel */

#endif /* _Gauss1DSMODEL_H */
