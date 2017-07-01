/** @file Gauss2DModel.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class declaration and inline and templated functions for Gauss2DModel.
 */

#ifndef _GAUSS2DMODEL_H
#define _GAUSS2DMODEL_H

#include "PointEmitterModel.h"
#include "ImageFormat2DBase.h"
#include "Gauss1DMAP.h"
#include "cGaussMLE/cGaussMLE.h"

namespace mappel {

/** @brief A base class for 2D Gaussian PSF with fixed but possibly asymmetric sigma.
 *
 */
class Gauss2DModel : public PointEmitterModel, public virtual ImageFormat2DBase {
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
        using ParamT = Gauss2DModel::ParamT;
        Gauss2DModel const *model;

        ParamT theta; 
        VecT dx, dy;
        VecT Gx, Gy;
        VecT X, Y;
        VecT DX, DY;
        VecT DXS, DYS;
        Stencil() {}
        Stencil(const Gauss2DModel &model, const ParamT &theta, bool compute_derivatives=true);
        void compute_derivatives();
        inline double x() const {return theta(0);}
        inline double y() const {return theta(1);}
        inline double I() const {return theta(2);}
        inline double bg() const {return theta(3);}
        friend std::ostream& operator<<(std::ostream &out, const Gauss2DModel::Stencil &s);
    };

    /* Static Data members */
    static const std::vector<std::string> param_names;
    static const std::vector<std::string> hyperparameter_names;
    
    /* Data Members */
    const VecT psf_sigma; /**< The standard deviation of the stymmetric gaussian PSF in units of pixels for X and Y */
    
    Gauss2DModel(const IVecT &size, const VecT &psf_sigma);

    StatsT get_stats() const;



    /* Make arrays for working with model data */
    using PointEmitterModel::make_param;
    ParamT make_param(double x, double y, double I, double bg) const;
    ParamT make_param(const ParamT &theta) const;
    Stencil make_stencil(const ParamT &theta, bool compute_derivatives=true) const;
    Stencil make_stencil(double x, double y, double I=1.0, double bg=0.0, bool compute_derivatives=true) const;


    /* Model Pixel Value And Derivatives */
    double pixel_model_value(int i, int j, const Stencil &s) const;
    void pixel_grad(int i, int j, const Stencil &s, ParamT &pgrad) const;
    void pixel_grad2(int i, int j, const Stencil &s, ParamT &pgrad2) const;
    void pixel_hess(int i, int j, const Stencil &s, ParamMatT &hess) const;
    void pixel_hess_update(int i, int j, const Stencil &s, double dm_ratio_m1, 
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
    Stencil heuristic_initial_theta_estimate(const ImageT &im, const ParamT &theta_init) const;
    Stencil seperable_initial_theta_estimate(const ImageT &im, const ParamT &theta_init, const std::string &estimator) const;

    /* Posterior Sampling */
    void sample_mcmc_candidate_theta(int sample_index, RNG &rng, ParamT &canidate_theta, double scale=1.0) const;

protected:
    VecT gaussian_Xstencil; /**< A stencil for gaussian filters with this size and psf*/
    VecT gaussian_Ystencil; /**< A stencil for gaussian filters with this size and psf*/
    Gauss1DMAP x_model, y_model;
};

/* Function Declarations */



/* Inline Method Definitions */

inline
Gauss2DModel::ParamT
Gauss2DModel::sample_prior(RNG &rng)
{
    ParamT theta = make_param();
    theta(0) = size(0)*pos_dist(rng);
    theta(1) = size(1)*pos_dist(rng);
    theta(2) = I_dist(rng);
    theta(3) = bg_dist(rng);
    bound_theta(theta);
    return theta;
}


inline
Gauss2DModel::ParamT
Gauss2DModel::make_param(double x, double y, double I, double bg) const
{
    ParamT theta = {x,y,I,bg};
    bound_theta(theta);
    return theta;
}

inline
Gauss2DModel::ParamT
Gauss2DModel::make_param(const ParamT &theta) const
{
    ParamT ntheta(theta);
    bound_theta(ntheta);
    return ntheta;
}

inline
Gauss2DModel::Stencil
Gauss2DModel::make_stencil(const ParamT &theta, bool compute_derivatives) const
{
//    return Stencil(*this,make_param(theta),compute_derivatives);
    //Remove implicit bounding.  This allows for computations outside of the limited region
    //And prevents false impression that LLH and grad and hessian do not change outside of boundaries
    return Stencil(*this,theta,compute_derivatives);
}

inline
Gauss2DModel::Stencil
Gauss2DModel::make_stencil(double x, double y, double I, double bg, bool compute_derivatives) const
{
    return Stencil(*this,make_param(x,y,I,bg),compute_derivatives);
}


inline
double Gauss2DModel::pixel_model_value(int i, int j, const Stencil &s) const
{
    return s.bg()+s.I()*s.X(i)*s.Y(j);
}

inline
void
Gauss2DModel::pixel_grad(int i, int j, const Stencil &s, ParamT &pgrad) const
{
    double I = s.I();
    pgrad(0) = I * s.DX(i) * s.Y(j);
    pgrad(1) = I * s.DY(j) * s.X(i);
    pgrad(2) = s.X(i) * s.Y(j);
    pgrad(3) = 1.;
}

inline
void
Gauss2DModel::pixel_grad2(int i, int j, const Stencil &s, ParamT &pgrad2) const
{
    double I = s.I();
    pgrad2(0) = I/psf_sigma(0) * s.DXS(i) * s.Y(j);
    pgrad2(1) = I/psf_sigma(1) * s.DYS(j) * s.X(i);
    pgrad2(2) = 0;
    pgrad2(3) = 0;
}

inline
void
Gauss2DModel::pixel_hess(int i, int j, const Stencil &s, ParamMatT &hess) const
{
    hess.zeros();
    double I = s.I();
    hess(0,0) = I/psf_sigma(0) * s.DXS(i) * s.Y(j);
    hess(0,1) = I * s.DX(i) * s.DY(j);
    hess(1,1) = I/psf_sigma(1) * s.DYS(j) * s.X(i);
    hess(0,2) = s.DX(i) * s.Y(j); 
    hess(1,2) = s.DY(j) * s.X(i); 
}

inline
Gauss2DModel::Stencil 
Gauss2DModel::initial_theta_estimate(const ImageT &im, const ParamT &theta_init) const
{
    return seperable_initial_theta_estimate(im, theta_init,"Newton");
}

/* Templated Overloads */



} /* namespace mappel */

#endif /* _GAUSS2DMODEL_H */
