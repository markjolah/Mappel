/** @file Blink2DsMAP.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class declaration and inline and templated functions for Blink2DsMAP.
 */

#ifndef _BLINK2DSMAP_H
#define _BLINK2DSMAP_H

#include "PointEmitter2DModel.h"
#include "BlinkModel.h"

/** @brief A base class for likelihood models for point emitters imaged in 
 * 2D with symmmetric PSF, where we estimate the apparent psf sigma accounting
 * for out-of-foucs emitters.
 *
 *
 *
 */
class Blink2DsMAP : public PointEmitter2DModel, public BlinkModel {
protected:
    /* Hyperparameters */
    double beta_pos=1.01; /**< The shape parameter for the Beta prior on the x and y positions. 0=Uniform, 1=Peaked  */
    double mean_I=1000.; /**< The mean of the intensity gamma prior */
    double kappa_I=2.;  /**< The shape parameter for the I prior gamma distribution 1=exponential 2-5=skewed large=normal */
    double mean_bg=3.; /**< The mean of the background gamma prior */
    double kappa_bg=2.;  /**< The shape parameter for the bg prior gamma distribution 1=exponential 2-5=skewed large=normal */
    double alpha_sigma=3.;  /**< The shape parameter for the bg prior gamma distribution 1=exponential 2-5=skewed large=normal */
    /* Distributions for drawing RND#s */
    BetaRNG pos_dist;
    GammaRNG I_dist;
    GammaRNG bg_dist;
    ParetoRNG sigma_dist;

    /* Precomputed Data  */
    double log_prior_pos_const; /**< This is -2*lgamma(beta_x)-lgamma(2*beta_x) */
    double log_prior_I_const; /**< This is kappa_I*(log(kappa_I)-1/mean_I-log(mean_I))-lgamma(kappa_I) */
    double log_prior_bg_const; /**< This is kappa_bg*(log(kappa_bg)-1/mean_bg-log(mean_bg))-lgamma(kappa_bg) */
    double log_prior_sigma_const;
public:
    /* Model matrix and vector types */
    typedef arma::vec ParamT; /**< A type for the set of parameters estimated by the model */
    typedef arma::mat ParamMatT; /**< A matrix type for the Hessian used by the CRLB estimation */
    
    std::vector<std::string> param_names;
    static const std::vector<std::string> hyperparameter_names;
    class Stencil {
    public:
        bool derivatives_computed=false;
        typedef Blink2DsMAP::ParamT ParamT;
        Blink2DsMAP const *model;
        ParamT theta;
        VecT dx, dy;
        VecT Gx, Gy;
        VecT X, Y;
        VecT DX, DY;
        VecT DXS, DYS;
        VecT DXS2, DYS2;
        VecT DXSX, DYSY;
        Stencil() {}
        Stencil(const Blink2DsMAP &model, const ParamT &theta, bool _compute_derivatives=true);
        void compute_derivatives();
        void set_duty(int i, double D){
            assert(model);
            assert(0<=i && i<model->size(0));
            assert(0<=D && D<=1);
            theta(5+i)=restrict_value_range(D, model->prior_epsilon, 1.-model->prior_epsilon);
        }
        inline double x() const {return theta(0);}
        inline double y() const {return theta(1);}
        inline double I() const {return theta(2);}
        inline double bg() const {return theta(3);}
        inline double sigma() const {return theta(4);}
        inline double sigmaX() const {return model->psf_sigma(0)*sigma();}
        inline double sigmaY() const {return model->psf_sigma(1)*sigma();}
        inline double D(int i) const {return theta(5+i);}
        friend std::ostream& operator<<(std::ostream &out, const Blink2DsMAP::Stencil &s);
    };

    /**
     * @brief An inner class for representing model values in a form that allows efficient recomputation
     * of relative_log_likelihoods when only Durt ratios change.  This elimintates many repeated
     * calls to log that occur in posterior sampling.
     *
     *
     */
    class ModelImage
    {
        const Blink2DsMAP *model;
        Blink2DsMAP::Stencil stencil;
        const Blink2DsMAP::ImageT *data_im;
        Blink2DsMAP::ImageT model_im;
        Blink2DsMAP::ImageT log_model_im;
    public:
        ModelImage(const Blink2DsMAP &model_,
                   const Blink2DsMAP::ImageT &data_im_);
        void set_stencil(const Blink2DsMAP::ParamT &theta);

        void set_duty(int i, double D);

        double relative_log_likelihood() const;
    };

    /* Constructor */
    Blink2DsMAP(const IVecT &size, const VecT &psf_sigma);

    /* Make ParamT ParamMatT and Stencil objects */
    ParamT make_param() const;
    ParamT make_param(const ParamT &theta) const;
    ParamT make_param(double x, double y, double I, double bg, double sigma) const;
    ParamT make_param(double x, double y, double I, double bg, double sigma, const VecT &Ds) const;
    ParamMatT make_param_mat() const;
    Stencil make_stencil(const ParamT &theta, bool compute_derivatives=true) const;
    Stencil make_stencil(double x, double y, double I, double bg, double sigma, bool compute_derivatives=true) const;
    Stencil make_stencil(double x, double y, double I, double bg, double sigma, const VecT &duty, bool compute_derivatives=true) const;

    /* Model information */
    std::string name() const {return "Blink2DsMAP";}
    StatsT get_stats() const;
    

    /* Model Derivatives */
    double model_value(int i, int j, const Stencil &s) const;
    void pixel_grad(int i, int j, const Stencil &s, ParamT &pgrad) const;
    void pixel_grad2(int i, int j, const Stencil &s, ParamT &pgrad2) const;
    void pixel_hess(int i, int j, const Stencil &s, ParamMatT &hess) const;
    void pixel_hess_update(int i, int j, const Stencil &s, double dm_ratio_m1, 
                           double dmm_ratio, ParamT &grad, ParamMatT &hess) const;
    
    /* Prior sampling and derivatives */
    ParamT sample_prior(RNG &rng);
    void set_hyperparameters(const VecT &hyperparameters);
    void bound_theta(ParamT &theta) const;
    bool theta_in_bounds(const ParamT &theta) const;

    double prior_log_likelihood(const Stencil &s) const;
    double prior_relative_log_likelihood(const Stencil &s) const;
    ParamT prior_grad(const Stencil &s) const;
    ParamT prior_grad2(const Stencil &s) const;
    ParamT prior_cr_lower_bound(const Stencil &s) const;

    /* Initialization */
    Stencil initial_theta_estimate(const ImageT &im, const ParamT &theta_init) const;

    /* Posterior Sampling */
    void sample_candidate_theta(int sample_index, RNG &rng, ParamT &candidate_theta, double scale=1.0) const;
    double compute_candidate_rllh(int sample_index, const ImageT &im,
                                  const ParamT &candidate_theta, ModelImage &model_image) const;


protected:
    VecT stencil_sigmas={1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8};
    MatT gaussian_Xstencils; /**< A stencils for gaussian filters with this size and psf*/
    MatT gaussian_Ystencils; /**< A stencils for gaussian filters with this size and psf*/

    double candidate_eta_sigma; /**< The standard deviation for the normally distributed pertebation to theta_sigma in the random walk MCMC sampling */
};

/* Template Specialization Declarations */
// template<>
// typename Blink2DsMAP::ParamT
// cr_lower_bound(const Blink2DsMAP &model, const typename Blink2DsMAP::Stencil &s);

template <>
typename Blink2DsMAP::ParamVecT
sample_posterior<Blink2DsMAP>(Blink2DsMAP &model, const typename Blink2DsMAP::ImageT &im,int max_samples,
                              typename Blink2DsMAP::Stencil &theta_init,  RNG &rng);

template <>
void sample_posterior_debug<Blink2DsMAP>(Blink2DsMAP &model, const typename Blink2DsMAP::ImageT &im,
                                         typename Blink2DsMAP::Stencil &theta_init,
                                         typename Blink2DsMAP::ParamVecT &sample,
                                         typename Blink2DsMAP::ParamVecT &candidates,
                                         RNG &rng);


/* Inline Methods */

inline
Blink2DsMAP::ParamT
Blink2DsMAP::make_param() const
{
    return ParamT(num_params);
}

inline
Blink2DsMAP::ParamT
Blink2DsMAP::make_param(double x, double y, double I, double bg, double sigma) const
{
    ParamT theta=make_param();
    theta(0)=x;
    theta(1)=y;
    theta(2)=I;
    theta(3)=bg;
    theta(4)=sigma;
    for(int i=0;i<size(0);i++) theta(5+i)=1.0;
    bound_theta(theta);
    return theta;
}

inline
Blink2DsMAP::ParamT
Blink2DsMAP::make_param(double x, double y, double I, double bg, double sigma, const VecT &Ds) const
{
    ParamT theta=make_param();
    theta(0)=x;
    theta(1)=y;
    theta(2)=I;
    theta(3)=bg;
    theta(4)=sigma;
    for(int i=0;i<size(0);i++) theta(5+i)=Ds(i);
    bound_theta(theta);
    return theta;
}

inline
Blink2DsMAP::ParamT
Blink2DsMAP::make_param(const ParamT &theta) const
{
    ParamT ntheta(theta);
    bound_theta(ntheta);
    return ntheta;
}


inline
Blink2DsMAP::ParamMatT
Blink2DsMAP::make_param_mat() const
{
    return ParamMatT(num_params, num_params);
}

inline
Blink2DsMAP::Stencil
Blink2DsMAP::make_stencil(const ParamT &theta, bool compute_derivatives) const
{
    return Stencil(*this,make_param(theta),compute_derivatives);
}

inline
Blink2DsMAP::Stencil
Blink2DsMAP::make_stencil(double x, double y, double I, double bg, double sigma, bool compute_derivatives) const
{
    return Stencil(*this,make_param(x,y,I,bg,sigma),compute_derivatives);
}


inline
Blink2DsMAP::Stencil
Blink2DsMAP::make_stencil(double x, double y, double I, double bg, double sigma, const VecT &Ds, bool compute_derivatives) const
{
    return Stencil(*this,make_param(x,y,I,bg,sigma,Ds),compute_derivatives);
}

inline
Blink2DsMAP::ParamT
Blink2DsMAP::sample_prior(RNG &rng)
{
    ParamT theta=make_param();
    theta(0)=size(0)*pos_dist(rng);
    theta(1)=size(1)*pos_dist(rng);
    theta(2)=I_dist(rng);
    theta(3)=bg_dist(rng);
    theta(4)=sigma_dist(rng);
    for(int i=0; i<size(0); i++) theta(5+i)=D_dist(rng);
    bound_theta(theta);
    return theta;
}


inline
bool Blink2DsMAP::theta_in_bounds(const ParamT &theta) const
{
    bool xOK = (theta(0)>=prior_epsilon) && (theta(0)<=size(0)-prior_epsilon);
    bool yOK = (theta(1)>=prior_epsilon) && (theta(1)<=size(1)-prior_epsilon);
    bool IOK = (theta(2)>=prior_epsilon);
    bool bgOK = (theta(3)>=prior_epsilon);
    bool sigmaOK = (theta(4)>=1.0);
    bool DOK=true;
    for(int i=0; i<size(0); i++) DOK &= (theta(5+i)>=prior_epsilon) && (theta(5+i)<=1.-prior_epsilon);
    return xOK && yOK && IOK && bgOK && sigmaOK && DOK;
}

inline
void Blink2DsMAP::bound_theta(ParamT &theta) const
{
    theta(0)=restrict_value_range(theta(0), prior_epsilon, size(0)-prior_epsilon); // Prior: Support on [0,size]
    theta(1)=restrict_value_range(theta(1), prior_epsilon, size(1)-prior_epsilon); // Prior: Support on [0,size]
    theta(2)=std::max(prior_epsilon,theta(2));// Prior: Support on [0, inf)
    theta(3)=std::max(prior_epsilon,theta(3));// Prior: Support on [0, inf)
    theta(4)=std::max(1.0,theta(4));// Prior: Support on [0, inf)
    for(int i=0;i<size(0);i++) theta(5+i)=restrict_value_range(theta(5+i), prior_epsilon, 1.-prior_epsilon);
}


inline
double Blink2DsMAP::model_value(int i, int j, const Stencil &s) const
{
    return s.bg()+s.D(i)*s.I()*s.X(i)*s.Y(j);
}


inline
void
Blink2DsMAP::pixel_grad(int i, int j, const Stencil &s, ParamT &pgrad) const
{
    double I=s.I();
    double Di=s.D(i);
    double DiI=Di*I;
    pgrad.zeros();
    pgrad(0) = DiI * s.DX(i) * s.Y(j);
    pgrad(1) = DiI * s.DY(j) * s.X(i);
    pgrad(2) = Di * s.X(i) * s.Y(j);
    pgrad(3) = 1.;
    pgrad(4) = DiI * (s.Y(j)*s.DXS(i) + s.X(i)*s.DYS(j));
    pgrad(5+i) = I * s.X(i) * s.Y(j);
}

inline
void
Blink2DsMAP::pixel_grad2(int i, int j, const Stencil &s, ParamT &pgrad2) const
{
    double I=s.I();
    double Di=s.D(i);
    double DiI=Di*I;
    pgrad2.zeros();
    pgrad2(0)= DiI/s.sigmaX() * s.DXS(i) * s.Y(j);
    pgrad2(1)= DiI/s.sigmaY() * s.DYS(j) * s.X(i);
    pgrad2(4)= DiI * (s.X(i)*s.DYS2(j) + 2.*s.DXS(i)*s.DYS(j) + s.Y(j)*s.DXS2(i));
}



inline
void
Blink2DsMAP::pixel_hess(int i, int j, const Stencil &s, ParamMatT &hess) const
{
    double I=s.I();
    double Di=s.D(i);
    double DiI=Di*I;
    hess.zeros();
    //On Diagonal
    hess(0,0)= DiI/s.sigmaX() * s.DXS(i) * s.Y(j); //xx
    hess(1,1)= DiI/s.sigmaY() * s.DYS(j) * s.X(i); //yy
    hess(4,4)= DiI*(s.X(i)*s.DYS2(j) + 2.*s.DXS(i)*s.DYS(j) + s.Y(j)*s.DXS2(i)); //SS
    //Off Diagonal
    hess(0,1)= DiI * s.DX(i) * s.DY(j); //xy
    hess(0,4)= DiI * (s.Y(j)*s.DXSX(i) + s.DX(i)*s.DYS(j)); //xS
    hess(1,4)= DiI * (s.X(i)*s.DYSY(j) + s.DY(j)*s.DXS(i)); //yS
    //Off Diagonal with respect to I
    hess(0,2)= Di * s.DX(i) * s.Y(j); //xI
    hess(1,2)= Di * s.DY(j) * s.X(i); //yI
    hess(2,4)= Di * (s.Y(j)*s.DXS(i) + s.X(i)*s.DYS(j)); //IS
    //Di terms
    hess(0,5+i)= I * s.DX(i) * s.Y(j); //xDi
    hess(1,5+i)= I * s.DY(j) * s.X(i); //yDi
    hess(2,5+i)= s.X(i) * s.Y(j); //IDi
    hess(4,5+i)= I * (s.Y(j)*s.DXS(i) + s.X(i)*s.DYS(j)); //SDi
}



template <>
inline
typename Blink2DsMAP::ParamVecT
sample_posterior<Blink2DsMAP>(Blink2DsMAP &model, const typename Blink2DsMAP::ImageT &im,int max_samples,
                             typename Blink2DsMAP::Stencil &theta_init,  RNG &rng)
{
    return sample_blink_posterior(model, im, max_samples, theta_init, rng);
}

template <>
inline
void sample_posterior_debug<Blink2DsMAP>(Blink2DsMAP &model, const typename Blink2DsMAP::ImageT &im,
                                         typename Blink2DsMAP::Stencil &theta_init,
                                         typename Blink2DsMAP::ParamVecT &sample,
                                         typename Blink2DsMAP::ParamVecT &candidates,
                                         RNG &rng)
{
    sample_blink_posterior_debug(model, im, theta_init, sample, candidates, rng);
}

template<>
inline
typename Blink2DsMAP::Stencil
SimulatedAnnealingMLE<Blink2DsMAP>::anneal(RNG &rng, const ImageT &im, Stencil &theta_init,
                                           ParamVecT &sequence)

{
    return blink_anneal(model, rng, im, theta_init, sequence, max_iterations, T_init, cooling_rate);
}

#endif /* _BLINK2DSMAP_H */
