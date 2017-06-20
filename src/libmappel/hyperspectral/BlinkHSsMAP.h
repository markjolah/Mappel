/** @file BlinkHSsMAP.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class declaration and inline and templated functions for BlinkHSsMAP.
 */

#ifndef _BLINKHSSMAP_H
#define _BLINKHSSMAP_H

#include "PointEmitterHSModel.h"
#include "BlinkModel.h"

/** @brief A base class for likelihood models for point emitters imaged in 
 * 2D with symmmetric PSF, where we estimate the apparent psf sigma accounting
 * for out-of-foucs emitters.
 *
 *
 *
 */
class BlinkHSsMAP : public PointEmitterHSModel, public BlinkModel {
protected:
    /* Hyperparameters */
    double alpha_sigma=3.;  /**< The shape parameter for the bg prior gamma distribution 1=exponential 2-5=skewed large=normal */
    double xi_sigmaL=2.;    /**< The standard deviation of sigmaL normal prior distribution */
    /* Distributions for drawing RND#s */
    ParetoRNG sigma_dist;
    NormalRNG sigmaL_dist;

    /* Precomputed Data  */
    double log_prior_sigma_const;
    double log_prior_sigmaL_const;
public:
    /* Model matrix and vector types */
    typedef arma::vec ParamT; /**< A type for the set of parameters estimated by the model */
    typedef arma::mat ParamMatT; /**< A matrix type for the Hessian used by the CRLB estimation */
    
    std::vector<std::string> param_names;
    static const std::vector<std::string> hyperparameter_names;

    class Stencil {
    public:
        bool derivatives_computed=false;
        typedef BlinkHSsMAP::ParamT ParamT;
        BlinkHSsMAP const *model;
        ParamT theta;
        VecT dx, dy, dL;
        VecT Gx, Gy, GL;
        VecT X, Y, L;
        VecT DX, DY, DL;
        VecT DXS, DYS, DLS;
        VecT DXS2, DYS2, DLS2;
        VecT DXSX, DYSY, DLSL;
        Stencil() {}
        Stencil(const BlinkHSsMAP &model, const ParamT &theta, bool _compute_derivatives=true);
        void compute_derivatives();
        void set_duty(int i, double D){
            assert(0<=i && i<model->size(0));
            assert(0<=D && D<=1);
            theta(7+i)=restrict_value_range(D, model->prior_epsilon, 1.-model->prior_epsilon);
        }
        inline double x() const {return theta(0);}
        inline double y() const {return theta(1);}
        inline double lambda() const {return theta(2);}
        inline double I() const {return theta(3);}
        inline double bg() const {return theta(4);}
        inline double sigma() const {return theta(5);}
        inline double sigmaX() const {return model->psf_sigma(0)*sigma();}
        inline double sigmaY() const {return model->psf_sigma(1)*sigma();}
        inline double sigmaL() const {return theta(6);}
        inline double D(int i) const {return theta(7+i);}
        friend std::ostream& operator<<(std::ostream &out, const BlinkHSsMAP::Stencil &s);
    private:
        inline int size(int i) const {return model->size(i);}
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
        const BlinkHSsMAP *model;
        BlinkHSsMAP::Stencil stencil;
        const BlinkHSsMAP::ImageT *data_im;
        BlinkHSsMAP::ImageT model_im;
        BlinkHSsMAP::ImageT log_model_im;
    public:
        ModelImage(const BlinkHSsMAP &model_,
                   const BlinkHSsMAP::ImageT &data_im_);
        void set_stencil(const BlinkHSsMAP::ParamT &theta);

        void set_duty(int i, double D);

        double relative_log_likelihood() const;
    };


    /* Constructor */
    BlinkHSsMAP(const IVecT &size,const VecT &sigma);

    /* Make ParamT ParamMatT and Stencil objects */
    ParamT make_param() const;
    ParamT make_param(const ParamT &theta) const;
    ParamT make_param(double x, double y, double L, double I, double bg, double sigma, double sigmaL) const;
    ParamT make_param(double x, double y, double L, double I, double bg, double sigma, double sigmaL,
                      const VecT &Ds) const;
    ParamMatT make_param_mat() const;
    Stencil make_stencil(const ParamT &theta, bool compute_derivatives=true) const;
    Stencil make_stencil(double x, double y, double L, double I, double bg, double sigma, double sigmaL,
                         bool compute_derivatives=true) const;
    Stencil make_stencil(double x, double y, double L, double I, double bg, double sigma, double sigmaL,
                         const VecT &duty, bool compute_derivatives=true) const;

    /* Model information */
    std::string name() const {return "BlinkHSsMAP";}
    StatsT get_stats() const;
    

    /* Model Derivatives */
    double model_value(int i, int j, int k, const Stencil &s) const;
    void pixel_grad(int i, int j, int k, const Stencil &s, ParamT &pgrad) const;
    void pixel_grad2(int i, int j, int k, const Stencil &s, ParamT &pgrad2) const;
    void pixel_hess(int i, int j, int k, const Stencil &s, ParamMatT &hess) const;
    void pixel_hess_update(int i, int j, int k, const Stencil &s, double dm_ratio_m1,
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
    Stencil initial_theta_estimate(const ImageT &im, const ParamVecT &theta_init) const;

    /* Posterior Sampling */
    void sample_candidate_theta(int sample_index, RNG &rng, ParamT &candidate_theta,double scale=1.0) const;
    double compute_candidate_rllh(int sample_index, const ImageT &im,
                                  const ParamT& candidate_theta, ModelImage &model_image) const;

protected:
    double candidate_eta_sigma; /**< The standard deviation for the normally distributed pertebation to theta_sigma in the random walk MCMC sampling */
    double candidate_eta_sigmaL; /**< The standard deviation for the normally distributed pertebation to theta_sigmaL in the random walk MCMC sampling */

    VecT stencil_sigmas;
    VecT stencil_sigmaLs;

    VecFieldT gaussian_stencils; /**< A structure for stencils for gaussian filters*/
    VecFieldT gaussian_Lstencils; /**< A structure for stencils for gaussian filters*/
};

/* Template Specialization Declarations */
// template<>
// typename BlinkHSsMAP::ParamT
// cr_lower_bound(const BlinkHSsMAP &model, const typename BlinkHSsMAP::Stencil &s);

template <>
typename BlinkHSsMAP::ParamVecT
sample_posterior<BlinkHSsMAP>(BlinkHSsMAP &model, const typename BlinkHSsMAP::ImageT &im,int max_samples,
                              typename BlinkHSsMAP::Stencil &theta_init,  RNG &rng);

template <>
void sample_posterior_debug<BlinkHSsMAP>(BlinkHSsMAP &model, const typename BlinkHSsMAP::ImageT &im,
                                         typename BlinkHSsMAP::Stencil &theta_init,
                                         typename BlinkHSsMAP::ParamVecT &sample,
                                         typename BlinkHSsMAP::ParamVecT &candidates,
                                         RNG &rng);


/* Inline Methods */

inline
BlinkHSsMAP::ParamT
BlinkHSsMAP::make_param() const
{
    return ParamT(num_params);
}

inline
BlinkHSsMAP::ParamT
BlinkHSsMAP::make_param(const ParamT &theta) const
{
    ParamT ntheta(theta);
    bound_theta(ntheta);
    return ntheta;
}

inline
BlinkHSsMAP::ParamT
BlinkHSsMAP::make_param(double x, double y, double L, double I, double bg,
                        double sigma, double sigmaL) const
{
    ParamT theta=make_param();
    theta(0)=x;
    theta(1)=y;
    theta(2)=L;
    theta(3)=I;
    theta(4)=bg;
    theta(5)=sigma;
    theta(6)=sigmaL;
    for(int i=0;i<size(0);i++) theta(7+i)=1.0;
    bound_theta(theta);
    return theta;
}

inline
BlinkHSsMAP::ParamT
BlinkHSsMAP::make_param(double x, double y, double L, double I, double bg, double sigma, double sigmaL, const VecT &Ds) const
{
    ParamT theta=make_param();
    theta(0)=x;
    theta(1)=y;
    theta(2)=L;
    theta(3)=I;
    theta(4)=bg;
    theta(5)=sigma;
    theta(6)=sigmaL;
    for(int i=0;i<size(0);i++) theta(7+i)=Ds(i);
    bound_theta(theta);
    return theta;
}




inline
BlinkHSsMAP::ParamMatT
BlinkHSsMAP::make_param_mat() const
{
    return ParamMatT(num_params, num_params);
}

inline
BlinkHSsMAP::Stencil
BlinkHSsMAP::make_stencil(const ParamT &theta, bool compute_derivatives) const
{
    return Stencil(*this,make_param(theta),compute_derivatives);
}

inline
BlinkHSsMAP::Stencil
BlinkHSsMAP::make_stencil(double x, double y, double L, double I, double bg,
                          double sigma, double sigmaL, bool compute_derivatives) const
{
    return Stencil(*this,make_param(x,y,L,I,bg,sigma,sigmaL),compute_derivatives);
}


inline
BlinkHSsMAP::Stencil
BlinkHSsMAP::make_stencil(double x, double y, double L, double I, double bg, double sigma,
                          double sigmaL, const VecT &Ds, bool compute_derivatives) const
{
    return Stencil(*this,make_param(x,y,L,I,bg,sigma,sigmaL,Ds),compute_derivatives);
}

inline
BlinkHSsMAP::ParamT
BlinkHSsMAP::sample_prior(RNG &rng)
{
    ParamT theta=make_param();
    theta(0)=size(0)*pos_dist(rng);
    theta(1)=size(1)*pos_dist(rng);
    theta(2)=size(2)*L_dist(rng);
    theta(3)=I_dist(rng);
    theta(4)=bg_dist(rng);
    theta(5)=sigma_dist(rng);
    theta(6)=sigmaL_dist(rng);
    for(int i=0; i<size(0); i++) theta(7+i)=D_dist(rng);
    bound_theta(theta);
    return theta;
}

inline
bool BlinkHSsMAP::theta_in_bounds(const ParamT &theta) const
{
    bool xOK = (theta(0)>=prior_epsilon) && (theta(0)<=size(0)-prior_epsilon);
    bool yOK = (theta(1)>=prior_epsilon) && (theta(1)<=size(1)-prior_epsilon);
    bool LOK = (theta(2)>=prior_epsilon) && (theta(2)<=prior_epsilon);
    bool IOK = (theta(3)>=prior_epsilon);
    bool bgOK = (theta(4)>=prior_epsilon);
    bool sigmaOK = (theta(5)>=1.0);
    bool sigmaLOK = (theta(6)>=prior_epsilon);
    bool DOK=true;
    for(int i=0; i<size(0); i++) DOK &= (theta(7+i)>=prior_epsilon) && (theta(7+i)<=1.-prior_epsilon);
    return xOK && yOK && LOK && IOK && bgOK && sigmaOK && sigmaLOK && DOK;
}

inline
void BlinkHSsMAP::bound_theta(ParamT &theta) const
{
    theta(0)=restrict_value_range(theta(0), prior_epsilon, size(0)-prior_epsilon);
    theta(1)=restrict_value_range(theta(1), prior_epsilon, size(1)-prior_epsilon);
    theta(2)=restrict_value_range(theta(2), prior_epsilon, size(2)-prior_epsilon );
    theta(3)=std::max(prior_epsilon,theta(3));
    theta(4)=std::max(prior_epsilon,theta(4));
    theta(5)=std::max(1.0,theta(5));
    theta(6)=std::max(prior_epsilon,theta(6));
    for(int i=0;i<size(0);i++) theta(7+i)=restrict_value_range(theta(7+i), prior_epsilon, 1.-prior_epsilon);
}


inline
double BlinkHSsMAP::model_value(int i, int j, int k, const Stencil &s) const
{
    return s.bg()+s.D(i)*s.I()*s.X(i)*s.Y(j)*s.L(k);
}

inline
double BlinkHSsMAP::prior_log_likelihood(const Stencil &s) const
{
    return prior_relative_log_likelihood(s)+log_prior_const;
}


inline
void
BlinkHSsMAP::pixel_grad(int i, int j,int k, const Stencil &s, ParamT &pgrad) const
{
    double I=s.I();
    double Di=s.D(i);
    double DiI=Di*I;

    pgrad.zeros();
    pgrad(0) = DiI * s.DX(i) * s.Y(j) * s.L(k);
    pgrad(1) = DiI * s.X(i) * s.DY(j) * s.L(k);
    pgrad(2) = DiI * s.X(i) * s.Y(j) * s.DL(k);
    pgrad(3) = Di * s.X(i) * s.Y(j) * s.L(k);
    pgrad(4) = 1.;
    pgrad(5) = DiI * s.L(k) * (s.X(i)*s.DYS(j) + s.Y(j)*s.DXS(i));
    pgrad(6) = DiI * s.X(i) * s.Y(j) * s.DLS(k);
    pgrad(7+i) = I * s.X(i) * s.Y(j) * s.L(k);
}

inline
void
BlinkHSsMAP::pixel_grad2(int i, int j,int k, const Stencil &s, ParamT &pgrad2) const
{
    double I=s.I();
    double Di=s.D(i);
    double DiI=Di*I;
    pgrad2.zeros();
    pgrad2(0)= DiI/s.sigmaX() * s.DXS(i) * s.Y(j) * s.L(k);
    pgrad2(1)= DiI/s.sigmaY() * s.X(i) * s.DYS(j) * s.L(k);
    pgrad2(2)= DiI/s.sigmaL() * s.X(i) * s.Y(j) * s.DLS(k);
    pgrad2(5)= DiI * s.L(k) * (s.X(i)*s.DYS2(j) + 2.*s.DXS(i)*s.DYS(j) + s.Y(j)*s.DXS2(i));
    pgrad2(6)= DiI * s.X(i) * s.Y(j) * s.DLS2(k);
}

inline
void
BlinkHSsMAP::pixel_hess(int i, int j, int k, const Stencil &s, ParamMatT &hess) const
{
    hess.zeros();
    double I=s.I();
    double Di=s.D(i);
    double DiI=Di*I;
    hess(0,0)= DiI/s.sigmaX() * s.DXS(i) * s.Y(j) * s.L(k);
    hess(1,1)= DiI/s.sigmaY() * s.X(i) * s.DYS(j) * s.L(k);
    hess(2,2)= DiI/s.sigmaL() * s.X(i) * s.Y(j) * s.DLS(k);
    hess(5,5)= DiI * s.L(k) * (s.X(i)*s.DYS2(j) + 2.*s.DXS(i)*s.DYS(j) + s.Y(j)*s.DXS2(i));
    hess(6,6)= DiI * s.X(i) * s.Y(j) * s.DLS2(k);

    hess(0,1)= DiI * s.DX(i) * s.DY(j) * s.L(k);
    hess(0,2)= DiI * s.DX(i) * s.Y(j) * s.DL(k);
    hess(1,2)= DiI * s.X(i) * s.DY(j) * s.DL(k);

    hess(0,5)= DiI * s.L(k)  * (s.Y(j)*s.DXSX(i) + s.DX(i)*s.DYS(j));
    hess(1,5)= DiI * s.L(k)  * (s.DY(j)*s.DXS(i) + s.X(i)*s.DYSY(j));
    hess(2,5)= DiI * s.DL(k) * (s.Y(j)*s.DXS(i)  + s.X(i)*s.DYS(j));

    hess(0,6)= DiI * s.DX(i) * s.Y(j) * s.DLS(k);
    hess(1,6)= DiI * s.X(i) * s.DY(j) * s.DLS(k);
    hess(2,6)= DiI * s.X(i) * s.Y(j) * s.DLSL(k);

    hess(0,3)= Di * s.DX(i) * s.Y(j) * s.L(k);
    hess(1,3)= Di * s.X(i) * s.DY(j) * s.L(k);
    hess(2,3)= Di * s.X(i) * s.Y(j) * s.DL(k);
    hess(3,5)= Di * s.L(k) * (s.X(i)*s.DYS(j) + s.Y(j)*s.DXS(i));
    hess(3,6)= Di * s.X(i) * s.Y(j) * s.DLS(k);
    
    hess(5,6)= DiI * s.DLS(k) * (s.X(i)  * s.DYS(j)  + s.Y(j)  * s.DXS(i));
    
    hess(0,7+i)=I * s.DX(i) * s.Y(j) * s.L(k); //xDi
    hess(1,7+i)=I * s.X(i) * s.DY(j) * s.L(k); //xDi
    hess(2,7+i)=I * s.X(i) * s.Y(j) * s.DL(k); //xDi
    hess(3,7+i)=s.X(i) * s.Y(j) * s.L(k); //xDi
    hess(5,7+i)=I * s.L(k) * (s.X(i)*s.DYS(j) + s.Y(j)*s.DXS(i)); //xDi
    hess(6,7+i)=I * s.X(i) * s.Y(j) * s.DLS(k); //xDi
}


template <>
inline
typename BlinkHSsMAP::ParamVecT
sample_posterior<BlinkHSsMAP>(BlinkHSsMAP &model, const typename BlinkHSsMAP::ImageT &im,int max_samples,
                              typename BlinkHSsMAP::Stencil &theta_init,  RNG &rng)
{
    return sample_blink_posterior(model, im, max_samples, theta_init, rng);
}

template <>
inline
void sample_posterior_debug<BlinkHSsMAP>(BlinkHSsMAP &model, const typename BlinkHSsMAP::ImageT &im,
                                         typename BlinkHSsMAP::Stencil &theta_init,
                                         typename BlinkHSsMAP::ParamVecT &sample,
                                         typename BlinkHSsMAP::ParamVecT &candidates,
                                         RNG &rng)
{
    sample_blink_posterior_debug(model, im, theta_init, sample, candidates, rng);
}

template<>
inline
typename BlinkHSsMAP::Stencil
SimulatedAnnealingMLE<BlinkHSsMAP>::anneal(RNG &rng, const ImageT &im, Stencil &theta_init,
                                           ParamVecT &sequence)

{
    return blink_anneal(model, rng, im, theta_init, sequence, max_iterations, T_init, cooling_rate);
}

#endif /* _BLINKHSSMAP_H */
