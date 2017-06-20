/** @file GaussHSsMAP.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 04-01-2014
 * @brief The class declaration and inline and templated functions for GaussHSsMAP.
 */

#ifndef _GAUSSHSSMAP_H
#define _GAUSSHSSMAP_H

#include "PointEmitterHSModel.h"

/** @brief A base class for likelihood models for point emitters imaged in 2D with symmmetric PSF.
 *
 * Eventually we want a multi-level baysean model for sigmaL.
 *
 */
class GaussHSsMAP : public PointEmitterHSModel {
private:
    double alpha_sigma=3.;  /**< The shape parameter for the bg prior gamma distribution 1=exponential 2-5=skewed large=normal */
    double xi_sigmaL=2.;  /**< The standard deviation of sigmaL normal prior distribution */
    /* Distributions for drawing RND#s */
    ParetoRNG sigma_dist;
    NormalRNG sigmaL_dist;

    double log_prior_sigma_const;
    double log_prior_sigmaL_const;
public:
    /* Model matrix and vector types */
    typedef arma::vec::fixed<7> ParamT; /**< A type for the set of parameters estimated by the model */
    typedef arma::mat::fixed<7,7> ParamMatT; /**< A matrix type for the Hessian used by the CRLB estimation */
    static const std::vector<std::string> param_names; /**<The parameter names for this class */
    static const std::vector<std::string> hyperparameter_names; /**<The hyperparameter names for this class */

    class Stencil {
    public:
        bool derivatives_computed=false;
        typedef GaussHSsMAP::ParamT ParamT;
        GaussHSsMAP const *model;
        ParamT theta;
        VecT dx, dy, dL;
        VecT Gx, Gy, GL;
        VecT X, Y, L;
        VecT DX, DY, DL;
        VecT DXS, DYS, DLS;
        VecT DXS2, DYS2, DLS2;
        VecT DXSX, DYSY, DLSL;
        Stencil() {}
        Stencil(const GaussHSsMAP &model, const ParamT &theta, bool _compute_derivatives=true);
        void compute_derivatives();
        inline double x() const {return theta(0);}
        inline double y() const {return theta(1);}
        inline double lambda() const {return theta(2);}
        inline double I() const {return theta(3);}
        inline double bg() const {return theta(4);}
        inline double sigma() const {return theta(5);}
        inline double sigmaX() const {return model->psf_sigma(0)*sigma();}
        inline double sigmaY() const {return model->psf_sigma(1)*sigma();}
        inline double sigmaL() const {return theta(6);}
        friend std::ostream& operator<<(std::ostream &out, const GaussHSsMAP::Stencil &s);
    private:
        inline int size(int i) const {return model->size(i);}
        
    };

    GaussHSsMAP(const IVecT &size,const VecT &sigma);

    ParamT make_param() const;
    ParamT make_param(const ParamT &theta) const;
    ParamT make_param(double x, double y, double L, double I, double bg, double sigma, double sigmaL) const;
    ParamMatT make_param_mat() const;
    Stencil make_stencil(const ParamT &theta, bool compute_derivatives=true) const;
    Stencil make_stencil(double x, double y, double L, double I, double bg, double sigma, double sigmaL, bool compute_derivatives=true) const;

    /* Model values setting and information */
    std::string name() const {return "GaussHSsMAP";}
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
    void sample_candidate_theta(int sample_index, RNG &rng, ParamT &candidate_theta, double scale=1.0) const;
protected:
    double candidate_eta_sigma; /**< The standard deviation for the normally distributed pertebation to theta_sigma in the random walk MCMC sampling */
    double candidate_eta_sigmaL; /**< The standard deviation for the normally distributed pertebation to theta_sigmaL in the random walk MCMC sampling */

    typedef arma::field<VecT> VecFieldT;
    VecT stencil_sigmas;
    VecT stencil_sigmaLs;
    
    VecFieldT gaussian_stencils; /**< A structure for stencils for gaussian filters*/
    VecFieldT gaussian_Lstencils; /**< A structure for stencils for gaussian filters*/
};


/* Inline Method Definitions */

inline
GaussHSsMAP::ParamT
GaussHSsMAP::make_param() const
{
    return ParamT();
}


inline
GaussHSsMAP::ParamT
GaussHSsMAP::make_param(const ParamT &theta) const
{
    ParamT ntheta(theta);
    bound_theta(ntheta);
    return ntheta;
}

inline
GaussHSsMAP::ParamT
GaussHSsMAP::make_param(double x, double y, double L, double I, double bg,
                        double sigma, double sigmaL) const
{
    ParamT theta;
    theta<<x<<y<<L<<I<<bg<<sigma<<sigmaL;
    bound_theta(theta);
    return theta;
}


inline
GaussHSsMAP::ParamMatT
GaussHSsMAP::make_param_mat() const
{
    return ParamMatT();
}

inline
GaussHSsMAP::Stencil
GaussHSsMAP::make_stencil(const ParamT &theta, bool compute_derivatives) const
{
    return Stencil(*this,make_param(theta),compute_derivatives);
}

inline
GaussHSsMAP::Stencil
GaussHSsMAP::make_stencil(double x, double y, double L, double I, double bg,
                          double sigma, double sigmaL, bool compute_derivatives) const
{
    return Stencil(*this,make_param(x,y,L,I,bg,sigma,sigmaL),compute_derivatives);
}


inline
GaussHSsMAP::ParamT
GaussHSsMAP::sample_prior(RNG &rng)
{
    ParamT theta=make_param();
    theta(0)=size(0)*pos_dist(rng);
    theta(1)=size(1)*pos_dist(rng);
    theta(2)=size(2)*L_dist(rng);
    theta(3)=I_dist(rng);
    theta(4)=bg_dist(rng);
    theta(5)=sigma_dist(rng);
    theta(6)=sigmaL_dist(rng);
    bound_theta(theta);
    return theta;
}

inline
bool GaussHSsMAP::theta_in_bounds(const ParamT &theta) const
{
    bool xOK = (theta(0)>=prior_epsilon) && (theta(0)<=size(0)-prior_epsilon);
    bool yOK = (theta(1)>=prior_epsilon) && (theta(1)<=size(1)-prior_epsilon);
    bool LOK = (theta(2)>=prior_epsilon) && (theta(2)<=size(2)-prior_epsilon);
    bool IOK = (theta(3)>=prior_epsilon);
    bool bgOK = (theta(4)>=prior_epsilon);
    bool sigmaOK = (theta(5)>=sigma_min);
    bool sigmaLOK = (theta(6)>=sigmaL_min);
    return xOK && yOK && LOK && IOK && bgOK && sigmaOK && sigmaLOK;
}

inline
void GaussHSsMAP::bound_theta(ParamT &theta) const
{
    theta(0)=restrict_value_range(theta(0), prior_epsilon, size(0)-prior_epsilon);
    theta(1)=restrict_value_range(theta(1), prior_epsilon, size(1)-prior_epsilon);
    theta(2)=restrict_value_range(theta(2), prior_epsilon, size(2)-prior_epsilon );
    theta(3)=std::max(prior_epsilon,theta(3));
    theta(4)=std::max(prior_epsilon,theta(4));
    theta(5)=std::max(sigma_min,theta(5));
    theta(6)=std::max(sigmaL_min,theta(6));
}



inline
double GaussHSsMAP::model_value(int i, int j, int k, const Stencil &s) const
{
    return s.bg()+s.I()*s.X(i)*s.Y(j)*s.L(k);
}


inline
double GaussHSsMAP::prior_log_likelihood(const Stencil &s) const
{
    return prior_relative_log_likelihood(s)+log_prior_const;
}

inline
void
GaussHSsMAP::pixel_grad(int i, int j, int k, const Stencil &s, ParamT &pgrad) const
{
    double I=s.I();
    pgrad(0) = I * s.DX(i) * s.Y(j) * s.L(k);
    pgrad(1) = I * s.X(i) * s.DY(j) * s.L(k);
    pgrad(2) = I * s.X(i) * s.Y(j) * s.DL(k);
    pgrad(3) = s.X(i) * s.Y(j) * s.L(k);
    pgrad(4) = 1.;
    pgrad(5) = I * s.L(k) * (s.X(i)*s.DYS(j) + s.Y(j)*s.DXS(i));
    pgrad(6) = I * s.X(i) * s.Y(j) * s.DLS(k);
}

inline
void
GaussHSsMAP::pixel_grad2(int i, int j, int k, const Stencil &s, ParamT &pgrad2) const
{
    double I=s.I();
    pgrad2(0)= I/s.sigmaX() * s.DXS(i) * s.Y(j) * s.L(k);
    pgrad2(1)= I/s.sigmaY() * s.X(i) * s.DYS(j) * s.L(k);
    pgrad2(2)= I/s.sigmaL() * s.X(i) * s.Y(j) * s.DLS(k);
    pgrad2(3)=0.;
    pgrad2(4)=0.;
    pgrad2(5)= I * s.L(k) * (s.X(i)*s.DYS2(j) + 2.*s.DXS(i)*s.DYS(j) + s.Y(j)*s.DXS2(i));
    pgrad2(6)= I * s.X(i) * s.Y(j) * s.DLS2(k);
}

inline
void
GaussHSsMAP::pixel_hess(int i, int j, int k, const Stencil &s, ParamMatT &hess) const
{
    hess.zeros();
    double I=s.I();
    hess(0,0)= I/s.sigmaX() * s.DXS(i) * s.Y(j) * s.L(k);
    hess(1,1)= I/s.sigmaY() * s.X(i) * s.DYS(j) * s.L(k);
    hess(2,2)= I/s.sigmaL() * s.X(i) * s.Y(j) * s.DLS(k);
    hess(5,5)= I * s.L(k) * (s.X(i)*s.DYS2(j) + 2.*s.DXS(i)*s.DYS(j) + s.Y(j)*s.DXS2(i));
    hess(6,6)= I * s.X(i) * s.Y(j) * s.DLS2(k);

    hess(0,1)= I * s.DX(i) * s.DY(j) * s.L(k);
    hess(0,2)= I * s.DX(i) * s.Y(j) * s.DL(k);
    hess(1,2)= I * s.X(i) * s.DY(j) * s.DL(k);

    hess(0,5)= I * s.L(k)  * (s.Y(j)*s.DXSX(i) + s.DX(i)*s.DYS(j));
    hess(1,5)= I * s.L(k)  * (s.DY(j)*s.DXS(i) + s.X(i)*s.DYSY(j));
    hess(2,5)= I * s.DL(k) * (s.Y(j)*s.DXS(i)  + s.X(i)*s.DYS(j));

    hess(0,6)= I * s.DX(i) * s.Y(j) * s.DLS(k);
    hess(1,6)= I * s.X(i) * s.DY(j) * s.DLS(k);
    hess(2,6)= I * s.X(i) * s.Y(j) * s.DLSL(k);

    hess(0,3)= s.DX(i) * s.Y(j) * s.L(k);
    hess(1,3)= s.X(i) * s.DY(j) * s.L(k);
    hess(2,3)= s.X(i) * s.Y(j) * s.DL(k);
    hess(3,5)= s.L(k) * (s.X(i)*s.DYS(j) + s.Y(j)*s.DXS(i));
    hess(3,6)= s.X(i) * s.Y(j) * s.DLS(k);

    hess(5,6)= I * s.DLS(k) * (s.X(i)  * s.DYS(j)  + s.Y(j)  * s.DXS(i));
}

#endif /* _GAUSSHSSMAP_H */
