/** @file GaussHSMAP.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class declaration and inline and templated functions for GaussHSMAP.
 */

#ifndef _GAUSSHSMAP_H
#define _GAUSSHSMAP_H

#include "PointEmitterHSModel.h"

/** @brief A base class for likelihood models for point emitters imaged in 2D with symmmetric PSF.
 *
 *
 *
 */
class GaussHSMAP : public PointEmitterHSModel {
public:
    /* Model matrix and vector types */
    typedef arma::vec::fixed<5> ParamT; /**< A type for the set of parameters estimated by the model */
    typedef arma::mat::fixed<5,5> ParamMatT; /**< A matrix type for the Hessian used by the CRLB estimation */
    static const std::vector<std::string> param_names; /**<The parameter names for this class */
    static const std::vector<std::string> hyperparameter_names; /**<The hyperparameter names for this class */

    class Stencil {
    public:
        bool derivatives_computed=false;
        typedef GaussHSMAP::ParamT ParamT;
        GaussHSMAP const *model;
        ParamT theta;
        VecT dx, dy, dL;
        VecT Gx, Gy, GL;
        VecT X, Y, L;
        VecT DX, DY, DL;
        VecT DXS, DYS, DLS;
        Stencil() {}
        Stencil(const GaussHSMAP &model, const ParamT &theta, bool _compute_derivatives=true);
        void compute_derivatives();
        inline double x() const {return theta(0);}
        inline double y() const {return theta(1);}
        inline double lambda() const {return theta(2);}
        inline double I() const {return theta(3);}
        inline double bg() const {return theta(4);}
        friend std::ostream& operator<<(std::ostream &out, const GaussHSMAP::Stencil &s);
    };

    GaussHSMAP(const IVecT &size,const VecT &sigma);

    ParamT make_param() const;
    ParamT make_param(const ParamT &theta) const;
    ParamT make_param(double x, double y, double L, double I, double bg) const;
    ParamMatT make_param_mat() const;
    Stencil make_stencil(const ParamT &theta, bool compute_derivatives=true) const;
    Stencil make_stencil(double x, double y, double L, double I, double bg, bool compute_derivatives=true) const;

    /* Model values setting and information */
    std::string name() const {return "GaussHSMAP";}
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
    VecFieldT gaussian_stencils; /**< A structure for stencils for gaussian filters*/
};


/* Inline Method Definitions */

inline
GaussHSMAP::ParamT
GaussHSMAP::make_param() const
{
    return ParamT();
}

inline
GaussHSMAP::ParamT
GaussHSMAP::make_param(const ParamT &theta) const
{
    ParamT ntheta(theta);
    bound_theta(ntheta);
    return ntheta;
}

inline
GaussHSMAP::ParamT
GaussHSMAP::make_param(double x, double y, double L, double I, double bg) const
{
    ParamT theta;
    theta<<x<<y<<L<<I<<bg;
    bound_theta(theta);
    return theta;
}

inline
GaussHSMAP::ParamMatT
GaussHSMAP::make_param_mat() const
{
    return ParamMatT();
}

inline
GaussHSMAP::Stencil
GaussHSMAP::make_stencil(const ParamT &theta, bool compute_derivatives) const
{
    return Stencil(*this,make_param(theta),compute_derivatives);
}

inline
GaussHSMAP::Stencil
GaussHSMAP::make_stencil(double x, double y, double L, double I, double bg, bool compute_derivatives) const
{
    return Stencil(*this,make_param(x,y,L,I,bg),compute_derivatives);
}

inline
GaussHSMAP::ParamT
GaussHSMAP::sample_prior(RNG &rng)
{
    ParamT theta=make_param();
    theta(0)=size[0]*pos_dist(rng);
    theta(1)=size[1]*pos_dist(rng);
    theta(2)=size[2]*L_dist(rng);
    theta(3)=I_dist(rng);
    theta(4)=bg_dist(rng);
    bound_theta(theta);
    return theta;
}

inline
bool GaussHSMAP::theta_in_bounds(const ParamT &theta) const
{
    bool xOK = (theta(0)>=prior_epsilon) && (theta(0)<=size(0)-prior_epsilon);
    bool yOK = (theta(1)>=prior_epsilon) && (theta(1)<=size(1)-prior_epsilon);
    bool LOK = (theta(2)>=prior_epsilon) && (theta(2)<=size(2)-prior_epsilon);
    bool IOK = (theta(3)>=prior_epsilon);
    bool bgOK = (theta(4)>=prior_epsilon);
    return xOK && yOK && LOK && IOK && bgOK;
}

inline
void GaussHSMAP::bound_theta(ParamT &theta) const
{
    theta(0)=restrict_value_range(theta(0), prior_epsilon, size(0)-prior_epsilon);
    theta(1)=restrict_value_range(theta(1), prior_epsilon, size(1)-prior_epsilon);
    theta(2)=restrict_value_range(theta(2), prior_epsilon, size(2)-prior_epsilon);
    theta(3)=std::max(prior_epsilon,theta(3));
    theta(4)=std::max(prior_epsilon,theta(4));
}


inline
double GaussHSMAP::model_value(int i, int j, int k, const Stencil &s) const
{
    return s.bg()+s.I()*s.X(i)*s.Y(j)*s.L(k);
}

inline
void
GaussHSMAP::pixel_grad(int i, int j, int k, const Stencil &s, ParamT &pgrad) const
{
    double I = s.I();
    pgrad(0) = I * s.DX(i) * s.Y(j) * s.L(k);
    pgrad(1) = I * s.X(i) * s.DY(j) * s.L(k);
    pgrad(2) = I * s.X(i) * s.Y(j) * s.DL(k);
    pgrad(3) = s.X(i) * s.Y(j) * s.L(k);
    pgrad(4) = 1.;
}

inline
void
GaussHSMAP::pixel_grad2(int i, int j, int k, const Stencil &s, ParamT &pgrad2) const
{
    double I = s.I();
    pgrad2(0) = I/psf_sigma(0) * s.DXS(i) * s.Y(j) * s.L(k);
    pgrad2(1) = I/psf_sigma(1) * s.X(i) * s.DYS(j) * s.L(k);
    pgrad2(2) = I/mean_sigmaL  * s.X(i) * s.Y(j) * s.DLS(k);
    pgrad2(3) = 0.;
    pgrad2(4) = 0.;
}

inline
void
GaussHSMAP::pixel_hess(int i, int j, int k, const Stencil &s, ParamMatT &hess) const
{
    hess.zeros();
    double I=s.I();
    hess(0,0)= I/psf_sigma(0) * s.DXS(i) * s.Y(j) * s.L(k);
    hess(1,1)= I/psf_sigma(1) * s.X(i) * s.DYS(j) * s.L(k);
    hess(2,2)= I/mean_sigmaL  * s.X(i) * s.Y(j) * s.DLS(k);
    hess(0,1)= I * s.DX(i) * s.DY(j) * s.L(k);
    hess(0,2)= I * s.DX(i) * s.Y(j) * s.DL(k);
    hess(1,2)= I * s.X(i) * s.DY(j) * s.DL(k);
    hess(0,3)= s.DX(i) * s.Y(j) * s.L(k);
    hess(1,3)= s.X(i) * s.DY(j) * s.L(k);
    hess(2,3)= s.X(i) * s.Y(j) * s.DL(k);
}


#endif /* _GAUSSHSMAP_H */
