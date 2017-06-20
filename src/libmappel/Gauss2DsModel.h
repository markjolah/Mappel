/** @file Gauss2DsModel.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class declaration and inline and templated functions for Gauss2DsModel.
 */

#ifndef _GAUSS2DSMODEL_H
#define _GAUSS2DSMODEL_H

#include "PointEmitter2DModel.h"
#include "estimator.h"
#include "cGaussMLE/cGaussMLE.h"
#include "cGaussMLE/GaussLib.h"

/** @brief A base class for likelihood models for point emitters imaged in 
 * 2D with symmmetric PSF, where we estimate the apparent psf sigma accounting
 * for out-of-foucs emitters.
 *
 *
 *
 */
class Gauss2DsModel : public PointEmitter2DModel {
public:
    /* Model matrix and vector types */
    typedef arma::vec::fixed<5> ParamT; /**< A type for the set of parameters estimated by the model */
    typedef arma::mat::fixed<5,5> ParamMatT; /**< A matrix type for the Hessian used by the CRLB estimation */
    static const std::vector<std::string> param_names;
    
    class Stencil {
    public:
        bool derivatives_computed=false;
        typedef Gauss2DsModel::ParamT ParamT;
        Gauss2DsModel const *model;
        ParamT theta;
        VecT dx, dy;
        VecT Gx, Gy;
        VecT X, Y;
        VecT DX, DY;
        VecT DXS, DYS;
        VecT DXS2, DYS2;
        VecT DXSX, DYSY;
        Stencil() {}
        Stencil(const Gauss2DsModel &model,const ParamT &theta, bool _compute_derivatives=true);
        void compute_derivatives();
        inline double x() const {return theta(0);}
        inline double y() const {return theta(1);}
        inline double I() const {return theta(2);}
        inline double bg() const {return theta(3);}
        inline double sigma() const {return theta(4);}
        inline double sigmaX() const {return model->psf_sigma(0)*sigma();}
        inline double sigmaY() const {return model->psf_sigma(1)*sigma();}
        friend std::ostream& operator<<(std::ostream &out, const Gauss2DsModel::Stencil &s);
    };

    Gauss2DsModel(const IVecT &size, const VecT &psf_sigma);

    /* Make arrays for working with model data */
    ParamT make_param() const;
    ParamT make_param(double x, double y, double I, double bg, double sigma) const;
    ParamT make_param(const ParamT &theta) const;
    ParamMatT make_param_mat() const;
    Stencil make_stencil(const ParamT &theta, bool compute_derivatives=true) const;
    Stencil make_stencil(double x, double y, double I, double bg, double sigma, bool compute_derivatives=true) const;

     /* Model Pixel Value And Derivatives */
    double model_value(int i, int j, const Stencil &s) const;
    void pixel_grad(int i, int j, const Stencil &s, ParamT &pgrad) const;
    void pixel_grad2(int i, int j, const Stencil &s, ParamT &pgrad2) const;
    void pixel_hess(int i, int j, const Stencil &s, ParamMatT &hess) const;
    void pixel_hess_update(int i, int j, const Stencil &s, double dm_ratio_m1, 
                           double dmm_ratio, ParamT &grad, ParamMatT &hess) const;

    ParamT bound_theta(const ParamT &theta) const;
    virtual void bound_theta(ParamT &theta) const=0;
    virtual double prior_relative_log_likelihood(const Stencil &s) const=0;
                           
    Stencil initial_theta_estimate(const ImageT &im, const ParamT &theta_init) const;

    /* Posterior Sampling */
    void sample_candidate_theta(int sample_index, RNG &rng, ParamT &canidate_theta, double scale=1.0) const;

protected:
    double candidate_eta_sigma; /**< The standard deviation for the normally distributed pertebation to theta_sigma in the random walk MCMC sampling */

    VecT stencil_sigmas={1.0, 1.3, 1.8, 2.3};
    MatT gaussian_Xstencils; /**< A stencils for gaussian filters with this size and psf*/
    MatT gaussian_Ystencils; /**< A stencils for gaussian filters with this size and psf*/
};


/* Inline Methods */

inline
Gauss2DsModel::ParamT
Gauss2DsModel::make_param() const
{
    return ParamT();
}

inline
Gauss2DsModel::ParamT
Gauss2DsModel::make_param(const ParamT &theta) const
{
    ParamT ntheta(theta);
    bound_theta(ntheta);
    return ntheta;
}

inline
Gauss2DsModel::ParamT
Gauss2DsModel::make_param(double x, double y, double I, double bg, double sigma) const
{
    ParamT theta;
    theta<<x<<y<<I<<bg<<sigma;
    bound_theta(theta);
    return theta;
}


inline
Gauss2DsModel::ParamMatT
Gauss2DsModel::make_param_mat() const
{
    return ParamMatT();
}

inline
Gauss2DsModel::Stencil
Gauss2DsModel::make_stencil(const ParamT &theta, bool compute_derivatives) const
{
    return Stencil(*this,make_param(theta),compute_derivatives);
}

inline
Gauss2DsModel::Stencil
Gauss2DsModel::make_stencil(double x, double y, double I, double bg, double sigma, bool compute_derivatives) const
{
    return Stencil(*this,make_param(x,y,I,bg,sigma),compute_derivatives);
}

inline
double Gauss2DsModel::model_value(int i, int j, const Stencil &s) const
{
    return s.bg()+s.I()*s.X(i)*s.Y(j);
}

inline
void
Gauss2DsModel::pixel_grad(int i, int j, const Stencil &s, ParamT &pgrad) const
{
    double I=s.I();
    pgrad(0) = I * s.DX(i) * s.Y(j);
    pgrad(1) = I * s.DY(j) * s.X(i);
    pgrad(2) = s.X(i) * s.Y(j);
    pgrad(3) = 1.;
    pgrad(4) = I * (s.Y(j)*s.DXS(i) + s.X(i)*s.DYS(j));
}

inline
void
Gauss2DsModel::pixel_grad2(int i, int j, const Stencil &s, ParamT &pgrad2) const
{
    double I=s.I();
    pgrad2(0)= I/s.sigmaX() * s.DXS(i) * s.Y(j);
    pgrad2(1)= I/s.sigmaY() * s.DYS(j) * s.X(i);
    pgrad2(2)= 0.;
    pgrad2(3)= 0.;
    pgrad2(4)= I * (s.X(i)*s.DYS2(j) + 2.*s.DXS(i)*s.DYS(j) + s.Y(j)*s.DXS2(i));
}

inline
void
Gauss2DsModel::pixel_hess(int i, int j, const Stencil &s, ParamMatT &hess) const
{
    hess.zeros();
    double I=s.I();
    //On Diagonal
    hess(0,0)= I/s.sigmaX() * s.DXS(i) * s.Y(j); //xx
    hess(1,1)= I/s.sigmaY() * s.DYS(j) * s.X(i); //yy
    hess(4,4)= I*(s.X(i)*s.DYS2(j) + 2.*s.DXS(i)*s.DYS(j) + s.Y(j)*s.DXS2(i)); //SS
    //Off Diagonal
    hess(0,1)= I * s.DX(i) * s.DY(j); //xy
    hess(0,4)= I * (s.Y(j)*s.DXSX(i) + s.DX(i)*s.DYS(j)); //xS
    hess(1,4)= I * (s.X(i)*s.DYSY(j) + s.DY(j)*s.DXS(i)); //yS
    //Off Diagonal with respect to I
    hess(0,2)=s.DX(i) * s.Y(j); //xI
    hess(1,2)=s.DY(j) * s.X(i); //yI
    hess(2,4)=s.Y(j)*s.DXS(i) + s.X(i)*s.DYS(j); //IS
}

#endif /* _GAUSS2DSMODEL_H */
