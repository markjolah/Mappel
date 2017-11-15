/** @file Gauss1DModel.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief The class declaration and inline and templated functions for Gauss1DModel.
 */

#ifndef _GAUSS1DMODEL_H
#define _GAUSS1DMODEL_H

#include "PointEmitterModel.h"
#include "ImageFormat1DBase.h"

namespace mappel {

/** @brief A base class for 1D Gaussian PSF with a fixed sigma (standard dev.)
 *
 */
class Gauss1DModel : public virtual PointEmitterModel, public virtual ImageFormat1DBase 
{   
public:
    /** @brief Stencil for 1D fixed-sigma models.
     */
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
   
    /* Non-static data Members */
    double psf_sigma; /**< Standard deviation of the fixed-sigma 1D Gaussian PSF in pixels */
    
    Gauss1DModel(IdxT size_, double psf_sigma);

    /* Prior construction */
    static CompositeDist make_prior(IdxT size);
    static CompositeDist make_prior(IdxT size, double beta_x, double mean_I,
                                    double kappa_I, double mean_bg, double kappa_bg);
    
    StatsT get_stats() const;

    Stencil make_stencil(const ParamT &theta, bool compute_derivatives=true) const;

    /* Model Pixel Value And Derivatives */
    double pixel_model_value(IdxT i,  const Stencil &s) const;
    void pixel_grad(IdxT i, const Stencil &s, ParamT &pgrad) const;
    void pixel_grad2(IdxT i, const Stencil &s, ParamT &pgrad2) const;
    void pixel_hess(IdxT i, const Stencil &s, ParamMatT &hess) const;
    void pixel_hess_update(IdxT i, const Stencil &s, double dm_ratio_m1, 
                           double dmm_ratio, ParamT &grad, ParamMatT &hess) const;
                           
    /** @brief Fast, heuristic estimate of initial theta */
    Stencil initial_theta_estimate(const ImageT &im, const ParamT &theta_init) const;

    /** @brief Posterior Sampling */
    void sample_mcmc_candidate_theta(IdxT sample_index, ParamT &canidate_theta, double scale=1.0) const;
};

/* Function Declarations */



/* Inline Method Definitions */
/*
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
}*/

inline
Gauss1DModel::Stencil
Gauss1DModel::make_stencil(const ParamT &theta, bool compute_derivatives) const
{
//    return Stencil(*this,make_param(theta),compute_derivatives);
    //Remove implicit bounding.  This allows for computations outside of the limited region
    //And prevents false impression that LLH and grad and hessian do not change outside of boundaries
    return Stencil(*this,theta,compute_derivatives);
}
/*
inline
Gauss1DModel::Stencil
Gauss1DModel::make_stencil(double x, double I, double bg, bool compute_derivatives) const
{
    return Stencil(*this,make_param(x,I,bg),compute_derivatives);
}
*/

inline
double Gauss1DModel::pixel_model_value(IdxT i,  const Stencil &s) const
{
    return s.bg()+s.I()*s.X(i);
}

inline
void Gauss1DModel::pixel_grad(IdxT i, const Stencil &s, ParamT &pgrad) const
{
    double I = s.I();
    pgrad(0) = I * s.DX(i);
    pgrad(1) = s.X(i);
    pgrad(2) = 1.;
}

inline
void Gauss1DModel::pixel_grad2(IdxT i, const Stencil &s, ParamT &pgrad2) const
{
    double I = s.I();
    pgrad2(0) = I/psf_sigma * s.DXS(i);
    pgrad2(1) = 0.;
    pgrad2(2) = 0.;
}

inline
void Gauss1DModel::pixel_hess(IdxT i, const Stencil &s, ParamMatT &hess) const
{
    hess.zeros();
    double I = s.I();
    hess(0,0) = I/psf_sigma * s.DXS(i);
    hess(0,1) = s.DX(i); 
}

} /* namespace mappel */

#endif /* _GAUSS1DMODEL_H */
