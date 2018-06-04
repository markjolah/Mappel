/** @file Gauss2DsxyModel.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2018
 * @brief The class declaration and inline and templated functions for Gauss2DsxyModel.
 */

#ifndef _MAPPEL_GAUSS2DSXYMODEL_H
#define _MAPPEL_GAUSS2DSXYMODEL_H

#include "PointEmitterModel.h"
#include "ImageFormat2DBase.h"
#include "Gauss1DsMAP.h"

namespace mappel {

/** @brief A base class for 2D Gaussian PSF with axis-aligned gaussian with free parameters for both sigma_x and sigma_y.
 * Gaussian sigma parameters sigma_x and sigma_y are measured in units of pixels.  The model has 6 parameters,
 * [x,y,I,bg,sigma_x,sigma_y].
 * 
 * Importantly sigma_x and sigma_y must be in the range given by parameters min_sigma, max_sigma.  Each is a 2-element vector, giving the minimum and maximum acceptable values for
 * the gaussian sigma.  It is important that min_sigma is at least 0.5 pixel, estimating gaussian centers when any component of the sigma is significantly
 * smaller than a pixel will lead to poor results anyways.
 * 
 * 
 */

class Gauss2DsxyModel : public virtual PointEmitterModel, public virtual ImageFormat2DBase 
{
public:
     /** @brief Stencil for 2D free-sigma (astigmatic) models.
     */
    class Stencil {
    public:
        bool derivatives_computed=false;
        typedef Gauss2DsxyModel::ParamT ParamT;
        Gauss2DsxyModel const *model;
        
        ParamT theta;
        VecT dx, dy;
        VecT Gx, Gy;
        VecT X, Y;
        VecT DX, DY;
        VecT DXSX, DYSX;
        VecT DXS, DYS;
        VecT DXS2, DYS2;
        VecT DXSX, DYSY;
        Stencil() {}
        Stencil(const Gauss2DsxyModel &model,const ParamT &theta, bool _compute_derivatives=true);
        void compute_derivatives();
        inline double x() const {return theta(0);}
        inline double y() const {return theta(1);}
        inline double I() const {return theta(2);}
        inline double bg() const {return theta(3);}
        inline double sigmaX() const {return theta(4);}
        inline double sigmaY() const {return theta(5);}
        friend std::ostream& operator<<(std::ostream &out, const Gauss2DsxyModel::Stencil &s);
    };

    using StencilVecT = std::vector<Stencil>;

    Gauss2DsxyModel(const ImageSizeT &size, const VecT &min_sigma, const VecT &max_sigma);
    /* Prior construction */
    static CompositeDist make_default_prior(const ImageSizeT &size, double max_sigma_ratio);
    static CompositeDist make_prior_beta_position(const ImageSizeT &size, double beta_xpos, double beta_ypos, 
                                                  double mean_I, double kappa_I, double mean_bg, double kappa_bg,
                                                  double max_sigma_ratio, double alpha_sigma);
    static CompositeDist make_prior_normal_position(const ImageSizeT &size, double sigma_xpos,double sigma_ypos, 
                                                    double mean_I, double kappa_I, double mean_bg, double kappa_bg,
                                                    double max_sigma_ratio, double alpha_sigma);

    /* Overrides of Base methods to enable resizing of 1D internal models */
    void set_hyperparams(const VecT &hyperparams);
    void set_prior(CompositeDist&& prior_);
    void set_size(const ImageSizeT &size_);

    /* min_sigma and max_sigma accessors */
    VecT get_min_sigma() const;
    double get_min_sigma(IdxT dim) const;
    VecT get_max_sigma() const;
    double get_max_sigma(IdxT dim) const;
    double get_max_sigma_ratio() const;
    void set_min_sigma(const VecT &min_sigma);
    void set_max_sigma(const VecT &max_sigma);
    void set_max_sigma_ratio(double max_sigma_ratio);

    StatsT get_stats() const;
    Stencil make_stencil(const ParamT &theta, bool compute_derivatives=true) const;
    
    /* Model Pixel Value And Derivatives */
    double pixel_model_value(int i, int j, const Stencil &s) const;
    void pixel_grad(int i, int j, const Stencil &s, ParamT &pgrad) const;
    void pixel_grad2(int i, int j, const Stencil &s, ParamT &pgrad2) const;
    void pixel_hess(int i, int j, const Stencil &s, MatT &hess) const;
    void pixel_hess_update(int i, int j, const Stencil &s, double dm_ratio_m1, 
                           double dmm_ratio, ParamT &grad, MatT &hess) const;


    /** @brief Fast, heuristic estimate of initial theta */
    Stencil initial_theta_estimate(const ImageT &im);
    Stencil initial_theta_estimate(const ImageT &im, const ParamT &theta_init);
    Stencil initial_theta_estimate(const ImageT &im, const ParamT &theta_init, const std::string &estimator);

    /* Posterior Sampling */
    void sample_mcmc_candidate_theta(int sample_index, ParamT &canidate_theta, double scale=1.0);

protected:
    double mcmc_candidate_eta_y; /**< Std-dev for the normal perturbations to theta_y under MCMC sampling */    
    double mcmc_candidate_eta_sigma; /**< The standard deviation for the normally distributed pertebation to theta_sigma in the random walk MCMC sampling */

    void update_internal_1D_estimators();
    static double compute_max_sigma_ratio(const VecT& min_sigma, const VecT& max_sigma);
    /* Non-static data Members */
    VecT min_sigma; /**< Gaussian PSF in pixels */
    Gauss1DsMAP x_model; /**< X-model fits 2D images X-axis (column sum).  Using variable sigma 1D model. */
    Gauss1DsMAP y_model; /**< Y-model fits 2D images Y-axis (row sum).  Using variable sigma 1D model. */

};

/* Inline Methods */

inline
VecT Gauss2DsxyModel::get_min_sigma() const
{ return min_sigma; }


inline
VecT Gauss2DsxyModel::get_max_sigma() const
{ return get_min_sigma() * get_max_sigma_ratio(); }

inline
double Gauss2DsxyModel::get_max_sigma(IdxT dim) const
{ return get_min_sigma(dim) * get_max_sigma_ratio(); }

inline
double Gauss2DsxyModel::get_max_sigma_ratio() const
{ return get_ubound()(4); }


/** @brief Make a new Model::Stencil object at theta.
 * 
 * Stencils store all of the important calculations necessary for evaluating the log-likelihood and its derivatives 
 * at a particular theta (parameter) value.
 * 
 * This allows re-use of the most expensive computations.  Stencils can be easily passed around by reference, and most
 * functions in the mappel::methods namespace accept a const Stencil reference in place of the model parameter.
 * 
 * Throws mappel::ModelBoundsError if not model.theta_in_bounds(theta).
 * 
 * If derivatives will not be computed with this stencil set compute_derivatives=false
 * 
 * @param theta Prameter to evaluate at
 * @param compute_derivatives True to also prepare for derivative computations
 * @return A new Stencil object ready to compute with
 */
inline
Gauss2DsxyModel::Stencil
Gauss2DsxyModel::make_stencil(const ParamT &theta, bool compute_derivatives) const
{
    if(!theta_in_bounds(theta)) {
        std::ostringstream msg;
        msg<<"Theta is not in bounds: "<<theta.t();
        throw ModelBoundsError(msg.str());
    }
    return Stencil(*this,theta,compute_derivatives);
}


/* Model Pixel Value And Derivatives */

inline
double Gauss2DsxyModel::pixel_model_value(int i, int j, const Stencil &s) const
{
    return s.bg()+s.I()*s.X(i)*s.Y(j);
}

inline
void
Gauss2DsxyModel::pixel_grad(int i, int j, const Stencil &s, ParamT &pgrad) const
{
    double I = s.I();
    pgrad(0) = I * s.DX(i) * s.Y(j);
    pgrad(1) = I * s.DY(j) * s.X(i);
    pgrad(2) = s.X(i) * s.Y(j);
    pgrad(3) = 1.;
    pgrad(4) = I * (s.Y(j)*s.DXS(i) + s.X(i)*s.DYS(j));
}

inline
void
Gauss2DsxyModel::pixel_grad2(int i, int j, const Stencil &s, ParamT &pgrad2) const
{
    double I = s.I();
    pgrad2(0) = I/s.sigmaX() * s.DXS(i) * s.Y(j);
    pgrad2(1) = I/s.sigmaY() * s.DYS(j) * s.X(i);
    pgrad2(2) = 0.;
    pgrad2(3) = 0.;
    pgrad2(4) = I * (s.X(i)*s.DYS2(j) + 2.*s.DXS(i)*s.DYS(j) + s.Y(j)*s.DXS2(i));
}

inline
void
Gauss2DsxyModel::pixel_hess(int i, int j, const Stencil &s, MatT &hess) const
{
    hess.zeros();
    double I = s.I();
    //On Diagonal
    hess(0,0) = I/s.sigmaX() * s.DXS(i) * s.Y(j); //xx
    hess(1,1) = I/s.sigmaY() * s.DYS(j) * s.X(i); //yy
    hess(4,4) = I*(s.X(i)*s.DYS2(j) + 2.*s.DXS(i)*s.DYS(j) + s.Y(j)*s.DXS2(i)); //SS
    //Off Diagonal
    hess(0,1) = I * s.DX(i) * s.DY(j); //xy
    hess(0,4) = I * (s.Y(j)*s.DXSX(i) + s.DX(i)*s.DYS(j)); //xS
    hess(1,4) = I * (s.X(i)*s.DYSY(j) + s.DY(j)*s.DXS(i)); //yS
    //Off Diagonal with respect to I
    hess(0,2) = s.DX(i) * s.Y(j); //xI
    hess(1,2) = s.DY(j) * s.X(i); //yI
    hess(2,4) = s.Y(j)*s.DXS(i) + s.X(i)*s.DYS(j); //IS
}

inline
Gauss2DsxyModel::Stencil 
Gauss2DsxyModel::initial_theta_estimate(const ImageT &im)
{
    return initial_theta_estimate(im, make_param(arma::fill::zeros), DefaultSeperableInitEstimator);
}

inline
Gauss2DsxyModel::Stencil 
Gauss2DsxyModel::initial_theta_estimate(const ImageT &im, const ParamT &theta_init)
{
    return initial_theta_estimate(im, theta_init, DefaultSeperableInitEstimator);
}

/* Templated Overloads */
template<class Model>
typename std::enable_if<std::is_base_of<Gauss2DsxyModel,Model>::value, StencilT<Model> >::type
cgauss_heuristic_compute_estimate(const Model &model, const ModelDataT<Model> &im, const ParamT<Model> &theta_init)
{
    auto size = model.get_size(0);
    auto psf_sigma = model.get_min_sigma(0);
    if(size != model.get_size(1)) throw NotImplementedError("CGaussHeuristicEstimator::Image size must be square.");
    if(psf_sigma != model.get_min_sigma(1)) throw NotImplementedError("CGaussHeuristicEstimator::PSF Sigma must be symmetric");
    arma::fvec theta_est(5);
    arma::fmat fim = arma::conv_to<arma::fmat>::from(im);  //Convert image to float from double
    cgauss::MLEInit_sigma(fim.memptr(),psf_sigma,size,theta_est.memptr());
    return model.make_stencil(cgauss::convertFromCGaussCoords_sigma(theta_est,psf_sigma));
}

template<class Model>
typename std::enable_if<std::is_base_of<Gauss2DsxyModel,Model>::value, StencilT<Model>>::type
cgauss_compute_estimate(Model &model, const ModelDataT<Model> &im, const ParamT<Model> &theta_init, int max_iterations)
{
    auto size = model.get_size(0);
    auto psf_sigma = model.get_min_sigma(0);
    if(size != model.get_size(1)) throw NotImplementedError("CGaussMLE::Image size must be square.");
    if(psf_sigma != model.get_min_sigma(1)) throw NotImplementedError("CGaussMLE::PSF Sigma must be symmetric");
    arma::fvec theta_est(5);
    arma::fmat fim = arma::conv_to<arma::fmat>::from(im);  //Convert image to float from double
    arma::fvec ftheta_init = cgauss::convertToCGaussCoords_sigma(theta_init,psf_sigma);
    cgauss::MLEFit_sigma(fim.memptr(), psf_sigma, size, max_iterations, ftheta_init, theta_est.memptr());
    return model.make_stencil(cgauss::convertFromCGaussCoords_sigma(theta_est,psf_sigma));
}

template<class Model>
typename std::enable_if<std::is_base_of<Gauss2DsxyModel,Model>::value, StencilT<Model>>::type
cgauss_compute_estimate_debug(const Model &model, const ModelDataT<Model> &im, 
                       const ParamT<Model> &theta_init, int max_iterations,
                       ParamVecT<Model> &sequence)
{
    auto size = model.get_size(0);
    auto psf_sigma = model.get_min_sigma(0);
    if(size != model.get_size(1)) throw NotImplementedError("CGaussMLE::Image size must be square.");
    if(psf_sigma != model.get_min_sigma(1)) throw NotImplementedError("CGaussMLE::PSF Sigma must be symmetric");
    arma::fvec theta_est(5);
    arma::fmat fim = arma::conv_to<arma::fmat>::from(im);  //Convert image to float from double
    arma::fvec ftheta_init = cgauss::convertToCGaussCoords_sigma(theta_init,psf_sigma);
    cgauss::MLEFit_sigma_debug(fim.memptr(), psf_sigma, size, max_iterations, ftheta_init, theta_est.memptr(), sequence);
    sequence = cgauss::convertFromCGaussCoords_sigma(sequence,psf_sigma);
    return model.make_stencil(cgauss::convertFromCGaussCoords_sigma(theta_est,psf_sigma));
}

} /* namespace mappel */

#endif /* _MAPPEL_GAUSS2DSXYMODEL_H */
