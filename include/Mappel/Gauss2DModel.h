/** @file Gauss2DModel.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2018
 * @brief The class declaration and inline and templated functions for Gauss2DModel.
 */

#ifndef _MAPPEL_GAUSS2DMODEL_H
#define _MAPPEL_GAUSS2DMODEL_H

#include "Mappel/PointEmitterModel.h"
#include "Mappel/ImageFormat2DBase.h"
#include "Mappel/MCMCAdaptor2D.h"
#include "Mappel/Gauss1DMAP.h"

namespace mappel {

/** @brief A base class for 2D Gaussian PSF with fixed but possibly asymmetric sigma.
 *
 */
class Gauss2DModel : public virtual PointEmitterModel, public virtual ImageFormat2DBase, public MCMCAdaptor2D
{    
public:
    /** @brief Stencil for 2D fixed-sigma models.
     */
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
    using StencilVecT = std::vector<Stencil>;
    

    /* Prior construction */
    static CompositeDist make_default_prior(const ImageSizeT &size);
    static CompositeDist make_prior_beta_position(const ImageSizeT &size, double beta_xpos, double beta_ypos, 
                                                  double mean_I, double kappa_I, double mean_bg, double kappa_bg);
    static CompositeDist make_prior_normal_position(const ImageSizeT &size, double sigma_xpos,double sigma_ypos, 
                                                  double mean_I, double kappa_I, double mean_bg, double kappa_bg);

    /* Overrides of Base methods to enable resizing of 1D internal models */
    void set_hyperparams(const VecT &hyperparams);
    void set_prior(CompositeDist&& prior_);
    void set_prior(const CompositeDist& prior_);
    void set_size(const ImageSizeT &size_);

    /* psf_sigma accessors */
    const VecT& get_psf_sigma() const;
    double get_psf_sigma(IdxT idx) const;
    void set_psf_sigma(double new_psf_sigma);
    void set_psf_sigma(const VecT& new_psf_sigma);
    
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

    Gauss1DSumModelT& debug_internal_sum_model_x() const {return x_model;}
    Gauss1DSumModelT& debug_internal_sum_model_y() const {return y_model;}
protected:
    //Abstract class cannot be instantiated
    Gauss2DModel(const ImageSizeT &size, const VecT &psf_sigma);
    Gauss2DModel(const Gauss2DModel &o);
    Gauss2DModel(Gauss2DModel &&o);
    Gauss2DModel& operator=(const Gauss2DModel &o);
    Gauss2DModel& operator=(Gauss2DModel &&o);
    

    using Gauss1DSumModelT = Gauss1DMAP; //Use a MAP estimator for the 1D initializer models
    void update_internal_1Dsum_estimators();
    static Gauss1DSumModelT make_internal_1Dsum_estimator(IdxT dim, const ImageSizeT &size, 
                                                const VecT &psf_sigma, const CompositeDist &prior);
    /* Non-static data Members */
    VecT psf_sigma; /**< Standard deviation of the fixed-sigma 1D Gaussian PSF in pixels */
    Gauss1DSumModelT x_model; /**< X-model fits 2D images X-axis (column sum) */
    Gauss1DSumModelT y_model; /**< Y-model fits 2D images Y-axis (row sum) */
};

/* Inline Method Definitions */

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
Gauss2DModel::Stencil
Gauss2DModel::make_stencil(const ParamT &theta, bool compute_derivatives) const
{
    if(!theta_in_bounds(theta)) {
        std::ostringstream msg;
        msg<<"Theta is not in bounds: "<<theta.t();
        throw ModelBoundsError(msg.str());
    }
    return Stencil(*this,theta,compute_derivatives);
}

inline
const VecT& Gauss2DModel::get_psf_sigma() const
{ return psf_sigma; }

inline
void Gauss2DModel::set_psf_sigma(double new_sigma)
{
    set_psf_sigma(VecT{new_sigma,new_sigma});
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
Gauss2DModel::pixel_hess(int i, int j, const Stencil &s, MatT &hess) const
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
Gauss2DModel::initial_theta_estimate(const ImageT &im)
{
    return initial_theta_estimate(im, make_param(arma::fill::zeros), DefaultSeperableInitEstimator);
}

inline
Gauss2DModel::Stencil 
Gauss2DModel::initial_theta_estimate(const ImageT &im, const ParamT &theta_init)
{
    return initial_theta_estimate(im, theta_init, DefaultSeperableInitEstimator);
}

/* Templated Overloads */
template<class Model>
typename std::enable_if<std::is_base_of<Gauss2DModel,Model>::value, StencilT<Model> >::type
cgauss_heuristic_compute_estimate(const Model &model, const ModelDataT<Model> &im, const ParamT<Model> &theta_init)
{
    auto size = model.get_size(0);
    auto psf_sigma = model.get_psf_sigma(0);
    if(size != model.get_size(1)) throw NotImplementedError("CGaussHeuristicEstimator::Image size must be square.");
    if(psf_sigma != model.get_psf_sigma(1)) throw NotImplementedError("CGaussHeuristicEstimator::PSF Sigma must be symmetric");
    arma::fvec theta_est(4);
    arma::fmat fim = arma::conv_to<arma::fmat>::from(im);  //Convert image to float from double
    cgauss::MLEInit(fim.memptr(),psf_sigma,size,theta_est.memptr());
    return model.make_stencil(cgauss::convertFromCGaussCoords(theta_est));
}

template<class Model>
typename std::enable_if<std::is_base_of<Gauss2DModel,Model>::value, StencilT<Model>>::type
cgauss_compute_estimate(Model &model, const ModelDataT<Model> &im, const ParamT<Model> &theta_init, int max_iterations)
{
    auto size = model.get_size(0);
    auto psf_sigma = model.get_psf_sigma(0);
    if(size != model.get_size(1)) throw NotImplementedError("CGaussMLE::Image size must be square.");
    if(psf_sigma != model.get_psf_sigma(1)) throw NotImplementedError("CGaussMLE::PSF Sigma nmust be symmetric");
    arma::fvec theta_est(4);
    arma::fmat fim = arma::conv_to<arma::fmat>::from(im);  //Convert image to float from double
    arma::fvec ftheta_init = cgauss::convertToCGaussCoords(theta_init);
    cgauss::MLEFit(fim.memptr(), psf_sigma, size, max_iterations, ftheta_init, theta_est.memptr());
    return model.make_stencil(cgauss::convertFromCGaussCoords(theta_est));
}

template<class Model>
typename std::enable_if<std::is_base_of<Gauss2DModel,Model>::value, StencilT<Model>>::type
cgauss_compute_estimate_debug(const Model &model, const ModelDataT<Model> &im, 
                       const ParamT<Model> &theta_init, int max_iterations,
                       ParamVecT<Model> &sequence)
{
    auto size = model.get_size(0);
    auto psf_sigma = model.get_psf_sigma(0);
    if(size != model.get_size(1)) throw NotImplementedError("CGaussMLE::Image size must be square.");
    if(psf_sigma != model.get_psf_sigma(1)) throw NotImplementedError("CGaussMLE::PSF Sigma must be symmetric");
    arma::fvec theta_est(4);
    arma::fmat fim = arma::conv_to<arma::fmat>::from(im);  //Convert image to float from double
    arma::fvec ftheta_init = cgauss::convertToCGaussCoords(theta_init);
    cgauss::MLEFit_debug(fim.memptr(), psf_sigma, size, max_iterations, ftheta_init, theta_est.memptr(), sequence);
    sequence = cgauss::convertFromCGaussCoords(sequence);
    return model.make_stencil(cgauss::convertFromCGaussCoords(theta_est));
}

} /* namespace mappel */

#endif /* _MAPPEL_GAUSS2DMODEL_H */
