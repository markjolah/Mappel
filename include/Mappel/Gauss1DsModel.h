/** @file Gauss1DsModel.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class declaration and inline and templated functions for Gauss1DsModel.
 */

#ifndef MAPPEL_GAUSS1DSMODEL_H
#define MAPPEL_GAUSS1DSMODEL_H

#include "Mappel/PointEmitterModel.h"
#include "Mappel/ImageFormat1DBase.h"
#include "Mappel/MCMCAdaptor1Ds.h"

namespace mappel {

/** @brief Base class for 1D Gaussian PSF with variable Gaussian sigma (standard deviation) measured in units of pixels.
 *
 */
class Gauss1DsModel : public virtual PointEmitterModel, public virtual ImageFormat1DBase, public MCMCAdaptor1Ds
{
public:
    /** @brief Stencil for 1D variable-sigma models.
     */
    class Stencil {
    public:
        bool derivatives_computed = false;
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
    using StencilVecT = std::vector<Stencil>;

    /* Prior construction */
    static const StringVecT prior_types;
    static const std::string DefaultPriorType;
    static CompositeDist make_default_prior(IdxT size, double min_sigma, double max_sigma, const std::string &prior_type);
    static CompositeDist make_default_prior_beta_position(IdxT size, double min_sigma, double max_sigma);
    static CompositeDist make_default_prior_normal_position(IdxT size, double min_sigma, double max_sigma);
    static CompositeDist make_prior_beta_position(IdxT size, double beta_xpos, double mean_I,
                                               double kappa_I, double mean_bg, double kappa_bg, 
                                               double min_sigma, double max_sigma, double alpha_sigma);
    static CompositeDist make_prior_normal_position(IdxT size, double sigma_xpos, double mean_I,
                                               double kappa_I, double mean_bg, double kappa_bg, 
                                               double min_sigma, double max_sigma, double alpha_sigma);

    /* min_sigma and max_sigma accessors */
    double get_min_sigma() const;
    double get_max_sigma() const;
    void set_min_sigma(double min_sigma);
    void set_max_sigma(double max_sigma);
    void set_min_sigma(const VecT &min_sigma);
    void set_max_sigma(const VecT &max_sigma);
    
    StatsT get_stats() const;
    Stencil make_stencil(const ParamT &theta, bool compute_derivatives=true) const;
    
    /* Model Pixel Value And Derivatives */
    double pixel_model_value(IdxT i,  const Stencil &s) const;
    void pixel_grad(IdxT i, const Stencil &s, ParamT &pgrad) const;
    void pixel_grad2(IdxT i, const Stencil &s, ParamT &pgrad2) const;
    void pixel_hess(IdxT i, const Stencil &s, MatT &hess) const;
    void pixel_hess_update(IdxT i, const Stencil &s, double dm_ratio_m1, 
                           double dmm_ratio, ParamT &grad, MatT &hess) const;
                           
    /** @brief Fast, heuristic estimate of initial theta */
    Stencil initial_theta_estimate(const ImageT &im) const;
    Stencil initial_theta_estimate(const ImageT &im, const ParamT &theta_init) const;
protected:
    explicit Gauss1DsModel(IdxT size_);
    Gauss1DsModel(const Gauss1DsModel &o);
    Gauss1DsModel(Gauss1DsModel &&o);
    Gauss1DsModel& operator=(const Gauss1DsModel &o);
    Gauss1DsModel& operator=(Gauss1DsModel &&o);
private:
    static void set_prior_variable_names(CompositeDist &prior);
};


/* Inline Method Definitions */

inline
double Gauss1DsModel::get_min_sigma() const
{ return prior.lbound()(3); }

inline
double Gauss1DsModel::get_max_sigma() const
{ return prior.ubound()(3); }

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
Gauss1DsModel::Stencil
Gauss1DsModel::make_stencil(const ParamT &theta, bool compute_derivatives) const
{
    if(!theta_in_bounds(theta)) {
        std::ostringstream msg;
        msg<<"Theta is not in bounds: "<<theta.t();
        throw ModelBoundsError(msg.str());
    }
    return Stencil(*this,theta,compute_derivatives);
}

inline
double Gauss1DsModel::pixel_model_value(IdxT i,  const Stencil &s) const
{
    return s.bg()+s.I()*s.X(i);
}

inline
void Gauss1DsModel::pixel_grad(IdxT i, const Stencil &s, ParamT &pgrad) const
{
    double I = s.I();
    pgrad(0) = I * s.DX(i);
    pgrad(1) = s.X(i);
    pgrad(2) = 1.;
    pgrad(3) = I * s.DXS(i);
}

inline
void Gauss1DsModel::pixel_grad2(IdxT i, const Stencil &s, ParamT &pgrad2) const
{
    double I = s.I();
    pgrad2(0) = I/s.sigma() * s.DXS(i);
    pgrad2(1) = 0.;
    pgrad2(2) = 0.;
    pgrad2(3) = I * s.DXS2(i);
}

inline
void Gauss1DsModel::pixel_hess(IdxT i, const Stencil &s, MatT &hess) const
{
    hess.zeros();
    double I = s.I();
    hess(0,0) = I/s.sigma() * s.DXS(i);
    hess(0,1) = s.DX(i); 
    hess(0,3) = I * s.DXSX(i);
    hess(1,3) = s.DXS(i);
    hess(3,3) = I * s.DXS2(i);
}

inline
Gauss1DsModel::Stencil 
Gauss1DsModel::initial_theta_estimate(const ImageT &im) const
{
    return initial_theta_estimate(im, make_param(arma::fill::zeros));
}


} /* namespace mappel */

#endif /* MAPPEL_GAUSS1DSMODEL_H */
