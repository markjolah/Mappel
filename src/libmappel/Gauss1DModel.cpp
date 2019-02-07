/** @file Gauss1DModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class definition and template Specializations for Gauss1DModel
 */

#include "Mappel/Gauss1DModel.h"
#include "Mappel/stencil.h"

namespace mappel {

Gauss1DModel::Gauss1DModel(IdxT size, double psf_sigma)
    : PointEmitterModel(), ImageFormat1DBase(size), //V-base calls ignored since a higher concrete class will call them
      MCMCAdaptor1D(),
      psf_sigma(psf_sigma)
{
    check_psf_sigma(psf_sigma);
}

Gauss1DModel::Gauss1DModel(const Gauss1DModel &o)
    : PointEmitterModel(o), ImageFormat1DBase(o), //V-base calls ignored since a higher concrete class will call them
      MCMCAdaptor1D(),
      psf_sigma(o.psf_sigma)
{ }

Gauss1DModel::Gauss1DModel(Gauss1DModel &&o)
    : PointEmitterModel(o), ImageFormat1DBase(o), //V-base calls ignored since a higher concrete class will call them
      MCMCAdaptor1D(),
      psf_sigma(o.psf_sigma)
{ }

Gauss1DModel& Gauss1DModel::operator=(const Gauss1DModel &o)
{
    //Don't copy virtual base classes.  This is called by superclass only.
    MCMCAdaptor1D::operator=(o);
    //Copy data memebers
    psf_sigma=o.psf_sigma;
    return *this;
}

Gauss1DModel& Gauss1DModel::operator=(Gauss1DModel &&o)
{
    //Don't copy virtual base classes.  This is called by superclass only.
    MCMCAdaptor1D::operator=(std::move(o));
    //Copy data memebers
    psf_sigma = o.psf_sigma;
    return *this;
}



/* Prior construction */
const StringVecT Gauss1DModel::prior_types = { "Beta", //Model the position as a symmetric Beta distribution scaled over (0,size)
                                                       "Normal"  //Model the position as a truncated Normal distribution centered at size/2 with domain (0,size)
                                                      };
const std::string Gauss1DModel::DefaultPriorType = "Normal";

CompositeDist
Gauss1DModel::make_default_prior(IdxT size, const std::string &prior_type)
{
    if(istarts_with(prior_type,"Normal")) {
        return make_default_prior_normal_position(size);
    } else if(istarts_with(prior_type,"Beta")) {
        return make_default_prior_beta_position(size);
    } else {
        std::ostringstream msg;
        msg<<"Unknown prior type: "<<prior_type;
        throw ParameterValueError(msg.str());
    }
}

void
Gauss1DModel::set_prior_variable_names(CompositeDist &pr)
{
    pr.set_component_names(StringVecT{"x_pos", "intensity", "background"});
    pr.set_dim_variables(StringVecT{"x","I","bg"});
}

CompositeDist
Gauss1DModel::make_default_prior_beta_position(IdxT size)
{
    CompositeDist d(make_prior_component_position_beta(size),
                    make_prior_component_intensity(),
                    make_prior_component_intensity(default_pixel_mean_bg*size)); //bg is summed over the other dimension leading to larger mean per 1D 'pixel'
    set_prior_variable_names(d);
    return d;
}

CompositeDist
Gauss1DModel::make_default_prior_normal_position(IdxT size)
{
    CompositeDist d(make_prior_component_position_normal(size),
                    make_prior_component_intensity(),
                    make_prior_component_intensity(default_pixel_mean_bg*size)); //bg is summed over the other dimension leading to larger mean per 1D 'pixel'
    set_prior_variable_names(d);
    return d;
}


CompositeDist
Gauss1DModel::make_prior_beta_position(IdxT size, double beta_xpos,
                                       double mean_I, double kappa_I,
                                       double mean_bg, double kappa_bg)
{
    CompositeDist d(make_prior_component_position_beta(size,beta_xpos),
                    make_prior_component_intensity(mean_I,kappa_I),
                    make_prior_component_intensity(mean_bg, kappa_bg)); //bg is summed over the other dimension leading to larger mean per 1D 'pixel'
    set_prior_variable_names(d);
    return d;
}


CompositeDist
Gauss1DModel::make_prior_normal_position(IdxT size, double sigma_xpos,
                                       double mean_I, double kappa_I,
                                       double mean_bg, double kappa_bg)
{
    CompositeDist d(make_prior_component_position_normal(size,sigma_xpos),
                    make_prior_component_intensity(mean_I,kappa_I),
                    make_prior_component_intensity(mean_bg, kappa_bg)); //bg is summed over the other dimension leading to larger mean per 1D 'pixel'
    set_prior_variable_names(d);
    return d;
}

void Gauss1DModel::set_psf_sigma(double new_sigma)
{ 
    check_psf_sigma(new_sigma);
    psf_sigma = new_sigma;
}

double Gauss1DModel::get_psf_sigma(IdxT idx) const
{
    if(idx > 0) {
        std::ostringstream msg;
        msg<<"Gauss1DModel::get_psf_sigma() idx="<<idx<<" is invalid.";
        throw ParameterValueError(msg.str());
    }
    return psf_sigma; 
}


Gauss1DModel::Stencil::Stencil(const Gauss1DModel &model_,
                               const Gauss1DModel::ParamT &theta,
                               bool _compute_derivatives)
        : model(&model_),theta(theta)
{
    IdxT szX = model->size;
    dx = make_d_stencil(szX, x());
    X = make_X_stencil(szX, dx,model->psf_sigma);
    if(_compute_derivatives) compute_derivatives();
}

void Gauss1DModel::Stencil::compute_derivatives()
{
    if(derivatives_computed) return;
    derivatives_computed=true;
    IdxT szX = model->size;
    double sigmaX = model->psf_sigma;
    Gx = make_G_stencil(szX, dx,sigmaX);
    DX = make_DX_stencil(szX, Gx,sigmaX);
    DXS = make_DXS_stencil(szX, dx, Gx,sigmaX);
}

std::ostream& operator<<(std::ostream &out, const Gauss1DModel::Stencil &s)
{
    IdxT w=8;
    print_vec_row(out,s.theta,"Theta:",w,TERM_WHITE);
    print_vec_row(out,s.dx,"dx:",w,TERM_CYAN);
    print_vec_row(out,s.X,"X:",w,TERM_CYAN);
    if(s.derivatives_computed) {
        print_vec_row(out,s.Gx,"Gx:",w,TERM_BLUE);
        print_vec_row(out,s.DX,"DX:",w,TERM_BLUE);
        print_vec_row(out,s.DXS,"DXS:",w,TERM_BLUE);
    }
    return out;
}

StatsT Gauss1DModel::get_stats() const
{
    auto stats = PointEmitterModel::get_stats();
    auto im_stats = ImageFormat1DBase::get_stats();
    auto mcmc_stats = MCMCAdaptor1D::get_stats();
    stats.insert(im_stats.begin(), im_stats.end());
    stats.insert(mcmc_stats.begin(), mcmc_stats.end());
    stats["psf_sigma"] = get_psf_sigma();
    return stats;
}

/** @brief pixel derivative inner loop calculations.
 */
void Gauss1DModel::pixel_hess_update(IdxT i, const Stencil &s, double dm_ratio_m1, double dmm_ratio, ParamT &grad, MatT &hess) const
{
    auto pgrad = make_param();
    pixel_grad(i,s,pgrad);
    double I = s.I();
    /* Update grad */
    grad += dm_ratio_m1*pgrad;
    /* Update hess */
    hess(0,0) += dm_ratio_m1 * I/psf_sigma * s.DXS(i);
    hess(0,1) += dm_ratio_m1 * pgrad(0) / I; 
    //This is the pixel-gradient dependent part of the hessian
    for(IdxT c=0; c<hess.n_cols; c++) for(IdxT r=0; r<=c; r++)
        hess(r,c) -= dmm_ratio * pgrad(r) * pgrad(c);
}

Gauss1DModel::Stencil 
Gauss1DModel::initial_theta_estimate(const ImageT &im, const ParamT &theta_init) const
{
    double x_pos = 0;
    double I = 0;
    double bg = 0;    
    if(theta_init.n_elem == num_params){
        x_pos = theta_init(0);
        I = theta_init(1);
        bg = theta_init(2);
    }
    
    if(x_pos <= 0 || x_pos > size || !std::isfinite(x_pos)) 
        x_pos = im.index_max()+0.5;
    if(bg <= 0 || !std::isfinite(bg)) 
        bg = std::max(1.0, 0.75*im.min());
    if(I <= 0 || !std::isfinite(I))  
        I = std::max(1.0, arma::sum(im) - bg*size);
    return make_stencil(ParamT{x_pos,  I, bg});
}

} /* namespace mappel */
