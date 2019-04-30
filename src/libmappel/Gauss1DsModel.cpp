/** @file Gauss1DsModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class definition and template Specializations for Gauss1DsModel
 */

#include "Mappel/Gauss1DsModel.h"
#include "Mappel/stencil.h"

namespace mappel {

Gauss1DsModel::Gauss1DsModel(IdxT size)
    :  PointEmitterModel(), ImageFormat1DBase(size), //V-base calls ignored since a higher concrete class will call them
      MCMCAdaptor1Ds()
{ }

Gauss1DsModel::Gauss1DsModel(const Gauss1DsModel &o)
    : PointEmitterModel(o), ImageFormat1DBase(o), //V-base calls ignored since a higher concrete class will call them
      MCMCAdaptor1Ds()
{ }

Gauss1DsModel::Gauss1DsModel(Gauss1DsModel &&o)
    : PointEmitterModel(o), ImageFormat1DBase(o), //V-base calls ignored since a higher concrete class will call them
      MCMCAdaptor1Ds()
{ }

Gauss1DsModel& Gauss1DsModel::operator=(const Gauss1DsModel &o)
{
    //Don't copy virtual base classes.  This is called by superclass only.
    if(this == &o) return *this; //Check for self assignment
    MCMCAdaptor1Ds::operator=(o);
    return *this;
}

Gauss1DsModel& Gauss1DsModel::operator=(Gauss1DsModel &&o)
{
    //Don't copy virtual base classes.  This is called by superclass only.
    if(this == &o) return *this; //Check for self assignment
    MCMCAdaptor1Ds::operator=(std::move(o));
    return *this;
}

/* Prior construction */
const StringVecT Gauss1DsModel::prior_types = { "Beta", //Model the position as a symmetric Beta distribution scaled over (0,size)
                                                       "Normal"  //Model the position as a truncated Normal distribution centered at size/2 with domain (0,size)
                                                      };
const std::string Gauss1DsModel::DefaultPriorType = "Normal";

CompositeDist
Gauss1DsModel::make_default_prior(IdxT size, double min_sigma, double max_sigma, const std::string &prior_type)
{
    if(istarts_with(prior_type,"Normal")) {
        return make_default_prior_normal_position(size, min_sigma, max_sigma);
    } else if(istarts_with(prior_type,"Beta")) {
        return make_default_prior_beta_position(size, min_sigma, max_sigma);
    } else {
        std::ostringstream msg;
        msg<<"Unknown prior type: "<<prior_type;
        throw ParameterValueError(msg.str());
    }
}

void
Gauss1DsModel::set_prior_variable_names(CompositeDist &pr)
{
    pr.set_component_names(StringVecT{"x_pos", "intensity", "background","psf_sigma"});
    pr.set_dim_variables(StringVecT{"x","I","bg","sigma"});
}


CompositeDist 
Gauss1DsModel::make_default_prior_beta_position(IdxT size, double min_sigma, double max_sigma)
{
    CompositeDist d(make_prior_component_position_beta(size),
                    make_prior_component_intensity(),
                    make_prior_component_intensity(DefaultPriorPixelMeanBG*size), //bg is summed over the other dimension leading to larger mean per 1D 'pixel'
                    make_prior_component_sigma(min_sigma,max_sigma));
    set_prior_variable_names(d);
    return d;
}

CompositeDist
Gauss1DsModel::make_default_prior_normal_position(IdxT size, double min_sigma, double max_sigma)
{
     CompositeDist d(make_prior_component_position_normal(size),
                    make_prior_component_intensity(),
                    make_prior_component_intensity(DefaultPriorPixelMeanBG*size), //bg is summed over the other dimension leading to larger mean per 1D 'pixel'
                    make_prior_component_sigma(min_sigma,max_sigma));
    set_prior_variable_names(d);
    return d;
}

CompositeDist 
Gauss1DsModel::make_prior_beta_position(IdxT size, double beta_xpos, 
                                       double mean_I, double kappa_I, 
                                       double mean_bg, double kappa_bg,
                                       double min_sigma, double max_sigma, double alpha_sigma)
{
    CompositeDist d(make_prior_component_position_beta(size,beta_xpos),
                    make_prior_component_intensity(mean_I,kappa_I),
                    make_prior_component_intensity(mean_bg, kappa_bg), //bg is summed over the other dimension leading to larger mean per 1D 'pixel'
                    make_prior_component_sigma(min_sigma,max_sigma, alpha_sigma));
    set_prior_variable_names(d);
    return d;
}

CompositeDist 
Gauss1DsModel::make_prior_normal_position(IdxT size, double sigma_xpos, 
                                       double mean_I, double kappa_I, 
                                       double mean_bg, double kappa_bg,
                                       double min_sigma, double max_sigma, double alpha_sigma)
{
    CompositeDist d(make_prior_component_position_normal(size,sigma_xpos),
                    make_prior_component_intensity(mean_I,kappa_I),
                    make_prior_component_intensity(mean_bg, kappa_bg), //bg is summed over the other dimension leading to larger mean per 1D 'pixel'
                    make_prior_component_sigma(min_sigma,max_sigma, alpha_sigma));
    set_prior_variable_names(d);
    return d;
}

/* min_sigma and max_sigma accessors */
void Gauss1DsModel::set_min_sigma(double new_sigma)
{
    check_psf_sigma(new_sigma);
    if(new_sigma >= get_max_sigma()) {
        std::ostringstream msg;
        msg<<"Invalid new min_sigma:"<<new_sigma<<" >= max_sigma:"<<get_max_sigma();
        throw ParameterValueError(msg.str());
    }
    auto lb = prior.lbound();
    lb(3) = new_sigma;
    set_lbound(lb);
}

void Gauss1DsModel::set_max_sigma(double new_sigma)
{
    check_psf_sigma(new_sigma);
    if(new_sigma <= get_min_sigma()) {
        std::ostringstream msg;
        msg<<"Invalid new max_sigma:"<<new_sigma<<" <= min_sigma:"<<get_min_sigma();
        throw ParameterValueError(msg.str());
    }
    auto ub = prior.ubound();
    ub(3) = new_sigma;
    set_ubound(ub);
}

void Gauss1DsModel::set_min_sigma(const VecT &min_sigma)
{
    set_min_sigma(min_sigma(0));
}

void Gauss1DsModel::set_max_sigma(const VecT &max_sigma)
{
    set_max_sigma(max_sigma(0));
}

/* Gauss1DsModel::Stencil member functions */
Gauss1DsModel::Stencil::Stencil(const Gauss1DsModel &model_,
                                const Gauss1DsModel::ParamT &theta,
                                bool _compute_derivatives)
: model(&model_),theta(theta)
{
    IdxT szX = model->size;
    dx = make_d_stencil(szX, x());
    X = make_X_stencil(szX, dx,sigma());
    if(_compute_derivatives) compute_derivatives();
}

void Gauss1DsModel::Stencil::compute_derivatives()
{
    if(derivatives_computed) return;
    derivatives_computed=true;
    IdxT szX = model->size;
    Gx = make_G_stencil(szX, dx, sigma());
    DX = make_DX_stencil(szX, Gx, sigma());
    DXS = make_DXS_stencil(szX, dx, Gx, sigma());
    DXS2 = make_DXS2_stencil(szX, dx, Gx, DXS, sigma());
    DXSX = make_DXSX_stencil(szX, dx, Gx, DX, sigma());
}

std::ostream& operator<<(std::ostream &out, const Gauss1DsModel::Stencil &s)
{
    int w=8;
    print_vec_row(out,s.theta,"Theta:",w,TERM_WHITE);
    print_vec_row(out,s.dx,"dx:",w,TERM_CYAN);
    print_vec_row(out,s.X,"X:",w,TERM_CYAN);
    if(s.derivatives_computed) {
        print_vec_row(out,s.Gx,"Gx:",w,TERM_BLUE);
        print_vec_row(out,s.DX,"DX:",w,TERM_BLUE);
        print_vec_row(out,s.DXS,"DXS:",w,TERM_BLUE);
        print_vec_row(out,s.DXS2,"DXS2:",w,TERM_BLUE);
        print_vec_row(out,s.DXSX,"DXSX:",w,TERM_BLUE);
    }
    return out;
}

StatsT Gauss1DsModel::get_stats() const
{
    auto stats = PointEmitterModel::get_stats();
    auto im_stats = ImageFormat1DBase::get_stats();
    auto mcmc_stats = MCMCAdaptor1Ds::get_stats();
    stats.insert(im_stats.begin(), im_stats.end());
    stats.insert(mcmc_stats.begin(), mcmc_stats.end());
    stats["min_sigma"] = get_min_sigma();
    stats["max_sigma"] = get_max_sigma();
    return stats;
}

void
Gauss1DsModel::pixel_hess_update(IdxT i, const Stencil &s, double dm_ratio_m1, double dmm_ratio, ParamT &grad, MatT &hess) const
{
    /* Caclulate pixel derivative */
    auto pgrad=make_param();
    pixel_grad(i,s,pgrad);
    double I = s.I();
    /* Update grad */
    grad += dm_ratio_m1*pgrad;
    /* Update hess */
    hess(0,0) += dm_ratio_m1 * I/s.sigma() * s.DXS(i);
    hess(0,1) += dm_ratio_m1 * s.DX(i);
    hess(0,3) += dm_ratio_m1 * I * s.DXSX(i);
    hess(1,3) += dm_ratio_m1 * s.DXS(i);
    hess(3,3) += dm_ratio_m1 * I * s.DXS2(i);
    //This is the pixel-gradient dependent part of the hessian
    for(IdxT c=0; c<hess.n_cols; c++) for(IdxT r=0; r<=c; r++)
        hess(r,c) -= dmm_ratio * pgrad(r) * pgrad(c);
}

Gauss1DsModel::Stencil
Gauss1DsModel::initial_theta_estimate(const ImageT &im, const ParamT &theta_init) const
{
    double x_pos=0, I=0, bg=0, sigma=0;
    if (!theta_init.is_empty()) {
        x_pos = theta_init(0);
        I = theta_init(1);
        bg = theta_init(2);
        sigma = theta_init(3);
    }
    if(x_pos<=0 || x_pos>size)  x_pos = im.index_max()+0.5; //Estimate position as the brightest pixel    
    
    double min_sigma = lbound(3);
    double max_sigma = ubound(3);
    if(sigma<min_sigma || sigma>max_sigma){
        //Pick an initial sigma in-between min and max for sigma
        //This is a rough approximation
        double eta=0.8; //Weight applies to min sigma in weighted average
        sigma = eta*min_sigma + (1-eta)*max_sigma;
    }    
    if(bg <= 0) bg = std::max(1.0, 0.75*im.min());
    if(I <= 0)  I = std::max(1.0, arma::sum(im) - bg*size);
    return make_stencil(ParamT{x_pos,  I, bg, sigma});
}

} /* namespace mappel */
