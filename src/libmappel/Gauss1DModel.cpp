/** @file Gauss1DModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief The class definition and template Specializations for Gauss1DModel
 */

#include "Gauss1DModel.h"
#include "stencil.h"

namespace mappel {

Gauss1DModel::Gauss1DModel(IdxT size, double psf_sigma)
    : ImageFormat1DBase(size), //Virtual base class call ignored
      psf_sigma(psf_sigma)
{
    /* Initialize MCMC step sizes */
    mcmc_num_candidate_sampling_phases=2;
    mcmc_candidate_eta_x = size*mcmc_candidate_sample_dist_ratio;
    mcmc_candidate_eta_I = find_hyperparam("mean_I",default_mean_I)*mcmc_candidate_sample_dist_ratio;
    mcmc_candidate_eta_bg = find_hyperparam("mean_bg",default_pixel_mean_bg)*mcmc_candidate_sample_dist_ratio;
}

CompositeDist 
Gauss1DModel::make_default_prior(IdxT size)
{
    return CompositeDist(make_prior_component_position_beta("x",size),
                         make_prior_component_intensity("I"),
                         make_prior_component_intensity("bg",default_pixel_mean_bg*size)); //bg is summed over the other dimension leading to larger mean per 1D 'pixel'
}

CompositeDist 
Gauss1DModel::make_prior_beta_position(IdxT size, double beta_xpos, 
                                       double mean_I, double kappa_I, 
                                       double mean_bg, double kappa_bg)
{
    return CompositeDist(make_prior_component_position_beta("x",size,beta_xpos),
                         make_prior_component_intensity("I",mean_I,kappa_I),
                         make_prior_component_intensity("bg",mean_bg, kappa_bg));
}

CompositeDist 
Gauss1DModel::make_prior_normal_position(IdxT size, double sigma_xpos, 
                                       double mean_I, double kappa_I, 
                                       double mean_bg, double kappa_bg)
{
    return CompositeDist(make_prior_component_position_normal("x",size, sigma_xpos),
                         make_prior_component_intensity("I",mean_I,kappa_I),
                         make_prior_component_intensity("bg",mean_bg, kappa_bg));
}

void Gauss1DModel::set_psf_sigma(double new_psf_sigma)
{ 
    if(new_psf_sigma<global_min_psf_sigma || 
       new_psf_sigma>global_max_psf_sigma || !std::isfinite(new_psf_sigma)) {
        std::ostringstream msg;
        msg<<"Invalid psf_sigma: "<<new_psf_sigma<<" Valid psf_sigma range:["
            <<global_min_psf_sigma<<","<<global_max_psf_sigma<<"]";
        throw ParameterValueError(msg.str());
    }
    psf_sigma = new_psf_sigma;
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
    stats.insert(im_stats.begin(), im_stats.end());
    return stats;
}


/** @brief pixel derivative inner loop calculations.
 */
void Gauss1DModel::pixel_hess_update(IdxT i, const Stencil &s, double dm_ratio_m1, double dmm_ratio, ParamT &grad, MatT &hess) const
{
    auto pgrad=make_param();
    pixel_grad(i,s,pgrad);
    double I = s.I();
    /* Update grad */
    grad += dm_ratio_m1*pgrad;
    /* Update hess */
    hess(0,0) += dm_ratio_m1 * I/psf_sigma * s.DXS(i);
    hess(0,1) += dm_ratio_m1 * pgrad(0) / I; 
    //This is the pixel-gradient dependent part of the hessian
    for(IdxT c=0; c<(IdxT)hess.n_cols; c++) for(IdxT r=0; r<=c; r++)
        hess(r,c) -= dmm_ratio * pgrad(r) * pgrad(c);
}

Gauss1DModel::Stencil 
Gauss1DModel::initial_theta_estimate(const ImageT &im, const ParamT &theta_init) const
{
    double x_pos=0, I=0, bg=0;
    if(theta_init.n_elem == num_params) {
        x_pos = theta_init(0);
        I = theta_init(1);
        bg = theta_init(2);
    }
    if(x_pos <= 0 || x_pos > size) x_pos = im.index_max()+0.5;
    if(bg <= 0) bg = std::max(1.0, 0.75*im.min());
    if(I <= 0)  I = std::max(1.0, arma::sum(im) - bg*size);
    return make_stencil(ParamT{x_pos,  I, bg});
}

void 
Gauss1DModel::sample_mcmc_candidate_theta(IdxT sample_index, ParamT &mcmc_candidate_theta, double scale)
{
    IdxT phase = sample_index%mcmc_num_candidate_sampling_phases;
    switch(phase) {
        case 0:  //change pos
            mcmc_candidate_theta(0) += rng_manager.randn()*mcmc_candidate_eta_x*scale;
            break;
        case 1: //change I, bg
            mcmc_candidate_theta(1) += rng_manager.randn()*mcmc_candidate_eta_I*scale;
            mcmc_candidate_theta(2) += rng_manager.randn()*mcmc_candidate_eta_bg*scale;
    }
}

} /* namespace mappel */
