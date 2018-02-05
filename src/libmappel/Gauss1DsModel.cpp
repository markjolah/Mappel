/** @file Gauss1DsModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief The class definition and template Specializations for Gauss1DsModel
 */

#include "Gauss1DsModel.h"
#include "stencil.h"

namespace mappel {

Gauss1DsModel::Gauss1DsModel(IdxT size_)
    : ImageFormat1DBase(size_)
{
    /* Initialize MCMC step sizes */
    mcmc_num_candidate_sampling_phases = 3;
    mcmc_candidate_eta_x = size*mcmc_candidate_sample_dist_ratio;
    mcmc_candidate_eta_I = find_hyperparam("mean_I",default_mean_I)*mcmc_candidate_sample_dist_ratio;
    mcmc_candidate_eta_bg = find_hyperparam("mean_bg",default_pixel_mean_bg)*mcmc_candidate_sample_dist_ratio;
    mcmc_candidate_eta_sigma = 1.0*mcmc_candidate_sample_dist_ratio;    
}

Gauss1DsModel::CompositeDist Gauss1DsModel::make_default_prior(IdxT size, double min_sigma, double max_sigma)
{
    return CompositeDist(make_prior_component_position_beta("x",size),
                         make_prior_component_intensity("I"),
                         make_prior_component_intensity("bg",default_pixel_mean_bg*size), //bg is summed over the other dimension leading to larger mean per 1D 'pixel'
                         make_prior_component_sigma("sigma",min_sigma,max_sigma));
}

Gauss1DsModel::CompositeDist 
Gauss1DsModel::make_prior_beta_position(IdxT size, double beta_xpos, 
                                       double mean_I, double kappa_I, 
                                       double mean_bg, double kappa_bg,
                                       double min_sigma, double max_sigma)
{
    return CompositeDist(make_prior_component_position_beta("x",size,beta_xpos),
                         make_prior_component_intensity("I",mean_I,kappa_I),
                         make_prior_component_intensity("bg",mean_bg, kappa_bg));
}

Gauss1DsModel::CompositeDist 
Gauss1DsModel::make_prior_normal_position(IdxT size, double sigma_xpos, 
                                       double mean_I, double kappa_I, 
                                       double mean_bg, double kappa_bg,
                                       double min_sigma, double max_sigma)
{
    return CompositeDist(make_prior_component_position_normal("x",size, sigma_xpos),
                         make_prior_component_intensity("I",mean_I,kappa_I),
                         make_prior_component_intensity("bg",mean_bg, kappa_bg));
}

void Gauss1DsModel::set_min_sigma(double min_sigma)
{
    auto lb = prior.lbound();
    lb(3) = min_sigma;
    set_lbound(lb);
}

void Gauss1DsModel::set_max_sigma(double max_sigma)
{
    auto ub = prior.ubound();
    ub(3) = max_sigma;
    set_ubound(ub);
}

void Gauss1DsModel::set_min_sigma(const VecT &min_sigma)
{
    auto lb = prior.lbound();
    lb(3) = min_sigma(0);
    set_lbound(lb);
}

void Gauss1DsModel::set_max_sigma(const VecT &max_sigma)
{
    auto ub = prior.ubound();
    ub(3) = max_sigma(0);
    set_ubound(ub);
}



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
    stats.insert(im_stats.begin(), im_stats.end());
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


void Gauss1DsModel::sample_mcmc_candidate_theta(IdxT sample_index, ParamT &mcmc_candidate_theta, double scale)
{
    int phase = sample_index%mcmc_num_candidate_sampling_phases;
    switch(phase) {
        case 0:  //change pos
            mcmc_candidate_theta(0) += rng_manager.randn()*mcmc_candidate_eta_x*scale;
            break;
        case 1: //change I, sigma
            mcmc_candidate_theta(1) += rng_manager.randn()*mcmc_candidate_eta_I*scale;
            mcmc_candidate_theta(3) += rng_manager.randn()*mcmc_candidate_eta_sigma*scale;
            break;
        case 2: //change I, bg
            mcmc_candidate_theta(1) += rng_manager.randn()*mcmc_candidate_eta_I*scale;
            mcmc_candidate_theta(2) += rng_manager.randn()*mcmc_candidate_eta_bg*scale;
    }
}



} /* namespace mappel */
