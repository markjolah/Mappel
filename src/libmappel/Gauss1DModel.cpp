/** @file Gauss1DModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief The class definition and template Specializations for Gauss1DModel
 */

#include "Gauss1DModel.h"
#include "stencil.h"
#include <PriorHessian/SymmetricBetaDist.h>
#include <PriorHessian/GammaDist.h>

namespace mappel {
using prior_hessian::SymmetricBetaDist;
using prior_hessian::GammaDist;

Gauss1DModel::Gauss1DModel(IdxT size_, double psf_sigma_)
    : ImageFormat1DBase(size_), //Virtual base class call ignored
      psf_sigma(psf_sigma_)
{
    /* Initialize MCMC step sizes */
    mcmc_num_candidate_sampling_phases=2;
    mcmc_candidate_eta_x = size*mcmc_candidate_sample_dist_ratio;
    auto hyperparams = get_hyperparams(); // [beta_x, mean_I, kappa_I, mean_bg, kappa_bg]
    double mean_I = hyperparams(1);
    double mean_bg = hyperparams(3);
    mcmc_candidate_eta_I = mean_I*mcmc_candidate_sample_dist_ratio;
    mcmc_candidate_eta_bg = mean_bg*mcmc_candidate_sample_dist_ratio;
}

Gauss1DModel::CompositeDist Gauss1DModel::make_prior(IdxT size)
{
    return CompositeDist(SymmetricBetaDist(default_pos_beta,0,size,"x"),
                         GammaDist(default_mean_I, default_kappa_I,"I"),
                         GammaDist(default_pixel_mean_bg*size, default_kappa_bg,"bg"));                        
}

Gauss1DModel::CompositeDist Gauss1DModel::make_prior(IdxT size, double beta_x, double mean_I, double kappa_I, double mean_bg, double kappa_bg)
{
    return CompositeDist(SymmetricBetaDist(beta_x,0,size,"x"),
                         GammaDist(mean_I,kappa_I,"I"),
                         GammaDist(mean_bg,kappa_bg,"bg"));
                         
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
    if (!theta_init.is_empty()) {
        x_pos = theta_init(0);
        I = theta_init(1);
        bg = theta_init(2);
    }
    if(x_pos<=0 || x_pos>size){ //Invalid position, estimate it as maximum column
        x_pos = im.index_max()+0.5;
    } 
    if (I<=0 || bg<=0) {
        bg = 0.75*im.min();
        I = arma::sum(im)-std::min(0.3,bg*size);
    }
    return make_stencil(ParamT{x_pos,  I, bg});
}

void 
Gauss1DModel::sample_mcmc_candidate_theta(IdxT sample_index, ParamT &mcmc_candidate_theta, double scale) const
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
