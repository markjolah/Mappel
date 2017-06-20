/** @file GaussHSMAP.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-28-2014
 * @brief The class definition and template Specializations for GaussHSMAP
 */
#include <algorithm>

#include "GaussHSMAP.h"

const std::vector<std::string>
GaussHSMAP::param_names={ "x", "y", "lambda", "I", "bg" };

const std::vector<std::string> GaussHSMAP::hyperparameter_names=
{ "beta_pos", "beta_L", "mean_I", "kappa_I", "mean_bg", "kappa_bg"};

    GaussHSMAP::GaussHSMAP(const IVecT &size,const VecT &sigma)
    : PointEmitterHSModel(5, size, sigma),
      gaussian_stencils(VecFieldT(ndim,1))
{
    /* Initialize MCMC step sizes */
    num_candidate_sampling_phases=2;

    gaussian_stencils(0,0)=make_gaussian_stencil(size(0),psf_sigma(0));
    gaussian_stencils(1,0)=make_gaussian_stencil(size(1),psf_sigma(1));
    gaussian_stencils(2,0)=make_gaussian_stencil(size(2),mean_sigmaL);
}

GaussHSMAP::Stencil::Stencil(const GaussHSMAP &model_,
                             const GaussHSMAP::ParamT &theta,
                             bool _compute_derivatives)
    : model(&model_), theta(theta)
{
    const IVecT &size=model->size;
    const VecT &psf_sigma=model->psf_sigma;
    dx=make_d_stencil(size(0), x());
    dy=make_d_stencil(size(1), y());
    dL=make_d_stencil(size(2), lambda());
    X=make_X_stencil(size(0), dx, psf_sigma(0));
    Y=make_X_stencil(size(1), dy, psf_sigma(1));
    L=make_X_stencil(size(2), dL, model->mean_sigmaL);
    if(_compute_derivatives) compute_derivatives();
}

void GaussHSMAP::Stencil::compute_derivatives()
{
    if(derivatives_computed) return;
    derivatives_computed=true;

    const IVecT &size=model->size;
    const VecT &psf_sigma=model->psf_sigma;

    Gx=make_G_stencil(size(0), dx, psf_sigma(0));
    Gy=make_G_stencil(size(1), dy, psf_sigma(1));
    GL=make_G_stencil(size(2), dL, model->mean_sigmaL);
    DX=make_DX_stencil(size(0), Gx, psf_sigma(0));
    DY=make_DX_stencil(size(1), Gy, psf_sigma(1));
    DL=make_DX_stencil(size(2), GL, model->mean_sigmaL);
    DXS=make_DXS_stencil(size(0), dx, Gx, psf_sigma(0));
    DYS=make_DXS_stencil(size(1), dy, Gy, psf_sigma(1));
    DLS=make_DXS_stencil(size(2), dL, GL, model->mean_sigmaL);
}

std::ostream& operator<<(std::ostream &out, const GaussHSMAP::Stencil &s)
{
    int w=8;
    print_vec_row(out,s.theta,"Theta:",w,TERM_WHITE);
    print_vec_row(out,s.dx,"dx:",w,TERM_CYAN);
    print_vec_row(out,s.dy,"dy:",w,TERM_CYAN);
    print_vec_row(out,s.dL,"dL:",w,TERM_CYAN);
    print_vec_row(out,s.X,"X:",w,TERM_CYAN);
    print_vec_row(out,s.Y,"Y:",w,TERM_CYAN);
    print_vec_row(out,s.L,"L:",w,TERM_CYAN);
    if(s.derivatives_computed){
        print_vec_row(out,s.Gx,"Gx:",w,TERM_BLUE);
        print_vec_row(out,s.Gy,"Gy:",w,TERM_BLUE);
        print_vec_row(out,s.GL,"GL:",w,TERM_BLUE);
        print_vec_row(out,s.DX,"DX:",w,TERM_BLUE);
        print_vec_row(out,s.DY,"DY:",w,TERM_BLUE);
        print_vec_row(out,s.DL,"DL:",w,TERM_BLUE);
        print_vec_row(out,s.DXS,"DXS:",w,TERM_BLUE);
        print_vec_row(out,s.DYS,"DYS:",w,TERM_BLUE);
        print_vec_row(out,s.DLS,"DLS:",w,TERM_BLUE);
    }
    return out;
}

GaussHSMAP::StatsT GaussHSMAP::get_stats() const
{
    auto stats=PointEmitterHSModel::get_stats();
    stats["hyperparameter.Beta_pos"]=beta_pos;
    stats["hyperparameter.Beta_L"]=beta_L;
    stats["hyperparameter.Mean_I"]=mean_I;
    stats["hyperparameter.Kappa_I"]=kappa_I;
    stats["hyperparameter.Mean_bg"]=mean_bg;
    stats["hyperparameter.Kappa_bg"]=kappa_bg;
    return stats;
}

void GaussHSMAP::set_hyperparameters(const VecT &hyperparameters)
{
    // Params are {beta_pos, mean_I, kappa_I, mean_bg, kappa_bg, mean_sigmaL}
    beta_pos=hyperparameters(0);
    beta_L=hyperparameters(1);
    mean_I=hyperparameters(2);
    kappa_I=hyperparameters(3);
    mean_bg=hyperparameters(4);
    kappa_bg=hyperparameters(5);
    log_prior_pos_const=log_prior_beta_const(beta_pos);
    log_prior_L_const=log_prior_beta_const(beta_L);
    log_prior_I_const=log_prior_gamma_const(kappa_I,mean_I);
    log_prior_bg_const=log_prior_gamma_const(kappa_bg,mean_bg);
    //Reset distributions
    pos_dist.set_params(beta_pos,beta_pos);
    L_dist.set_params(beta_L,beta_L);
    I_dist.kappa(kappa_I);
    I_dist.theta(mean_I/kappa_I);
    bg_dist.kappa(mean_bg);
    bg_dist.theta(mean_bg/kappa_bg);
}


void
GaussHSMAP::pixel_hess_update(int i, int j, int k, const Stencil &s, double dm_ratio_m1,
                                double dmm_ratio, ParamT &grad, ParamMatT &hess) const
{
    /* Caclulate pixel derivative */
    auto pgrad=make_param();
    pixel_grad(i,j,k,s,pgrad);
    double I=s.I();
    /* Update grad */
    grad+=dm_ratio_m1*pgrad;
    /* Update hess */
    hess(0,0)+=dm_ratio_m1 * I/psf_sigma(0) * s.DXS(i) * s.Y(j) * s.L(k);
    hess(1,1)+=dm_ratio_m1 * I/psf_sigma(1) * s.X(i) * s.DYS(j) * s.L(k);
    hess(2,2)+=dm_ratio_m1 * I/mean_sigmaL * s.X(i) * s.Y(j) * s.DLS(k);
    hess(0,1)+=dm_ratio_m1 * I * s.DX(i) * s.DY(j) * s.L(k);
    hess(0,2)+=dm_ratio_m1 * I * s.DX(i) * s.Y(j) * s.DL(k);
    hess(1,2)+=dm_ratio_m1 * I * s.X(i) * s.DY(j) * s.DL(k);
    hess(0,3)+=dm_ratio_m1 * pgrad(0) / I;
    hess(1,3)+=dm_ratio_m1 * pgrad(1) / I;
    hess(2,3)+=dm_ratio_m1 * pgrad(2) / I;
    //This is the pixel-gradient dependenent part of the hessian
    for(int c=0; c<(int)hess.n_cols; c++) for(int r=0; r<=c; r++)
        hess(r,c) -= dmm_ratio * pgrad(r) * pgrad(c);
}


GaussHSMAP::Stencil
GaussHSMAP::initial_theta_estimate(const ImageT &im, const ParamVecT &theta_init) const
{
    int pos[3];
    double min_bg;
    estimate_gaussian_3Dmax(im, gaussian_stencils, pos, min_bg);
    auto unit_im=unit_model_HS_image(size,pos,psf_sigma(0),psf_sigma(1), mean_sigmaL);
    double bg=estimate_background(im, unit_im);
    double I=estimate_intensity(im, unit_im, bg);
//     std::cout<<"I: "<<I<<" bg:"<<bg<<std::endl;
    return make_stencil(pos[0]+.5,pos[1]+.5,pos[2]+.5,I,bg);
}


double GaussHSMAP::prior_log_likelihood(const Stencil &s) const
{
    double rllh=prior_relative_log_likelihood(s);
    return rllh + 2*log_prior_pos_const + log_prior_L_const + log_prior_I_const + log_prior_bg_const;
}

double GaussHSMAP::prior_relative_log_likelihood(const Stencil &s) const
{
    double xrllh=rllh_beta_prior(beta_pos, s.x(), size(0));
    double yrllh=rllh_beta_prior(beta_pos, s.y(), size(1));
    double Lrllh=rllh_beta_prior(beta_L, s.lambda(), size(2));
    double Irllh=rllh_gamma_prior(kappa_I, mean_I, s.I());
    double bgrllh=rllh_gamma_prior(kappa_bg, mean_bg, s.bg());
//     std::cout<<"PRLLH: "<<xrllh<<", "<<yrllh<<", "<<Lrllh<<", "<<Irllh<<", "<<bgrllh<<std::endl;
    return xrllh+yrllh+Lrllh+Irllh+bgrllh;
}

GaussHSMAP::ParamT
GaussHSMAP::prior_grad(const Stencil &s) const
{
    ParamT grad=make_param();
    grad(0)=beta_prior_grad(beta_pos, s.x(), size(0));
    grad(1)=beta_prior_grad(beta_pos, s.y(), size(1));
    grad(2)=beta_prior_grad(beta_L, s.lambda(), size(2));
    grad(3)=gamma_prior_grad(kappa_I, mean_I, s.I());
    grad(4)=gamma_prior_grad(kappa_bg, mean_bg, s.bg());
    return grad;
}

GaussHSMAP::ParamT
GaussHSMAP::prior_grad2(const Stencil &s) const
{
    ParamT grad2=make_param();
    grad2(0)=beta_prior_grad2(beta_pos, s.x(), size(0));
    grad2(1)=beta_prior_grad2(beta_pos, s.y(), size(1));
    grad2(2)=beta_prior_grad2(beta_L, s.lambda(), size(2));
    grad2(3)=gamma_prior_grad2(kappa_I, s.I());
    grad2(4)=gamma_prior_grad2(kappa_bg, s.bg());
    return grad2;
}

GaussHSMAP::ParamT
GaussHSMAP::prior_cr_lower_bound(const Stencil &s) const
{
    //TODO complete these calculations
    ParamT pcrlb=make_param();
    pcrlb.zeros();
    return pcrlb;
}

void GaussHSMAP::sample_candidate_theta(int sample_index, RNG &rng, ParamT &candidate_theta, double scale) const
{
    int phase=sample_index%num_candidate_sampling_phases;
    switch(phase) {
        case 0:  //change x,y, L
            candidate_theta(0)+=generate_normal(rng,0.0,candidate_eta_x*scale);
            candidate_theta(1)+=generate_normal(rng,0.0,candidate_eta_y*scale);
            candidate_theta(2)+=generate_normal(rng,0.0,candidate_eta_L*scale);
            break;
        case 1: //change I, bg
            candidate_theta(3)+=generate_normal(rng,0.0,candidate_eta_I*scale);
            candidate_theta(4)+=generate_normal(rng,0.0,candidate_eta_bg*scale);
    }
}


