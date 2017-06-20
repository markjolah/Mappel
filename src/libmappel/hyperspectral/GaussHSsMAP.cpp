/** @file GaussHSsMAP.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-28-2014
 * @brief The class definition and template Specializations for GaussHSsMAP
 */
#include <algorithm>

#include "GaussHSsMAP.h"

const std::vector<std::string>
GaussHSsMAP::param_names={ "x", "y", "lambda", "I", "bg", "sigma", "sigmaL" };

const std::vector<std::string> GaussHSsMAP::hyperparameter_names=
    { "beta_pos", "beta_L", "mean_I", "kappa_I", "mean_bg", "kappa_bg", "alpha_sigma",
      "mean_sigmaL", "xi_sigmaL"};

GaussHSsMAP::GaussHSsMAP(const IVecT &size,const VecT &sigma)
     : PointEmitterHSModel(7, size, sigma),
      sigma_dist(ParetoRNG(alpha_sigma, 1.0)),
      sigmaL_dist(NormalRNG(mean_sigmaL,xi_sigmaL)),
      log_prior_sigma_const(log_prior_pareto_const(alpha_sigma,1.0)),
      log_prior_sigmaL_const(log_prior_normal_const(xi_sigmaL))
{
    /* Precompute log prior constants */
    log_prior_const+=log_prior_sigma_const+log_prior_sigmaL_const;

    /* Initialize MCMC step sizes */
    num_candidate_sampling_phases=4;
    candidate_eta_sigma=1.0*candidate_sample_dist_ratio;
    candidate_eta_sigmaL=mean_sigmaL*candidate_sample_dist_ratio;

    /* Stencil setup for Heuristic estimation */
    stencil_sigmas={1.0,1.3,1.6,2.0,2.5};
    stencil_sigmaLs={std::max(mean_sigmaL/2,mean_sigmaL-1.5*xi_sigmaL), mean_sigmaL, mean_sigmaL+1.5*xi_sigmaL};

    gaussian_stencils=VecFieldT(stencil_sigmas.n_elem,2);
    gaussian_Lstencils=VecFieldT(stencil_sigmaLs.n_elem);
    for(unsigned i=0; i<stencil_sigmas.n_elem; i++){
        gaussian_stencils(i,0)=make_gaussian_stencil(size(0),psf_sigma(0)*stencil_sigmas(i));
        gaussian_stencils(i,1)=make_gaussian_stencil(size(1),psf_sigma(1)*stencil_sigmas(i));
    }
    for(unsigned i=0; i<stencil_sigmaLs.n_elem; i++){
        gaussian_Lstencils(i)=make_gaussian_stencil(size(2),stencil_sigmaLs(i));
    }
}

GaussHSsMAP::Stencil::Stencil(const GaussHSsMAP &model_,
                             const GaussHSsMAP::ParamT &theta,
                             bool _compute_derivatives)
    : model(&model_), theta(theta)
{
    dx=make_d_stencil(size(0), x());
    dy=make_d_stencil(size(1), y());
    dL=make_d_stencil(size(2), lambda());
    X=make_X_stencil(size(0), dx, sigmaX());
    Y=make_X_stencil(size(1), dy, sigmaY());
    L=make_X_stencil(size(2), dL, sigmaL());
    if(_compute_derivatives) compute_derivatives();
}

void GaussHSsMAP::Stencil::compute_derivatives()
{
    if(derivatives_computed) return;
    derivatives_computed=true;
    Gx=make_G_stencil(size(0), dx, sigmaX());
    Gy=make_G_stencil(size(1), dy, sigmaY());
    GL=make_G_stencil(size(2), dL, sigmaL());
    DX=make_DX_stencil(size(0), Gx, sigmaX());
    DY=make_DX_stencil(size(1), Gy, sigmaY());
    DL=make_DX_stencil(size(2), GL, sigmaL());
    DXS=make_DXS_stencil(size(0), dx, Gx, sigmaX());
    DYS=make_DXS_stencil(size(1), dy, Gy, sigmaY());
    DLS=make_DXS_stencil(size(2), dL, GL, sigmaL());
    DXS2=make_DXS2_stencil(size(0), dx, Gx, DXS, sigmaX());
    DYS2=make_DXS2_stencil(size(1), dy, Gy, DYS, sigmaY());
    DLS2=make_DXS2_stencil(size(2), dL, GL, DLS, sigmaL());
    DXSX=make_DXSX_stencil(size(0), dx, Gx, DX, sigmaX());
    DYSY=make_DXSX_stencil(size(1), dy, Gy, DY, sigmaY());
    DLSL=make_DXSX_stencil(size(2), dL, GL, DL, sigmaL());
}

std::ostream& operator<<(std::ostream &out, const GaussHSsMAP::Stencil &s)
{
    int w=18;
    char str[64];
    print_vec_row(out,s.theta,"Theta:",w,TERM_WHITE);
    print_vec_row(out,s.dx,"dx:",w,TERM_CYAN);
    print_vec_row(out,s.dy,"dy:",w,TERM_CYAN);
    print_vec_row(out,s.dL,"dL:",w,TERM_CYAN);
    snprintf(str, 64, "sigmaX:%.3f X:",s.sigmaX());
    print_vec_row(out,s.X,str,w,TERM_CYAN);
    snprintf(str, 64, "sigmaY:%.3f Y:",s.sigmaY());
    print_vec_row(out,s.Y,str,w,TERM_CYAN);
    snprintf(str, 64, "sigmaL:%.3f L:",s.sigmaL());
    print_vec_row(out,s.L,str,w,TERM_CYAN);
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
        print_vec_row(out,s.DXS2,"DXS2:",w,TERM_DIM_BLUE);
        print_vec_row(out,s.DYS2,"DYS2:",w,TERM_DIM_BLUE);
        print_vec_row(out,s.DLS2,"DLS2:",w,TERM_DIM_BLUE);
        print_vec_row(out,s.DXSX,"DXSX:",w,TERM_DIM_BLUE);
        print_vec_row(out,s.DYSY,"DYSY:",w,TERM_DIM_BLUE);
        print_vec_row(out,s.DLSL,"DLSL:",w,TERM_DIM_BLUE);
    }
    return out;
}

GaussHSsMAP::StatsT GaussHSsMAP::get_stats() const
{
    auto stats=PointEmitterHSModel::get_stats();
    stats["hyperparameter.Beta_pos"]=beta_pos;
    stats["hyperparameter.Beta_L"]=beta_L;
    stats["hyperparameter.Mean_I"]=mean_I;
    stats["hyperparameter.Kappa_I"]=kappa_I;
    stats["hyperparameter.Mean_bg"]=mean_bg;
    stats["hyperparameter.Kappa_bg"]=kappa_bg;
    stats["hyperparameter.Alpha_sigma"]=kappa_bg;
    stats["hyperparameter.Mean_sigmaL"]=mean_sigmaL;
    stats["hyperparameter.Xi_sigmaL"]=xi_sigmaL;
    stats["candidate.etaSigma"]=candidate_eta_sigma;
    stats["candidate.etaSigmaL"]=candidate_eta_sigmaL;
    return stats;
}

void GaussHSsMAP::set_hyperparameters(const VecT &hyperparameters)
{
    // Params are {beta_pos, mean_I, kappa_I, mean_bg, kappa_bg}
    beta_pos=hyperparameters(0);
    beta_L=hyperparameters(1);
    mean_I=hyperparameters(2);
    kappa_I=hyperparameters(3);
    mean_bg=hyperparameters(4);
    kappa_bg=hyperparameters(5);
    alpha_sigma=hyperparameters(6);
    mean_sigmaL=hyperparameters(7);
    xi_sigmaL=hyperparameters(8);
    log_prior_pos_const=log_prior_beta_const(beta_pos);
    log_prior_L_const=log_prior_beta_const(beta_L);
    log_prior_I_const=log_prior_gamma_const(kappa_I,mean_I);
    log_prior_bg_const=log_prior_gamma_const(kappa_bg,mean_bg);
//     log_prior_sigma_const=log_prior_pareto_const(alpha_sigma,1.0);
    log_prior_sigma_const=0;
    log_prior_sigmaL_const=log_prior_normal_const(xi_sigmaL);
    log_prior_const=2*log_prior_pos_const+log_prior_L_const+log_prior_I_const+
                    log_prior_bg_const+log_prior_sigma_const+log_prior_sigmaL_const;
    //Reset distributions
    pos_dist.set_params(beta_pos,beta_pos);
    L_dist.set_params(beta_L,beta_L);
    I_dist.kappa(kappa_I);
    I_dist.theta(mean_I/kappa_I);
    bg_dist.kappa(mean_bg);
    bg_dist.theta(mean_bg/kappa_bg);
    sigma_dist.gamma(alpha_sigma); //gamma=alpha for trng's powerlaw dist
    sigmaL_dist.mu(mean_sigmaL);
    sigmaL_dist.sigma(xi_sigmaL);
}

void
GaussHSsMAP::pixel_hess_update(int i, int j, int k, const Stencil &s, double dm_ratio_m1,
                                double dmm_ratio, ParamT &grad, ParamMatT &hess) const
{
    /* Caclulate pixel derivative */
    auto pgrad=make_param();
    pixel_grad(i,j,k,s,pgrad);
    double I=s.I();
    /* Update grad */
    grad+=dm_ratio_m1*pgrad;
    /* Update hess */
    hess(0,0)+=dm_ratio_m1 * I/s.sigmaX() * s.DXS(i) * s.Y(j) * s.L(k);
    hess(1,1)+=dm_ratio_m1 * I/s.sigmaY() * s.X(i) * s.DYS(j) * s.L(k);
    hess(2,2)+=dm_ratio_m1 * I/s.sigmaL() * s.X(i) * s.Y(j) * s.DLS(k);
    hess(5,5)+=dm_ratio_m1 * I * s.L(k) * (s.X(i)*s.DYS2(j) + 2.*s.DXS(i)*s.DYS(j) + s.Y(j)*s.DXS2(i));
    hess(6,6)+=dm_ratio_m1 * I * s.X(i) * s.Y(j) * s.DLS2(k);

    hess(0,1)+=dm_ratio_m1 * I * s.DX(i) * s.DY(j) * s.L(k);
    hess(0,2)+=dm_ratio_m1 * I * s.DX(i) * s.Y(j) * s.DL(k);
    hess(1,2)+=dm_ratio_m1 * I * s.X(i) * s.DY(j) * s.DL(k);
    
    hess(0,5)+=dm_ratio_m1 * I * s.L(k)  * (s.Y(j)*s.DXSX(i) + s.DX(i)*s.DYS(j));
    hess(1,5)+=dm_ratio_m1 * I * s.L(k)  * (s.DY(j)*s.DXS(i) + s.X(i)*s.DYSY(j));
    hess(2,5)+=dm_ratio_m1 * I * s.DL(k) * (s.Y(j)*s.DXS(i)  + s.X(i)*s.DYS(j));

    hess(0,6)+=dm_ratio_m1 * I * s.DX(i) * s.Y(j) * s.DLS(k);
    hess(1,6)+=dm_ratio_m1 * I * s.X(i) * s.DY(j) * s.DLS(k);
    hess(2,6)+=dm_ratio_m1 * I * s.X(i) * s.Y(j) * s.DLSL(k);


    hess(0,3)+=dm_ratio_m1 * pgrad(0) / I;
    hess(1,3)+=dm_ratio_m1 * pgrad(1) / I;
    hess(2,3)+=dm_ratio_m1 * pgrad(2) / I;
    hess(3,5)+=dm_ratio_m1 * pgrad(5) / I;
    hess(3,6)+=dm_ratio_m1 * pgrad(6) / I;

    hess(5,6)+=dm_ratio_m1 * I * s.DLS(k) * (s.X(i)  * s.DYS(j)  + s.Y(j)  * s.DXS(i));

    //This is the pixel-gradient dependenent part of the hessian
    for(int c=0; c<(int)hess.n_cols; c++) for(int r=0; r<=c; r++)
        hess(r,c) -= dmm_ratio * pgrad(r) * pgrad(c);
}


GaussHSsMAP::Stencil
GaussHSsMAP::initial_theta_estimate(const ImageT &im, const ParamVecT &theta_init) const
{
    int pos[3];
    double min_bg;
    Stencil theta;
    VecFieldT field(3);
    Stencil theta_max;
    VecT eff_psf_sigma(3);
    double rllh_max=-INFINITY;
    bool found_new_max=true;
    for(unsigned i=0; i<stencil_sigmas.n_elem; i++) {
        if(not found_new_max) break;
        found_new_max=false;
        double sigma=stencil_sigmas(i);
        eff_psf_sigma(0)=sigma*psf_sigma(0);
        eff_psf_sigma(1)=sigma*psf_sigma(1);
        field(0)=gaussian_stencils(i,0);
        field(1)=gaussian_stencils(i,1);
        for(unsigned k=0; k<stencil_sigmaLs.n_elem; k++){
            double sigmaL=stencil_sigmaLs(k);
            eff_psf_sigma(2)=sigmaL;
            field(2)=gaussian_Lstencils(k);
            if(i+k==0){
                estimate_gaussian_3Dmax(im, field, pos, min_bg);
                refine_gaussian_3Dmax(im, field, pos);
            }
            auto unit_im=unit_model_HS_image(size,pos,eff_psf_sigma(0),eff_psf_sigma(1),eff_psf_sigma(2));
            double bg=estimate_background(im, unit_im);
            double I= estimate_intensity(im, unit_im,bg);
            auto theta=make_stencil(pos[0]+.5,pos[1]+.5,pos[2]+.5,I,bg, sigma, sigmaL);
            double rllh=relative_log_likelihood(*this, im, theta);
            if(rllh>rllh_max) {
                theta_max=theta;
                rllh_max=rllh;
                found_new_max=true;
            }
        }
    }
    theta_max.compute_derivatives();
    return theta_max;
}


double GaussHSsMAP::prior_relative_log_likelihood(const Stencil &s) const
{
    double xrllh=rllh_beta_prior(beta_pos, s.x(), size(0));
    double yrllh=rllh_beta_prior(beta_pos, s.y(), size(1));
    double Lrllh=rllh_beta_prior(beta_L, s.lambda(), size(2));
    double Irllh=rllh_gamma_prior(kappa_I, mean_I, s.I());
    double bgrllh=rllh_gamma_prior(kappa_bg, mean_bg, s.bg());
    double sigmallh=0; // temp fix
//     double sigmallh=rllh_pareto_prior(alpha_sigma, s.sigma());
    double sigmaLllh=rllh_normal_prior(mean_sigmaL, xi_sigmaL, s.sigmaL());
    return xrllh+yrllh+Lrllh+Irllh+bgrllh+sigmallh+sigmaLllh;
}

GaussHSsMAP::ParamT
GaussHSsMAP::prior_grad(const Stencil &s) const
{
    ParamT grad=make_param();
    grad(0)=beta_prior_grad(beta_pos, s.x(), size(0));
    grad(1)=beta_prior_grad(beta_pos, s.y(), size(1));
    grad(2)=beta_prior_grad(beta_L, s.lambda(), size(2));
    grad(3)=gamma_prior_grad(kappa_I, mean_I, s.I());
    grad(4)=gamma_prior_grad(kappa_bg, mean_bg, s.bg());
//     grad(5)=pareto_prior_grad(alpha_sigma, s.sigma());
    grad(5) = 0;
    grad(5)=pareto_prior_grad(alpha_sigma, s.sigma());
    grad(6)=normal_prior_grad(mean_sigmaL, xi_sigmaL, s.sigmaL());
    return grad;
}

GaussHSsMAP::ParamT
GaussHSsMAP::prior_grad2(const Stencil &s) const
{
    ParamT grad2=make_param();
    grad2(0)=beta_prior_grad2(beta_pos, s.x(), size(0));
    grad2(1)=beta_prior_grad2(beta_pos, s.y(), size(1));
    grad2(2)=beta_prior_grad2(beta_L, s.lambda(), size(2));
    grad2(3)=gamma_prior_grad2(kappa_I, s.I());
    grad2(4)=gamma_prior_grad2(kappa_bg, s.bg());
//     grad2(5)=pareto_prior_grad2(alpha_sigma, s.sigma());
    grad2(5)=0;
    grad2(6)=normal_prior_grad2(xi_sigmaL);
    return grad2;
}

GaussHSsMAP::ParamT
GaussHSsMAP::prior_cr_lower_bound(const Stencil &s) const
{
    //TODO complete these calculations
    ParamT pcrlb=make_param();
    pcrlb.zeros();
    return pcrlb;
}


void GaussHSsMAP::sample_candidate_theta(int sample_index, RNG &rng, ParamT &candidate_theta, double scale) const
{
    int phase=sample_index%num_candidate_sampling_phases;
    switch(phase) {
        case 0:  //change x,y, L
            candidate_theta(0)+=generate_normal(rng,0.0,candidate_eta_x*scale);
            candidate_theta(1)+=generate_normal(rng,0.0,candidate_eta_y*scale);
            candidate_theta(2)+=generate_normal(rng,0.0,candidate_eta_L*scale);
            break;
        case 1: //change sigma, I
            candidate_theta(3)+=generate_normal(rng,0.0,candidate_eta_I*scale);
            candidate_theta(5)+=generate_normal(rng,0.0,candidate_eta_sigma*scale);
            break;
        case 2: //change sigmaL, I
            candidate_theta(3)+=generate_normal(rng,0.0,candidate_eta_I*scale);
            candidate_theta(6)+=generate_normal(rng,0.0,candidate_eta_sigmaL*scale);
            break;
        case 3: //change I, bg
            candidate_theta(3)+=generate_normal(rng,0.0,candidate_eta_I*scale);
            candidate_theta(4)+=generate_normal(rng,0.0,candidate_eta_bg*scale);
    }
}

