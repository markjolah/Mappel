/** @file BlinkHSsMAP.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class definition and template Specializations for BlinkHSsMAP
 */
#include <algorithm>

#include "BlinkHSsMAP.h"

/* Constant model estimator names: These are the estimator names we have defined for this class */
const std::vector<std::string> 
BlinkHSsMAP::hyperparameter_names({
    "beta_pos", "mean_I", "kappa_I", "mean_bg", "kappa_bg", "alpha_sigma", "beta_D0", "beta_D1" });


BlinkHSsMAP::BlinkHSsMAP(const IVecT &size,const VecT &sigma)
    : PointEmitterHSModel(7+size(0),size,sigma),
      BlinkModel(candidate_sample_dist_ratio),
      sigma_dist(ParetoRNG(alpha_sigma, 1.0)),
      sigmaL_dist(NormalRNG(mean_sigmaL,xi_sigmaL)),
      log_prior_sigma_const(log_prior_pareto_const(alpha_sigma, 1.0)),
      log_prior_sigmaL_const(log_prior_normal_const(xi_sigmaL)),
      param_names(std::vector<std::string>(7+size(0)))
{
    /* Precompute log prior constants */
    log_prior_const+=log_prior_sigma_const+log_prior_sigmaL_const+size(0)*log_prior_D_const;

    /* Fill out parameter names */
    param_names[0]="x";
    param_names[1]="y";
    param_names[2]="L";
    param_names[3]="I";
    param_names[4]="bg";
    param_names[5]="sigma";
    param_names[6]="sigmaL";
    for(int i=0;i<size(0);i++) {
        std::ostringstream stringStream;
        stringStream <<"D"<<i;
        param_names[7+i]= stringStream.str();
    }

    /* Initialize MCMC step sizes */
    num_candidate_sampling_phases=4+size(0);
    candidate_eta_sigma=1.0*candidate_sample_dist_ratio;
    candidate_eta_sigmaL=mean_sigmaL*candidate_sample_dist_ratio;
    

    /* Stencil setup for Heuristic estimation */
    stencil_sigmas={1.0,1.3,1.6,2.0,2.5};
    stencil_sigmaLs={std::max(mean_sigmaL/2,mean_sigmaL-xi_sigmaL), mean_sigmaL, mean_sigmaL+xi_sigmaL};
    gaussian_stencils=VecFieldT(stencil_sigmas.n_elem,2);
    gaussian_Lstencils=VecFieldT(stencil_sigmaLs.n_elem);
    for(unsigned i=0; i<stencil_sigmas.n_elem; i++){
        gaussian_stencils(i,0)=make_gaussian_stencil(size(0),psf_sigma(0)*stencil_sigmas(i));
        gaussian_stencils(i,1)=make_gaussian_stencil(size(1),psf_sigma(1)*stencil_sigmas(i));
    }
    for(unsigned i=0; i<stencil_sigmaLs.n_elem; i++){
        assert(stencil_sigmaLs(i)>0);
        gaussian_Lstencils(i)=make_gaussian_stencil(size(2),stencil_sigmaLs(i));
    }
}

BlinkHSsMAP::Stencil::Stencil(const BlinkHSsMAP &model_,
                              const BlinkHSsMAP::ParamT &theta,
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

void BlinkHSsMAP::Stencil::compute_derivatives()
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

/* BlinkHSsMAP::ModelImage ---  Inner class method definitions */
BlinkHSsMAP::ModelImage::ModelImage(const BlinkHSsMAP &model_,
                                    const BlinkHSsMAP::ImageT &data_im_)
    : model(&model_), data_im(&data_im_),
      model_im(model->make_image()),
      log_model_im(model->make_image())
{
}

void BlinkHSsMAP::ModelImage::set_stencil(const BlinkHSsMAP::ParamT &theta)
{
    stencil=model->make_stencil(theta,false);
    for(int k=0; k<model->size(2); k++) for(int j=0; j<model->size(1); j++) for(int i=0; i<model->size(0); i++) {
        double val=model->model_value(i,j,k,stencil);
        model_im(i,j,k)=val;
        log_model_im(i,j,k)= (val==0.0) ? 0. : log(val);
    }
}



void BlinkHSsMAP::ModelImage::set_duty(int i, double D)
{
    D=restrict_value_range(D, model->prior_epsilon, 1.-model->prior_epsilon);
    stencil.set_duty(i,D);
    int size_L=model->size(2);
    int size_y=model->size(1);

    for(int k=0; k<size_L; k++) for(int j=0; j<size_y; j++) {
        double val=model->model_value(i,j,k,stencil);
        log_model_im(i,j,k)= (val==0.0) ? 0. : log(val);
    }
}


double BlinkHSsMAP::ModelImage::relative_log_likelihood() const
{
    double rllh=0;
    for(int k=0; k<model->size(2); k++) for(int j=0; j<model->size(1); j++) for(int i=0; i<model->size(0); i++) { //Col major ordering for armadillo
        double model_val=model_im(i,j,k);
        double data_val=(*data_im)(i,j,k);
        if(model_val==0.) continue; //Probability here is below machine epsilon
        if(data_val==0.) { //Skip multiplication by zero
            rllh+=-model_val;
            continue;
        }
        double log_model_val=log_model_im(i,j,k);
        rllh+=data_val*log_model_val-model_val;
    }
    double prllh=model->prior_relative_log_likelihood(stencil);
    return rllh+prllh;
}


std::ostream& operator<<(std::ostream &out, const BlinkHSsMAP::Stencil &s)
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

BlinkHSsMAP::StatsT BlinkHSsMAP::get_stats() const
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
    stats["hyperparameter.Beta_D0"]=beta_D0;
    stats["hyperparameter.Beta_D1"]=beta_D1;
    stats["candidate.etaSigma"]=candidate_eta_sigma;
    stats["candidate.etaSigmaL"]=candidate_eta_sigmaL;
    stats["candidate.etaD"]=candidate_eta_D;

    return stats;
}

void BlinkHSsMAP::set_hyperparameters(const VecT &hyperparameters)
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
    beta_D0=hyperparameters(9);
    beta_D1=hyperparameters(10);
    log_prior_pos_const=log_prior_beta_const(beta_pos);
    log_prior_L_const=log_prior_beta_const(beta_L);
    log_prior_I_const=log_prior_gamma_const(kappa_I,mean_I);
    log_prior_bg_const=log_prior_gamma_const(kappa_bg,mean_bg);
    log_prior_sigma_const=log_prior_pareto_const(alpha_sigma,1.0);
    log_prior_sigmaL_const=log_prior_normal_const(xi_sigmaL);
    log_prior_D_const=log_prior_beta2_const(beta_D0,beta_D1);
    log_prior_const=2*log_prior_pos_const+log_prior_L_const+log_prior_I_const+
                    log_prior_bg_const+log_prior_sigma_const+log_prior_sigmaL_const+
                    size(0)*log_prior_D_const;

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
    D_dist.set_params(beta_D1, beta_D0);
}


void
BlinkHSsMAP::pixel_hess_update(int i, int j, int k, const Stencil &s,
                                double dm_ratio_m1, double dmm_ratio, 
                                ParamT &grad, ParamMatT &hess) const
{
    auto pgrad=make_param();
    pixel_grad(i,j,k,s,pgrad);
    double I=s.I();
    double Di=s.D(i);
    double DiI=Di*I;
    /* Update grad */
    grad+=dm_ratio_m1*pgrad;
    //Update Hessian
    //On Diagonal
    hess(0,0)+=dm_ratio_m1 * DiI/s.sigmaX() * s.DXS(i) * s.Y(j) * s.L(k);
    hess(1,1)+=dm_ratio_m1 * DiI/s.sigmaY() * s.X(i) * s.DYS(j) * s.L(k);
    hess(2,2)+=dm_ratio_m1 * DiI/s.sigmaL() * s.X(i) * s.Y(j) * s.DLS(k);
    hess(5,5)+=dm_ratio_m1 * DiI * s.L(k) * (s.X(i)*s.DYS2(j) + 2.*s.DXS(i)*s.DYS(j) + s.Y(j)*s.DXS2(i));
    hess(6,6)+=dm_ratio_m1 * DiI * s.X(i) * s.Y(j) * s.DLS2(k);

    hess(0,1)+=dm_ratio_m1 * DiI * s.DX(i) * s.DY(j) * s.L(k);
    hess(0,2)+=dm_ratio_m1 * DiI * s.DX(i) * s.Y(j) * s.DL(k);
    hess(1,2)+=dm_ratio_m1 * DiI * s.X(i) * s.DY(j) * s.DL(k);

    hess(0,5)+=dm_ratio_m1 * DiI * s.L(k) * (s.Y(j)*s.DXSX(i) + s.DX(i)*s.DYS(j));
    hess(1,5)+=dm_ratio_m1 * DiI * s.L(k) * (s.X(i)*s.DYSY(j) + s.DY(j)*s.DXS(i));
    hess(2,5)+=dm_ratio_m1 * DiI * s.DL(k) * (s.Y(j)*s.DXS(i)  + s.X(i)*s.DYS(j));

    hess(0,6)+=dm_ratio_m1 * DiI * s.DX(i) * s.Y(j) * s.DLS(k);
    hess(1,6)+=dm_ratio_m1 * DiI * s.X(i) * s.DY(j) * s.DLS(k);
    hess(2,6)+=dm_ratio_m1 * DiI * s.X(i) * s.Y(j) * s.DLSL(k);

    hess(0,3)+=dm_ratio_m1 * pgrad(0) / I;
    hess(1,3)+=dm_ratio_m1 * pgrad(1) / I;
    hess(2,3)+=dm_ratio_m1 * pgrad(2) / I;
    hess(3,5)+=dm_ratio_m1 * pgrad(5) / I;
    hess(3,6)+=dm_ratio_m1 * pgrad(6) / I;
    hess(5,6)+=dm_ratio_m1 * DiI * s.DLS(k) * (s.X(i)  * s.DYS(j)  + s.Y(j)  * s.DXS(i));

    //Di terms
    hess(0,7+i)+=dm_ratio_m1 * pgrad(0) / Di; //xDi
    hess(1,7+i)+=dm_ratio_m1 * pgrad(1) / Di; //xDi
    hess(2,7+i)+=dm_ratio_m1 * pgrad(2) / Di; //xDi
    hess(3,7+i)+=dm_ratio_m1 * pgrad(3) / Di; //xDi
    hess(5,7+i)+=dm_ratio_m1 * pgrad(5) / Di; //xDi
    hess(6,7+i)+=dm_ratio_m1 * pgrad(6) / Di; //xDi
    //This is the pixel-gradient dependenent part of the hessian
    for(int c=0; c<(int)hess.n_cols; c++) {
        if(pgrad(c)==0.0) continue;
        for(int r=0; r<=c; r++) {
            if(pgrad(r)==0.0) continue;
            hess(r,c) -= dmm_ratio * pgrad(r) * pgrad(c);
        }
    }
}


BlinkHSsMAP::Stencil
BlinkHSsMAP::initial_theta_estimate(const ImageT &im, const ParamVecT &theta_init) const
{
    int pos[3];
    double min_bg;
    Stencil theta;
    VecFieldT field(3);
    Stencil theta_max;
    double rllh_max=-INFINITY;
    bool found_new_max=true;
    for(unsigned i=0; i<stencil_sigmas.n_elem; i++) {
        if(not found_new_max) break;
        found_new_max=false;
        double sigma=stencil_sigmas(i);
        field(0)=gaussian_stencils(i,0);
        field(1)=gaussian_stencils(i,1);
        for(unsigned k=0; k<stencil_sigmaLs.n_elem; k++){
            double sigmaL=stencil_sigmaLs(k);
            field(2)=gaussian_Lstencils(k);
            if(i+k==0){
                estimate_gaussian_3Dmax(im, field, pos, min_bg);
                refine_gaussian_3Dmax(im, field, pos);
            }
            auto unit_im=unit_model_HS_image(size,pos,sigma*psf_sigma(0),sigma*psf_sigma(1), sigmaL);
            auto duty=estimate_HS_duty_ratios(im-min_bg, unit_im);
            double bg=estimate_background(im, unit_im);
            double I= estimate_intensity(im, unit_im,bg);
            auto theta=make_stencil(pos[0]+.5,pos[1]+.5,pos[2]+.5,I,bg, sigma, sigmaL, duty);
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

double BlinkHSsMAP::prior_relative_log_likelihood(const Stencil &s) const
{
    double xrllh=rllh_beta_prior(beta_pos, s.x(), size(0));
    double yrllh=rllh_beta_prior(beta_pos, s.y(), size(1));
    double Lrllh=rllh_beta_prior(beta_L, s.lambda(), size(2));
    double Irllh=rllh_gamma_prior(kappa_I, mean_I, s.I());
    double bgrllh=rllh_gamma_prior(kappa_bg, mean_bg, s.bg());
    double sigmallh=rllh_pareto_prior(alpha_sigma, s.sigma());
    double sigmaLllh=rllh_normal_prior(mean_sigmaL, xi_sigmaL, s.sigmaL());
    double Dirllh=0.;
    for(int i=0;i<size(0);i++) Dirllh+=rllh_beta2_prior(beta_D0, beta_D1, s.D(i));
    return xrllh+yrllh+Lrllh+Irllh+bgrllh+sigmallh+sigmaLllh+Dirllh;
}

BlinkHSsMAP::ParamT
BlinkHSsMAP::prior_grad(const Stencil &s) const
{
    ParamT grad=make_param();
    grad(0)=beta_prior_grad(beta_pos, s.x(), size(0));
    grad(1)=beta_prior_grad(beta_pos, s.y(), size(1));
    grad(2)=beta_prior_grad(beta_L, s.lambda(), size(2));
    grad(3)=gamma_prior_grad(kappa_I, mean_I, s.I());
    grad(4)=gamma_prior_grad(kappa_bg, mean_bg, s.bg());
    grad(5)=pareto_prior_grad(alpha_sigma, s.sigma());
    grad(6)=normal_prior_grad(mean_sigmaL, xi_sigmaL, s.sigmaL());
    for(int i=0;i<size(0);i++) grad(7+i)=beta2_prior_grad(beta_D0, beta_D1, s.D(i));
    return grad;
}

BlinkHSsMAP::ParamT
BlinkHSsMAP::prior_grad2(const Stencil &s) const
{
    ParamT grad2=make_param();
    grad2(0)=beta_prior_grad2(beta_pos, s.x(), size(0));
    grad2(1)=beta_prior_grad2(beta_pos, s.y(), size(1));
    grad2(2)=beta_prior_grad2(beta_L, s.lambda(),size(2));
    grad2(3)=gamma_prior_grad2(kappa_I, s.I());
    grad2(4)=gamma_prior_grad2(kappa_bg, s.bg());
    grad2(5)=pareto_prior_grad2(alpha_sigma, s.sigma());
    grad2(6)=normal_prior_grad2(xi_sigmaL);
    for(int i=0;i<size(0);i++) grad2(7+i)=beta2_prior_grad2(beta_D0, beta_D1, s.D(i));
    return grad2;
}

BlinkHSsMAP::ParamT
BlinkHSsMAP::prior_cr_lower_bound(const Stencil &s) const
{
    //TODO complete these calculations
    ParamT pcrlb=make_param();
    pcrlb.zeros();
    return pcrlb;
}

void BlinkHSsMAP::sample_candidate_theta(int sample_index, RNG &rng, ParamT &candidate_theta,double scale) const
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
            break;
        default:
            int col=phase-(num_candidate_sampling_phases-size(0));
            candidate_theta(7+col)+=generate_normal(rng,0.0,candidate_eta_D*scale);
    }
}

double BlinkHSsMAP::compute_candidate_rllh(int sample_index, const ImageT &im,
                                           const ParamT &candidate_theta, ModelImage &model_image) const
{
    assert(theta_in_bounds(candidate_theta));

    int phase=sample_index%num_candidate_sampling_phases;
    switch(phase) {
        case 0:
        case 1:
        case 2:
            return relative_log_likelihood(*this, im, candidate_theta);
        case 3:
            model_image.set_stencil(candidate_theta);
            return model_image.relative_log_likelihood();
        default:
            int col=phase-(num_candidate_sampling_phases-size(0));
            model_image.set_duty(col, candidate_theta(7+col));
            return model_image.relative_log_likelihood();
    }
}

/* Template Specializations */
/*
template<>
typename BlinkHSsMAP::ParamT
cr_lower_bound(const BlinkHSsMAP &model, const typename BlinkHSsMAP::Stencil &s)
{
    auto crlb=model.make_param();
    crlb.zeros();
    auto pgrad=model.make_param();
    for(int k=0; k<model.size(2); k++) for(int j=0; j<model.size(1); j++) for(int i=0; i<model.size(0); i++) { //Col major ordering for armadillo
        double model_val=model.model_value(i,j,k,s);
        model.pixel_grad(i,j,k,s,pgrad);
        for(int n=0; n<model.num_params; n++) {
            double g=pgrad(n);
            crlb(n)+=g*g/model_val;
        }
    }
    for(int n=0; n<model.num_params; n++)  crlb(n)= (crlb(n)<1e-6) ? 1e6  : 1./crlb(n);
    return crlb;
}

*/

