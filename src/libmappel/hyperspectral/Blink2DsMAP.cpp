/** @file Blink2DsMAP.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class definition and template Specializations for Blink2DsMAP
 */
#include <algorithm>

#include "Blink2DsMAP.h"
#include "cGaussMLE/cGaussMLE.h"
#include "cGaussMLE/GaussLib.h"


/* Constant model estimator names: These are the estimator names we have defined for this class */
const std::vector<std::string> 
Blink2DsMAP::hyperparameter_names({ 
    "beta_pos", "mean_I", "kappa_I", "mean_bg", "kappa_bg", "alpha_sigma", "beta_D0", "beta_D1" });


Blink2DsMAP::Blink2DsMAP(const IVecT &size, const VecT &psf_sigma)
    : PointEmitter2DModel(5+size(0),size,psf_sigma),
      BlinkModel(candidate_sample_dist_ratio),
      pos_dist(BetaRNG(beta_pos,beta_pos)),
      I_dist(GammaRNG(kappa_I,mean_I/kappa_I)),
      bg_dist(GammaRNG(kappa_bg,mean_bg/kappa_bg)),
      sigma_dist(ParetoRNG(alpha_sigma, 1.0)),
      log_prior_pos_const(log_prior_beta_const(beta_pos)),
      log_prior_I_const(log_prior_gamma_const(kappa_I,mean_I)),
      log_prior_bg_const(log_prior_gamma_const(kappa_bg,mean_bg)),
      log_prior_sigma_const(log_prior_pareto_const(alpha_sigma,1.0)),
      param_names(std::vector<std::string>(5+size(0))),
      gaussian_Xstencils(arma::mat(2*size(0)-1,stencil_sigmas.n_rows)),
      gaussian_Ystencils(arma::mat(2*size(1)-1,stencil_sigmas.n_rows))
{
    param_names[0]="x";
    param_names[1]="y";
    param_names[2]="I";
    param_names[3]="bg";
    param_names[4]="sigma";

    /* Initialize MCMC step sizes */
    num_candidate_sampling_phases=3+size(0);
    candidate_eta_I=mean_I*candidate_sample_dist_ratio;
    candidate_eta_bg=mean_bg*candidate_sample_dist_ratio;
    candidate_eta_sigma=1.0*candidate_sample_dist_ratio;

    for(int i=0;i<size(0);i++) {
        std::ostringstream stringStream;
        stringStream <<"D"<<i;
        param_names[5+i]= stringStream.str();
    }
    
    for(unsigned i=0;i<stencil_sigmas.n_rows;i++){
        gaussian_Xstencils.col(i)=make_gaussian_stencil(size(0),psf_sigma(0)*stencil_sigmas(i));
        gaussian_Ystencils.col(i)=make_gaussian_stencil(size(1),psf_sigma(1)*stencil_sigmas(i));
    }
}



Blink2DsMAP::Stencil::Stencil(const Blink2DsMAP &model_, const ParamT &theta, bool _compute_derivatives)
      : model(&model_), theta(theta)
{
    int szX=model->size(0);
    int szY=model->size(1);
    dx=make_d_stencil(szX, x());
    dy=make_d_stencil(szY, y());
    X=make_X_stencil(szX, dx, sigmaX());
    Y=make_X_stencil(szY, dy, sigmaY());
    if(_compute_derivatives) compute_derivatives();
}

void Blink2DsMAP::Stencil::compute_derivatives()
{
    if(derivatives_computed) return;
    derivatives_computed=true;
    int szX=model->size(0);
    int szY=model->size(1);
    Gx=make_G_stencil(szX, dx, sigmaX());
    Gy=make_G_stencil(szY, dy, sigmaY());
    DX=make_DX_stencil(szX, Gx, sigmaX());
    DY=make_DX_stencil(szY, Gy, sigmaY());
    DXS=make_DXS_stencil(szX, dx, Gx, sigmaX());
    DYS=make_DXS_stencil(szY, dy, Gy, sigmaY());
    DXS2=make_DXS2_stencil(szX, dx, Gx, DXS, sigmaX());
    DYS2=make_DXS2_stencil(szY, dy, Gy, DYS, sigmaY());
    DXSX=make_DXSX_stencil(szX, dx, Gx, DX, sigmaX());
    DYSY=make_DXSX_stencil(szY, dy, Gy, DY, sigmaY());
}


Blink2DsMAP::ModelImage::ModelImage(const Blink2DsMAP &model_,
                                    const Blink2DsMAP::ImageT &data_im_)
    : model(&model_), data_im(&data_im_),
      model_im(model->make_image()),
      log_model_im(model->make_image())
{
}


void Blink2DsMAP::ModelImage::set_stencil(const Blink2DsMAP::ParamT &theta)
{
    stencil=model->make_stencil(theta,false);
    for(int i=0; i<model->size(0); i++) for(int j=0; j<model->size(1); j++)  {
        double val=model->model_value(i,j,stencil);
        model_im(j,i)=val;
        log_model_im(j,i)= (val==0.0) ? 0. : log(val);
    }
}


void Blink2DsMAP::ModelImage::set_duty(int i, double D)
{
    D=restrict_value_range(D, model->prior_epsilon, 1.-model->prior_epsilon);
    stencil.set_duty(i,D);
    for(int j=0; j<model->size(1); j++) {
        double val=model->model_value(i,j,stencil);
        log_model_im(j,i)= (val==0.0) ? 0. : log(val);
    }
}


double Blink2DsMAP::ModelImage::relative_log_likelihood() const
{
    double rllh=0;
    for(int i=0;i<model->size(0);i++) for(int j=0;j<model->size(1);j++) {  // i=x position=column; j=yposition=row
        double model_val=model_im(j,i);
        double data_val= (*data_im)(j,i);
        if(model_val==0.) continue; //Probability here is below machine epsilon
        if(data_val==0.) { //Skip multiplication by zero
            rllh+=-model_val;
            continue;
        }
        double log_model_val=log_model_im(j,i);
        rllh+=data_val*log_model_val-model_val;
    }
    double prllh=model->prior_relative_log_likelihood(stencil);
    return rllh+prllh;
}


std::ostream& operator<<(std::ostream &out, const Blink2DsMAP::Stencil &s)
{
    int w=8;
    print_vec_row(out,s.theta,"Theta:",w,TERM_WHITE);
    print_vec_row(out,s.dx,"dx:",w,TERM_CYAN);
    print_vec_row(out,s.dy,"dy:",w,TERM_CYAN);
    print_vec_row(out,s.X,"X:",w,TERM_CYAN);
    print_vec_row(out,s.Y,"Y:",w,TERM_CYAN);
    if(s.derivatives_computed){
        print_vec_row(out,s.Gx,"Gx:",w,TERM_BLUE);
        print_vec_row(out,s.Gy,"Gy:",w,TERM_BLUE);
        print_vec_row(out,s.DX,"DX:",w,TERM_BLUE);
        print_vec_row(out,s.DY,"DY:",w,TERM_BLUE);
        print_vec_row(out,s.DXS,"DXS:",w,TERM_BLUE);
        print_vec_row(out,s.DYS,"DYS:",w,TERM_BLUE);
        print_vec_row(out,s.DXS2,"DXS2:",w,TERM_BLUE);
        print_vec_row(out,s.DYS2,"DYS2:",w,TERM_BLUE);
        print_vec_row(out,s.DXSX,"DXSX:",w,TERM_BLUE);
        print_vec_row(out,s.DYSY,"DYSY:",w,TERM_BLUE);
    }
    return out;
}

Blink2DsMAP::StatsT Blink2DsMAP::get_stats() const
{
    auto stats=PointEmitter2DModel::get_stats();
    stats["hyperparameter.Beta_x"]=beta_pos;
    stats["hyperparameter.Mean_I"]=mean_I;
    stats["hyperparameter.Kappa_I"]=kappa_I;
    stats["hyperparameter.Mean_bg"]=mean_bg;
    stats["hyperparameter.Kappa_bg"]=kappa_bg;
    stats["hyperparameter.Alpha_sigma"]=alpha_sigma;
    stats["hyperparameter.Beta_D0"]=beta_D0;
    stats["hyperparameter.Beta_D1"]=beta_D1;
    stats["candidate.etaSigma"]=candidate_eta_sigma;
    stats["candidate.etaD"]=candidate_eta_D;

    return stats;
}


void Blink2DsMAP::set_hyperparameters(const VecT &hyperparameters)
{
    // Params are {beta_pos, mean_I, kappa_I, mean_bg, kappa_bg}
    beta_pos=hyperparameters(0);
    mean_I=hyperparameters(1);
    kappa_I=hyperparameters(2);
    mean_bg=hyperparameters(3);
    kappa_bg=hyperparameters(4);
    alpha_sigma=hyperparameters(5);
    beta_D0=hyperparameters(6);
    beta_D1=hyperparameters(7);
    log_prior_pos_const=log_prior_beta_const(beta_pos);
    log_prior_I_const=log_prior_gamma_const(kappa_I,mean_I);
    log_prior_bg_const=log_prior_gamma_const(kappa_bg,mean_bg);
    log_prior_sigma_const=log_prior_pareto_const(alpha_sigma, 1.0);
    log_prior_D_const=log_prior_beta2_const(beta_D1, beta_D0);
    //Reset distributions
    pos_dist.set_params(beta_pos,beta_pos);
    I_dist.kappa(kappa_I);
    I_dist.theta(mean_I/kappa_I);
    bg_dist.kappa(mean_bg);
    bg_dist.theta(mean_bg/kappa_bg);
    sigma_dist.gamma(alpha_sigma); //gamma=alpha for trng's powerlaw dist
    D_dist.set_params(beta_D1,beta_D0);
}


void
Blink2DsMAP::pixel_hess_update(int i, int j, const Stencil &s, 
                                double dm_ratio_m1, double dmm_ratio, 
                                ParamT &grad, ParamMatT &hess) const
{
    auto pgrad=make_param();
    pixel_grad(i,j,s,pgrad);
    double I=s.I();
    double Di=s.D(i);
    double DiI=Di*I;
    /* Update grad */
    grad+=dm_ratio_m1*pgrad;
    //Update Hessian
    //On Diagonal
    hess(0,0)+=dm_ratio_m1 * DiI/s.sigmaX() * s.DXS(i)*s.Y(j); //xx
    hess(1,1)+=dm_ratio_m1 * DiI/s.sigmaY() * s.DYS(j)*s.X(i); //yy
    hess(4,4)+=dm_ratio_m1 * DiI*(s.X(i)*s.DYS2(j) + 2*s.DXS(i)*s.DYS(j) + s.Y(j)*s.DXS2(i)); //SS
    //Off Diagonal
    hess(0,1)+=dm_ratio_m1 * DiI * s.DX(i)*s.DY(j); //xy
    hess(0,4)+=dm_ratio_m1 * DiI * (s.Y(j)*s.DXSX(i) + s.DX(i)*s.DYS(j)); //xS
    hess(1,4)+=dm_ratio_m1 * DiI * (s.X(i)*s.DYSY(j) + s.DY(j)*s.DXS(i)); //yS
    //Off Diagonal with respect to I
    hess(0,2)+=dm_ratio_m1 * pgrad(0) / I; //xI
    hess(1,2)+=dm_ratio_m1 * pgrad(1) / I; //yI
    hess(2,4)+=dm_ratio_m1 * pgrad(4) / I; //IS
    //Di terms
    hess(0,5+i)+=dm_ratio_m1 * pgrad(0) / Di; //xDi
    hess(1,5+i)+=dm_ratio_m1 * pgrad(1) / Di; //xDi
    hess(2,5+i)+=dm_ratio_m1 * pgrad(2) / Di; //xDi
    hess(4,5+i)+=dm_ratio_m1 * pgrad(4) / Di; //xDi
    //This is the pixel-gradient dependenent part of the hessian
    for(int c=0; c<static_cast<int>(hess.n_cols); c++) {
        if(pgrad(c)==0.0) continue;
        for(int r=0; r<=c; r++) {
            if(pgrad(r)==0.0) continue;
            hess(r,c) -= dmm_ratio * pgrad(r) * pgrad(c);
        }
    }
}


Blink2DsMAP::Stencil
Blink2DsMAP::initial_theta_estimate(const ImageT &im, const ParamT &theta_init) const
{
    int pos[2];
    double min_bg;
    Stencil theta;
    double rllh=-INFINITY;
    for(unsigned n=0; n<gaussian_Xstencils.n_cols; n++) {
        double sigma=stencil_sigmas(n);
        if(n==0){
            estimate_gaussian_2Dmax(im, gaussian_Xstencils.col(n),gaussian_Ystencils.col(n), pos,min_bg);
        } else {
            refine_gaussian_2Dmax(im, gaussian_Xstencils.col(n),gaussian_Ystencils.col(n), pos);
        }
        auto unit_im=unit_model_image(size,pos,stencil_sigmas(n)*psf_sigma);
        auto duty=estimate_duty_ratios(im-min_bg, unit_im);
        unit_im.each_row()%=duty.t();
        double bg=estimate_background(im, unit_im, min_bg);
        double I= estimate_intensity(im, unit_im, bg);
        auto ntheta=make_stencil(pos[0]+.5,pos[1]+.5,I,bg,sigma,duty,false);
        if(ntheta.sigma() != sigma) break; //Past the largest sigma allowed [cannot occur if n==0]
        double nrllh=relative_log_likelihood(*this, im, ntheta);
        if(nrllh>rllh){
            theta=ntheta;
            rllh=nrllh;
        }
    }
    theta.compute_derivatives();
    return theta;
}



double Blink2DsMAP::prior_log_likelihood(const Stencil &s) const
{
    double rllh=prior_relative_log_likelihood(s);
    return rllh+ 2*log_prior_pos_const + log_prior_I_const + log_prior_bg_const + log_prior_sigma_const + size(0)*log_prior_D_const;
}

double Blink2DsMAP::prior_relative_log_likelihood(const Stencil &s) const
{
    double xrllh=rllh_beta_prior(beta_pos, s.x(), size(0));
    double yrllh=rllh_beta_prior(beta_pos, s.y(), size(1));
    double Irllh=rllh_gamma_prior(kappa_I, mean_I, s.I());
    double bgrllh=rllh_gamma_prior(kappa_bg, mean_bg, s.bg());
    double sigmallh=rllh_pareto_prior(alpha_sigma, s.sigma());
    double Dirllh=0.;
    for(int i=0;i<size(0);i++) Dirllh+=rllh_beta2_prior(beta_D0, beta_D1, s.D(i));
    return xrllh+yrllh+Irllh+bgrllh+sigmallh+Dirllh;
}

Blink2DsMAP::ParamT
Blink2DsMAP::prior_grad(const Stencil &s) const
{
    ParamT grad=make_param();
    grad(0)=beta_prior_grad(beta_pos, s.x(), size(0));
    grad(1)=beta_prior_grad(beta_pos, s.y(), size(1));
    grad(2)=gamma_prior_grad(kappa_I, mean_I, s.I());
    grad(3)=gamma_prior_grad(kappa_bg, mean_bg, s.bg());
    grad(4)=pareto_prior_grad(alpha_sigma, s.sigma());
    for(int i=0;i<size(0);i++) grad(5+i)=beta2_prior_grad(beta_D0, beta_D1, s.D(i));
    return grad;
}

Blink2DsMAP::ParamT
Blink2DsMAP::prior_grad2(const Stencil &s) const
{
    ParamT grad2=make_param();
    grad2(0)= beta_prior_grad2(beta_pos, s.x(), size(0));
    grad2(1)= beta_prior_grad2(beta_pos, s.y(), size(1));
    grad2(2)= gamma_prior_grad2(kappa_I, s.I());
    grad2(3)= gamma_prior_grad2(kappa_bg, s.bg());
    grad2(4)= pareto_prior_grad2(alpha_sigma, s.sigma());
    for(int i=0;i<size(0);i++) grad2(5+i)=beta2_prior_grad2(beta_D0, beta_D1, s.D(i));
    return grad2;
}

Blink2DsMAP::ParamT
Blink2DsMAP::prior_cr_lower_bound(const Stencil &s) const
{
    //TODO complete these calculations
    ParamT pcrlb=make_param();
    pcrlb.zeros();
    return pcrlb;
}


void Blink2DsMAP::sample_candidate_theta(int sample_index, RNG &rng, ParamT &candidate_theta,double scale) const
{
    int phase=sample_index%num_candidate_sampling_phases;
    switch(phase) {
        case 0:  //change x,y
            candidate_theta(0)+=generate_normal(rng,0.0,candidate_eta_x*scale);
            candidate_theta(1)+=generate_normal(rng,0.0,candidate_eta_y*scale);
            break;
        case 1: //change sigma, I
            candidate_theta(2)+=generate_normal(rng,0.0,candidate_eta_I*scale);
            candidate_theta(4)+=generate_normal(rng,0.0,candidate_eta_sigma*scale);
            break;
        case 2: //change I, bg
            candidate_theta(2)+=generate_normal(rng,0.0,candidate_eta_I*scale);
            candidate_theta(3)+=generate_normal(rng,0.0,candidate_eta_bg*scale);
            break;
        default:
            int col=phase-(num_candidate_sampling_phases-size(0));
            candidate_theta(5+col)+=generate_normal(rng,0.0,candidate_eta_D*scale);
    }
}

double Blink2DsMAP::compute_candidate_rllh(int sample_index, const ImageT &im,
                                           const ParamT &candidate_theta, ModelImage &model_image) const
{
    assert(theta_in_bounds(candidate_theta));
    return log_likelihood(*this, im, candidate_theta);
//FOOOOOO fix me
//     int phase=sample_index%num_candidate_sampling_phases;
//     switch(phase) {
//         case 0:
//         case 1:
//             return relative_log_likelihood(*this, im, candidate_theta);
//         case 2:
//             model_image.set_stencil(candidate_theta);
//             return model_image.relative_log_likelihood();
//         default:
//             int col=phase-(num_candidate_sampling_phases-size(0));
//             model_image.set_duty(col, candidate_theta(5+col));
//             return model_image.relative_log_likelihood();
//     }
}


/* Template Specialization Definitions */


template<>
Blink2DsMAP::Stencil
CGaussHeuristicMLE<Blink2DsMAP>::compute_estimate(const ImageT &im, const ParamT &theta_init)
{
    ParamT theta_est(model.num_params,arma::fill::zeros);
    if(model.size(0)==model.size(1) && model.psf_sigma(0)==model.psf_sigma(1)){ //only works for square images and iso-tropic psf
        float Nmax;
        arma::fvec ftheta_est(model.num_params);
        //Convert from double
        arma::fmat fim=arma::conv_to<arma::fmat>::from(im);
        //Compute
        CenterofMass2D(model.size(0), fim.memptr(), &ftheta_est(0), &ftheta_est(1));
        GaussFMaxMin2D(model.size(0), model.psf_sigma(0), fim.memptr(), &Nmax, &ftheta_est(3));
        ftheta_est(2)=std::max(0., (Nmax-ftheta_est(3)) * 2 * arma::datum::pi * model.psf_sigma(0) * model.psf_sigma(0));
        ftheta_est(4)=model.psf_sigma(0);
        for(int i=0;i<model.size(0);i++) ftheta_est(5+i)=1.0;//Fill in blink parameters
        //Back to double
        theta_est=arma::conv_to<arma::mat>::from(ftheta_est);
        //Shift to account for change of coordinates
        theta_est(0)+=0.5;
        theta_est(1)+=0.5;
    }
    return model.make_stencil(theta_est);
}

template<>
void
CGaussMLE<Blink2DsMAP>::compute_estimate(const ImageT &im, const ParamT &theta_init, ParamT &theta, ParamT &crlb, double &llh)
{
    if(model.size(0)==model.size(1) && model.psf_sigma(0)==model.psf_sigma(1)){//only works for square images and iso-tropic psf
        float fllh;
        arma::fvec fcrlb(model.num_params);
        arma::fvec ftheta(model.num_params);
        //Convert from double
        arma::fmat fim=arma::conv_to<arma::fmat>::from(im);
        //Compute
        MLEFit_sigma(fim.memptr(), model.psf_sigma(0), model.size(0), max_iterations,
                    ftheta.memptr(), fcrlb.memptr(), &fllh);
        for(int i=0;i<model.size(0);i++) {
            fcrlb(5+i)=0.0; //Fill in blink parameters
            ftheta(5+i)=1.0; //Fill in blink parameters
        }
        //Back to double
        theta=arma::conv_to<arma::vec>::from(ftheta);
        crlb=arma::conv_to<arma::vec>::from(fcrlb);
        theta(0)+=0.5;
        theta(1)+=0.5;
        llh=log_likelihood(model, im,model.make_stencil(theta));
    } else {
        theta.zeros();
        crlb.zeros();
        llh=0.0;
    }
}


// template<>
// typename Blink2DsMAP::ParamT
// cr_lower_bound(const Blink2DsMAP &model, const typename Blink2DsMAP::Stencil &s)
// {
// //     auto crlb=model.make_param();
// //     crlb.zeros();
//     auto pgrad=model.make_param();
//     auto fisherM=model.make_param_mat();
//     fisherM.zeros();
//     auto hess=model.make_param_mat();
// 
//     for(int j=0; j<model.size(1); j++) for(int i=0; i<model.size(0); i++) { //Col major ordering for armadillo
//         double model_val=model.model_value(i,j,s);
//         model.pixel_grad(i,j,s,pgrad);
//         model.pixel_hess(i,j,s,hess); /* Fill upper diagonal.  Zeroes whole matrix.  Could be made more efficient with an update only call */
//         std::cout<<"Pixel: ["<<i<<","<<j<<"] ModelVal:"<<model_val<<" Pixel Grad:"<<pgrad.t().eval();
// //         double min_grad=1e-6;
// //         for(int n=0; n<model.num_params; n++) {
// //             if(fabs(pgrad(n))<min_grad)
// //                 pgrad(n)=0;
// //         }
//         std::cout<<"Pixel: ["<<i<<","<<j<<"] ModelVal:"<<model_val<<" Pixel Grad:"<<pgrad.t().eval();
//         std::cout<<"Stencil: "<<s<<std::endl;
//         //Fill lower triangular elements (including diagonal) col-major order for aramadillo
// //         for(int n=0; n<model.num_params; n++) {
// //             double g=pgrad(n);
// //             crlb(n)+=g*g/model_val;
// //         }
//         for(int c=0; c<model.num_params; c++) for(int r=0; r<=c; r++) {
//             fisherM(r,c) += (pgrad(r)*pgrad(c))/model_val;
//         }
//     }
//     std::cout<<"FISHERM:"<<std::endl<<arma::symmatu(fisherM).eval();
//     typename Blink2DsMAP::ParamT crlb=arma::symmatu(fisherM).i().eval().diag();
// //     print_vec_row(std::cout,crlb2,"SQRT(CRLB): ",15,TERM_DIM_BLUE);
// //     for(int n=0; n<model.num_params; n++)  crlb(n)= (crlb(n)<1e-6) ? 1e6  : 1./crlb(n);
//     print_vec_row(std::cout,crlb,"APPROXCRLB: ",10,TERM_DIM_CYAN);
//     return crlb;
// }
// 


