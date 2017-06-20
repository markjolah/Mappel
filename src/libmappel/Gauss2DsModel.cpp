/** @file Gauss2DsModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class definition and template Specializations for Gauss2DsModel
 */

#include "Gauss2DsModel.h"
#include "stencil.h"


const std::vector<std::string> Gauss2DsModel::param_names({ "x", "y", "I", "bg", "sigma" });


Gauss2DsModel::Gauss2DsModel(const IVecT &size, const VecT &psf_sigma)
    : PointEmitter2DModel(5,size,psf_sigma),
      gaussian_Xstencils(arma::mat(2*size(0)-1,stencil_sigmas.n_rows)),
      gaussian_Ystencils(arma::mat(2*size(1)-1,stencil_sigmas.n_rows))
{
    /* Initialize MCMC step sizes */
    num_candidate_sampling_phases=3;
    candidate_eta_sigma=1.0*candidate_sample_dist_ratio;

    for(unsigned i=0;i<stencil_sigmas.n_elem;i++){
        gaussian_Xstencils.col(i)=make_gaussian_stencil(size(0),psf_sigma(0)*stencil_sigmas(i));
        gaussian_Ystencils.col(i)=make_gaussian_stencil(size(1),psf_sigma(1)*stencil_sigmas(i));
    }
}


Gauss2DsModel::Stencil::Stencil(const Gauss2DsModel &model_, const ParamT &theta, bool _compute_derivatives)
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

void Gauss2DsModel::Stencil::compute_derivatives()
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

std::ostream& operator<<(std::ostream &out, const Gauss2DsModel::Stencil &s)
{
    int w=8;
    print_vec_row(out,s.theta,"Theta:",w,TERM_WHITE);
    print_vec_row(out,s.dx,"dx:",w,TERM_CYAN);
    print_vec_row(out,s.dy,"dy:",w,TERM_CYAN);
    print_vec_row(out,s.X,"X:",w,TERM_CYAN);
    print_vec_row(out,s.Y,"Y:",w,TERM_CYAN);
    if(s.derivatives_computed) {
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

void
Gauss2DsModel::pixel_hess_update(int i, int j, const Stencil &s, 
                                double dm_ratio_m1, double dmm_ratio, 
                                ParamT &grad, ParamMatT &hess) const
{
    auto pgrad=make_param();
    pixel_grad(i,j,s,pgrad);
    double I=s.I();
    /* Update grad */
    grad+=dm_ratio_m1*pgrad;
    //Update Hessian
    //On Diagonal
    hess(0,0)+=dm_ratio_m1 * I/s.sigmaX() * s.DXS(i) * s.Y(j); //xx
    hess(1,1)+=dm_ratio_m1 * I/s.sigmaY() * s.DYS(j) * s.X(i); //yy
    hess(4,4)+=dm_ratio_m1 * I * (s.X(i)*s.DYS2(j) + 2.*s.DXS(i)*s.DYS(j) + s.Y(j)*s.DXS2(i)); //SS
    //Off Diagonal
    hess(0,1)+=dm_ratio_m1 * I * s.DX(i) * s.DY(j); //xy
    hess(0,4)+=dm_ratio_m1 * I * (s.Y(j)*s.DXSX(i) + s.DX(i)*s.DYS(j)); //xS
    hess(1,4)+=dm_ratio_m1 * I * (s.X(i)*s.DYSY(j) + s.DY(j)*s.DXS(i)); //yS
    //Off Diagonal with respect to I
    hess(0,2)+=dm_ratio_m1 * pgrad(0) / I; //xI
    hess(1,2)+=dm_ratio_m1 * pgrad(1) / I; //yI
    hess(2,4)+=dm_ratio_m1 * pgrad(4) / I; //IS
    //This is the pixel-gradient dependenent part of the hessian
    for(int c=0; c<(int)hess.n_cols; c++) for(int r=0; r<=c; r++)
        hess(r,c) -= dmm_ratio * pgrad(r) * pgrad(c);
}

Gauss2DsModel::Stencil
Gauss2DsModel::initial_theta_estimate(const ImageT &im, const ParamT &theta_init) const
{
    double x_pos=0, y_pos=0, I=0, bg=0, sigma=0;
    double min_bg=1; //default minimum background.  Will be updated only if estimate_gaussian_2Dmax is called.
    if (!theta_init.is_empty()) {
        x_pos = theta_init(0);
        y_pos = theta_init(1);
        I = theta_init(2);
        bg = theta_init(3);
        sigma = theta_init(4);
    }
    Stencil theta;
//     std::cout<<"Theta_init: "<<theta_init.t()<<" -->";
    if(x_pos<=0 || x_pos>size(0) || y_pos<=0 || y_pos>size(1)){ //Invlaid positions, estimate them
//         std::cout<<"Full init\n";
        double rllh=-INFINITY;
        for(unsigned n=0; n<gaussian_Xstencils.n_cols; n++) {
            int pos[2];
            VecT sigma=psf_sigma*stencil_sigmas(n);
            if(n==0){
                estimate_gaussian_2Dmax(im, gaussian_Xstencils.col(n), gaussian_Ystencils.col(n), pos,min_bg);
            } else {
                refine_gaussian_2Dmax(im, gaussian_Xstencils.col(n), gaussian_Ystencils.col(n), pos);
            }
            auto unit_im=unit_model_image(size,pos,psf_sigma*stencil_sigmas(n));
            double bg=estimate_background(im, unit_im, min_bg);
            double I= estimate_intensity(im, unit_im, bg);
            auto ntheta=make_stencil(pos[0]+.5,pos[1]+.5,I,bg,stencil_sigmas(n),false);
            if(ntheta.sigma() != stencil_sigmas(n)) break; //Past the largest sigma allowed [cannot occur if n==0]
            double nrllh=relative_log_likelihood(*this, im, ntheta);
            if((n>0) && (nrllh<=rllh)) break;
            theta=ntheta;
            rllh=nrllh;
        }
    } else if(sigma<prior_epsilon) {//Valid x_pos/y_pos but invalid sigma
//         std::cout<<"Sigma init\n";
        double rllh=-INFINITY;
        for (unsigned n=0; n<stencil_sigmas.n_cols; n++) {
            auto unit_im=unit_model_image(size,x_pos,y_pos, psf_sigma(0)*stencil_sigmas(n), psf_sigma(1)*stencil_sigmas(n));
            double bg=estimate_background(im, unit_im, min_bg);
            double I= estimate_intensity(im, unit_im, bg);
            auto ntheta = make_stencil(x_pos,y_pos,I,bg,stencil_sigmas(n),false);
            if(ntheta.sigma() != stencil_sigmas(n)) break; //Past the largest sigma allowed [cannot occur if n==0]
            double nrllh=relative_log_likelihood(*this, im, ntheta);
            if((n>0) && (nrllh<=rllh)) break;
            theta=ntheta;
            rllh=nrllh;
        }
    } else if(I<=0 || bg<=0) { //Valid x_pos, y_pos, and sigma given
//         std::cout<<"Intenisty init\n";
        auto unit_im=unit_model_image(size,x_pos,y_pos, psf_sigma(0)*sigma, psf_sigma(1)*sigma);
        double bg=estimate_background(im, unit_im, min_bg);
        double I= estimate_intensity(im, unit_im, bg);
        theta = make_stencil(x_pos,y_pos,I,bg,sigma);
    } else {
//         std::cout<<"Null init\n";
        theta = make_stencil(x_pos,y_pos,I,bg,sigma);
    }
    theta.compute_derivatives();
//     std::cout<<"ThetaFinal: "<<theta.theta.t()<<"\n";
    return theta;
}

void
Gauss2DsModel::sample_candidate_theta(int sample_index, RNG &rng, ParamT &candidate_theta, double scale) const
{
    int phase=sample_index%num_candidate_sampling_phases;
    switch(phase) {
        case 0:  //change x,y
            candidate_theta(0)+=generate_normal(rng,0.0,candidate_eta_x*scale);
            candidate_theta(1)+=generate_normal(rng,0.0,candidate_eta_y*scale);
            break;
        case 1: //change I, sigma
            candidate_theta(2)+=generate_normal(rng,0.0,candidate_eta_I*scale);
            candidate_theta(4)+=generate_normal(rng,0.0,candidate_eta_sigma*scale);
            break;
        case 2: //change I, bg
            candidate_theta(2)+=generate_normal(rng,0.0,candidate_eta_I*scale);
            candidate_theta(3)+=generate_normal(rng,0.0,candidate_eta_bg*scale);
    }
}




