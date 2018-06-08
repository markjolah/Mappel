/** @file Gauss2DModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2018
 * @brief The class definition and template Specializations for Gauss2DModel
 */

#include "Mappel/Gauss2DModel.h"
#include "Mappel/stencil.h"

namespace mappel {

Gauss2DModel::Gauss2DModel(const ImageSizeT &size, const VecT &psf_sigma)
    :  PointEmitterModel(), ImageFormat2DBase(size), //V-base calls ignored since a higher concrete class will call them
      MCMCAdaptor2D(),
      psf_sigma(psf_sigma),
      x_model{make_internal_1Dsum_estimator(0,size,psf_sigma,prior)},
      y_model{make_internal_1Dsum_estimator(1,size,psf_sigma,prior)}
{ }

Gauss2DModel::Gauss2DModel(const Gauss2DModel &o)
    : PointEmitterModel(o), ImageFormat2DBase(o), //V-base calls ignored since a higher concrete class will call them
      MCMCAdaptor2D(o),
      psf_sigma(o.psf_sigma),
      x_model{make_internal_1Dsum_estimator(0,size,psf_sigma,prior)},
      y_model{make_internal_1Dsum_estimator(1,size,psf_sigma,prior)}
{ }

Gauss2DModel::Gauss2DModel(Gauss2DModel &&o)
    : PointEmitterModel(std::move(o)), ImageFormat2DBase(std::move(o)), //V-base calls ignored since a higher concrete class will call them
      MCMCAdaptor2D(std::move(o)),
      psf_sigma(o.psf_sigma),
      x_model{make_internal_1Dsum_estimator(0,size,psf_sigma,prior)},
      y_model{make_internal_1Dsum_estimator(1,size,psf_sigma,prior)}
{ }

Gauss2DModel& Gauss2DModel::operator=(const Gauss2DModel &o)
{
    //Don't copy virtual base classes.  This is called by superclass only.
    MCMCAdaptor2D::operator=(o);
    //Copy data memebers
    psf_sigma = o.psf_sigma;
    x_model = make_internal_1Dsum_estimator(0,size,psf_sigma,prior);
    y_model = make_internal_1Dsum_estimator(1,size,psf_sigma,prior);
    return *this;
}

Gauss2DModel& Gauss2DModel::operator=(Gauss2DModel &&o)
{
    //Don't copy virtual base classes.  This is called by superclass only.
    MCMCAdaptor2D::operator=(std::move(o));
    //Copy data memebers
    psf_sigma = o.psf_sigma;
    x_model = make_internal_1Dsum_estimator(0,size,psf_sigma,prior);
    y_model = make_internal_1Dsum_estimator(1,size,psf_sigma,prior);
    return *this;
}

Gauss2DModel::Gauss1DSumModelT 
Gauss2DModel::make_internal_1Dsum_estimator(IdxT dim, const ImageSizeT &size, const VecT &psf_sigma, const CompositeDist &prior)
{
    std::type_index pos_dist = prior.component_types()[dim];
    std::type_index beta_dist(typeid(prior_hessian::SymmetricBetaDist));
    std::type_index normal_dist(typeid(prior_hessian::NormalDist));
    auto hyperparams = prior.params();
    if(pos_dist == beta_dist){
        double beta_pos = hyperparams(dim);
        double mean_I = hyperparams(2);
        double kappa_I = hyperparams(3);
        double mean_bg = size((dim+1)%2) * hyperparams(4); //Amplify bg to account for summation
        double kappa_bg = hyperparams(5);
        return {size(dim),psf_sigma(dim),Gauss1DSumModelT::make_prior_beta_position(
                    size(dim), beta_pos, mean_I, kappa_I, mean_bg, kappa_bg)};
    } else if(pos_dist == normal_dist) {
        double sigma_pos = hyperparams(2*dim+1);
        double mean_I = hyperparams(4);
        double kappa_I = hyperparams(5);
        double mean_bg = size((dim+1)%2) * hyperparams(6); //Amplify bg to account for summation
        double kappa_bg = hyperparams(7);
        return {size(dim),psf_sigma(dim),Gauss1DSumModelT::make_prior_normal_position(
                    size(dim), sigma_pos, mean_I, kappa_I, mean_bg, kappa_bg)};
    } else {
        std::ostringstream msg;
        msg<<"Unknown position distribution: "<<pos_dist.name();
        throw ParameterValueError(msg.str());
    }    
}

void Gauss2DModel::update_internal_1Dsum_estimators()
{
    x_model = make_internal_1Dsum_estimator(0,size,psf_sigma,get_prior());
    y_model = make_internal_1Dsum_estimator(1,size,psf_sigma,get_prior());
}

void Gauss2DModel::set_prior(CompositeDist&& prior_)
{
    PointEmitterModel::set_prior(std::move(prior_));
    update_internal_1Dsum_estimators();
}

void Gauss2DModel::set_prior(const CompositeDist& prior_)
{
    PointEmitterModel::set_prior(prior_);
    update_internal_1Dsum_estimators();
}

void Gauss2DModel::set_hyperparams(const VecT &hyperparams)
{
    PointEmitterModel::set_hyperparams(hyperparams);
    update_internal_1Dsum_estimators();
}

void Gauss2DModel::set_size(const ImageSizeT &size_)
{
    ImageFormat2DBase::set_size(size_);
    //Reset initializer models' sizes
    x_model.set_size(size(0));
    y_model.set_size(size(1));
}

void Gauss2DModel::set_psf_sigma(const VecT& new_sigma)
{ 
    check_psf_sigma(new_sigma);
    psf_sigma = new_sigma;
    //Reset initializer models' psf_sigma
    x_model.set_psf_sigma(psf_sigma(0));
    y_model.set_psf_sigma(psf_sigma(1));
}

double Gauss2DModel::get_psf_sigma(IdxT dim) const
{
    if(dim > 1) {
        std::ostringstream msg;
        msg<<"Gauss2DModel::get_psf_sigma() dim="<<dim<<" is invalid.";
        throw ParameterValueError(msg.str());
    }
    return psf_sigma(dim); 
}

CompositeDist 
Gauss2DModel::make_default_prior(const ImageSizeT &size)
{
    return CompositeDist(std::make_tuple(
                            make_prior_component_position_beta("x",size(0)),
                            make_prior_component_position_beta("y",size(1)),
                            make_prior_component_intensity("I"),
                            make_prior_component_intensity("bg",default_pixel_mean_bg)
                                        ));
}

CompositeDist 
Gauss2DModel::make_prior_beta_position(const ImageSizeT &size, double beta_xpos,double beta_ypos,
                                       double mean_I, double kappa_I, 
                                       double mean_bg, double kappa_bg)
{
    return CompositeDist(std::make_tuple(
                            make_prior_component_position_beta("x",size(0),beta_xpos),
                            make_prior_component_position_beta("y",size(1),beta_ypos),
                            make_prior_component_intensity("I",mean_I,kappa_I),
                            make_prior_component_intensity("bg",mean_bg, kappa_bg)
                                        ));
}

CompositeDist 
Gauss2DModel::make_prior_normal_position(const ImageSizeT &size, double sigma_xpos, double sigma_ypos,
                                       double mean_I, double kappa_I, 
                                       double mean_bg, double kappa_bg)
{
    return CompositeDist(std::make_tuple(
                            make_prior_component_position_normal("x",size(0), sigma_xpos),
                            make_prior_component_position_normal("y",size(1), sigma_ypos),
                            make_prior_component_intensity("I",mean_I,kappa_I),
                            make_prior_component_intensity("bg",mean_bg, kappa_bg)
                                        ));
}

Gauss2DModel::Stencil::Stencil(const Gauss2DModel &model_,
                               const Gauss2DModel::ParamT &theta,
                               bool _compute_derivatives)
: model(&model_),theta(theta)
{
    int szX = model->size(0);
    int szY = model->size(1);
    dx = make_d_stencil(szX, x());
    dy = make_d_stencil(szY, y());
    X = make_X_stencil(szX, dx,model->psf_sigma(0));
    Y = make_X_stencil(szY, dy,model->psf_sigma(1));
    if(_compute_derivatives) compute_derivatives();
}

void Gauss2DModel::Stencil::compute_derivatives()
{
    if(derivatives_computed) return;
    derivatives_computed=true;
    int szX = model->size(0);
    int szY = model->size(1);
    double sigmaX = model->psf_sigma(0);
    double sigmaY = model->psf_sigma(1);
    Gx = make_G_stencil(szX, dx,sigmaX);
    Gy = make_G_stencil(szY, dy,sigmaY);
    DX = make_DX_stencil(szX, Gx,sigmaX);
    DY = make_DX_stencil(szY, Gy,sigmaY);
    DXS = make_DXS_stencil(szX, dx, Gx,sigmaX);
    DYS = make_DXS_stencil(szY, dy, Gy,sigmaY);
}


std::ostream& operator<<(std::ostream &out, const Gauss2DModel::Stencil &s)
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
    }
    return out;
}

StatsT Gauss2DModel::get_stats() const
{
    auto stats = PointEmitterModel::get_stats();
    auto im_stats = ImageFormat2DBase::get_stats();
    auto mcmc_stats = MCMCAdaptor2D::get_stats();
    stats.insert(im_stats.begin(), im_stats.end());
    stats.insert(mcmc_stats.begin(), mcmc_stats.end());
    stats["psf_sigma.0"] = get_psf_sigma(0);
    stats["psf_sigma.1"] = get_psf_sigma(1);
    return stats;
}

/** @brief pixel derivative inner loop calculations.
 */
void Gauss2DModel::pixel_hess_update(int i, int j, const Stencil &s, double dm_ratio_m1, double dmm_ratio, 
                                     ParamT &grad, MatT &hess) const
{
    /* Caclulate pixel derivative */
    auto pgrad = make_param();
    pixel_grad(i,j,s,pgrad);
    double I = s.I();
    /* Update grad */
    grad += dm_ratio_m1*pgrad;
    /* Update hess */
    hess(0,0) += dm_ratio_m1 * I/psf_sigma(0) * s.DXS(i) * s.Y(j);
    hess(0,1) += dm_ratio_m1 * I * s.DX(i) * s.DY(j);
    hess(1,1) += dm_ratio_m1 * I/psf_sigma(1) * s.DYS(j) * s.X(i);
    hess(0,2) += dm_ratio_m1 * pgrad(0) / I; 
    hess(1,2) += dm_ratio_m1 * pgrad(1) / I; 
    //This is the pixel-gradient dependent part of the hessian
    for(IdxT c=0; c<hess.n_cols; c++) for(IdxT r=0; r<=c; r++)
        hess(r,c) -= dmm_ratio * pgrad(r) * pgrad(c);
}

Gauss2DModel::Stencil
Gauss2DModel::initial_theta_estimate(const ImageT &im, const ParamT &theta_init, 
                                               const std::string &estimator_method)
{
    double x_pos = 0;
    double y_pos = 0;
    double I = 0;
    double bg = 0;    
    if(theta_init.n_elem == num_params){
        if(theta_in_bounds(theta_init)) return make_stencil(theta_init);
        x_pos = theta_init(0);
        y_pos = theta_init(1);
        I = theta_init(2);
        bg = theta_init(3);
    }
    Gauss1DModel::ImageT x_im = arma::sum(im,0).t();
    Gauss1DModel::ImageT y_im = arma::sum(im,1);
    auto x_est = methods::estimate_max(x_model,x_im,estimator_method);
    auto y_est = methods::estimate_max(y_model,y_im,estimator_method);
    
    if(x_pos <= lbound(0) || x_pos >= ubound(0) || !std::isfinite(x_pos)) x_pos = x_est.theta(0);
    if(y_pos <= lbound(1) || y_pos >= ubound(1) || !std::isfinite(y_pos)) y_pos = y_est.theta(0);
    if(I <= lbound(2) || I >= ubound(2) || !std::isfinite(I)) {
        //max of X and Y est of I
        I = std::max(x_est.theta(1), y_est.theta(1)); 
    }
    if(bg <= lbound(3) || bg >= ubound(3) || !std::isfinite(bg)) {
        //mean of X and Y est of bg corrected for 1D vs 2D interpretation of bg
        bg = .5*(x_est.theta(2)/size(1) + y_est.theta(2)/size(0));
    }
    return make_stencil(ParamT{x_pos, y_pos, I, bg});
}

} /* namespace mappel */
