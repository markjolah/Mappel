/** @file Gauss2DsModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class definition and template Specializations for Gauss2DsModel
 */

#include "Mappel/Gauss2DsModel.h"
#include "Mappel/stencil.h"

namespace mappel {

Gauss2DsModel::Gauss2DsModel(const ImageSizeT &size, const VecT &min_sigma, const VecT &max_sigma)
    :  PointEmitterModel(), ImageFormat2DBase(size), //V-base calls ignored since a higher concrete class will call them
       MCMCAdaptor2Ds(),
       min_sigma(min_sigma),
       x_model{make_internal_1Dsum_estimator(0,size,min_sigma,max_sigma,prior)},
       y_model{make_internal_1Dsum_estimator(1,size,min_sigma,max_sigma,prior)}
{
    check_psf_sigma(min_sigma);
    check_psf_sigma(max_sigma);
    if(!arma::all(min_sigma < max_sigma)){
        std::ostringstream msg;
        msg<<"Got bad sigma min_sigma:"<<min_sigma<<" >= max_sigma:"<<max_sigma;
        throw ParameterValueError(msg.str());
    }
}

Gauss2DsModel::Gauss2DsModel(const Gauss2DsModel &o)
    : PointEmitterModel(o), ImageFormat2DBase(o), //V-base calls ignored since a higher concrete class will call them
      MCMCAdaptor2Ds(o),
      min_sigma(o.min_sigma),
      x_model{make_internal_1Dsum_estimator(0,size,min_sigma,get_max_sigma(),prior)},
      y_model{make_internal_1Dsum_estimator(1,size,min_sigma,get_max_sigma(),prior)}
{ }

Gauss2DsModel::Gauss2DsModel(Gauss2DsModel &&o)
    : PointEmitterModel(std::move(o)), ImageFormat2DBase(std::move(o)), //V-base calls ignored since a higher concrete class will call them
      MCMCAdaptor2Ds(std::move(o)),
      min_sigma(o.min_sigma),
      x_model{make_internal_1Dsum_estimator(0,size,min_sigma,get_max_sigma(),prior)},
      y_model{make_internal_1Dsum_estimator(1,size,min_sigma,get_max_sigma(),prior)}
{ }

Gauss2DsModel& Gauss2DsModel::operator=(const Gauss2DsModel &o)
{
    //Don't copy virtual base classes.  This is called by superclass only.
    MCMCAdaptor2Ds::operator=(o);
    //Copy data memebers
    min_sigma = o.min_sigma;
    auto max_sigma = get_max_sigma();
    x_model = o.x_model;
    y_model = o.y_model;
    return *this;
}

Gauss2DsModel& Gauss2DsModel::operator=(Gauss2DsModel &&o)
{
    //Don't copy virtual base classes.  This is called by superclass only.
    MCMCAdaptor2Ds::operator=(std::move(o));
    //Copy data memebers
    min_sigma = o.min_sigma;
    //Move sub models
    x_model = std::move(o.x_model);
    y_model = std::move(o.y_model);
    return *this;
}

Gauss2DsModel::Gauss1DSumModelT 
Gauss2DsModel::make_internal_1Dsum_estimator(IdxT dim, const ImageSizeT &size, const VecT &min_sigma,const VecT &max_sigma, const CompositeDist &prior)
{
    std::type_index pos_dist = prior.component_types()[dim];
    std::type_index beta_dist(typeid(prior_hessian::ScaledSymmetricBetaDist));
    std::type_index normal_dist(typeid(prior_hessian::TruncatedNormalDist));
    auto hyperparams = prior.params();
    if(pos_dist == beta_dist){
        double beta_pos = hyperparams(dim);
        double mean_I = hyperparams(2);
        double kappa_I = hyperparams(3);
        double mean_bg = size((dim+1)%2) * hyperparams(4); //Amplify bg to account for summation
        double kappa_bg = hyperparams(5);
        double alpha_sigma = hyperparams(6);
        return {size(dim),Gauss1DSumModelT::make_prior_beta_position(
                    size(dim), beta_pos, mean_I, kappa_I, mean_bg, kappa_bg, min_sigma(dim), max_sigma(dim),alpha_sigma)};
    } else if(pos_dist == normal_dist) {
        double sigma_pos = hyperparams(2*dim+1);
        double mean_I = hyperparams(4);
        double kappa_I = hyperparams(5);
        double mean_bg = size((dim+1)%2) * hyperparams(6); //Amplify bg to account for summation
        double kappa_bg = hyperparams(7);
        double alpha_sigma = hyperparams(8);
        return {size(dim),Gauss1DSumModelT::make_prior_normal_position(
                    size(dim), sigma_pos, mean_I, kappa_I, mean_bg, kappa_bg, min_sigma(dim), max_sigma(dim),alpha_sigma)};
    } else {
        std::ostringstream msg;
        msg<<"Unknown position distribution: "<<pos_dist.name();
        throw ParameterValueError(msg.str());
    }    
}

void Gauss2DsModel::update_internal_1Dsum_estimators()
{
    auto max_sigma = get_max_sigma();
    x_model = make_internal_1Dsum_estimator(0,size,min_sigma,max_sigma,get_prior());
    y_model = make_internal_1Dsum_estimator(1,size,min_sigma,max_sigma,get_prior());
}

void Gauss2DsModel::set_prior(CompositeDist&& prior_)
{
    PointEmitterModel::set_prior(std::move(prior_));
    update_internal_1Dsum_estimators();
}

void Gauss2DsModel::set_prior(const CompositeDist& prior_)
{
    PointEmitterModel::set_prior(prior_);
    update_internal_1Dsum_estimators();
}

void Gauss2DsModel::set_hyperparams(const VecT &hyperparams)
{
    PointEmitterModel::set_hyperparams(hyperparams);
    update_internal_1Dsum_estimators();
}

void Gauss2DsModel::set_size(const ImageSizeT &size_)
{
    ImageFormat2DBase::set_size(size_);
    //Reset initializer model sizes
    x_model.set_size(size(0));
    y_model.set_size(size(1));
}

/**
 * Set the minimum sigma, keeping the max_sigma_ratio the same.
 * 
 */
void Gauss2DsModel::set_min_sigma(const VecT& new_sigma)
{ 
    check_psf_sigma(new_sigma);
    min_sigma = new_sigma;
    //Reset initializer model sigma ranges
    auto max_sigma = get_max_sigma();
    x_model.set_min_sigma(min_sigma(0));
    x_model.set_max_sigma(max_sigma(0));
    
    y_model.set_min_sigma(min_sigma(1));
    y_model.set_max_sigma(max_sigma(1));
}


/**
 * Set the max_sigma_ratio based on the new max_sigma's ratio with the current min_sigma.
 * 
 */
void Gauss2DsModel::set_max_sigma(const VecT& new_sigma)
{ 
    check_psf_sigma(new_sigma);
    double new_max_sigma_ratio = compute_max_sigma_ratio(get_min_sigma(), new_sigma);  
    set_max_sigma_ratio(new_max_sigma_ratio);
}

double Gauss2DsModel::compute_max_sigma_ratio(const VecT& min_sigma, const VecT& max_sigma)
{
    VecT ratio = max_sigma/min_sigma;
    if(std::fabs(ratio(0)-ratio(1)) > std::sqrt(std::numeric_limits<double>::epsilon())) {
        std::ostringstream msg;
        msg<<"Invalid sigma ratio min_sigma:"<<min_sigma.t()<<" max_sigma:"<<max_sigma.t()<<" ratio:["
            <<std::setprecision(15)<<ratio(0)<<","<<std::setprecision(15)<<ratio(1)<<"] with delta:"<<std::setprecision(15)<<ratio(0)-ratio(1)<<" are unequal."
           <<"  Sigma ratios must be equal for a scalar-valued variable gaussian PSF sigma model."
           <<"  Make max_sigma an exact multiple of min_sigma (psf_sigma).";
        throw ParameterValueError(msg.str());
    }
    return ratio(0);
}

void Gauss2DsModel::set_max_sigma_ratio(double new_max_sigma_ratio)
{
    if(new_max_sigma_ratio <= 1.0+bounds_epsilon || !std::isfinite(new_max_sigma_ratio)){
        std::ostringstream msg;
        msg<<"Gauss2DsModel::set_max_sigma_ratio() max_sigma_ratio:"<<new_max_sigma_ratio<<" is invalid.";
        throw ParameterValueError(msg.str());        
    }
    auto ub = get_ubound();
    ub(4) = new_max_sigma_ratio;
    set_ubound(ub);

    x_model.set_max_sigma(get_max_sigma(0));
    y_model.set_max_sigma(get_max_sigma(1));
}

double Gauss2DsModel::get_min_sigma(IdxT dim) const
{ 
    if(dim > 1) {
        std::ostringstream msg;
        msg<<"Gauss2DsModel::get_min_sigma() dim="<<dim<<" is invalid.";
        throw ParameterValueError(msg.str());
    }
    return min_sigma(dim); 
}

/* Prior construction */
const StringVecT Gauss2DsModel::prior_types = { "Beta", //Model the position as a symmetric Beta distribution scaled over (0,size)
                                                "Normal"  //Model the position as a truncated Normal distribution centered at size/2 with domain (0,size)
                                                 };
const std::string Gauss2DsModel::DefaultPriorType = "Normal";

CompositeDist
Gauss2DsModel::make_default_prior(const ImageSizeT &size, double max_sigma_ratio, const std::string &prior_type)
{
    if(istarts_with(prior_type,"Normal")) {
        return make_default_prior_normal_position(size, max_sigma_ratio);
    } else if(istarts_with(prior_type,"Beta")) {
        return make_default_prior_beta_position(size, max_sigma_ratio);
    } else {
        std::ostringstream msg;
        msg<<"Unknown prior type: "<<prior_type;
        throw ParameterValueError(msg.str());
    }
}

void
Gauss2DsModel::set_prior_variable_names(CompositeDist &pr)
{
    pr.set_component_names(StringVecT{"x_pos", "y_pos", "intensity", "background","psf_sigma_ratio"});
    pr.set_dim_variables(StringVecT{"x","y","I","bg","sigma_ratio"});
}

CompositeDist
Gauss2DsModel::make_default_prior_beta_position(const ImageSizeT &size, double max_sigma_ratio)
{
    CompositeDist d(make_prior_component_position_beta(size(0)),
                    make_prior_component_position_beta(size(1)),
                    make_prior_component_intensity(),
                    make_prior_component_intensity(default_pixel_mean_bg),
                    make_prior_component_sigma(1.0, max_sigma_ratio));
    set_prior_variable_names(d);
    return d;
}

CompositeDist
Gauss2DsModel::make_default_prior_normal_position(const ImageSizeT &size, double max_sigma_ratio)
{
    CompositeDist d(make_prior_component_position_normal(size(0)),
                    make_prior_component_position_normal(size(1)),
                    make_prior_component_intensity(),
                    make_prior_component_intensity(default_pixel_mean_bg),
                    make_prior_component_sigma(1.0, max_sigma_ratio));
    set_prior_variable_names(d);
    return d;
}

CompositeDist
Gauss2DsModel::make_prior_beta_position(const ImageSizeT &size, double beta_xpos, double beta_ypos,
                                       double mean_I, double kappa_I,
                                       double mean_bg, double kappa_bg,
                                       double max_sigma_ratio, double alpha_sigma)
{
   CompositeDist d(make_prior_component_position_beta(size(0),beta_xpos),
                   make_prior_component_position_beta(size(1), beta_ypos),
                   make_prior_component_intensity(mean_I,kappa_I),
                   make_prior_component_intensity(mean_bg, kappa_bg),
                   make_prior_component_sigma(1.0, max_sigma_ratio,alpha_sigma));
    set_prior_variable_names(d);
    return d;
}

CompositeDist
Gauss2DsModel::make_prior_normal_position(const ImageSizeT &size, double sigma_xpos, double sigma_ypos,
                                       double mean_I, double kappa_I,
                                       double mean_bg, double kappa_bg,
                                       double max_sigma_ratio, double alpha_sigma)
{
    CompositeDist d(make_prior_component_position_normal(size(0), sigma_xpos),
                    make_prior_component_position_normal(size(1), sigma_ypos),
                    make_prior_component_intensity(mean_I,kappa_I),
                    make_prior_component_intensity(mean_bg, kappa_bg),
                    make_prior_component_sigma(1.0, max_sigma_ratio,alpha_sigma));
    set_prior_variable_names(d);
    return d;
}

Gauss2DsModel::Stencil::Stencil(const Gauss2DsModel &model_, 
                                const ParamT &theta, 
                                bool _compute_derivatives)
      : model(&model_), theta(theta)
{
    int szX = model->size(0);
    int szY = model->size(1);
    dx = make_d_stencil(szX, x());
    dy = make_d_stencil(szY, y());
    X = make_X_stencil(szX, dx, sigmaX());
    Y = make_X_stencil(szY, dy, sigmaY());
    if(_compute_derivatives) compute_derivatives();
}

void Gauss2DsModel::Stencil::compute_derivatives()
{
    if(derivatives_computed) return;
    derivatives_computed=true;
    int szX = model->size(0);
    int szY = model->size(1);
    Gx = make_G_stencil(szX, dx, sigmaX());
    Gy = make_G_stencil(szY, dy, sigmaY());
    DX = make_DX_stencil(szX, Gx, sigmaX());
    DY = make_DX_stencil(szY, Gy, sigmaY());
    DXS = make_DXS_stencil(szX, dx, Gx, sigmaX());
    DYS = make_DXS_stencil(szY, dy, Gy, sigmaY());
    DXS2 = make_DXS2_stencil(szX, dx, Gx, DXS, sigmaX());
    DYS2 = make_DXS2_stencil(szY, dy, Gy, DYS, sigmaY());
    DXSX = make_DXSX_stencil(szX, dx, Gx, DX, sigmaX());
    DYSY = make_DXSX_stencil(szY, dy, Gy, DY, sigmaY());
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

StatsT Gauss2DsModel::get_stats() const
{
    auto stats = PointEmitterModel::get_stats();
    auto im_stats = ImageFormat2DBase::get_stats();
    auto mcmc_stats = MCMCAdaptor2Ds::get_stats();
    stats.insert(im_stats.begin(), im_stats.end());
    stats.insert(mcmc_stats.begin(), mcmc_stats.end());
    stats["min_sigma.1"] = get_min_sigma(0);
    stats["min_sigma.2"] = get_min_sigma(1);
    stats["max_sigma.1"] = get_max_sigma(0);
    stats["max_sigma.2"] = get_max_sigma(1);
    stats["max_sigma_ratio"] = get_max_sigma_ratio();
    return stats;
}

/** @brief pixel derivative inner loop calculations.
 */
void Gauss2DsModel::pixel_hess_update(int i, int j, const Stencil &s, double dm_ratio_m1, double dmm_ratio, 
                                      ParamT &grad, MatT &hess) const
{
    auto pgrad = make_param();
    pixel_grad(i,j,s,pgrad);
    double I = s.I();
    /* Update grad */
    grad += dm_ratio_m1*pgrad;
    //Update Hessian
    //On Diagonal
    hess(0,0) += dm_ratio_m1 * I/s.sigmaX() * s.DXS(i) * s.Y(j); //xx
    hess(1,1) += dm_ratio_m1 * I/s.sigmaY() * s.DYS(j) * s.X(i); //yy
    hess(4,4) += dm_ratio_m1 * I * (s.X(i)*s.DYS2(j) + 2.*s.DXS(i)*s.DYS(j) + s.Y(j)*s.DXS2(i)); //SS
    //Off Diagonal
    hess(0,1) += dm_ratio_m1 * I * s.DX(i) * s.DY(j); //xy
    hess(0,4) += dm_ratio_m1 * I * (s.Y(j)*s.DXSX(i) + s.DX(i)*s.DYS(j)); //xS
    hess(1,4) += dm_ratio_m1 * I * (s.X(i)*s.DYSY(j) + s.DY(j)*s.DXS(i)); //yS
    //Off Diagonal with respect to I
    hess(0,2) += dm_ratio_m1 * pgrad(0) / I; //xI
    hess(1,2) += dm_ratio_m1 * pgrad(1) / I; //yI
    hess(2,4) += dm_ratio_m1 * pgrad(4) / I; //IS
    //This is the pixel-gradient dependenent part of the hessian
    for(IdxT c=0; c<hess.n_cols; c++) for(IdxT r=0; r<=c; r++)
        hess(r,c) -= dmm_ratio * pgrad(r) * pgrad(c);
}

Gauss2DsModel::Stencil
Gauss2DsModel::initial_theta_estimate(const ImageT &im, const ParamT &theta_init, 
                                      const std::string &estimator_method) const
{
    double x_pos = 0;
    double y_pos = 0;
    double I = 0;
    double bg = 0;
    double sigma_ratio = 0;
    if(theta_init.n_elem == num_params){
        if(theta_in_bounds(theta_init)) return make_stencil(theta_init);
        x_pos = theta_init(0);
        y_pos = theta_init(1);
        I = theta_init(2);
        bg = theta_init(3);
        sigma_ratio = theta_init(4);
    }
    Gauss2DsModel::ImageT x_im = arma::sum(im,0).t();
    Gauss2DsModel::ImageT y_im = arma::sum(im,1);
    estimator::MLEData x_est, y_est;
    methods::estimate_max(x_model,x_im,estimator_method,x_est);
    methods::estimate_max(y_model,y_im,estimator_method,y_est);
    
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
    if(sigma_ratio <= lbound(4) || sigma_ratio >= ubound(4) || !std::isfinite(sigma_ratio)) {
        //mean of X and Y sigma_ratio. 1D models report sigma in pixels, convert to ratio to min_sigma.
        sigma_ratio = .5*(x_est.theta(3)/min_sigma(0) + y_est.theta(3)/min_sigma(1));
    }
    return make_stencil(ParamT{x_pos, y_pos, I, bg, sigma_ratio});
}

} /* namespace mappel */
