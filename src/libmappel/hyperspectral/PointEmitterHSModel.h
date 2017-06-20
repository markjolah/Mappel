/** @file PointEmitterHSModel.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-26-2014
 * @brief The class declaration and inline and templated functions for PointEmitterHSModel.
 *
 * The base class for all point emitter localization models
 */

#ifndef _POINTEMITTERHSMODEL_H
#define _POINTEMITTERHSMODEL_H

#include "PointEmitterModel.h"
#include "hypercube.h"

/** @brief A Base type for point emitter localization models that use 3d hyperspectral images
 *
 * We don't assume much here, so that it is possible to have a wide range of HS models
 *
 * 
 * 
 */
class PointEmitterHSModel : public PointEmitterModel {
public:
    typedef arma::cube ImageT; /**< A type to represent image data as a size X size floating-point array */
    typedef hypercube ImageStackT; /**< A type to represent image data stacks */

    static const std::vector<std::string> estimator_names;     /**< Estimator Names defined for this class */

    /* Model parameters */
    const int ndim; /**< Number of spatial dimensions */
    IVecT size; /**< The size of an image in [X Y L] (shape 3x1).  Important: images are indexed [L Y X]*/
    VecT psf_sigma; /**< The standard deviation of the gaussian PSF along the [X Y] axis in units of pixels (shape 2x1) */
    double mean_sigmaL;/**< The mean of the expected lambda sigma (nm).  This is a constant in this model.  We do not estimate it. */

    PointEmitterHSModel(int num_params, const IVecT &size, const VecT &psf_sigma);

    StatsT get_stats() const;

    ImageT make_image() const;
    ImageStackT make_image_stack(int n) const;
protected:
    double sigma_min=0.5;
    double sigmaL_min=0.5;
    
    double beta_pos=1.01; /**< The shape parameter for the Beta prior on the x and y positions. 0=Uniform, 1=Peaked  */
    double beta_L=1.01; /**< The shape parameter for the Beta prior on the lambda positions. 0=Uniform, 1=Peaked  */
    double mean_I=1000.; /**< The mean of the intensity gamma prior */
    double kappa_I=2.;  /**< The shape parameter for the I prior gamma distribution 1=exponential 2-5=skewed large=normal */
    double mean_bg=3.; /**< The mean of the background gamma prior */
    double kappa_bg=2.;  /**< The shape parameter for the bg prior gamma distribution 1=exponential 2-5=skewed large=normal */
    BetaRNG pos_dist;
    BetaRNG L_dist;
    GammaRNG I_dist;
    GammaRNG bg_dist;

    double log_prior_pos_const; /**< This is -2*lgamma(beta_pos)-lgamma(2*beta_pos) */
    double log_prior_L_const; /**< This is -2*lgamma(beta_L)-lgamma(2*beta_L) */
    double log_prior_I_const; /**< This is kappa_I*(log(kappa_I)-1/mean_I-log(mean_I))-lgamma(kappa_I) */
    double log_prior_bg_const; /**< This is kappa_bg*(log(kappa_bg)-1/mean_bg-log(mean_bg))-lgamma(kappa_bg) */
    double log_prior_const;

    double candidate_eta_L; /**< The standard deviation for the normally distributed pertebation to theta_L in the random walk MCMC sampling */
};

/* Inline Method Definitions */

inline
PointEmitterHSModel::ImageT
PointEmitterHSModel::make_image() const
{
    return ImageT(size(2),size(1),size(0));
}

inline
PointEmitterHSModel::ImageStackT
PointEmitterHSModel::make_image_stack(int n) const
{
    return ImageStackT(size(2),size(1),size(0),n);
}


/* Templated Function Definitions */

template<class Model>
typename std::enable_if<std::is_base_of<PointEmitterHSModel,Model>::value,typename Model::ImageT>::type
model_image(const Model &model, const typename Model::Stencil &s) 
{
    auto im=model.make_image();
    for(int i=0; i<model.size(0); i++) for(int j=0; j<model.size(1); j++) for(int k=0; k<model.size(2); k++) { //Col major ordering for armadillo
        im(k,j,i) = model.model_value(i,j,k,s);
        assert(im(k,j,i)>0.);//Model value must be positive for grad to be defined
    }
    return im;
}


/** @brief Simulate an image using the PSF model, by generating Poisson noise
 * @param[out] image An image to populate.
 * @param[in] theta The parameter values to us
 * @param[in,out] rng An initialized random number generator
 */
template<class Model, class rng_t>
typename std::enable_if<std::is_base_of<PointEmitterHSModel,Model>::value,typename Model::ImageT>::type
simulate_image(const Model &model, const typename Model::Stencil &s, rng_t &rng)
{
    auto sim_im=model.make_image();
    for(int i=0; i<model.size(0); i++) for(int j=0; j<model.size(1); j++) for(int k=0; k<model.size(2); k++){ //Col major ordering for armadillo
        sim_im(k,j,i) = generate_poisson(rng,model.model_value(i,j,k,s));
    }
    return sim_im;
}

template<class Model, class rng_t>
typename std::enable_if<std::is_base_of<PointEmitterHSModel,Model>::value,typename Model::ImageT>::type
simulate_image(const Model &model, const typename Model::ImageT &model_im, rng_t &rng)
{
    auto sim_im=model.make_image();
    for(int i=0; i<model.size(0); i++) for(int j=0; j<model.size(1); j++) for(int k=0; k<model.size(2); k++){ //Col major ordering for armadillo
        sim_im(k,j,i)=generate_poisson(rng,model_im(i,j,k));
    }
    return sim_im;
}

template<class Model>
typename std::enable_if<std::is_base_of<PointEmitterHSModel,Model>::value>::type
model_grad(const Model &model, const typename Model::ImageT &im,
           const typename Model::Stencil &s, typename Model::ParamT &grad) 
{
    auto pgrad=model.make_param();
    grad.zeros();
    for(int i=0; i<model.size(0); i++) for(int j=0; j<model.size(1); j++) for(int k=0; k<model.size(2); k++){ //Col major ordering for armadillo
        model.pixel_grad(i,j,k,s,pgrad);
        double model_val = model.model_value(i,j,k,s);
        assert(model_val>0.);//Model value must be positive for grad to be defined
        double dm_ratio_m1 = im(k,j,i)/model_val - 1.;
        grad += dm_ratio_m1*pgrad;
    }
    grad += model.prior_grad(s);
    assert(grad.is_finite());
}

template<class Model>
typename std::enable_if<std::is_base_of<PointEmitterHSModel,Model>::value>::type
model_grad2(const Model &model, const typename Model::ImageT &im,
            const typename Model::Stencil &s, 
            typename Model::ParamT &grad, typename Model::ParamT &grad2) 
{
    grad.zeros();
    grad2.zeros();
    auto pgrad = model.make_param();
    auto pgrad2 = model.make_param();
    for(int i=0; i<model.size(0); i++) for(int j=0; j<model.size(1); j++) for(int k=0; k<model.size(2); k++){ //Col major ordering for armadillo
        /* Compute model value and ratios */
        double model_val = model.model_value(i,j,k,s);
        assert(model_val>0.);//Model value must be positive for grad to be defined
        double dm_ratio = im(k,j,i)/model_val;
        double dm_ratio_m1 = dm_ratio-1;
        double dmm_ratio = dm_ratio/model_val;
        model.pixel_grad(i,j,k,s,pgrad);
        model.pixel_grad2(i,j,k,s,pgrad2);
        grad  += dm_ratio_m1*pgrad;
        grad2 += dm_ratio_m1*pgrad2 - dmm_ratio*pgrad%pgrad;
    }
    grad += model.prior_grad(s);
    grad2 += model.prior_grad2(s);
    assert(grad.is_finite()); 
    assert(grad2.is_finite()); 
}


template<class Model>
typename std::enable_if<std::is_base_of<PointEmitterHSModel,Model>::value>::type
model_hessian(const Model &model, const typename Model::ImageT &im,
              const typename Model::Stencil &s, 
              typename Model::ParamT &grad, typename Model::ParamMatT &hess) 
{
    grad.zeros();
    hess.zeros();
    for(int i=0; i<model.size(0); i++) for(int j=0; j<model.size(1); j++) for(int k=0; k<model.size(2); k++){ //Col major ordering for armadillo
        /* Compute model value and ratios */
        double model_val=model.model_value(i,j,k,s);
        assert(model_val>0.);//Model value must be positive for grad to be defined
        double dm_ratio = im(k,j,i)/model_val;
        double dm_ratio_m1 = dm_ratio-1;
        double dmm_ratio = dm_ratio/model_val;
        model.pixel_hess_update(i,j,k,s,dm_ratio_m1,dmm_ratio,grad,hess);
    }
    grad += model.prior_grad(s);
    hess.diag() += model.prior_grad2(s);
    assert(grad.is_finite()); 
    assert(hess.is_finite()); 
}



template<class Model>
typename std::enable_if<std::is_base_of<PointEmitterHSModel,Model>::value,double>::type
log_likelihood(const Model &model, const typename Model::ImageT &data_im,
               const typename Model::Stencil &s)
{
    double llh=0;
    for(int i=0; i<model.size(0); i++) for(int j=0; j<model.size(1); j++) for(int k=0; k<model.size(2); k++){ //Col major ordering for armadillo
        llh += log_likelihood_at_pixel(model.model_value(i,j,k,s), data_im(k,j,i));
    }
    double pllh=model.prior_log_likelihood(s);
    return llh+pllh;
}


template<class Model>
typename std::enable_if<std::is_base_of<PointEmitterHSModel,Model>::value,double>::type
relative_log_likelihood(const Model &model, const typename Model::ImageT &data_im,
                        const typename Model::Stencil &s)
{
    double rllh=0;
    for(int i=0; i<model.size(0); i++) for(int j=0; j<model.size(1); j++) for(int k=0; k<model.size(2); k++){ //Col major ordering for armadillo
        rllh += relative_log_likelihood_at_pixel(model.model_value(i,j,k,s), data_im(k,j,i));
    }
    double prllh = model.prior_relative_log_likelihood(s);
    return rllh+prllh;
}

/** @brief  */
template<class Model>
typename std::enable_if<std::is_base_of<PointEmitterHSModel,Model>::value,typename Model::MatT>::type
fisher_information(const Model &model, const typename Model::Stencil &s)
{
    auto fisherI = model.make_param_mat();
    fisherI.zeros();
    auto pgrad = model.make_param();
    for(int k=0; k<model.size(2); k++) for(int j=0; j<model.size(1); j++) for(int i=0; i<model.size(0); i++) { //Col major ordering for armadillo
        double model_val = model.model_value(i,j,k,s);
        model.pixel_grad(i,j,k,s,pgrad);
        for(int c=0; c<model.num_params; c++) for(int r=0; r<=c; r++) {
            fisherI(r,c) += pgrad(r)*pgrad(c)/model_val; //Fill upper triangle
        }
    }
    return fisherI;
}

template<class Model>
typename std::enable_if<std::is_base_of<PointEmitterHSModel,Model>::value,
    std::shared_ptr<Estimator<Model>>>::type
make_estimator(Model &model, std::string ename)
{
    using std::make_shared;
    const char *name=ename.c_str();
    if (istarts_with(name,"Heuristic")) {
        return make_shared<HeuristicMLE<Model>>(model);
    } else if (istarts_with(name,"NewtonRaphson")) {
        return make_shared<NewtonRaphsonMLE<Model>>(model);
    } else if (istarts_with(name,"QuasiNewton")) {
        return make_shared<QuasiNewtonMLE<Model>>(model);
    } else if (istarts_with(name,"Newton")) {
        return make_shared<NewtonMLE<Model>>(model);
//     } else if (istarts_with(name,"TrustRegion")) {
//         return make_shared<TrustRegionMLE<Model>>(model);
    } else if (istarts_with(name,"SimulatedAnnealing")) {
        return make_shared<SimulatedAnnealingMLE<Model>>(model);
    } else {
        return std::shared_ptr<Estimator<Model>>();
    }
}

#endif /* _POINTEMITTERHSMODEL_H */
