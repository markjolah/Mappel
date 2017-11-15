/** @file PoissonNoise1DObjective.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 05-2017
 * @brief The class declaration and inline and templated functions for PoissonNoise1DObjective.
 *
 * The base class for all point emitter localization models
 */

#ifndef _POISSONNOISE1DOBJECTIVE_H
#define _POISSONNOISE1DOBJECTIVE_H

#include "ImageFormat1DBase.h"
#include "estimator.h"

namespace mappel {

/** @brief A base class for 1D objectives with Poisson read noise.
 *
 * Only the simulate_image functions are defined here
 * 
 */
class PoissonNoise1DObjective : public virtual ImageFormat1DBase {
    /**
     * This objective function and its subclasses are for models where the only source of noise is the "shot" or 
     * "counting" or Poisson noise inherent to a discrete capture of phontons given a certain mean rate of 
     * incidence on each pixel.
     * 
     * Subclasses: PoissonNoise1DObjective and PoissonNoise2DMLEObjective complete the objective specification and they
     *  allow differentiation of models with (MAP) and without (MLE) a prior.
     * 
     * The objective for Poisson noise is just the (gain-corrected) image converted to give approximate photons counts 
     * with purely Poisson noise.
     * 
     * The objective data type required for optimization (DataT) is simply a 2D double precision image with 
     * approximate photon counts.
     * 
     */
public:
    static const std::vector<std::string> estimator_names;
    using ModelDataT = typename ImageFormat1DBase::ImageT; /**< Objective function data type: 1D double precision image, gain-corrected to approximate photons counts */
    using ModelDataStackT = ImageFormat1DBase::ImageStackT; /**< Objective function data stack type: 1D double precision image stack, of images gain-corrected to approximate photons counts */
    template<class T> using IsSubclassT = typename std::enable_if<std::is_base_of<PoissonNoise1DObjective,T>::value>::type;
};

/* Inline Method Definitions */
/** @brief Simulate an image using the PSF model, by generating Poisson noise
 * @param[out] image An image to populate.
 * @param[in] theta The parameter values to us
 * @param[in,out] rng An initialized random number generator
 */
template<class Model, class rng_t, typename=typename PoissonNoise1DObjective::IsSubclassT<Model> >
typename Model::ImageT
simulate_image(const Model &model, const typename Model::Stencil &s, rng_t &rng)
{
    auto sim_im=model.make_image();
    for(typename Model::ImageSizeT i=0;i<model.size;i++) 
        sim_im(i)=generate_poisson(rng,model.pixel_model_value(i,s));
    return sim_im;
}

template<class Model, class rng_t>
typename std::enable_if<std::is_base_of<PoissonNoise1DObjective,Model>::value,typename Model::ImageT>::type
simulate_image_from_model(const Model &model, const typename Model::ImageT &model_im, rng_t &rng)
{
    auto sim_im=model.make_image();
    for(typename Model::ImageSizeT i=0;i<model.size;i++) sim_im(i)=generate_poisson(rng,model_im(i));
    return sim_im;
}

/* Inline Method Definitions */
template<class Model>
typename std::enable_if<std::is_base_of<PoissonNoise1DObjective,Model>::value,double>::type
log_likelihood(const Model &model, const typename Model::ImageT &data_im, 
               const typename Model::Stencil &s)
{
    double llh=0.;
    for(typename Model::ImageSizeT i=0;i<model.size;i++) 
        llh+=poisson_log_likelihood(model.pixel_model_value(i,s), data_im(i));
    double pllh=model.prior_log_likelihood(s.theta); /* MAP: Add log of prior for params theta */
    return llh+pllh;
}

template<class Model>
inline
typename std::enable_if<std::is_base_of<PoissonNoise1DObjective,Model>::value,double>::type
relative_log_likelihood(const Model &model, const typename Model::ImageT &data_im,
                        const typename Model::Stencil &s)
{
    double rllh=0.;
    for(typename Model::ImageSizeT i=0;i<model.size;i++)
        rllh+=relative_poisson_log_likelihood(model.pixel_model_value(i,s), data_im(i));
    double prllh=model.prior_relative_log_likelihood(s.theta); /* MAP: Add relative log of prior for params theta */
    return rllh+prllh;
}

template<class Model>
typename std::enable_if<std::is_base_of<PoissonNoise1DObjective,Model>::value>::type
model_grad(const Model &model, const typename Model::ImageT &im,
           const typename Model::Stencil &s, typename Model::ParamT &grad) 
{
    auto pgrad=model.make_param();
    grad.zeros();
    for(typename Model::ImageSizeT i=0;i<model.size;i++) {
        if(!std::isfinite(im(i))) continue; /* Skip non-finite image values as they are assumed masked */
        model.pixel_grad(i,s,pgrad);
        double model_val=model.pixel_model_value(i,s);
        double dm_ratio_m1=im(i)/model_val - 1.;
        grad+=dm_ratio_m1*pgrad;
    }
    model.prior_grad_accumulate(s.theta, grad); /* As appropriate for MAP/MLE: Add grad of log of prior for params theta */
}

template<class Model>
typename std::enable_if<std::is_base_of<PoissonNoise1DObjective,Model>::value>::type
model_grad2(const Model &model, const typename Model::ImageT &im, 
            const typename Model::Stencil &s, 
            typename Model::ParamT &grad, typename Model::ParamT &grad2) 
{
    grad.zeros();
    grad2.zeros();
    auto pgrad=model.make_param();
    auto pgrad2=model.make_param();
    for(typename Model::ImageSizeT i=0;i<model.size;i++){
        if(!std::isfinite(im(i))) continue; /* Skip non-finite image values as they are assumed masked */
        /* Compute model value and ratios */
        double model_val = model.pixel_model_value(i,s);
        double dm_ratio = im(i)/model_val;
        double dm_ratio_m1 = dm_ratio-1;
        double dmm_ratio = dm_ratio/model_val;
        model.pixel_grad(i,s,pgrad);
        model.pixel_grad2(i,s,pgrad2);
        grad  += dm_ratio_m1*pgrad;
        grad2 += dm_ratio_m1*pgrad2 - dmm_ratio*pgrad%pgrad;
    }
    model.prior_grad_grad2_accumulate(s.theta,grad,grad2); /* As appropriate for MAP/MLE: Add grad of log of prior for params theta */
}

template<class Model>
typename std::enable_if<std::is_base_of<PoissonNoise1DObjective,Model>::value>::type
model_hessian(const Model &model, const typename Model::ImageT &im, 
              const typename Model::Stencil &s, 
              typename Model::ParamT &grad, MatT &hess) 
{
    /* Returns hessian as an upper triangular matrix */
    grad.zeros();
    hess.zeros();
    for(typename Model::ImageSizeT i=0;i<model.size;i++) { 
        if(!std::isfinite(im(i))) continue; /* Skip non-finite image values as they are assumed masked */
        /* Compute model value and ratios */
        double model_val = model.pixel_model_value(i,s);
        double dm_ratio = im(i)/model_val;
        double dm_ratio_m1 = dm_ratio-1;
        double dmm_ratio = dm_ratio/model_val;
        model.pixel_hess_update(i,s,dm_ratio_m1,dmm_ratio,grad,hess);
    }
    model.prior_grad_hess_accumulate(s.theta,grad,hess); /* As appropriate for MAP/MLE: Add grad of log of prior for params theta */
}



/** @brief  */
template<class Model>
typename std::enable_if<std::is_base_of<PoissonNoise1DObjective,Model>::value,MatT>::type
fisher_information(const Model &model, const typename Model::Stencil &s)
{
    auto fisherI=model.make_param_mat();
    fisherI.zeros();
    auto pgrad=model.make_param();
    for(typename Model::ImageSizeT i=0;i<model.size;i++) {  
        double model_val=model.pixel_model_value(i,s);
        model.pixel_grad(i,s,pgrad);
        for(IdxT c=0; c<model.get_num_params(); c++) for(IdxT r=0; r<=c; r++) {
            fisherI(r,c) += pgrad(r)*pgrad(c)/model_val; //Fill upper triangle
        }
    }
    //TODO Fix for prior
//     model.prior.hess_accumulate(s.theta,fisherI); /* As appropriate for MAP/MLE: Add diagonal hession of log of prior for params theta */
    return fisherI;
}


template<class Model>
typename std::enable_if<std::is_base_of<PoissonNoise1DObjective,Model>::value,std::shared_ptr<Estimator<Model>>>::type
make_estimator(Model &model, std::string ename)
{
    using std::make_shared;
    const char *name=ename.c_str();
    if (istarts_with(name,"NewtonDiagonal")) {
        return make_shared<NewtonDiagonalMaximizer<Model>>(model);
    } else if (istarts_with(name,"QuasiNewton")) {
        return make_shared<QuasiNewtonMaximizer<Model>>(model);
    } else if (istarts_with(name,"Newton")) {
        return make_shared<NewtonMaximizer<Model>>(model);
    } else if (istarts_with(name,"TrustRegion")) {
        return make_shared<TrustRegionMaximizer<Model>>(model);
    } else {
        throw std::logic_error("Unknown estimator name");
    }
}


} /* namespace mappel */

#endif /* _POISSONNOISE1DOBJECTIVE_H */
