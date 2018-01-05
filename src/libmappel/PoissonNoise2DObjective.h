/** @file PoissonNoise2DObjective.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-26-2014
 * @brief The class declaration and inline and templated functions for PoissonNoise2DObjective.
 *
 * The base class for all point emitter localization models
 */

#ifndef _POISSONNOISE2DOBJECTIVE_H
#define _POISSONNOISE2DOBJECTIVE_H

#include "ImageFormat2DBase.h"
#include "estimator.h"

namespace mappel {

/** @brief A base class for 2D objectives with Poisson read noise.
 *
 * Only the simulate_image functions are defined here
 * 
 */
class PoissonNoise2DObjective : public virtual ImageFormat2DBase {
    /**
     * This objective function and its subclasses are for models where the only source of noise is the "shot" or 
     * "counting" or Poisson noise inherent to a discrete capture of phontons given a certain mean rate of 
     * incidence on each pixel.
     * 
     * Subclasses: PoissonNoise2DObjective and PoissonNoise2DMLEObjective complete the objective specification and they
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
    PoissonNoise2DObjective(const IVecT &size) : ImageFormat2DBase(size) {};
    
    using ModelDataT = typename ImageFormat2DBase::ImageT; /**< Objective function data type: 2D double precision image, gain-corrected to approximate photons counts */
    using ModelDataStackT = ImageFormat2DBase::ImageStackT; /**< Objective function data stack type: 2D double precision image stack, of images gain-corrected to approximate photons counts */

    /** A helper template for enabling member function */
    template<class T> using IsPoissonNoise2DObjectiveT = typename std::enable_if<std::is_base_of<PoissonNoise2DObjective,T>::value>::type;
    
};

namespace methods {

    /** @brief Simulate an image at a given theta stencil under noise model
    * @param[in] model Model object
    * @param[in] s The stencil computed at theta.
    * @param[in,out] rng A random number generator
    * @returns A simulated image at theta under the noise model.
    */
    template<class Model, class rng_t, typename=PoissonNoise2DObjective::IsPoissonNoise2DObjectiveT<Model>>
    typename Model::ImageT 
    simulate_image(const Model &model, const typename Model::Stencil &s, rng_t &rng)
    {
        auto sim_im=model.make_image();
        for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
            sim_im(j,i)=generate_poisson(rng,model.pixel_model_value(i,j,s));
        }
        return sim_im;
    }

    /** @brief Simulate an image using model mean values provided by a "model image" at some theta
    * @param[in] model Model object
    * @param[in] model_im Image giving the model mean value at each pixel computed at some theta.
    * @param[in,out] rng A random number generator.
    * @returns A simulated image at theta under the noise model.
    */
    template<class Model, class rng_t, typename=PoissonNoise2DObjective::IsPoissonNoise2DObjectiveT<Model>>
    typename Model::ImageT 
    simulate_image(const Model &model, const typename Model::ImageT &model_im, rng_t &rng)
    {
        auto sim_im=model.make_image();
        for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
            sim_im(j,i)=generate_poisson(rng,model_im(j,i));
        }
        return sim_im;
    }

    namespace likelihood_func {
        template<class Model, typename=PoissonNoise2DObjective::IsPoissonNoise2DObjectiveT<Model>>
        double absolute_log_likelihood(const Model &model, const typename Model::ImageT &data_im, 
                            const typename Model::Stencil &s)
        {
            double llh=0.;
            for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
                llh+=poisson_log_likelihood(model.pixel_model_value(i,j,s), data_im(j,i));
            }
//            double pllh=model.prior_log_likelihood(s.theta); /* MAP: Add log of prior for params theta */
//            return llh+pllh;
        }

template<class Model, typename=PoissonNoise2DObjective::IsPoissonNoise2DObjectiveT<Model>>
typename std::enable_if<std::is_base_of<PoissonNoise2DObjective,Model>::value,double>::type
relative_log_likelihood(const Model &model, const typename Model::ImageT &data_im,
                        const typename Model::Stencil &s)
{
    double rllh=0.;
    for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
        rllh+=relative_poisson_log_likelihood(model.pixel_model_value(i,j,s), data_im(j,i));
    }
    double prllh=model.prior_relative_log_likelihood(s.theta); /* MAP: Add relative log of prior for params theta */
    return rllh+prllh;
}

template<class Model>
typename std::enable_if<std::is_base_of<PoissonNoise2DObjective,Model>::value>::type
model_grad(const Model &model, const typename Model::ImageT &im,
           const typename Model::Stencil &s, typename Model::ParamT &grad) 
{
    auto pgrad=model.make_param();
    grad.zeros();
    for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
        if(!std::isfinite(im(j,i))) continue; /* Skip non-finite image values as they are assumed masked */
        model.pixel_grad(i,j,s,pgrad);
        double model_val=model.pixel_model_value(i,j,s);
//         assert(model_val>0.);//Model value must be positive for grad to be defined
        double dm_ratio_m1=im(j,i)/model_val - 1.;
        grad+=dm_ratio_m1*pgrad;
    }
    model.prior_grad_update(s.theta, grad); /* As appropriate for MAP/MLE: Add grad of log of prior for params theta */
}

template<class Model>
typename std::enable_if<std::is_base_of<PoissonNoise2DObjective,Model>::value>::type
model_grad2(const Model &model, const typename Model::ImageT &im, 
            const typename Model::Stencil &s, 
            typename Model::ParamT &grad, typename Model::ParamT &grad2) 
{
    grad.zeros();
    grad2.zeros();
    auto pgrad=model.make_param();
    auto pgrad2=model.make_param();
    for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
        if(!std::isfinite(im(j,i))) continue; /* Skip non-finite image values as they are assumed masked */
        /* Compute model value and ratios */
        double model_val=model.pixel_model_value(i,j,s);
//         assert(model_val>0.);//Model value must be positive for grad to be defined
        double dm_ratio=im(j,i)/model_val;
        double dm_ratio_m1=dm_ratio-1;
        double dmm_ratio=dm_ratio/model_val;
        model.pixel_grad(i,j,s,pgrad);
        model.pixel_grad2(i,j,s,pgrad2);
        grad +=dm_ratio_m1*pgrad;
        grad2+=dm_ratio_m1*pgrad2 - dmm_ratio*pgrad%pgrad;
    }
    model.prior_grad_update(s.theta,grad); /* As appropriate for MAP/MLE: Add grad of log of prior for params theta */
    model.prior_grad2_update(s.theta,grad2); /* As appropriate for MAP/MLE: Add grad2 of log of prior for params theta */
}

template<class Model>
typename std::enable_if<std::is_base_of<PoissonNoise2DObjective,Model>::value>::type
model_hessian(const Model &model, const typename Model::ImageT &im, 
              const typename Model::Stencil &s, 
              typename Model::ParamT &grad, typename Model::MatT &hess) 
{
    /* Returns hessian as an upper triangular matrix */
    grad.zeros();
    hess.zeros();
    for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
        if(!std::isfinite(im(j,i))) continue; /* Skip non-finite image values as they are assumed masked */
        /* Compute model value and ratios */
        double model_val=model.pixel_model_value(i,j,s);
//         assert(model_val>0.);//Model value must be positive for grad to be defined
        double dm_ratio=im(j,i)/model_val;
        double dm_ratio_m1=dm_ratio-1;
        double dmm_ratio=dm_ratio/model_val;
        model.pixel_hess_update(i,j,s,dm_ratio_m1,dmm_ratio,grad,hess);
    }
    model.prior_grad_update(s.theta,grad); /* As appropriate for MAP/MLE: Add grad of log of prior for params theta */
    model.prior_hess_update(s.theta,hess); /* As appropriate for MAP/MLE: Add hessian of log of prior for params theta */
}



/** @brief  */
template<class Model>
typename std::enable_if<std::is_base_of<PoissonNoise2DObjective,Model>::value,typename Model::MatT>::type
fisher_information(const Model &model, const typename Model::Stencil &s)
{
    auto fisherI=model.make_param_mat();
    fisherI.zeros();
    auto pgrad=model.make_param();
    for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
        double model_val=model.pixel_model_value(i,j,s);
        model.pixel_grad(i,j,s,pgrad);
        for(int c=0; c<model.num_params; c++) for(int r=0; r<=c; r++) {
            fisherI(r,c) += pgrad(r)*pgrad(c)/model_val; //Fill upper triangle
        }
    }
    model.prior_hess_update(s.theta,fisherI); /* As appropriate for MAP/MLE: Add diagonal hession of log of prior for params theta */
    return fisherI;
}


template<class Model>
typename std::enable_if<std::is_base_of<PoissonNoise2DObjective,Model>::value,std::shared_ptr<Estimator<Model>>>::type
make_estimator(Model &model, std::string ename)
{
    using std::make_shared;
    const char *name=ename.c_str();
    
    } else if (istarts_with(name,"Seperable")) {
        return  make_shared<SeperableHeuristicEstimator<Model>>(model);
    } else if (istarts_with(name,"CGaussHeuristic")) {
        return  make_shared<CGaussHeuristicEstimator<Model>>(model);
    } else if (istarts_with(name,"CGauss")) {
        return make_shared<CGaussMLE<Model>>(model);
    } else if (istarts_with(name,"NewtonDiagonal")) {
        return make_shared<NewtonDiagonalMaximizer<Model>>(model);
    } else if (istarts_with(name,"QuasiNewton")) {
        return make_shared<QuasiNewtonMaximizer<Model>>(model);
    } else if (istarts_with(name,"Newton")) {
        return make_shared<NewtonMaximizer<Model>>(model);
    } else if (istarts_with(name,"TrustRegion")) {
        return make_shared<TrustRegionMaximizer<Model>>(model);
    } else if (istarts_with(name,"SimulatedAnnealing")) {
        return make_shared<SimulatedAnnealingMaximizer<Model>>(model);
    } else {
        return std::shared_ptr<Estimator<Model>>();
    }
}


} /* namespace mappel */

#endif /* _POISSONNOISE2DOBJECTIVE_H */
