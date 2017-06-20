/** @file PoissonGaussianNoiseMAPObjective.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 04-2017
 * @brief The class declaration and inline and templated functions for PoissonGaussianNoiseMAPObjective.
 *
 * 
 */

#ifndef _POISSONGAUSSIANNOISEMAPOBJECTIVE_H
#define _POISSONGAUSSIANNOISEMAPOBJECTIVE_H

#include "PoissonNoise2D.h"


/** @brief A Base type for point emitter localization models that use 2d images
 *
 * We don't assume much here, so that it is possible to have a wide range of 2D models
 *
 * 
 * 
 */
template<typename ModelBase>
class PoissonGaussianNoiseMAPObjective : public PointNoise2D {
public:
    typedef arma::mat ImageT; /**< A type to represent image data*/
    typedef arma::cube ImageStackT; /**< A type to represent image data stacks */

    static const std::vector<std::string> estimator_names;     /**< Estimator Names defined for this class */

    
    /* Model parameters */
    const IVecT size; /**< The number of pixels in the X and Y directions, given as [X,Y].  Note that images have shape [size(1),size(0)], Y is rows X is columns.   */
    const VecT psf_sigma; /**< The standard deviation of the stymmetric gaussian PSF in units of pixels for X and Y */

    PoissonGaussianNoiseMAPObjective(int num_params, const IVecT &size, const VecT &psf_sigma);
    StatsT get_stats() const;

    ImageT make_image() const;
    ImageStackT make_image_stack(int n) const;
};

/* Inline Method Definitions */

inline
PoissonGaussianNoiseMAPObjective::ImageT
PoissonGaussianNoiseMAPObjective::make_image() const
{
    return ImageT(size(1),size(0)); //Images are size [Y x]
}

inline
PoissonGaussianNoiseMAPObjective::ImageStackT
PoissonGaussianNoiseMAPObjective::make_image_stack(int n) const
{
    return ImageStackT(size(1),size(0),n);
}

/* Templated Function Definitions */

template<class Model>
typename std::enable_if<std::is_base_of<PoissonGaussianNoiseMAPObjective,Model>::value,typename Model::ImageT>::type
model_image(const Model &model, const typename Model::Stencil &s)
{
    auto im=model.make_image();
    for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=xposition=column; j=yposition=row
        im(j,i)=model.model_value(i,j,s);
        assert(im(j,i)>0.);//Model value must be positive for grad to be defined
    }
    return im;
}



/** @brief Simulate an image using the PSF model, by generating Poisson noise
 * @param[out] image An image to populate.
 * @param[in] theta The parameter values to us
 * @param[in,out] rng An initialized random number generator
 */
template<class Model, class rng_t>
typename std::enable_if<std::is_base_of<PoissonGaussianNoiseMAPObjective,Model>::value,typename Model::ImageT>::type
simulate_image(const Model &model, const typename Model::Stencil &s, rng_t &rng)
{
    auto sim_im=model.make_image();
    for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
        sim_im(j,i)=generate_poisson(rng,model.model_value(i,j,s));
    }
    return sim_im;
}

template<class Model, class rng_t>
typename std::enable_if<std::is_base_of<PoissonGaussianNoiseMAPObjective,Model>::value,typename Model::ImageT>::type
simulate_image(const Model &model, const typename Model::ImageT &model_im, rng_t &rng)
{
    auto sim_im=model.make_image();
    for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
        sim_im(j,i)=generate_poisson(rng,model_im(j,i));
    }
    return sim_im;
}


template<class Model>
typename std::enable_if<std::is_base_of<PoissonGaussianNoiseMAPObjective,Model>::value>::type
model_grad(const Model &model, const typename Model::ImageT &im,
           const typename Model::Stencil &s, typename Model::ParamT &grad) 
{
    auto pgrad=model.make_param();
    grad.zeros();
    for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
        model.pixel_grad(i,j,s,pgrad);
        double model_val=model.model_value(i,j,s);
        assert(model_val>0.);//Model value must be positive for grad to be defined
        double dm_ratio_m1=im(j,i)/model_val - 1.;
        grad+=dm_ratio_m1*pgrad;
    }
    grad+=model.prior_grad(s);
    assert(grad.is_finite()); 
}

template<class Model>
typename std::enable_if<std::is_base_of<PoissonGaussianNoiseMAPObjective,Model>::value>::type
model_grad2(const Model &model, const typename Model::ImageT &im, 
            const typename Model::Stencil &s, 
            typename Model::ParamT &grad, typename Model::ParamT &grad2) 
{
    grad.zeros();
    grad2.zeros();
    auto pgrad=model.make_param();
    auto pgrad2=model.make_param();
    for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
        /* Compute model value and ratios */
        double model_val=model.model_value(i,j,s);
        assert(model_val>0.);//Model value must be positive for grad to be defined
        double dm_ratio=im(j,i)/model_val;
        double dm_ratio_m1=dm_ratio-1;
        double dmm_ratio=dm_ratio/model_val;
        model.pixel_grad(i,j,s,pgrad);
        model.pixel_grad2(i,j,s,pgrad2);
        grad +=dm_ratio_m1*pgrad;
        grad2+=dm_ratio_m1*pgrad2 - dmm_ratio*pgrad%pgrad;
    }
    grad+=model.prior_grad(s);
    grad2+=model.prior_grad2(s);
    assert(grad.is_finite());
    assert(grad2.is_finite());
}


template<class Model>
typename std::enable_if<std::is_base_of<PoissonGaussianNoiseMAPObjective,Model>::value>::type
model_hessian(const Model &model, const typename Model::ImageT &im, 
              const typename Model::Stencil &s, 
              typename Model::ParamT &grad, typename Model::ParamMatT &hess) 
{
    /* Returns hess as an upper triangular matrix */
    grad.zeros();
    hess.zeros();
    for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
        /* Compute model value and ratios */
        double model_val=model.model_value(i,j,s);
        assert(model_val>0.);//Model value must be positive for grad to be defined
        double dm_ratio=im(j,i)/model_val;
        double dm_ratio_m1=dm_ratio-1;
        double dmm_ratio=dm_ratio/model_val;
        model.pixel_hess_update(i,j,s,dm_ratio_m1,dmm_ratio,grad,hess);
    }
    grad+=model.prior_grad(s);
    hess.diag()+=model.prior_grad2(s);
    assert(grad.is_finite());
    assert(hess.is_finite());
}

template<class Model>
typename std::enable_if<std::is_base_of<PoissonGaussianNoiseMAPObjective,Model>::value,double>::type
log_likelihood(const Model &model, const typename Model::ImageT &data_im, 
               const typename Model::Stencil &s)
{
    double llh=0;
    for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
        llh+=poisson_log_likelihood(model.model_value(i,j,s), data_im(j,i));
    }
    double pllh=model.prior_log_likelihood(s);
    return llh+pllh;
}

template<class Model>
typename std::enable_if<std::is_base_of<PoissonGaussianNoiseMAPObjective,Model>::value,double>::type
relative_log_likelihood(const Model &model, const typename Model::ImageT &data_im,
                        const typename Model::Stencil &s)
{
    double rllh=0;
    for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
        rllh+=relative_poisson_log_likelihood(model.model_value(i,j,s), data_im(j,i));
    }
    double prllh=model.prior_relative_log_likelihood(s);
    return rllh+prllh;
}


/** @brief  */
template<class Model>
typename std::enable_if<std::is_base_of<PoissonGaussianNoiseMAPObjective,Model>::value,typename Model::MatT>::type
fisher_information(const Model &model, const typename Model::Stencil &s)
{
    typename Model::MatT fisherI=model.make_param_mat();
    fisherI.zeros();
    auto pgrad=model.make_param();
    for(int i=0;i<model.size(0);i++) for(int j=0;j<model.size(1);j++) {  // i=x position=column; j=yposition=row
        double model_val=model.model_value(i,j,s);
        model.pixel_grad(i,j,s,pgrad);
        for(int c=0; c<model.num_params; c++) for(int r=0; r<=c; r++) {
            fisherI(r,c) += pgrad(r)*pgrad(c)/model_val; //Fill upper triangle
        }
    }
    return fisherI;
}

template<class Model>
typename std::enable_if<std::is_base_of<PoissonGaussianNoiseMAPObjective,Model>::value,std::shared_ptr<Estimator<Model>>>::type
make_estimator(Model &model, std::string ename)
{
    using std::make_shared;
    const char *name=ename.c_str();
    if (istarts_with(name,"Heuristic")) {
        return make_shared<HeuristicEstimator<Model>>(model);
    } else if (istarts_with(name,"CGaussHeuristic")) {
        return  make_shared<CGaussHeuristicEstimator<Model>>(model);
    } else if (istarts_with(name,"CGauss")) {
        return make_shared<CGaussMLE<Model>>(model);
    } else if (istarts_with(name,"NewtonRaphson")) {
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

#endif /* _POISSONGAUSSIANNOISEMAPOBJECTIVE_H */
