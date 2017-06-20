
/** @file stackcomp.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @brief
 *
 * OpenMP computation for stacked Model operations on verctor data.
 *
 *  * Design Decisions
 *  * Because we want to integrate as seamlessly as possible with matlab, we use the armadillo
 *    package which stores arrays in column major order.
 *  * Therefore in the *_stack operations, if they are to be parallelized, we want the data
 *    stored as a nParms X n matrix, i.e. each column is a parameter matrix.  Simillarly
 *    stacks are size X size X n, so that contiguous images sequences are contiguous in
 *    memory.  This avoids false sharing.
 *
 */

#ifndef _STACKCOMP_H
#define _STACKCOMP_H

#include <omp.h>
#include "rng.h"

namespace mappel {

/* Stack methods to move to base class */
template<class Model>
void sample_prior_stack(Model &model, typename Model::ParamVecT &theta_stack)
{
    int nthetas = static_cast<int>(theta_stack.n_cols);
    unsigned long seed=make_seed();
    #pragma omp parallel
    {
        RNG rng=make_parallel_rng_stream(seed);
        #pragma omp for
        for(int n=0; n<nthetas; n++){
            theta_stack.col(n) = model.sample_prior(rng);
        }
    }
}

template<class Model>
void model_image_stack(const Model &model,
                       const typename Model::ParamVecT &theta_stack,
                       typename Model::ImageStackT &image_stack)
{
    int nthetas = static_cast<int>(theta_stack.n_cols);
    #pragma omp parallel for
    for(int n=0; n<nthetas; n++)
        image_stack.slice(n) = model_image(model, theta_stack.col(n));
}



/**
 * @brief
 * This accepts either a single theta and a stack of images, or a stack of thetas
 * and a stack of images.
 *
 */
template<class Model>
void simulate_image_stack(const Model &model,
                          const typename Model::ParamVecT &theta_stack,
                          typename Model::ImageStackT &image_stack)
{
    int nimages = static_cast<int>(image_stack.n_slices);
    int nthetas = static_cast<int>(theta_stack.n_cols);
    unsigned long seed=make_seed();
    if (nimages==1 && nthetas==1) {
        RNG rng=make_parallel_rng_stream(seed);
        image_stack.slice(0)=simulate_image(model,model.make_param(theta_stack.col(0)),rng);
    } else if (nthetas==1) {
        auto model_im=model_image(model, theta_stack.col(0));
        #pragma omp parallel
        {
            RNG rng=make_parallel_rng_stream(seed);
            #pragma omp for
            for(int n=0; n<nimages; n++)
                image_stack.slice(n)=simulate_image(model, model_im,rng);
        }
    } else {
        #pragma omp parallel
        {
            RNG rng=make_parallel_rng_stream(seed);
            #pragma omp for
            for(int n=0; n<nimages; n++)
                image_stack.slice(n)=simulate_image(model,model.make_param(theta_stack.col(n)),rng);
        }
    }
}

/* This version works on a single image only */
template<class Model>
void log_likelihood_stack(const Model &model,
                          const typename Model::ImageT &image,
                          const typename Model::ParamVecT &theta_stack,
                          VecT &llh_stack)
{
    int nthetas = static_cast<int>(theta_stack.n_cols);
    #pragma omp parallel for
    for(int n=0; n<nthetas; n++)
        llh_stack(n) = log_likelihood(model, image, theta_stack.col(n));
}

/* This version works when both the images and thetas are of stack type */
template<class Model>
void log_likelihood_stack(const Model &model,
                          const typename Model::ImageStackT &image_stack,
                          const typename Model::ParamVecT &theta_stack,
                          VecT &llh_stack)
{
    int nimages = static_cast<int>(image_stack.n_slices);
    int nthetas = static_cast<int>(theta_stack.n_cols);
    if (nimages==1 && nthetas==1) {
        llh_stack(0) = log_likelihood(model, image_stack.slice(0), theta_stack.col(0));
    } else if (nthetas==1) {
        auto s=model.make_stencil(theta_stack.col(0));
        #pragma omp parallel for
        for(int n=0; n<nimages; n++)
            llh_stack(n) = log_likelihood(model, image_stack.slice(n), s);
    } else if (nimages==1) {
        #pragma omp parallel for
        for(int n=0; n<nthetas; n++)
            llh_stack(n) = log_likelihood(model, image_stack.slice(0), theta_stack.col(n));
    } else {
        #pragma omp parallel for
        for(int n=0; n<nimages; n++)
            llh_stack(n) = log_likelihood(model, image_stack.slice(n), theta_stack.col(n));
    }
}

template<class Model>
void model_grad_stack(const Model &model,
                          const typename Model::ImageStackT &image_stack,
                          const typename Model::ParamVecT &theta_stack,
                          typename Model::ParamVecT &grad_stack)
{
    int nimages = static_cast<int>(image_stack.n_slices);
    int nthetas = static_cast<int>(theta_stack.n_cols);
    if (nimages==1 && nthetas==1) {
        grad_stack.col(0) = model_grad(model, image_stack.slice(0), theta_stack.col(0));
    } else if (nthetas==1) { //Single theta multiple images
        auto s = model.make_stencil(theta_stack.col(0));
        #pragma omp parallel
        {
            auto grad = model.make_param();
            #pragma omp for
            for(int n=0; n<nimages; n++) {
                model_grad(model, image_stack.slice(n), s, grad);
                grad_stack.col(n) = grad;
            }
        }
    } else if (nimages==1) { //Single image multiple thetas
        #pragma omp parallel for
        for(int n=0; n<nthetas; n++)
            grad_stack.col(n) = model_grad(model, image_stack.slice(0), theta_stack.col(n));
    } else {
        #pragma omp parallel for
        for(int n=0; n<nthetas; n++)
            grad_stack.col(n) = model_grad(model, image_stack.slice(n), theta_stack.col(n));
    }
}

template<class Model>
void model_hessian_stack(const Model &model,
                          const typename Model::ImageStackT &image_stack,
                          const typename Model::ParamVecT &theta_stack,
                          typename Model::ParamMatStackT &hessian_stack)
{
    int nimages = static_cast<int>(image_stack.n_slices);
    int nthetas = static_cast<int>(theta_stack.n_cols);
    if (nimages==1 && nthetas==1) {
        hessian_stack.slice(0)=model_hessian(model, image_stack.slice(0), theta_stack.col(0));
    } else if (nthetas==1) { //Single theta multiple images
        auto s=model.make_stencil(theta_stack.col(0));
        #pragma omp parallel
        {
            auto grad=model.make_param();
            auto hess=model.make_param_mat();
            #pragma omp for
            for(int n=0; n<nimages; n++) {
                model_hessian(model, image_stack.slice(n), s, grad, hess);
                hessian_stack.slice(n) = hess;
            }
        }
    } else if (nimages==1) { //Single image multiple thetas
        #pragma omp parallel for
        for(int n=0; n<nthetas; n++)
            hessian_stack.slice(n) = model_hessian(model, image_stack.slice(0), theta_stack.col(n));
    } else {
        #pragma omp parallel for
        for(int n=0; n<nthetas; n++)
            hessian_stack.slice(n) = model_hessian(model, image_stack.slice(n), theta_stack.col(n));
    }
}

template<class Model>
void model_positive_hessian_stack(const Model &model,
                          const typename Model::ImageStackT &image_stack,
                          const typename Model::ParamVecT &theta_stack,
                          typename Model::ParamMatStackT &hessian_stack)
{
    int nimages=image_stack.n_slices;
    int nthetas=theta_stack.n_cols;
    if (nimages==1 && nthetas==1) {
        hessian_stack.slice(0)=model_positive_hessian(model, image_stack.slice(0), theta_stack.col(0));
    } else if (nthetas==1) { //Single theta multiple images
        //Less efficient but this is mainly for debugging anyways
        #pragma omp parallel for
        for(int n=0; n<nimages; n++) {
            hessian_stack.slice(n) = model_positive_hessian(model, image_stack.slice(n), theta_stack.col(0));
        }
    } else if (nimages==1) { //Single image multiple thetas
        #pragma omp parallel for
        for(int n=0; n<nthetas; n++)
            hessian_stack.slice(n) = model_positive_hessian(model, image_stack.slice(0), theta_stack.col(n));
    } else {
        #pragma omp parallel for
        for(int n=0; n<nthetas; n++)
            hessian_stack.slice(n) = model_positive_hessian(model, image_stack.slice(n), theta_stack.col(n));
    }
}


template<class Model>
void cr_lower_bound_stack(const Model &model,
                          const typename Model::ParamVecT &theta_stack,
                          typename Model::ParamVecT &crlb_stack)
{
    int nthetas = static_cast<int>(theta_stack.n_cols);
    #pragma omp parallel for
    for(int n=0; n<nthetas; n++)
        crlb_stack.col(n) = cr_lower_bound(model,theta_stack.col(n));
}

template<class Model>
void fisher_information_stack(const Model &model,
                          const typename Model::ParamVecT &theta_stack,
                          typename Model::ParamMatStackT &fisherI_stack)
{
    int nthetas = static_cast<int>(theta_stack.n_cols);
    #pragma omp parallel for
    for(int n=0; n<nthetas; n++)
        fisherI_stack.slice(n) = fisher_information(model,theta_stack.col(n));
}

} /* namespace mappel */

#endif /* _STACKCOMP_H */
