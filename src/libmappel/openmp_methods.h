
/** @file openmp_methods.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2013-2017
 * @brief Namespaces for OpenMP parallelized verions of the mappel::model namespace functions (external methods)
 *
 * OpenMP computation for stacked Model operations on verctor data.
 *
 *  Design Decisions
 *  * OpenMP vectorized versions are implemented as templated external methods in inline namespaces openmp.
 *    This allows easy future replacement with other parallelization mechanisms (CUDA, C++11 threads, etc..).
 *    Also allows the vectorized versions to directly overload with the non-vectorized base-versions.
 *  * Because we want to integrate as seamlessly as possible with matlab, we use the armadillo
 *    package which stores arrays in column major order.
 *  * Therefore in the *_stack operations, if they are to be parallelized, we want the data
 *    stored as a nParms X n matrix, i.e. each column is a parameter matrix.  Simillarly
 *    stacks are size X size X n, so that contiguous images sequences are contiguous in
 *    memory.  This avoids false sharing.
 *
 */

#ifndef _MAPPEL_OPENMP_METHODS
#define _MAPPEL_OPENMP_METHODS

#include <omp.h>
#include "model_methods.h"

namespace mappel {

    inline namespace openmp { 
        /* model::llh - openmp vectorized */
        template<class Model>
        VecT llh(const Model &model, const ModelDataT<Model> &data, const ParamVecT<Model> &thetas);

        template<class Model>
        VecT llh(const Model &model, const ModelDataT<Model> &data, const StencilVecT<Model> &thetas);
        
        template<class Model>
        VecT llh(const Model &model, const ModelDataStackT<Model> &datas, const ParamT<Model> &theta);

        template<class Model>
        VecT llh(const Model &model, const ModelDataStackT<Model> &datas, const StencilT<Model> &theta);

        template<class Model>
        VecT llh(const Model &model, const ModelDataStackT<Model> &datas, const ParamVecT<Model> &thetas);

        template<class Model>
        VecT llh(const Model &model, const ModelDataStackT<Model> &datas, const StencilVecT<Model> &thetas);
        
    } /* namespace openmp */
    
    namespace likelihood_func {
        inline namespace openmp { 

            
        } /* namespace openmp */                
    } /* namespace likelihood_func */
    namespace prior {
        inline namespace openmp { 

            
        } /* namespace openmp */                
    } /* namespace prior */
} /* namespace mappel */
/** @brief Parallel sampling of the model prior.
 * 
 * Use: model.make_param_stack() to make a parameter stack of appropriate dimensions for the model
 * 
 * @tparam Model  A concrete subclass of PointEmitterModel 
 * @param[in] model, A PointEmitterModel object.
 * @param[out] theta_stack, A sequence of sampled thetas.  Size: [model.num_params, nSamples]
 */
template<class Model>
void sample_prior_stack(Model &model, typename Model::ParamVecT &theta_stack)
{
    int nthetas = static_cast<int>(theta_stack.n_cols);
    #pragma omp parallel
    {
        auto rng = rng_manager.generator();
        #pragma omp for
        for(int n=0; n<nthetas; n++){
            theta_stack.col(n) = model.sample_prior(rng);
        }
    }
}

/** @brief Parallel computation of the model image.
 * 
 * The model image is the expected photon count at each pixel under parameter theta.
 *
 * Use: model.make_param_stack() to make a parameter stack of appropriate dimensions for the model
 * Use: model.make_image_stack() to make an image stack of appropriate dimensions for the model
 * 
 * @tparam Model  A concrete subclass of PointEmitterModel 
 * @param[in] model      A PointEmitterModel object.
 * @param[in] theta_stack    Sequence of thetas for which to generate images.  Size: [model.num_params, nThetas]
 * @param[out] image_stack   Sequence of model images generated.
 */
template<class Model>
void model_image_stack(const Model &model,
                       const typename Model::ParamVecT &theta_stack,
                       typename Model::ImageStackT &image_stack)
{
    int nthetas = static_cast<int>(theta_stack.n_cols);
    #pragma omp parallel for
    for(int n=0; n<nthetas; n++)
        model.get_image_from_stack(image_stack,n) = model_image(model, theta_stack.col(n));
}



/** @brief Parallel simulation of images from one or more theta.
 * 
 * This accepts either a single theta and a stack of images, or a stack of thetas
 * and a stack of images.
 * 
 * Use: model.make_param_stack() to make a parameter stack of appropriate dimensions for the model
 * Use: model.make_image_stack() to make an image stack of appropriate dimensions for the model
 * @tparam Model  A concrete subclass of PointEmitterModel 
 * @param[in] model      A PointEmitterModel object.
 * @param[in] theta_stack    Single theta or a sequence of thetas.  Size: [model.num_params, nThetas]
 * @param[out] image_stack   Sequence of model images generated.
 */
template<class Model>
void simulate_image_stack(const Model &model,
                    const typename Model::ParamVecT &theta_stack,
                    typename Model::ImageStackT &image_stack)
{
    int nimages = model.size_image_stack(image_stack);
    int nthetas = static_cast<int>(theta_stack.n_cols);
    if (nimages==1 && nthetas==1) {
        auto rng = rng_manager.generator();
        model.get_image_from_stack(image_stack,0) = simulate_image(model,theta_stack.col(0),rng);
    } else if (nthetas==1) {
        auto model_im=model_image(model, theta_stack.col(0));
        #pragma omp parallel
        {
            auto rng = rng_manager.generator();
            #pragma omp for
            for(int n=0; n<nimages; n++)
                model.get_image_from_stack(image_stack,n) = simulate_image_from_model(model, model_im,rng);
        }
    } else {
        #pragma omp parallel
        {
            auto rng = rng_manager.generator();
            #pragma omp for
            for(int n=0; n<nimages; n++)
                model.get_image_from_stack(image_stack,n) = simulate_image(model,theta_stack.col(n),rng);
        }
    }
}

/** @brief Parallel log_likelihood calculations for a single image.
 * 
 * Compute log-likelihood for multiple thetas using the same image
 * 
 * Use: model.make_param_stack() to make a parameter stack of appropriate dimensions for the model
 * @tparam Model  A concrete subclass of PointEmitterModel 
 * @param[in] model   A PointEmitterModel object.
 * @param[in] image   An image.
 * @param[in] theta_stack    Sequence of thetas.  Size: [model.num_params, nThetas]
 * @param[out] llh_stack Sequence of llh values computed.
 */
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

/** @brief Parallel log_likelihood calculations for a stack of images.
 * 
 * Compute log-likelihood for multiple image, theta pairs.
 * 
 * Use: model.make_param_stack() to make a parameter stack of appropriate dimensions for the model
 * @tparam Model  A concrete subclass of PointEmitterModel 
 * @param[in] model   A PointEmitterModel object.
 * @param[in] image_stack  Sequence of images.
 * @param[in] theta_stack    Sequence of thetas.  Size: [model.num_params, nThetas]
 * @param[out] llh_stack Sequence of llh values computed. Size: [n]
 */
template<class Model>
void log_likelihood_stack(const Model &model,
                    const typename Model::ImageStackT &image_stack,
                    const typename Model::ParamVecT &theta_stack,
                    VecT &llh_stack)
{
    int nimages = model.size_image_stack(image_stack);
    int nthetas = static_cast<int>(theta_stack.n_cols);
    if (nimages==1 && nthetas==1) {
        llh_stack(0) = log_likelihood(model, model.get_image_from_stack(image_stack,0), theta_stack.col(0));
    } else if (nthetas==1) {
        auto s=model.make_stencil(theta_stack.col(0));
        #pragma omp parallel for
        for(int n=0; n<nimages; n++)
            llh_stack(n) = log_likelihood(model, model.get_image_from_stack(image_stack,n), s);
    } else if (nimages==1) {
        #pragma omp parallel for
        for(int n=0; n<nthetas; n++)
            llh_stack(n) = log_likelihood(model, model.get_image_from_stack(image_stack,0), theta_stack.col(n));
    } else {
        #pragma omp parallel for
        for(int n=0; n<nimages; n++)
            llh_stack(n) = log_likelihood(model, model.get_image_from_stack(image_stack,n), theta_stack.col(n));
    }
}

/** @brief Parallel model gradient calculations for a stack of images.
 * 
 * Compute gradient of log-likelihood for multiple image, theta pairs.
 * 
 * Use: model.make_param_stack() to make a parameter stack of appropriate dimensions for the model gradients.
 * @tparam Model  A concrete subclass of PointEmitterModel 
 * @param[in] model   A PointEmitterModel object.
 * @param[in] image_stack  Sequence of images.
 * @param[in] theta_stack  Sequence of thetas.  
 * @param[out] grad_stack  Sequence of grad vectors values computed. Size: [model.num_params, n]
 */
template<class Model>
void model_grad_stack(const Model &model,
                          const typename Model::ImageStackT &image_stack,
                          const typename Model::ParamVecT &theta_stack,
                          typename Model::ParamVecT &grad_stack)
{
    int nimages = model.size_image_stack(image_stack);
    int nthetas = static_cast<int>(theta_stack.n_cols);
    if (nimages==1 && nthetas==1) {
        grad_stack.col(0) = model_grad(model, model.get_image_from_stack(image_stack,0), theta_stack.col(0));
    } else if (nthetas==1) { //Single theta multiple images
        auto s = model.make_stencil(theta_stack.col(0));
        #pragma omp parallel
        {
            auto grad = model.make_param();
            #pragma omp for
            for(int n=0; n<nimages; n++) {
                model_grad(model, model.get_image_from_stack(image_stack,n), s, grad);
                grad_stack.col(n) = grad;
            }
        }
    } else if (nimages==1) { //Single image multiple thetas
        #pragma omp parallel for
        for(int n=0; n<nthetas; n++)
            grad_stack.col(n) = model_grad(model, model.get_image_from_stack(image_stack,0), theta_stack.col(n));
    } else {
        #pragma omp parallel for
        for(int n=0; n<nthetas; n++)
            grad_stack.col(n) = model_grad(model, model.get_image_from_stack(image_stack,n), theta_stack.col(n));
    }
}

/** @brief Parallel model Hessian calculations for a stack of images.
 * 
 * Compute Hessian of log-likelihood for multiple image, theta pairs.
 * 
 * Use: model.make_param_mat_stack() to make a parameter matrix stack of appropriate dimensions for the model Hessian.
 * @tparam Model  A concrete subclass of PointEmitterModel 
 * @param[in] model   A PointEmitterModel object.
 * @param[in] image_stack  Sequence of images.
 * @param[in] theta_stack  Sequence of thetas.  Size: [model.num_params, nThetas]
 * @param[out] hess_stack  Sequence of Hessian matrices computed. Size: [model.num_params, model.num_params, n]
 */
template<class Model>
void model_hessian_stack(const Model &model,
                          const typename Model::ImageStackT &image_stack,
                          const typename Model::ParamVecT &theta_stack,
                          CubeT &hessian_stack)
{
    int nimages = model.size_image_stack(image_stack);
    int nthetas = static_cast<int>(theta_stack.n_cols);
    if (nimages==1 && nthetas==1) {
        hessian_stack.slice(0)=model_hessian(model, model.get_image_from_stack(image_stack,0), theta_stack.col(0));
    } else if (nthetas==1) { //Single theta multiple images
        auto s=model.make_stencil(theta_stack.col(0));
        #pragma omp parallel
        {
            auto grad=model.make_param();
            auto hess=model.make_param_mat();
            #pragma omp for
            for(int n=0; n<nimages; n++) {
                model_hessian(model, model.get_image_from_stack(image_stack,n), s, grad, hess);
                hessian_stack.slice(n) = hess;
            }
        }
    } else if (nimages==1) { //Single image multiple thetas
        #pragma omp parallel for
        for(int n=0; n<nthetas; n++)
            hessian_stack.slice(n) = model_hessian(model, model.get_image_from_stack(image_stack,0), theta_stack.col(n));
    } else {
        #pragma omp parallel for
        for(int n=0; n<nthetas; n++)
            hessian_stack.slice(n) = model_hessian(model, model.get_image_from_stack(image_stack,n), theta_stack.col(n));
    }
}

/** @brief Parallel model positive-definite Hessian approximation calculations for a stack of images.
 * 
 * Compute Hessian a positive-definite Hessian using a modified cholesky decompositions. 
 * Computes for multiple image, theta pairs.
 * 
 * Use: model.make_param_mat_stack() to make a parameter matrix stack of appropriate dimensions for the model Hessian.
 * @tparam Model  A concrete subclass of PointEmitterModel 
 * @param[in] model   A PointEmitterModel object.
 * @param[in] image_stack  Sequence of images.
 * @param[in] theta_stack  Sequence of thetas.  Size: [model.num_params, nThetas]
 * @param[out] hess_stack  Sequence of approximate Hessian matrices computed. Size: [model.num_params, model.num_params, n]
 */
template<class Model>
void model_positive_hessian_stack(const Model &model,
                          const typename Model::ImageStackT &image_stack,
                          const typename Model::ParamVecT &theta_stack,
                          CubeT &hessian_stack)
{
    int nimages = model.size_image_stack(image_stack);
    int nthetas = theta_stack.n_cols;
    if (nimages==1 && nthetas==1) {
        hessian_stack.slice(0) = model_positive_hessian(model, model.get_image_from_stack(image_stack,0), theta_stack.col(0));
    } else if (nthetas==1) { //Single theta multiple images
        //Less efficient but this is mainly for debugging anyways
        #pragma omp parallel for
        for(int n=0; n<nimages; n++) {
            hessian_stack.slice(n) = model_positive_hessian(model, model.get_image_from_stack(image_stack,n), theta_stack.col(0));
        }
    } else if (nimages==1) { //Single image multiple thetas
        #pragma omp parallel for
        for(int n=0; n<nthetas; n++)
            hessian_stack.slice(n) = model_positive_hessian(model, model.get_image_from_stack(image_stack,0), theta_stack.col(n));
    } else {
        #pragma omp parallel for
        for(int n=0; n<nthetas; n++)
            hessian_stack.slice(n) = model_positive_hessian(model, model.get_image_from_stack(image_stack,n), theta_stack.col(n));
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
                          CubeT &fisherI_stack)
{
    int nthetas = static_cast<int>(theta_stack.n_cols);
    #pragma omp parallel for
    for(int n=0; n<nthetas; n++)
        fisherI_stack.slice(n) = fisher_information(model,theta_stack.col(n));
}

} /* namespace mappel */

#endif /* _MAPPEL_OPENMP_METHODS */
