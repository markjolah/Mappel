
/** @file openmp_methods.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2013-2019
 * @brief Namespaces for OpenMP parallelized versions of the mappel::model namespace functions (external methods)
 *
 * OpenMP computation for stacked Model operations on vector data.
 *
 *  Design Decisions
 *  * OpenMP vectorized versions are implemented as templated external methods in inline namespaces openmp.
 *    This allows easy future replacement with other palatalization mechanisms (CUDA, C++11 threads, etc..).
 *    Also allows the vectorized versions to directly overload with the non-vectorized base-versions.
 *  * Because we want to integrate as seamlessly as possible with matlab, we use the armadillo
 *    package which stores arrays in column major order.
 *  * Therefore in the *_stack operations, if they are to be parallelized, we want the data
 *    stored as a nParms X n matrix, i.e. each column is a parameter matrix.  Similarly
 *    stacks are size X size X n, so that contiguous images sequences are contiguous in
 *    memory.  This avoids false sharing.
 *
 */

#ifndef MAPPEL_OPENMP_METHODS
#define MAPPEL_OPENMP_METHODS

#include <omp.h>
#include "Mappel/OMPExceptionCatcher/OMPExceptionCatcher.h"
#include "Mappel/util.h"
#include "Mappel/mcmc.h"

using omp_exception_catcher::Strategy;

namespace mappel {
namespace methods {
inline namespace openmp {

/** @brief Parallel sampling of the model prior.
 * 
 * Use: model.make_param_stack() to make a parameter stack of appropriate dimensions for the model
 * 
 * @tparam Model  A concrete subclass of PointEmitterModel 
 * @param[in] model, A PointEmitterModel object.
 * @param[out] theta_stack, A sequence of sampled thetas.  Size: [model.num_params, nSamples]
 */
template<class Model>
void sample_prior_stack(Model &model, ParamVecT<Model> &theta_stack)
{
    int nthetas = static_cast<int>(theta_stack.n_cols);
    omp_exception_catcher::OMPExceptionCatcher catcher(omp_exception_catcher::Strategy::Continue);
    #pragma omp parallel
    {
        auto &rng = model.get_rng_generator();
        #pragma omp for
        for(int n=0; n<nthetas; n++){
            catcher.run([&]{theta_stack.col(n) = model.sample_prior(rng);});
        }
    }
    catcher.rethrow();
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
                       const ParamVecT<Model> &theta_stack,
                       ImageStackT<Model> &image_stack)
{
    int nthetas = static_cast<int>(theta_stack.n_cols);
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel for
    for(int n=0; n<nthetas; n++)
        catcher.run([&]{
            model.set_image_in_stack(image_stack,n, model_image(model, theta_stack.col(n)));
        });
    catcher.rethrow();
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
void simulate_image_stack(Model &model,
                    const ParamVecT<Model> &theta_stack,
                    ImageStackT<Model> &image_stack)
{
    int nimages = model.get_size_image_stack(image_stack);
    int nthetas = static_cast<int>(theta_stack.n_cols);
    if (nimages==1 && nthetas==1) {
        auto &rng = model.get_rng_generator();
        model.set_image_in_stack(image_stack,0,simulate_image(model,theta_stack.col(0),rng));
    } else {
        omp_exception_catcher::OMPExceptionCatcher catcher;
        if (nthetas==1) {
            auto model_im=model_image(model, theta_stack.col(0));
            #pragma omp parallel
            {
                auto &rng = model.get_rng_generator();
                #pragma omp for
                for(int n=0; n<nimages; n++)
                    catcher.run([&]{
                        model.set_image_in_stack(image_stack,n,simulate_image_from_model(model, model_im,rng));
                    });
            }
        } else {
            #pragma omp parallel
            {
                auto &rng = model.get_rng_generator();
                #pragma omp for
                for(int n=0; n<nimages; n++)
                    catcher.run([&]{
                        model.set_image_in_stack(image_stack,n,simulate_image(model,theta_stack.col(n),rng));
                    });
            }
        }
        catcher.rethrow();
    }
}

template<class Model>
void cr_lower_bound_stack(const Model &model,
                          const ParamVecT<Model> &theta_stack,
                          ParamVecT<Model> &crlb_stack)
{
    int nthetas = static_cast<int>(theta_stack.n_cols);
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel for
    for(int n=0; n<nthetas; n++)
        catcher.run([&]{
            crlb_stack.col(n) = methods::cr_lower_bound(model,theta_stack.col(n));
        });
    catcher.rethrow();
}

template<class Model>
void expected_information_stack(const Model &model,
                          const ParamVecT<Model> &theta_stack,
                          CubeT &fisherI_stack)
{
    int nthetas = static_cast<int>(theta_stack.n_cols);
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel for
    for(int n=0; n<nthetas; n++)
        catcher.run([&]{
            fisherI_stack.slice(n) = methods::expected_information(model,theta_stack.col(n));
        });
    catcher.rethrow();
}

template<class Model>
void estimate_max_stack(Model &model, const ModelDataStackT<Model> &data_stack, const std::string &method,
                        ParamVecT<Model> &theta_max_stack, VecT &theta_max_rllh, CubeT &obsI_stack)
{
    auto estimator = make_estimator(model,method);
    estimator->estimate_max_stack(data_stack, theta_max_stack, theta_max_rllh, obsI_stack);
}

template<class Model>
void estimate_max_stack(Model &model, const ModelDataStackT<Model> &data_stack, const std::string &method,
                        ParamVecT<Model> &theta_max_stack, VecT &theta_max_rllh, CubeT &obsI_stack, StatsT &stats)
{
    auto estimator = make_estimator(model,method);
    estimator->estimate_max_stack(data_stack, theta_max_stack, theta_max_rllh, obsI_stack);
    stats = estimator->get_stats();
}

template<class Model>
void estimate_max_stack(Model &model, const ModelDataStackT<Model> &data_stack, const std::string &method, ParamVecT<Model> &theta_init_stack,
                        ParamVecT<Model> &theta_max_stack, VecT &theta_max_rllh, CubeT &obsI_stack)
{
    auto estimator = make_estimator(model,method);
    estimator->estimate_max_stack(data_stack, theta_init_stack, theta_max_stack, theta_max_rllh, obsI_stack);
}

template<class Model>
void estimate_max_stack(Model &model, const ModelDataStackT<Model> &data_stack, const std::string &method, ParamVecT<Model> &theta_init_stack,
                        ParamVecT<Model> &theta_max_stack, VecT &theta_max_rllh, CubeT &obsI_stack, StatsT &stats)
{
    auto estimator = make_estimator(model,method);
    estimator->estimate_max_stack(data_stack, theta_init_stack, theta_max_stack, theta_max_rllh, obsI_stack);
    stats = estimator->get_stats();
}

//default theta_init without stats
template<class Model>
void estimate_profile_likelihood(Model &model, const ModelDataT<Model> &data, const IdxVecT &fixed_parameters, const MatT &fixed_values, const std::string &method,
                                 VecT &profile_likelihood, ParamVecT<Model> &profile_parameters)
{
    auto estimator = make_estimator(model,method);
    auto theta_init = model.make_param_vec(fixed_values.n_elem);
    estimator->estimate_profile_stack(data, fixed_parameters, fixed_values, theta_init, profile_likelihood, profile_parameters);
}

//default theta_init with stats
template<class Model>
void estimate_profile_likelihood(Model &model, const ModelDataT<Model> &data, const IdxVecT &fixed_parameters, const MatT &fixed_values, const std::string &method,
                                 VecT &profile_likelihood, ParamVecT<Model> &profile_parameters, StatsT &stats)
{
    auto estimator = make_estimator(model,method);
    auto theta_init = model.make_param_vec(fixed_values.n_elem);
    estimator->estimate_profile_stack(data, fixed_parameters, fixed_values, theta_init, profile_likelihood, profile_parameters);
    stats = estimator->get_stats();
}

//theta_init without stats
template<class Model>
void estimate_profile_likelihood(Model &model, const ModelDataT<Model> &data, const IdxVecT &fixed_parameters, const MatT &fixed_values, const std::string &method,
                                 const ParamVecT<Model> &theta_init,
                                 VecT &profile_likelihood, ParamVecT<Model> &profile_parameters)
{
    auto estimator = make_estimator(model,method);
    estimator->estimate_profile_stack(data, fixed_parameters, fixed_values, theta_init, profile_likelihood, profile_parameters);
}

//theta_init with stats
template<class Model>
void estimate_profile_likelihood(Model &model, const ModelDataT<Model> &data, const IdxVecT &fixed_parameters, const MatT &fixed_values, const std::string &method,
                                 const ParamVecT<Model> &theta_init,
                                 VecT &profile_likelihood, ParamVecT<Model> &profile_parameters, StatsT &stats)
{
    auto estimator = make_estimator(model,method);
    estimator->estimate_profile_stack(data, fixed_parameters, fixed_values, theta_init, profile_likelihood, profile_parameters);
    stats = estimator->get_stats();
}

template <class Model>
void estimate_mcmc_sample_stack(Model &model, const ModelDataStackT<Model> &data_stack, const ParamVecT<Model> &theta_init_stack,
                                IdxT Nsamples, IdxT Nburnin, IdxT thin, CubeT &sample_stack, MatT &sample_rllh_stack)
{
    IdxT count = model.get_size_image_stack(data_stack);
    sample_stack.set_size(model.get_num_params(), Nsamples, count);
    sample_rllh_stack.set_size(Nsamples,count);
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        auto sample = model.make_param_stack(Nsamples);
        VecT sample_rllh(Nsamples);
        #pragma omp for
        for(IdxT n=0; n<count; n++)
            catcher.run([&]{
                estimate_mcmc_sample(model, model.get_image_from_stack(data_stack,n),
                                 theta_init_stack.col(n), Nsamples, Nburnin, thin, 
                                 sample, sample_rllh);
                sample_stack.slice(n) = sample;
                sample_rllh_stack.col(n) = sample_rllh;
            });
    }
    catcher.rethrow();
}

template <class Model>
void estimate_mcmc_sample_stack(Model &model, const ModelDataStackT<Model> &data_stack,
                                IdxT Nsamples, IdxT Nburnin, IdxT thin, CubeT &sample, MatT &sample_rllh)
{
    IdxT count = model.get_size_image_stack(data_stack);
    auto theta_init_stack = model.make_param_stack(count,arma::fill::zeros);
    estimate_mcmc_sample_stack(model, data_stack, theta_init_stack, Nsamples, Nburnin, thin, sample, sample_rllh);
}

template <class Model>
void estimate_mcmc_posterior_stack(Model &model, const ModelDataStackT<Model> &data_stack, const ParamVecT<Model> &theta_init_stack,
                         IdxT Nsamples, IdxT Nburnin, IdxT thin, MatT &theta_mean_stack, CubeT &theta_cov_stack)
{
    IdxT count = model.get_size_image_stack(data_stack);
    theta_mean_stack.set_size(model.get_num_params(), count);
    theta_cov_stack.set_size(model.get_num_params(),model.get_num_params(),count);
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        auto sample = model.make_param_stack(Nsamples);
        VecT sample_rllh(Nsamples);
        auto theta_mean = model.make_param();
        auto theta_cov = model.make_param_mat();
        #pragma omp for
        for(IdxT n=0; n<count; n++)
            catcher.run([&]{
                estimate_mcmc_sample(model, model.get_image_from_stack(data_stack,n),
                                 theta_init_stack.col(n), Nsamples, Nburnin, thin, 
                                 sample, sample_rllh);
                mcmc::estimate_sample_posterior(sample,theta_mean, theta_cov);
                theta_mean_stack.col(n) = theta_mean;
                theta_cov_stack.slice(n) = theta_cov;
            });
    }
    catcher.rethrow();
}

template <class Model>
void estimate_mcmc_posterior_stack(Model &model, const ModelDataStackT<Model> &data_stack,
                         IdxT Nsamples, IdxT Nburnin, IdxT thin, MatT &theta_mean_stack, CubeT &theta_cov_stack)
{
    IdxT count = model.get_size_image_stack(data_stack);
    auto theta_init_stack = model.make_param_stack(count,arma::fill::zeros);
    estimate_mcmc_posterior_stack(model, data_stack, theta_init_stack, Nsamples, Nburnin, thin, theta_mean_stack, theta_cov_stack);
}

template<class Model>
void error_bounds_expected_stack(const Model &model, const MatT &theta_est_stack, double confidence,
                            MatT &theta_lb_stack, MatT &theta_ub_stack)
{
    IdxT count = theta_est_stack.n_cols;
    theta_lb_stack.set_size(model.get_num_params(), count);
    theta_ub_stack.set_size(model.get_num_params(), count);
    auto crlb_stack = model.make_param_stack(count);
    cr_lower_bound_stack(model, theta_est_stack, crlb_stack);
    double z = normal_quantile_twosided(confidence);
    auto sqrt_crlb_stack = arma::sqrt(crlb_stack);
    theta_lb_stack = theta_est_stack - z*sqrt_crlb_stack;
    theta_ub_stack = theta_est_stack + z*sqrt_crlb_stack;        
}

template<class Model>
void error_bounds_observed_stack(const Model &model, const MatT &theta_est_stack, CubeT &obsI_stack, double confidence,
                           MatT &theta_lb_stack, MatT &theta_ub_stack)
{
    IdxT count = theta_est_stack.n_cols;
    theta_lb_stack.set_size(model.get_num_params(), count);
    theta_ub_stack.set_size(model.get_num_params(), count);
    if(obsI_stack.n_slices != count) {
        std::ostringstream msg;
        msg<<"Got inconsistent sizes.  Num theta_est:"<<count<<" #obsI:"<<obsI_stack.n_slices;
        ArrayShapeError(msg.str());
    }
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        auto obsI = model.make_param_mat();
        auto theta_lb = model.make_param();
        auto theta_ub = model.make_param();
        #pragma omp  for
        for(IdxT n=0; n<count; n++)
            catcher.run([&]{
                obsI = obsI_stack.slice(n);
                error_bounds_observed(model, theta_est_stack.col(n), obsI, confidence,
                                    theta_lb, theta_ub);
                theta_lb_stack.col(n) = theta_lb;
                theta_ub_stack.col(n) = theta_ub;
            });
    }
    catcher.rethrow();
}

template<class Model>
void error_bounds_posterior_credible_stack(const Model &model, const CubeT &sample_stack, double confidence,
                                     MatT &theta_mean_stack, MatT &theta_lb_stack, MatT &theta_ub_stack)
{
    IdxT count = sample_stack.n_slices;
    auto Np = model.get_num_params();
    theta_mean_stack.set_size(Np, count);
    theta_lb_stack.set_size(Np, count);
    theta_ub_stack.set_size(Np, count);
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        MatT sample(sample_stack.n_rows,sample_stack.n_cols);
        auto theta_mean = model.make_param();
        auto theta_lb = model.make_param();
        auto theta_ub = model.make_param();
        #pragma omp for
        for(IdxT n=0; n<count; n++)
            catcher.run([&]{
                sample = sample_stack.slice(n);
                error_bounds_posterior_credible(model, sample, confidence,
                                                theta_mean, theta_lb, theta_ub);
                theta_mean_stack.col(n) = theta_mean;
                theta_lb_stack.col(n) = theta_lb;
                theta_ub_stack.col(n) = theta_ub;
            });
    }
    catcher.rethrow();
}

} /* namespace mappel::methods::openmp */


namespace objective {

inline namespace openmp {

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
void llh_stack(const Model &model, const ImageT<Model> &image, const ParamVecT<Model> &theta_stack, VecT &llh_stack)
{
    auto nthetas = theta_stack.n_cols;
    llh_stack.set_size(nthetas);
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel for
    for(IdxT n=0; n<theta_stack.n_cols; n++)
        catcher.run([&]{
             llh_stack(n) = methods::objective::llh(model, image, theta_stack.col(n));
        });
    catcher.rethrow();
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
void llh_stack(const Model &model, const ImageStackT<Model> &image_stack, const ParamVecT<Model> &theta_stack, VecT &llh_stack)
{
    IdxT nimages = model.get_size_image_stack(image_stack);
    IdxT nthetas = theta_stack.n_cols;
    model.check_param_shape(theta_stack);
    model.check_image_shape(image_stack);
    llh_stack.set_size(std::max(nthetas,nimages));
    if (nimages==1 && nthetas==1) {
        llh_stack(0) = objective::llh(model, model.get_image_from_stack(image_stack,0), theta_stack.col(0));
    } else {
        omp_exception_catcher::OMPExceptionCatcher catcher;
        if (nthetas==1) {
            auto s = model.make_stencil(theta_stack.col(0));
            #pragma omp parallel for
            for(IdxT n=0; n<nimages; n++)
                catcher.run([&]{
                    llh_stack(n) = objective::llh(model, model.get_image_from_stack(image_stack,n), s);
                });
        } else if (nimages==1) {
            #pragma omp parallel for
            for(IdxT n=0; n<nthetas; n++)
                catcher.run([&]{
                    llh_stack(n) = objective::llh(model, model.get_image_from_stack(image_stack,0), theta_stack.col(n));
                });
        } else {
            #pragma omp parallel for
            for(IdxT n=0; n<nimages; n++)
                catcher.run([&]{
                    llh_stack(n) = objective::llh(model, model.get_image_from_stack(image_stack,n), theta_stack.col(n));
                });
        }
        catcher.rethrow();
    }
}

/** @brief Parallel relative log_likelihood calculations for a stack of images.
 * 
 * Compute relative log-likelihood for multiple image, theta pairs.
 * 
 * Use: model.make_param_stack() to make a parameter stack of appropriate dimensions for the model
 * @tparam Model  A concrete subclass of PointEmitterModel 
 * @param[in] model   A PointEmitterModel object.
 * @param[in] image_stack  Sequence of images.
 * @param[in] theta_stack    Sequence of thetas.  Size: [model.num_params, nThetas]
 * @param[out] rllh_stack Sequence of rllh values computed. Size: [n]
 */
template<class Model>
void rllh_stack(const Model &model, const ImageStackT<Model> &image_stack, const ParamVecT<Model> &theta_stack, VecT &rllh_stack)
{
    IdxT nimages = model.get_size_image_stack(image_stack);
    IdxT nthetas = theta_stack.n_cols;
    model.check_param_shape(theta_stack);
    model.check_image_shape(image_stack);
    rllh_stack.set_size(std::max(nthetas,nimages));
    if (nimages==1 && nthetas==1) {
        rllh_stack(0) = objective::rllh(model, model.get_image_from_stack(image_stack,0), theta_stack.col(0));
    } else {
        omp_exception_catcher::OMPExceptionCatcher catcher;
        if (nthetas==1) {
            auto s = model.make_stencil(theta_stack.col(0));
            #pragma omp parallel for
            for(IdxT n=0; n<nimages; n++)
                catcher.run([&]{
                    rllh_stack(n) = objective::rllh(model, model.get_image_from_stack(image_stack,n), s);
                });
        } else if (nimages==1) {
            #pragma omp parallel for
            for(IdxT n=0; n<nthetas; n++)
                catcher.run([&]{
                    rllh_stack(n) = objective::rllh(model, model.get_image_from_stack(image_stack,0), theta_stack.col(n));
                });
        } else {
            #pragma omp parallel for
            for(IdxT n=0; n<nimages; n++)
                catcher.run([&]{
                    rllh_stack(n) = objective::rllh(model, model.get_image_from_stack(image_stack,n), theta_stack.col(n));
                });
        }
        catcher.rethrow();
    }
}

template<class Model>
void rllh_stack(const Model &model, const ImageT<Model> &image, const ParamVecT<Model> &theta_stack, VecT &rllh_stack)
{
    IdxT nthetas = theta_stack.n_cols;
    model.check_param_shape(theta_stack);
    model.check_image_shape(image);
    rllh_stack.set_size(nthetas);
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel for
    for(IdxT n=0; n<nthetas; n++)
        catcher.run([&]{
            rllh_stack(n) = objective::rllh(model, image, theta_stack.col(n));
        });
    catcher.rethrow();
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
void grad_stack(const Model &model, const ImageStackT<Model> &image_stack, const ParamVecT<Model> &theta_stack,
                ParamVecT<Model> &grad_stack)
{
    IdxT nimages = model.get_size_image_stack(image_stack);
    IdxT nthetas = theta_stack.n_cols;
    model.check_param_shape(theta_stack);
    model.check_image_shape(image_stack);
    grad_stack.set_size(model.get_num_params(),std::max(nthetas,nimages));
    if (nimages==1 && nthetas==1) {
        grad_stack.col(0) = objective::grad(model, model.get_image_from_stack(image_stack,0), theta_stack.col(0));
    } else {
        omp_exception_catcher::OMPExceptionCatcher catcher;
        if (nthetas==1) { //Single theta multiple images
            auto s = model.make_stencil(theta_stack.col(0));
            #pragma omp for
            for(IdxT n=0; n<nimages; n++)
                catcher.run([&]{
                    grad_stack.col(n) = objective::grad(model, model.get_image_from_stack(image_stack,n), s);
                });
        } else if (nimages==1) { //Single image multiple thetas
            #pragma omp parallel for
            for(IdxT n=0; n<nthetas; n++)
                catcher.run([&]{
                    grad_stack.col(n) = objective::grad(model, model.get_image_from_stack(image_stack,0), theta_stack.col(n));
                });
        } else {
            #pragma omp parallel for
            for(IdxT n=0; n<nthetas; n++)
                catcher.run([&]{
                    grad_stack.col(n) = objective::grad(model, model.get_image_from_stack(image_stack,n), theta_stack.col(n));
                });
        }
        catcher.rethrow();
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
void hessian_stack(const Model &model, const ImageStackT<Model> &image_stack, const ParamVecT<Model> &theta_stack,
                    CubeT &hessian_stack)
{
    IdxT nimages = model.get_size_image_stack(image_stack);
    IdxT nthetas = theta_stack.n_cols;
    model.check_param_shape(theta_stack);
    model.check_image_shape(image_stack);
    hessian_stack.set_size(model.get_num_params(),model.get_num_params(),std::max(nthetas,nimages));
    if (nimages==1 && nthetas==1) {
        hessian_stack.slice(0) = objective::hessian(model, model.get_image_from_stack(image_stack,0), theta_stack.col(0));
    } else {
        omp_exception_catcher::OMPExceptionCatcher catcher;
        if (nthetas==1) { //Single theta multiple images
            auto s = model.make_stencil(theta_stack.col(0));
            #pragma omp for
            for(IdxT n=0; n<nimages; n++)
                catcher.run([&]{
                    hessian_stack.slice(n) = objective::hessian(model, model.get_image_from_stack(image_stack,n), s);
                });
        } else if (nimages==1) { //Single image multiple thetas
            #pragma omp parallel for
            for(IdxT n=0; n<nthetas; n++)
                catcher.run([&]{
                    hessian_stack.slice(n) = objective::hessian(model, model.get_image_from_stack(image_stack,0), theta_stack.col(n));
                });
        } else {
            #pragma omp parallel for
            for(IdxT n=0; n<nthetas; n++)
                catcher.run([&]{
                    hessian_stack.slice(n) = objective::hessian(model, model.get_image_from_stack(image_stack,n), theta_stack.col(n));
                });
        }
        catcher.rethrow();
    }
}

/** @brief Parallel model negative_definite Hessian approximation calculations for a stack of images.
 * 
 * Compute Hessian a negative_definite Hessian using a modified cholesky decompositions. 
 * Computes for multiple image, theta pairs.
 * 
 * Use: model.make_param_mat_stack() to make a parameter matrix stack of appropriate dimensions for the model Hessian.
 * @tparam Model  A concrete subclass of PointEmitterModel 
 * @param[in] model   A PointEmitterModel object.
 * @param[in] image_stack  Sequence of images.
 * @param[in] theta_stack  Sequence of thetas.  Size: [model.num_params, nThetas]
 * @param[out] hess_stack  Sequence of approximate Hessian negative definite matrices computed. Size: [model.num_params, model.num_params, n]
 */
template<class Model>
void negative_definite_hessian_stack(const Model &model, const ImageStackT<Model> &image_stack, const ParamVecT<Model> &theta_stack,
                                     CubeT &hessian_stack)
{
    IdxT nimages = model.get_size_image_stack(image_stack);
    IdxT nthetas = theta_stack.n_cols;
    model.check_param_shape(theta_stack);
    model.check_image_shape(image_stack);
    hessian_stack.set_size(model.get_num_params(),model.get_num_params(),std::max(nthetas,nimages));
    if (nimages==1 && nthetas==1) {
        hessian_stack.slice(0) = objective::negative_definite_hessian(model, model.get_image_from_stack(image_stack,0), theta_stack.col(0));
    } else {
        omp_exception_catcher::OMPExceptionCatcher catcher;
        if (nthetas==1) { //Single theta multiple images
            auto s=model.make_stencil(theta_stack.col(0));
            #pragma omp parallel for
            for(IdxT n=0; n<nimages; n++)
                catcher.run([&]{
                    hessian_stack.slice(n) = objective::negative_definite_hessian(model, model.get_image_from_stack(image_stack,n), s);
                });
        } else if (nimages==1) { //Single image multiple thetas
            #pragma omp parallel for
            for(IdxT n=0; n<nthetas; n++)
                catcher.run([&]{
                    hessian_stack.slice(n) = objective::negative_definite_hessian(model, model.get_image_from_stack(image_stack,0), theta_stack.col(n));
                });
        } else {
            #pragma omp parallel for
            for(IdxT n=0; n<nthetas; n++)
                catcher.run([&]{
                    hessian_stack.slice(n) = objective::negative_definite_hessian(model, model.get_image_from_stack(image_stack,n), theta_stack.col(n));
                });
        }
        catcher.rethrow();
    }
}

} /* namespace mappel::methods::objective::openmp */
} /* namespace mappel::methods::objective */
} /* namespace mappel::methods */
} /* namespace mappel */

#endif /* MAPPEL_OPENMP_METHODS */
