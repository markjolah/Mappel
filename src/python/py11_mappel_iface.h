/** @file py11_mappel_iface.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 01-2018
 * @brief Definitions for the py11_armadillo namespace, numpy to armadillo conversions
 */

#ifndef _PY11_ARMADILLO_IFACE
#define _PY11_ARMADILLO_IFACE

#include <cstdint>
#include <sstream>
#include <type_traits>
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "python_error.h"
#include "py11_armadillo_iface.h"
#include "Gauss1DMLE.h"

namespace mappel {
namespace python {

namespace py = pybind11;
namespace pyarma = py11_armadillo;
using pyarma::IdxT;
using pyarma::ArrayT;
using pyarma::ArrayDoubleT;
using pyarma::ColumnMajorOrder;


template<class Model>
typename std::enable_if<std::is_base_of<ImageFormat1DBase,Model>::value, ImageT<Model>>::type
imageAsArma(ArrayDoubleT &im)
{
    return {static_cast<double*>(im.mutable_data(0)), static_cast<IdxT>(im.size()), false, true};
}

template<class Model>
typename std::enable_if<std::is_base_of<ImageFormat1DBase,Model>::value, ImageStackT<Model>>::type
imageStackAsArma(ArrayDoubleT &im)
{
    return {static_cast<double*>(im.mutable_data(0,0)), static_cast<IdxT>(im.shape(0)), static_cast<IdxT>(im.shape(1)), false, true};
}


template<class ElemT=double>
ArrayT<ElemT, ColumnMajorOrder> 
makeImageArray(uint32_t shape)
{
    return ArrayT<ElemT, ColumnMajorOrder>({shape});
}


template<class ElemT=double, class IntT=uint32_t>
ArrayT<ElemT, ColumnMajorOrder> 
makeImageArray(arma::Col<IntT> shape)
{
    switch(shape.n_elem) {
        case 1:
            return ArrayT<ElemT, ColumnMajorOrder>({shape(0)});
        case 2:
            return ArrayT<ElemT, ColumnMajorOrder>({shape(0),shape(1)});
        case 3:
            return ArrayT<ElemT, ColumnMajorOrder>({shape(0),shape(1), shape(2)});
        case 4:
            return ArrayT<ElemT, ColumnMajorOrder>({shape(0),shape(1), shape(2), shape(3)});
        default:
            std::ostringstream msg;
            msg<<"Unable to create array of size:"<<shape.n_elem<<" val:"<<shape.t();
            throw PythonError("ConversionError",msg.str());
    }
}

template<class ElemT=double>
ArrayT<ElemT, ColumnMajorOrder> 
makeImageStackArray(uint32_t shape, uint32_t count)
{
    return ArrayT<ElemT, ColumnMajorOrder>({shape,count});
}

template<class ElemT=double, class IntT=uint32_t>
ArrayT<ElemT, ColumnMajorOrder> 
makeImageStackArray(arma::Col<IntT> shape, IntT count)
{
    switch(shape.n_elem) {
        case 1:
            return ArrayT<ElemT, ColumnMajorOrder>({shape(0),count});
        case 2:
            return ArrayT<ElemT, ColumnMajorOrder>({shape(0),shape(1),count});
        case 3:
            return ArrayT<ElemT, ColumnMajorOrder>({shape(0),shape(1), shape(2),count});
        default:
            std::ostringstream msg;
            msg<<"Unable to create array stack of size:"<<shape.n_elem<<" val:"<<shape.t();
            throw PythonError("ConversionError",msg.str());
    }
}


template<class Model>
class ModelWrapper
{
    using ImageCoordT = typename Model::ImageCoordT;
public:
    static ArrayDoubleT get_hyperparams(Model &model);
    static void set_hyperparams(Model &model, ArrayDoubleT &arr);
    static ArrayDoubleT get_lbound(Model &model);
    static void set_lbound(Model &model, ArrayDoubleT &arr);
    static ArrayDoubleT get_ubound(Model &model);
    static void set_ubound(Model &model, ArrayDoubleT &arr);
    static ArrayDoubleT bounded_theta(Model &model, ArrayDoubleT &arr);
    static ArrayDoubleT reflected_theta(Model &model, ArrayDoubleT &arr);
    static bool theta_in_bounds(Model &model, ArrayDoubleT &arr);
    static double find_hyperparam(Model &model, std::string param_name, double default_val);

    static ArrayDoubleT sample_prior(Model &model); 
    static ArrayDoubleT sample_prior_stack(Model &model, IdxT count);

    static ArrayDoubleT model_image_stack(Model &model, ArrayDoubleT &thetas);
    static ArrayDoubleT simulate_image_stack(Model &model, ArrayDoubleT &thetas, IdxT count);
    static ArrayDoubleT objective_llh_stack(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
    static ArrayDoubleT objective_rllh_stack(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
    static ArrayDoubleT objective_grad_stack(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
    static ArrayDoubleT objective_hessian_stack(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
    static ArrayDoubleT objective_negative_definite_hessian_stack(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
    static py::tuple objective(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta);
    static py::tuple likelihood_objective(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta);
    static py::tuple prior_objective(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta);
    static py::tuple aposteriori_objective(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta);
    
    static ArrayDoubleT cr_lower_bound(Model &model, ArrayDoubleT &thetas);
    static ArrayDoubleT expected_information(Model &model, ArrayDoubleT &thetas);
    static ArrayDoubleT observed_information(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta_mode);
    
    static py::tuple estimate_max(Model &model, ArrayDoubleT &image_stack, std::string method, ArrayDoubleT &theta_init_stack, bool return_stats );
//     static py::tuple estimate_profile_max(Model &model, ArrayDoubleT &images, std::string method, 
//                                           ArrayDoubleT &fixed_theta_stack, ArrayDoubleT &theta_init_stack, bool return_stats );
    static py::tuple estimate_mcmc_posterior(Model &model, ArrayDoubleT &image_stack, ArrayDoubleT &theta_init_stack, IdxT Nsample, IdxT Nburnin, IdxT thin);
    static py::tuple estimate_mcmc_sample(Model &model, ArrayDoubleT &image_stack, ArrayDoubleT &theta_init_stack, IdxT Nsample, IdxT Nburnin, IdxT thin);

    static py::tuple error_bounds_expected(Model &model, ArrayDoubleT &thetas, double confidence);
    static py::tuple error_bounds_observed(Model &model, ArrayDoubleT &thetas, ArrayDoubleT &obsI_stack, double confidence);
//     static py::tuple error_bounds_profile(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas, double confidence);
    static py::tuple error_bounds_posterior_credible(Model &model, ArrayDoubleT &sample_stack, double confidence);

    static ArrayDoubleT objective_llh_components(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta);
    static ArrayDoubleT objective_rllh_components(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta);
    static ArrayDoubleT objective_grad_components(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta);
    static ArrayDoubleT objective_hessian_components(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta);
        
    static py::tuple estimate_max_debug(Model &model, ArrayDoubleT &image, std::string method, ArrayDoubleT &theta_init);
    static py::tuple estimate_mcmc_debug(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta_init, IdxT Nsample);
};

template<class Model>
void bindMappelModel(py::module &M)
{
    py::class_<Model> model(M, Model::name.c_str(), py::multiple_inheritance());
    if(std::is_base_of<Gauss1DModel,Model>::value) {
        model.def(py::init<typename Model::ImageCoordT,double>());
        model.def_property("size",[](Model &model) {return model.get_size();},[](Model &model,IdxT size) { model.set_size(size); },"1D-Image size in pixels." );
        model.def_property("psf_sigma",[](Model &model) {return model.get_psf_sigma();},[](Model &model,double sigma) { model.set_psf_sigma(sigma); },"Sigma of emitter (PSF) Gaussian approximation [pixels]." );
    }
    model.def_property_readonly("name",[](Model &model) {return Model::name;},
                                         "Model name.");
    model.def_property_readonly("estimator_names",[](Model &model) {return Model::estimator_names;},
                                         "Available MLE/MAP estimator names.");
    model.def_property_readonly("num_params",[](Model &model) {return model.get_num_params();},
                                  "Number of model parameters (dimensionality).");
    model.def_property_readonly("num_hyperparams",[](Model &model) {return model.get_num_hyperparams();},
                                  "Number of prior distribution parameters." );
    model.def_property("params_desc",[](Model &model) {return model.get_params_desc();},[](Model &model,StringVecT &desc) { model.set_params_desc(desc); },
                         "Names of model parameters." );
    model.def_property("hyperparams_desc",[](Model &model) {return model.get_hyperparams_desc();}, [](Model &model,StringVecT &desc) { model.set_hyperparams_desc(desc); },
                         "Names of prior distribution parameters."  );
    model.def_property("hyperparams",&ModelWrapper<Model>::get_hyperparams, &ModelWrapper<Model>::set_hyperparams,
                         "Prior distribution parameters.");
    model.def_property("lbound",&ModelWrapper<Model>::get_lbound, &ModelWrapper<Model>::set_lbound,
                         "Parameter box-constraints lower-bounds.");
    model.def_property("ubound",&ModelWrapper<Model>::get_ubound, &ModelWrapper<Model>::set_ubound,
                         "Parameter box-constraints upper-bounds.");
 
    model.def("find_hyperparam",&ModelWrapper<Model>::find_hyperparam,py::arg("name"),py::arg("default_val")=-1,
                "Find a hyperparameter value by name or return a default_val.");
    model.def("bounded_theta", &ModelWrapper<Model>::bounded_theta,py::arg("theta"),
                "Bound parameter theta to be within the box-constraints by truncation.");
    model.def("reflected_theta", &ModelWrapper<Model>::reflected_theta,py::arg("theta"),
                "Bound parameter theta to be within the box-constraints by reflection.");
    model.def("theta_in_bounds",&ModelWrapper<Model>::theta_in_bounds,py::arg("theta"),
                "True if parameter theta is within the box-constraints.");
    
    model.def("get_stats",&Model::get_stats,
                "Get Model description and settings.");
    
    model.def("sample_prior",&ModelWrapper<Model>::sample_prior,
                "Sample a single parameter value from the prior.");
    model.def("sample_prior",&ModelWrapper<Model>::sample_prior_stack, py::arg("count"),
                "Sample a vector of parameter values from the prior. [OpenMP]");
    
    model.def("model_image",&ModelWrapper<Model>::model_image_stack, py::arg("theta"),
                "Generate a stack of model images (expected photon count) at a stack of thetas. [OpenMP]");
    model.def("simulate_image",&ModelWrapper<Model>::simulate_image_stack, py::arg("thetas"), py::arg("count")=1,
                "Simulate a stack of images with noise using one or more parameters thetas. [OpenMP]");
    
    model.def("objective_llh",&ModelWrapper<Model>::objective_llh_stack, py::arg("images"), py::arg("thetas"),
                "Calculate the full log-likelihood for one or more images at one or more thetas under the  model objective. [OpenMP]");
    model.def("objective_rllh",&ModelWrapper<Model>::objective_rllh_stack, py::arg("images"), py::arg("thetas"),
                "Calculate the relative log-likelihood for one or more images at one or more thetas under the  model objective. [OpenMP]");
    model.def("objective_grad",&ModelWrapper<Model>::objective_grad_stack, py::arg("images"), py::arg("thetas"),
                "Calculate the gadiant of the relative log-likelihood for one or more images at one or more thetas under the  model objective. [OpenMP]");
    model.def("objective_hessian",&ModelWrapper<Model>::objective_hessian_stack, py::arg("images"), py::arg("thetas"),
                "Calculate the hessian of the relative log-likelihood for one or more images at one or more thetas under the  model objective. [OpenMP]");
    model.def("objective_negative_definite_hessian",&ModelWrapper<Model>::objective_negative_definite_hessian_stack, py::arg("images"), py::arg("thetas"),
                "Calculate the best negative-definite approximation to the hessian of the relative log-likelihood for one or more images at one or more thetas under the  model objective. [OpenMP]");
    
    model.def("objective",&ModelWrapper<Model>::objective, py::arg("image"), py::arg("theta"),
                "Returns the tuple (rllh, grad, hessian) of the model objective with respect to a single image, evaluated at a single theta.  The objective depends on the Estimator type (MLE) or (MAP).  This should be called as the objective function to maximize in optimization algorithms.");
    model.def("likelihood_objective",&ModelWrapper<Model>::likelihood_objective, py::arg("image"), py::arg("theta"),
                "Returns the tuple (rllh, grad, hessian) of the pure log-likelihood function with respect to a single image, evaluated at a single theta.");
    model.def("prior_objective",&ModelWrapper<Model>::prior_objective, py::arg("image"), py::arg("theta"),
                "Returns the tuple (rllh, grad, hessian) of the prior log-likelihood with respect, evaluated at a single theta.");
    model.def("aposteriori_objective",&ModelWrapper<Model>::aposteriori_objective, py::arg("image"), py::arg("theta"),
                "Returns the tuple (rllh, grad, hessian) of the log-aposteriori function  (the log_likelihood + log_prior) with respect to a single image , evaluated at a single theta.");
    
    model.def("cr_lower_bound",&ModelWrapper<Model>::cr_lower_bound, py::arg("thetas"),
                "Returns the Cramer-Rao lower-bound at one or more thetas.");
    model.def("expected_information",&ModelWrapper<Model>::expected_information, py::arg("thetas"),
                "Returns the Expected Fisher information matrix at one or more thetas.");
    model.def("observed_information",&ModelWrapper<Model>::observed_information, py::arg("image"), py::arg("theta_mode"),
                "Returns the Observed Fisher information matrix with respect to a single image, evaluated at the estimated mode theta_mode.");

    model.def("estimate_max",&ModelWrapper<Model>::estimate_max,  
              py::arg("images"),  py::arg("method"),  py::arg("theta_init")=ArrayDoubleT(), py::arg("return_stats")=false,
              "Returns (theta_max_stack,rllh_stack,observedI_stack). Estimates the maximum of the model objective.  This is Maximum likelihood estimation (MLE) or maximum-aposeteriori (MAP) estimation depending on the model.  fixed_theta is a vector of fixed values free values are indicated by inf or nan. [OpenMP]");
//     model.deg("estimate_profile_max",&ModelWrapper<Model>::estimate_profile_max,  
//               py::arg("image"),  py::arg("fixed_theta_stack"), py::arg("method"), py::arg("theta_mle"), py::arg("return_stats")=false,
//               "Returns (theta_profile_max_stack,rllh_stack,observedI_stack) estimating the maximum of the model objective for each image using given method and theta_init. [OpenMP]");
              
    
    model.def("estimate_mcmc_posterior",&ModelWrapper<Model>::estimate_mcmc_posterior, 
              py::arg("images"), py::arg("theta_init")=ArrayDoubleT(), py::arg("Nsamples")=1000,  py::arg("Nburnin")=100, py::arg("thin")=0,
              "Returns the summarized MCMC postrerior mean and covariance: (theta_mean_stack, theta_cov_stack) [OpenMP]");
    model.def("estimate_mcmc_samples",&ModelWrapper<Model>::estimate_mcmc_sample, 
              py::arg("images"), py::arg("Nsamples")=1000, py::arg("theta_init")=ArrayDoubleT(), py::arg("Nburnin")=100, py::arg("thin")=0,
              "Returns the full MCMC sample: (sample_stack, sample_rllh_stack).  This can be used to estimate credible intervals. [OpenMP]");

    model.def("error_bounds_expected",&ModelWrapper<Model>::error_bounds_expected,  
              py::arg("theta_est"), py::arg("confidence")=0.95,
              "Returns error bounds for each parameter (theta_lb, theta_ub) for one or more estimated theta values, using the Expected Fisher Information. Important: These bounds are only valid if the estimator errors are normally distributed (i.e., the objective function is regular near the maximum).  Additionally, the estimator must be unbiased and approach the CRLB in accuracy. [OpenMP]");
    model.def("error_bounds_observed",&ModelWrapper<Model>::error_bounds_observed,  
              py::arg("theta_est"), py::arg("obsI"), py::arg("confidence")=0.95,
              "Returns error bounds for each parameter (theta_lb, theta_ub) for one or more estimated theta values, using the Observed Fisher Information (negative hessian).  Important: These bounds are only valid if the estimator errors are normally distributed (i.e., the objective function is regular near the maximum). [OpenMP]");
//     model.deg("error_bounds_profile",&ModelWrapper<Model>::error_bounds_profile,  
//               py::arg("images"),py::arg("theta_est"), py::arg("confidence")=0.95,
//               " Returns error bounds for each parameter (theta_lb, theta_ub). Make no assumptions about the Normality of the errors or regularity of the objective and use a pure-likelihood based approach to find the estimated error bounds. [OpenMP]");
    model.def("error_bounds_posterior_credible",&ModelWrapper<Model>::error_bounds_posterior_credible,  
              py::arg("samples"), py::arg("confidence")=0.95,
              "Returns error bounds for each parameter (theta_mean, theta_lb, theta_ub) for one or more images, using an MCMC sample.  Assuming sufficient sample size and mcmc mixing, these bounds are valid even for non-regular posterior distributions. [OpenMP]");

        
    /* Debugging methods (single threaded) */
    model.def("objective_llh_components",&ModelWrapper<Model>::objective_llh_components, 
              py::arg("image"), py::arg("theta"),
                "[DEBUGGING] Calculate for each component (each pixel and each prior parameter) the full log-likelihood contribution, for a single image and theta.");
    model.def("objective_rllh_components",&ModelWrapper<Model>::objective_rllh_components, 
              py::arg("image"), py::arg("theta"),
                "[DEBUGGING] Calculate for each component (each pixel and each prior parameter) the relative log-likelihood contribution, for a single image and theta.");
    model.def("objective_grad_components",&ModelWrapper<Model>::objective_grad_components, 
              py::arg("image"), py::arg("theta"),
                "[DEBUGGING] Calculate for each component (each pixel and each prior parameter) the contribution to the gradient of the log-likelihood, for a single image and theta.");
    model.def("objective_hessian_components",&ModelWrapper<Model>::objective_hessian_components, 
              py::arg("image"), py::arg("theta"),
                "[DEBUGGING] Calculate for each component (each pixel and each prior parameter) the contribution to the hessian of the log-likelihood, for a single image and theta.");
    
    model.def("estimate_max_debug",&ModelWrapper<Model>::estimate_max_debug,
              py::arg("image"),  py::arg("method"),  py::arg("theta_init")=ArrayDoubleT(),
              "[DEBUGGING] Returns (theta_max, observedI, stats, sequence, sequence_rllh) For a single image.  The returned sequence is all evaluated points in sequence.");    
//     model.def("estimate_profile_max_debug",&ModelWrapper<Model>::estimate_max_debug,
//               py::arg("image"),  py::arg("fixed_theta"), py::arg("method"), py::arg("theta_mle"), py::arg("return_stats")=false,
//               "[DEBUGGING] Returns (theta_profile_max, rllh_stack, stats, sequence, sequence_rllh) For a single image.  The returned sequence is all evaluated points in sequence.");    
    model.def("estimate_mcmc_debug",&ModelWrapper<Model>::estimate_mcmc_debug, 
              py::arg("image"), py::arg("Nsamples"), py::arg("theta_init")=ArrayDoubleT(),
              "[DEBUGGING] Returns (sample, sample_rllh, candidates, candidates_rllh).  Running MCMC for a single image.  No thinning or burnin is performed. Candidates are the condisdered candidates for each iteration");
}

template<class Model>
ArrayDoubleT
ModelWrapper<Model>::get_hyperparams(Model &model)
{
    auto arr = pyarma::makeArray(model.get_num_hyperparams());
    auto theta = pyarma::asVec(arr);
    theta = model.get_hyperparams();
    return arr;
}

template<class Model>
double 
ModelWrapper<Model>:: find_hyperparam(Model &model, std::string param_name, double default_val)
{
    return model.find_hyperparam(param_name,default_val);
}

template<class Model>
void
ModelWrapper<Model>::set_hyperparams(Model &model,ArrayDoubleT &arr )
{
    auto theta = pyarma::asVec(arr);
    model.set_hyperparams(theta);
}

template<class Model>
ArrayDoubleT
ModelWrapper<Model>::get_lbound(Model &model)
{
    auto arr = pyarma::makeArray(model.get_num_params());
    auto bd = pyarma::asVec(arr);
    bd = model.get_lbound();
    return arr;
}

template<class Model>
void
ModelWrapper<Model>::set_lbound(Model &model,ArrayDoubleT &arr )
{
    auto bd = pyarma::asVec(arr);
    model.set_lbound(bd);
}

template<class Model>
ArrayDoubleT
ModelWrapper<Model>::get_ubound(Model &model)
{
    auto arr = pyarma::makeArray(model.get_num_params());
    auto bd = pyarma::asVec(arr);
    bd = model.get_ubound();
    return arr;
}

template<class Model>
void
ModelWrapper<Model>::set_ubound(Model &model,ArrayDoubleT &arr )
{
    auto bd = pyarma::asVec(arr);
    model.set_ubound(bd);
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::bounded_theta(Model &model, ArrayDoubleT &arr)
{
    auto theta = pyarma::asVec(arr);
    auto out = pyarma::makeArray(model.get_num_params());
    auto bd_theta = pyarma::asVec(out);
    bd_theta = model.bounded_theta(theta);    
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::reflected_theta(Model &model, ArrayDoubleT &arr)
{
    auto theta = pyarma::asVec(arr);
    auto out = pyarma::makeArray(model.get_num_params());
    auto bd_theta = pyarma::asVec(out);
    bd_theta = model.reflected_theta(theta);    
    return out;
}

template<class Model>    
bool 
ModelWrapper<Model>::theta_in_bounds(Model &model, ArrayDoubleT &arr)
{
    auto theta = pyarma::asVec(arr);
    return model.theta_in_bounds(theta);
}


template<class Model>
ArrayDoubleT
ModelWrapper<Model>::sample_prior(Model &model)
{
    auto out = pyarma::makeArray(model.get_num_params());
    auto theta = pyarma::asVec(out);
    theta = model.sample_prior();
    return out;
}

template<class Model>
ArrayDoubleT
ModelWrapper<Model>::sample_prior_stack(Model &model, IdxT count)
{
    auto out = pyarma::makeArray(model.get_num_params(),count);
    auto theta = pyarma::asMat(out);
    methods::sample_prior_stack(model,theta);
    return out;
}


template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::model_image_stack(Model &model, ArrayDoubleT &thetas_arr)
{
    auto thetas = pyarma::asMat(thetas_arr);
    auto out = makeImageStackArray(model.get_size(), thetas.n_cols);
    auto ims = imageStackAsArma<Model>(out);
    methods::model_image_stack(model,thetas, ims);
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::simulate_image_stack(Model &model, ArrayDoubleT &thetas_arr, IdxT count)
{
    auto thetas = pyarma::asMat(thetas_arr);
    if(count>1 && thetas.n_cols>1) {
        std::ostringstream msg;
        msg<<"Simulate image got N="<<thetas.n_cols<<" count="<<count;
        throw PythonError("ParameterError",msg.str());
    }
    count = std::max(count, thetas.n_cols);
    auto out = makeImageStackArray(model.get_size(), count);
    auto ims = imageStackAsArma<Model>(out);
    methods::simulate_image_stack(model,thetas, ims);
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::objective_llh_stack(Model &model, ArrayDoubleT &ims_arr, ArrayDoubleT &thetas_arr)
{
    auto thetas = pyarma::asMat(thetas_arr);
    auto ims = imageStackAsArma<Model>(ims_arr);
    IdxT count = std::max(thetas.n_cols, static_cast<IdxT>(model.get_size_image_stack(ims))); 
    auto out = pyarma::makeArray(count);
    auto llh = pyarma::asVec(out);
    methods::objective::llh_stack(model,ims, thetas, llh);
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::objective_rllh_stack(Model &model, ArrayDoubleT &ims_arr, ArrayDoubleT &thetas_arr)
{
    auto thetas = pyarma::asMat(thetas_arr);
    auto ims = imageStackAsArma<Model>(ims_arr);
    IdxT count = std::max(thetas.n_cols, static_cast<IdxT>(model.get_size_image_stack(ims))); 
    auto out = pyarma::makeArray(count);
    auto rllh = pyarma::asVec(out);
    methods::objective::rllh_stack(model,ims, thetas, rllh);
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::objective_grad_stack(Model &model, ArrayDoubleT &ims_arr, ArrayDoubleT &thetas_arr)
{
    auto thetas = pyarma::asMat(thetas_arr);
    auto ims = imageStackAsArma<Model>(ims_arr);
    IdxT count = std::max(thetas.n_cols, static_cast<IdxT>(model.get_size_image_stack(ims))); 
    auto out = pyarma::makeArray(model.get_num_params(), count);
    auto grad = pyarma::asMat(out);
    methods::objective::grad_stack(model,ims, thetas, grad);
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::objective_hessian_stack(Model &model, ArrayDoubleT &ims_arr, ArrayDoubleT &thetas_arr)
{
    auto thetas = pyarma::asMat(thetas_arr);
    auto ims = imageStackAsArma<Model>(ims_arr);
    IdxT count = std::max(thetas.n_cols, static_cast<IdxT>(model.get_size_image_stack(ims))); 
    auto out = pyarma::makeArray(model.get_num_params(), model.get_num_params(), count);
    auto hess = pyarma::asCube(out);
    methods::objective::hessian_stack(model,ims, thetas, hess);
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::objective_negative_definite_hessian_stack(Model &model, ArrayDoubleT &ims_arr, ArrayDoubleT &thetas_arr)
{
    auto thetas = pyarma::asMat(thetas_arr);
    auto ims = imageStackAsArma<Model>(ims_arr);
    IdxT count = std::max(thetas.n_cols, static_cast<IdxT>(model.get_size_image_stack(ims))); 
    auto out = pyarma::makeArray(model.get_num_params(), model.get_num_params(), count);
    auto hess = pyarma::asCube(out);
    methods::objective::negative_definite_hessian_stack(model,ims, thetas, hess);
    return out;
}

template<class Model>
py::tuple 
ModelWrapper<Model>::objective(Model &model, ArrayDoubleT &im_arr, ArrayDoubleT &theta_arr)
{
    auto theta = pyarma::asVec(theta_arr);
    auto im = imageAsArma<Model>(im_arr);
    auto s = model.make_stencil(theta);
    double rllh = methods::objective::rllh(model, im, s);
    auto N = model.get_num_params();
    auto grad_arr = pyarma::makeArray(N);
    auto grad = pyarma::asVec(grad_arr);
    grad = methods::objective::grad(model, im, s);
    auto hess_arr = pyarma::makeArray(N,N);
    auto hess = pyarma::asMat(hess_arr);
    hess = methods::objective::hessian(model, im, s);
    py::tuple out(3);
    out[0] = rllh;
    out[1] = grad_arr;
    out[2] = hess_arr;
    return out; 
}

template<class Model>
py::tuple 
ModelWrapper<Model>::likelihood_objective(Model &model, ArrayDoubleT &im_arr, ArrayDoubleT &theta_arr)
{
    auto theta = pyarma::asVec(theta_arr);
    auto im = imageAsArma<Model>(im_arr);
    double rllh;
    auto N = model.get_num_params();
    auto grad_arr = pyarma::makeArray(N);
    auto grad = pyarma::asVec(grad_arr);
    auto hess_arr = pyarma::makeArray(N,N);
    auto hess = pyarma::asMat(hess_arr);
    methods::likelihood_objective(model,im,theta,rllh,grad,hess);
    
    py::tuple out(3);
    out[0] = rllh;
    out[1] = grad_arr;
    out[2] = hess_arr;
    return out; 
}

template<class Model>
py::tuple 
ModelWrapper<Model>::prior_objective(Model &model, ArrayDoubleT &im_arr, ArrayDoubleT &theta_arr)
{
    auto theta = pyarma::asVec(theta_arr);
    auto im = imageAsArma<Model>(im_arr);
    double rllh;
    auto N = model.get_num_params();
    auto grad_arr = pyarma::makeArray(N);
    auto grad = pyarma::asVec(grad_arr);
    auto hess_arr = pyarma::makeArray(N,N);
    auto hess = pyarma::asMat(hess_arr);
    methods::prior_objective(model,im,theta,rllh,grad,hess);
    
    py::tuple out(3);
    out[0] = rllh;
    out[1] = grad_arr;
    out[2] = hess_arr;
    return out; 
}

template<class Model>
py::tuple 
ModelWrapper<Model>::aposteriori_objective(Model &model, ArrayDoubleT &im_arr, ArrayDoubleT &theta_arr)
{
    auto theta = pyarma::asVec(theta_arr);
    auto im = imageAsArma<Model>(im_arr);
    double rllh;
    auto N = model.get_num_params();
    auto grad_arr = pyarma::makeArray(N);
    auto grad = pyarma::asVec(grad_arr);
    auto hess_arr = pyarma::makeArray(N,N);
    auto hess = pyarma::asMat(hess_arr);
    methods::aposteriori_objective(model,im,theta,rllh,grad,hess);
    
    py::tuple out(3);
    out[0] = rllh;
    out[1] = grad_arr;
    out[2] = hess_arr;
    return out; 
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::cr_lower_bound(Model &model, ArrayDoubleT &thetas_arr)
{
    auto thetas = pyarma::asMat(thetas_arr);
    IdxT count = thetas.n_cols;
    auto out = pyarma::makeArray(model.get_num_params(), count);
    auto cr_vec = pyarma::asMat(out);
    methods::cr_lower_bound_stack(model, thetas, cr_vec);
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::expected_information(Model &model, ArrayDoubleT &thetas_arr)
{
    auto thetas = pyarma::asMat(thetas_arr);
    IdxT count = thetas.n_cols;
    auto out = pyarma::makeArray(model.get_num_params(), model.get_num_params(), count);
    auto I_stack = pyarma::asCube(out);
    methods::expected_information_stack(model, thetas, I_stack);
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::observed_information(Model &model, ArrayDoubleT &im_arr, ArrayDoubleT &theta_mode_arr)
{
    auto theta = pyarma::asVec(theta_mode_arr);
    auto im = imageAsArma<Model>(im_arr);
    auto out = pyarma::makeArray(model.get_num_params(), model.get_num_params());
    auto hess = pyarma::asMat(out);
    methods::objective::hessian(model, im, theta, hess);
    return out;
}

template<class Model>
py::tuple 
ModelWrapper<Model>::estimate_max(Model &model, ArrayDoubleT &images_arr, std::string method, 
                                  ArrayDoubleT &theta_init_arr, bool return_stats)
{
    auto image_stack = imageStackAsArma<Model>(images_arr);
    IdxT count = model.get_size_image_stack(image_stack);
    MatT theta_init_stack;
    if(theta_init_arr.ndim()==0 || theta_init_arr.size()==0) {
        //Initialize empty
        theta_init_stack.set_size(model.get_num_params(),count);
        theta_init_stack.zeros();
    } else {
        theta_init_stack = pyarma::asMat(theta_init_arr);
        model.check_param_shape(theta_init_stack);
        if(theta_init_stack.n_cols != count) {
            std::ostringstream msg;
            msg<<"Got inconsistent counts: #images="<<count<<" #theta_inits="<<theta_init_stack.n_cols;
            throw PythonError(msg.str());
        }
    }
    auto theta_max_arr = pyarma::makeArray(model.get_num_params(),count);
    auto theta_max_stack = pyarma::asMat(theta_max_arr);
    
    auto rllh_arr = pyarma::makeArray(count);
    auto rllh_stack = pyarma::asVec(rllh_arr);
    
    auto obsI_arr = pyarma::makeArray(model.get_num_params(),model.get_num_params(),count);
    auto obsI_stack = pyarma::asCube(obsI_arr);
    
    if(return_stats){
        StatsT stats;
        methods::estimate_max_stack(model, image_stack, method, theta_init_stack, theta_max_stack, rllh_stack, obsI_stack, stats);
        py::tuple out(4);
        out[0] = theta_max_arr;
        out[1] = rllh_arr;
        out[2] = obsI_arr;
        out[3] = stats;
        return out; 
    } else {
        methods::estimate_max_stack(model, image_stack, method, theta_init_stack, theta_max_stack, rllh_stack, obsI_stack);
        py::tuple out(3);
        out[0] = theta_max_arr;
        out[1] = rllh_arr;
        out[2] = obsI_arr;
        return out; 
    }
    
}

// template<class Model>
// py::tuple 
// ModelWrapper<Model>::estimate_profile_max(Model &model, ArrayDoubleT &images, std::string method, 
//                                           ArrayDoubleT &fixed_theta_stack, ArrayDoubleT &theta_init_stack, bool return_stats )
// {
// }

template<class Model>
py::tuple 
ModelWrapper<Model>::estimate_mcmc_sample(Model &model, ArrayDoubleT &images_arr, ArrayDoubleT &theta_init_arr, 
                                          IdxT Nsample, IdxT Nburnin, IdxT thin)
{
    auto image_stack = imageStackAsArma<Model>(images_arr);
    IdxT count = model.get_size_image_stack(image_stack);
    MatT theta_init_stack;
    if(theta_init_arr.ndim()==0 || theta_init_arr.size()==0) {
        //Initialize empty
        theta_init_stack.set_size(model.get_num_params(),count);
        theta_init_stack.zeros();
    } else {
        theta_init_stack = pyarma::asMat(theta_init_arr);
        model.check_param_shape(theta_init_stack);
        if(theta_init_stack.n_cols != count) {
            std::ostringstream msg;
            msg<<"Got inconsistent counts: #images="<<count<<" #theta_inits="<<theta_init_stack.n_cols;
            throw PythonError(msg.str());
        }
    }
    auto sample_arr = pyarma::makeArray(model.get_num_params(), Nsample, count);
    auto sample_stack = pyarma::asCube(sample_arr);
    auto sample_rllh_arr = pyarma::makeArray(Nsample, count);
    auto sample_rllh_stack = pyarma::asMat(sample_rllh_arr);
    methods::estimate_mcmc_sample_stack(model,image_stack,theta_init_stack, Nsample, Nburnin, thin, sample_stack, sample_rllh_stack);
    py::tuple out(2);
    out[0] = sample_arr;
    out[1] = sample_rllh_arr;
    return out; 
}

template<class Model>
py::tuple 
ModelWrapper<Model>::estimate_mcmc_posterior(Model &model, ArrayDoubleT &images_arr, ArrayDoubleT &theta_init_arr, IdxT Nsample, IdxT Nburnin, IdxT thin)
{
    auto image_stack = imageStackAsArma<Model>(images_arr);
    IdxT count = model.get_size_image_stack(image_stack);
    MatT theta_init_stack;
    if(theta_init_arr.ndim()==0 || theta_init_arr.size()==0) {
        //Initialize empty
        theta_init_stack.set_size(model.get_num_params(),count);
        theta_init_stack.zeros();
    } else {
        theta_init_stack = pyarma::asMat(theta_init_arr);
        model.check_param_shape(theta_init_stack);
        if(theta_init_stack.n_cols != count) {
            std::ostringstream msg;
            msg<<"Got inconsistent counts: #images="<<count<<" #theta_inits="<<theta_init_stack.n_cols;
            throw PythonError(msg.str());
        }
    }
    auto theta_mean_arr = pyarma::makeArray(model.get_num_params(), count);
    auto theta_mean_stack = pyarma::asMat(theta_mean_arr);
    auto theta_cov_arr = pyarma::makeArray(model.get_num_params(), model.get_num_params(), count);
    auto theta_cov_stack = pyarma::asCube(theta_cov_arr);
    methods::estimate_mcmc_posterior_stack(model,image_stack, theta_init_stack, Nsample, Nburnin, thin, 
                                           theta_mean_stack, theta_cov_stack);
    py::tuple out(2);
    out[0] = theta_mean_arr;
    out[1] = theta_cov_arr;
    return out; 
}


template<class Model>
py::tuple 
ModelWrapper<Model>::error_bounds_expected(Model &model, ArrayDoubleT &theta_arr, double confidence)
{
    auto theta_stack = pyarma::asMat(theta_arr);
    IdxT count = theta_stack.n_cols;
    auto theta_lb_arr = pyarma::makeArray(model.get_num_params(), count);
    auto theta_lb_stack = pyarma::asMat(theta_lb_arr);
    auto theta_ub_arr = pyarma::makeArray(model.get_num_params(), count);
    auto theta_ub_stack = pyarma::asMat(theta_ub_arr);
    methods::error_bounds_expected_stack(model, theta_stack, confidence, theta_lb_stack, theta_ub_stack);
    py::tuple out(2);
    out[0] = theta_lb_arr;
    out[1] = theta_ub_arr;
    return out;     
}

template<class Model>
py::tuple 
ModelWrapper<Model>::error_bounds_observed(Model &model, ArrayDoubleT &theta_arr, ArrayDoubleT &obsI_arr, double confidence)
{
    auto theta_stack = pyarma::asMat(theta_arr);
    IdxT count = theta_stack.n_cols;
    auto obsI_stack = pyarma::asCube(obsI_arr);
    if(obsI_stack.n_slices != count) {
        std::ostringstream msg;
        msg<<"Got inconsistent counts: #theta="<<count<<" #obsI="<<obsI_stack.n_slices;
        throw PythonError(msg.str());
    }
    auto theta_lb_arr = pyarma::makeArray(model.get_num_params(), count);
    auto theta_lb_stack = pyarma::asMat(theta_lb_arr);
    auto theta_ub_arr = pyarma::makeArray(model.get_num_params(), count);
    auto theta_ub_stack = pyarma::asMat(theta_ub_arr);
    methods::error_bounds_observed_stack(model, theta_stack, obsI_stack, confidence,  theta_lb_stack, theta_ub_stack);
    py::tuple out(2);
    out[0] = theta_lb_arr;
    out[1] = theta_ub_arr;
    return out;     
}

// template<class Model>
// py::tuple 
// ModelWrapper<Model>::error_bounds_profile(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas, double confidence)
// {
// }


template<class Model>
py::tuple 
ModelWrapper<Model>::error_bounds_posterior_credible(Model &model, ArrayDoubleT &sample_arr, double confidence)
{
    auto sample_stack = pyarma::asCube(sample_arr);
    IdxT count = sample_stack.n_slices;
    auto theta_mean_arr = pyarma::makeArray(model.get_num_params(), count);
    auto theta_mean_stack = pyarma::asMat(theta_mean_arr);
    auto theta_lb_arr = pyarma::makeArray(model.get_num_params(), count);
    auto theta_lb_stack = pyarma::asMat(theta_lb_arr);
    auto theta_ub_arr = pyarma::makeArray(model.get_num_params(), count);
    auto theta_ub_stack = pyarma::asMat(theta_ub_arr);
    methods::error_bounds_posterior_credible_stack(model, sample_stack, confidence, theta_mean_stack, theta_lb_stack, theta_ub_stack);
    py::tuple out(3);
    out[0] = theta_mean_arr;
    out[1] = theta_lb_arr;
    out[2] = theta_ub_arr;
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::objective_llh_components(Model &model, ArrayDoubleT &im_arr, ArrayDoubleT &theta_arr)
{
    auto im = imageAsArma<Model>(im_arr);
    auto theta = pyarma::asVec(theta_arr);
    auto llh_comps = methods::objective::llh_components(model,im, theta);
    IdxT count = llh_comps.n_elem;
    auto llh_arr = pyarma::makeArray(count);
    auto llh_stack = pyarma::asVec(llh_arr);
    llh_stack = llh_comps;
    return llh_arr;    
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::objective_rllh_components(Model &model, ArrayDoubleT &im_arr, ArrayDoubleT &theta_arr)
{
    auto im = imageAsArma<Model>(im_arr);
    auto theta = pyarma::asVec(theta_arr);
    auto rllh_comps = methods::objective::rllh_components(model,im, theta);
    IdxT count = rllh_comps.n_elem;
    auto rllh_arr = pyarma::makeArray(count);
    auto rllh_stack = pyarma::asVec(rllh_arr);
    rllh_stack = rllh_comps;
    return rllh_arr;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::objective_grad_components(Model &model, ArrayDoubleT &im_arr, ArrayDoubleT &theta_arr)
{
    auto im = imageAsArma<Model>(im_arr);
    auto theta = pyarma::asVec(theta_arr);
    auto grad_comps = methods::objective::grad_components(model,im, theta);
    IdxT count = grad_comps.n_cols;
    auto grad_arr = pyarma::makeArray(model.get_num_params(),count);
    auto grad_stack = pyarma::asMat(grad_arr);
    grad_stack = grad_comps;
    return grad_arr;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::objective_hessian_components(Model &model, ArrayDoubleT &im_arr, ArrayDoubleT &theta_arr)
{
    auto im = imageAsArma<Model>(im_arr);
    auto theta = pyarma::asVec(theta_arr);
    auto hess_comps = methods::objective::hessian_components(model,im, theta);
    IdxT count = hess_comps.n_slices;
    auto hess_arr = pyarma::makeArray(model.get_num_params(),model.get_num_params(),count);
    auto hess_stack = pyarma::asCube(hess_arr);
    hess_stack = hess_comps;
    return hess_arr;
}

template<class Model>
py::tuple 
ModelWrapper<Model>::estimate_max_debug(Model &model, ArrayDoubleT &image_arr, std::string method, ArrayDoubleT &theta_init_arr)
{
    auto image = imageAsArma<Model>(image_arr);
    auto theta_init = pyarma::asVec(theta_init_arr);
    auto theta_est_arr = pyarma::makeArray(model.get_num_params());
    auto theta_est = pyarma::asVec(theta_est_arr);
    auto obsI_arr = pyarma::makeArray(model.get_num_params(),model.get_num_params());
    auto obsI = pyarma::asMat(obsI_arr);
    StatsT stats;
    MatT sequence;
    VecT sequence_rllh;
    methods::estimate_max_debug(model, image, method, theta_init, theta_est, obsI, sequence, sequence_rllh, stats);
    IdxT Nseq = sequence.n_cols;
    auto sequence_arr = pyarma::makeArray(model.get_num_params(), Nseq);
    auto sequence_out = pyarma::asMat(sequence_arr);
    sequence_out = sequence;
    auto sequence_rllh_arr = pyarma::makeArray(Nseq);
    auto sequence_rllh_out = pyarma::asVec(sequence_rllh_arr);
    sequence_rllh_out = sequence_rllh;
    
    py::tuple out(5);
    out[0] = theta_est_arr;
    out[1] = obsI_arr;
    out[2] = stats;
    out[3] = sequence_arr;
    out[4] = sequence_rllh_arr;    
    return out;    
}

template<class Model>
py::tuple 
ModelWrapper<Model>::estimate_mcmc_debug(Model &model, ArrayDoubleT &image_arr, ArrayDoubleT &theta_init_arr, IdxT Nsample)
{
    auto image = imageAsArma<Model>(image_arr);
    auto theta_init = pyarma::asVec(theta_init_arr);
    auto sample_arr = pyarma::makeArray(model.get_num_params(), Nsample);
    auto sample = pyarma::asMat(sample_arr);
    auto candidates_arr = pyarma::makeArray(model.get_num_params(), Nsample);
    auto candidates = pyarma::asMat(candidates_arr);
    auto sample_rllh_arr = pyarma::makeArray(Nsample);
    auto sample_rllh = pyarma::asVec(sample_rllh_arr);
    auto candidates_rllh_arr = pyarma::makeArray(Nsample);
    auto candidates_rllh = pyarma::asVec(candidates_rllh_arr);
    
    methods::estimate_mcmc_sample_debug(model, image, theta_init, Nsample, 
                                        sample, sample_rllh, candidates, candidates_rllh);
    py::tuple out(4);
    out[0] = sample_arr;
    out[1] = sample_rllh_arr;
    out[2] = candidates_arr;
    out[3] = candidates_rllh_arr;
    return out;    
}

    

} /* namespace mappel::python */
} /* namespace mappel */

#endif /*_PY11_ARMADILLO_IFACE */
