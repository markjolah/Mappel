/** @file py11_mappel_iface.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
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

#include <PriorHessian/BaseDist.h>
#include <BacktraceException/BacktraceException.h>

#include "python_error.h"
#include "py11_armadillo_iface.h"

#include "Mappel/Gauss1DMLE.h"
#include "Mappel/Gauss1DMAP.h"
#include "Mappel/Gauss1DsMLE.h"
#include "Mappel/Gauss1DsMAP.h"

#include "Mappel/Gauss2DMLE.h"
#include "Mappel/Gauss2DMAP.h"
#include "Mappel/Gauss2DsMLE.h"
#include "Mappel/Gauss2DsMAP.h"

namespace mappel {
namespace python {

namespace py = pybind11;
namespace pyarma = py11_armadillo;
using pyarma::IdxT;
using pyarma::ArrayT;
using pyarma::ArrayDoubleT;
using pyarma::ColumnMajorOrder;
using python_error::PythonError;
using parallel_rng::SeedT; //RNG seed type from ParallelRng package


inline
VecT thetaAsArma(ArrayDoubleT &theta)
{
     if(theta.size()==0) return {0};
     switch(theta.ndim()) {
        case 1:
            return {static_cast<double*>(theta.mutable_data(0)), static_cast<IdxT>(theta.size()), false, true};
        case 2:
            if(theta.shape(1) != 1 ) {
                std::ostringstream msg;
                msg<<"Expected single theta. Got numpy ndarray dim="<<theta.ndim()<<" #columns:"<<theta.shape(1);
                throw PythonError("ConversionError",msg.str());
            }
            return {static_cast<double*>(theta.mutable_data(0,0)), static_cast<IdxT>(theta.size()), false, true};
        default:
            std::ostringstream msg;
            msg<<"Expected single theta. Got numpy ndarray dim="<<theta.ndim();
            throw PythonError("ConversionError",msg.str());
    }
}

MatT thetaStackAsArma(ArrayDoubleT &theta)
{
     switch(theta.ndim()) {
        case 1:
            return {static_cast<double*>(theta.mutable_data(0)), static_cast<IdxT>(theta.size()), 1, false, true};
        case 2:
            return {static_cast<double*>(theta.mutable_data(0,0)), static_cast<IdxT>(theta.shape(0)), static_cast<IdxT>(theta.shape(1)), false, true};
        default:
            std::ostringstream msg;
            msg<<"Expected stack of 1 or more theta. Got numpy ndarray dim="<<theta.ndim();
            throw PythonError("ConversionError",msg.str());
    }
}


template<class Model>
ReturnIfSubclassT<ImageT<Model>, Model, ImageFormat1DBase>
imageAsArma(ArrayDoubleT &im)
{
     switch(im.ndim()) {
        case 1:
            return {static_cast<double*>(im.mutable_data(0)), static_cast<IdxT>(im.size()), false, true};
        case 2:
            if(im.shape(Model::num_dim) != 1 ) {
                std::ostringstream msg;
                msg<<"Expected single 1D image. Got numpy ndarray dim="<<im.ndim()<<" #images:"<<im.shape(Model::num_dim);
                throw PythonError("ConversionError",msg.str());
            }
            return {static_cast<double*>(im.mutable_data(0,0)), static_cast<IdxT>(im.size()), false, true};
        default:
            std::ostringstream msg;
            msg<<"Expected single image for model with num_dim:"<<Model::num_dim<<". Got numpy ndarray dim="<<im.ndim();
            throw PythonError("ConversionError",msg.str());
    }
}

template<class Model>
ReturnIfSubclassT<ImageT<Model>, Model, ImageFormat2DBase>
imageAsArma(ArrayDoubleT &im)
{
    switch(im.ndim()) {
        case 2:
            return {static_cast<double*>(im.mutable_data(0,0)), static_cast<IdxT>(im.shape(0)), static_cast<IdxT>(im.shape(1)), false, true};
        case 3:
            if(im.shape(Model::num_dim) != 1 ) {
                std::ostringstream msg;
                msg<<"Expected single 2D image. Got numpy ndarray dim="<<im.ndim()<<" #images:"<<im.shape(Model::num_dim);
                throw PythonError("ConversionError",msg.str());
            }
            return {static_cast<double*>(im.mutable_data(0,0,0)), static_cast<IdxT>(im.shape(0)), static_cast<IdxT>(im.shape(1)), false, true};
        default:
            std::ostringstream msg;
            msg<<"Expected single image for model with num_dim:"<<Model::num_dim<<". Got numpy ndarray dim="<<im.ndim();
            throw PythonError("ConversionError",msg.str());
    }
}


template<class Model>
ReturnIfSubclassT<ImageStackT<Model>, Model, ImageFormat1DBase>
imageStackAsArma(ArrayDoubleT &im)
{
    switch(im.ndim()) {
        case 1:
            return {static_cast<double*>(im.mutable_data(0)), static_cast<IdxT>(im.shape(0)), 1, false, true};
        case 2:
            return {static_cast<double*>(im.mutable_data(0,0)), static_cast<IdxT>(im.shape(0)), static_cast<IdxT>(im.shape(1)), false, true};
        default:
            std::ostringstream msg;
            msg<<"Expected stack of 1 or more 1D images. Got numpy ndarray dim="<<im.ndim();
            throw PythonError("ConversionError",msg.str());
    }
}

template<class Model>
ReturnIfSubclassT<ImageStackT<Model>, Model, ImageFormat2DBase>
imageStackAsArma(ArrayDoubleT &im)
{
    switch(im.ndim()) {
        case 2:
            return {static_cast<double*>(im.mutable_data(0,0)), static_cast<IdxT>(im.shape(0)), static_cast<IdxT>(im.shape(1)), 1, false, true};
        case 3:
            return {static_cast<double*>(im.mutable_data(0,0,0)), static_cast<IdxT>(im.shape(0)), static_cast<IdxT>(im.shape(1)), static_cast<IdxT>(im.shape(2)), false, true};
        default:
            std::ostringstream msg;
            msg<<"Expected stack of 1 or more 2D images. Got numpy ndarray dim="<<im.ndim();
            throw PythonError("ConversionError",msg.str());
    }
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
    if(count == 1)  return ArrayT<ElemT, ColumnMajorOrder>({shape});  //Squeeze out last dim
    else            return ArrayT<ElemT, ColumnMajorOrder>({shape,count});
}


/**
 * 
 * @param size The image size as returned by Model.get_size(). [sizeX, sizeY] for 2D models
 */
template<class ElemT=double, class IntT=uint32_t>
ArrayT<ElemT, ColumnMajorOrder> 
makeImageStackArray(arma::Col<IntT> size, IdxT count)
{
    switch(count) {
        case 0:
            throw PythonError("ConversionError","Attempt to make image stack of count==0");
        case 1:
            switch(size.n_elem) {
                case 1:
                    return ArrayT<ElemT, ColumnMajorOrder>({size(0)});
                case 2:
                    return ArrayT<ElemT, ColumnMajorOrder>({size(1),size(0)});
                case 3:
                    return ArrayT<ElemT, ColumnMajorOrder>({size(2),size(1), size(0)});
                default:
                    break;
            }
        default:
            switch(size.n_elem) {
                case 1:
                    return ArrayT<ElemT, ColumnMajorOrder>({size(0),static_cast<IntT>(count)});
                case 2:
                    return ArrayT<ElemT, ColumnMajorOrder>({size(1),size(0),static_cast<IntT>(count)});
                case 3:
                    return ArrayT<ElemT, ColumnMajorOrder>({size(2),size(1), size(0),static_cast<IntT>(count)});
                default:
                    break;
            }
    }
    std::ostringstream msg;
    msg<<"Unable to create array stack for images of size:"<<size.t();
    throw PythonError("ConversionError",msg.str());
}

inline
void register_exceptions()
{
    py::register_exception_translator([](std::exception_ptr err_ptr) {
        try { if(err_ptr) std::rethrow_exception(err_ptr); }
        catch (const ParameterValueError &err) { PyErr_SetString(PyExc_ValueError, err.what()); }
        catch (const ArrayShapeError &err)     { PyErr_SetString(PyExc_ValueError, err.what()); }
        catch (const ArraySizeError &err)      { PyErr_SetString(PyExc_ValueError, err.what()); }
        catch (const ModelBoundsError &err)    { PyErr_SetString(PyExc_RuntimeError, err.what()); }
        catch (const NumericalError &err)      { PyErr_SetString(PyExc_RuntimeError, err.what()); }
        catch (const LogicalError &err)        { PyErr_SetString(PyExc_RuntimeError, err.what()); }
        catch (const NotImplementedError &err) { PyErr_SetString(PyExc_NotImplementedError, err.what()); }
        });
    python_error::register_exceptions();        
    //Exception backtraces are a problem with python and pdb
    //Disable them globally here for any python code.
    backtrace_exception::disable_backtraces(); 
}

template<class Model>
class ModelWrapper
{
    using ImageCoordT = typename Model::ImageCoordT;
public:
    static Model construct_fixed2D(ArrayT<ImageCoordT> &size, ArrayDoubleT &psf_sigma);
    static Model construct_variable2D(ArrayT<ImageCoordT> &size, ArrayDoubleT &min_sigma, ArrayDoubleT &max_sigma);
    static ArrayT<ImageCoordT> get_size(Model &model);
    static void set_size(Model &model, ArrayT<ImageCoordT> &size);
    static ArrayDoubleT get_psf_sigma(Model &model);
    static void set_psf_sigma(Model &model, ArrayDoubleT &psf_sigma);
    static ArrayDoubleT get_min_sigma(Model &model);
    static void set_min_sigma(Model &model, ArrayDoubleT &min_sigma);
    static ArrayDoubleT get_max_sigma(Model &model);
    static void set_max_sigma(Model &model, ArrayDoubleT &max_sigma);
    
    static ArrayDoubleT get_hyperparams(Model &model);
    static void set_hyperparams(Model &model, ArrayDoubleT &arr);
    static ArrayDoubleT get_lbound(Model &model);
    static void set_lbound(Model &model, ArrayDoubleT &arr);
    static ArrayDoubleT get_ubound(Model &model);
    static void set_ubound(Model &model, ArrayDoubleT &arr);
    static ArrayDoubleT bounded_theta(Model &model, ArrayDoubleT &arr);
    static ArrayDoubleT reflected_theta(Model &model, ArrayDoubleT &arr);
    static ArrayT<BoolT> theta_in_bounds(Model &model, ArrayDoubleT &arr);
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
    static py::tuple prior_objective(Model &model, ArrayDoubleT &theta);
    static py::tuple aposteriori_objective(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta);
    
    static ArrayDoubleT cr_lower_bound(Model &model, ArrayDoubleT &thetas);
    static ArrayDoubleT expected_information(Model &model, ArrayDoubleT &thetas);
    static ArrayDoubleT observed_information(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta_mode);
    
    static py::tuple estimate_max(Model &model, ArrayDoubleT &image_stack, std::string method, ArrayDoubleT &theta_init_stack, bool return_stats );
//     static py::tuple estimate_profile_max(Model &model, ArrayDoubleT &images, std::string method, 
//                                           ArrayDoubleT &fixed_theta_stack, ArrayDoubleT &theta_init_stack, bool return_stats );
    static py::tuple estimate_mcmc_posterior(Model &model, ArrayDoubleT &image_stack, IdxT Nsample, ArrayDoubleT &theta_init_stack,  IdxT Nburnin, IdxT thin);
    static py::tuple estimate_mcmc_sample(Model &model, ArrayDoubleT &image_stack, IdxT Nsample, ArrayDoubleT &theta_init_stack,  IdxT Nburnin, IdxT thin);

    static py::tuple error_bounds_expected(Model &model, ArrayDoubleT &thetas, double confidence);
    static py::tuple error_bounds_observed(Model &model, ArrayDoubleT &thetas, ArrayDoubleT &obsI_stack, double confidence);
//     static py::tuple error_bounds_profile(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas, double confidence);
    static py::tuple error_bounds_posterior_credible(Model &model, ArrayDoubleT &sample_stack, double confidence);

    static ArrayDoubleT objective_llh_components(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta);
    static ArrayDoubleT objective_rllh_components(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta);
    static ArrayDoubleT objective_grad_components(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta);
    static ArrayDoubleT objective_hessian_components(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta);
        
    static py::tuple estimate_max_debug(Model &model, ArrayDoubleT &image, std::string method, ArrayDoubleT &theta_init);
    static py::tuple estimate_mcmc_debug(Model &model, ArrayDoubleT &image, IdxT Nsample, ArrayDoubleT &theta_init);
    static ArrayDoubleT initial_theta_estimate(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta_init);
};

template<class Model>
EnableIfSubclassT<Model,Gauss1DModel>
bindMappelModelBase(py::module &M, py::class_<Model> &model)
{
        model.def(py::init<typename Model::ImageCoordT,double>(), py::arg("size"), py::arg("psf_sigma"));
        model.def_property("psf_sigma",[](Model &model) {return model.get_psf_sigma();},
                                       [](Model &model,double sigma) { model.set_psf_sigma(sigma); },
                           "Sigma of emitter (PSF) Gaussian approximation [pixels]." );
}

template<class Model>
EnableIfSubclassT<Model,Gauss1DsModel>
bindMappelModelBase(py::module &M, py::class_<Model> &model)
{
        model.def(py::init<typename Model::ImageCoordT,double,double>(), py::arg("size"), py::arg("min_sigma"), py::arg("max_sigma"));
        model.def_property("min_sigma",[](Model &model) {return model.get_min_sigma();},
                                       [](Model &model,double sigma) { model.set_min_sigma(sigma); },
                           "Minimum Gaussian sigma of emitter PSF in pixels." );
        model.def_property("max_sigma",[](Model &model) {return model.get_max_sigma();},
                                       [](Model &model,double sigma) { model.set_max_sigma(sigma); },
                           "Maximum Gaussian sigma of emitter PSF in pixels." );
}

template<class Model>
EnableIfSubclassT<Model,Gauss2DModel>
bindMappelModelBase(py::module &M, py::class_<Model> &model)
{
        model.def(py::init(&ModelWrapper<Model>::construct_fixed2D), 
                  py::arg("size"), py::arg("psf_sigma"));
        model.def_property("psf_sigma",&ModelWrapper<Model>::get_psf_sigma,&ModelWrapper<Model>::set_psf_sigma,
                           "2D Sigma of emitter (PSF) Gaussian approximation in pixels [X, Y]." );
}


template<class Model>
EnableIfSubclassT<Model,Gauss2DsModel>
bindMappelModelBase(py::module &M, py::class_<Model> &model)
{
        model.def(py::init(&ModelWrapper<Model>::construct_variable2D), py::arg("size"), py::arg("min_sigma"), py::arg("max_sigma"));
                    
        model.def_property("min_sigma",&ModelWrapper<Model>::get_min_sigma, &ModelWrapper<Model>::set_min_sigma,
                           "Minimum Gaussian sigma of emitter PSF in pixels." );
        model.def_property("max_sigma",&ModelWrapper<Model>::get_max_sigma, &ModelWrapper<Model>::set_max_sigma,
                           "Maximum Gaussian sigma of emitter PSF in pixels (must be exact multiple of min_sigma.)" );
        model.def_property("max_sigma_ratio",[](Model &model) { return model.get_max_sigma_ratio(); },
                                             [](Model &model, double max_sigma_ratio) { model.set_max_sigma_ratio(max_sigma_ratio); },
                           "Ratio of Maximum Gaussian sigma to PSF (min_sigma).  Must be greater than 1.0.");
}


template<class Model>
EnableIfSubclassT<Model,ImageFormat1DBase>
bindMappelModelImageBase(py::module &M, py::class_<Model> &model)
{    
        model.def_property("size",[](Model &model) {return model.get_size();},
                                  [](Model &model,IdxT size) { model.set_size(size); },
                           "1D-Image size in pixels." );
}

template<class Model>
EnableIfSubclassT<Model,ImageFormat2DBase>
bindMappelModelImageBase(py::module &M, py::class_<Model> &model)
{    
        model.def_property("size",&ModelWrapper<Model>::get_size,&ModelWrapper<Model>::set_size,
                           "2D-Image size in pixels [X, Y]." );
}

template<class Model>
void bindMappelModel(py::module &M)
{
    register_exceptions(); //Register Mappel exceptions
    
    py::class_<Model> model(M, Model::name.c_str(), py::multiple_inheritance());
    
    bindMappelModelBase<Model>(M,model);
    bindMappelModelImageBase(M,model);
    
    model.def_property_readonly("num_dim",[](Model &model) { return Model::num_dim; },
                                         "Number of image dimensions.");
    model.def_property_readonly("name",[](Model &model) { return Model::name; },
                                         "Model name.");
    model.def_property_readonly("global_min_size",[](Model &model) { return model.global_min_size; },
                                  "Global constraint on the minimum size along any image dimension in pixels.");
    model.def_property_readonly("global_min_psf_sigma",[](Model &model) { return model.global_min_psf_sigma; },
                                  "Global constraint on the minimum psf size in pixels along any dimension.");
    model.def_property_readonly("global_max_psf_sigma",[](Model &model) { return model.global_max_psf_sigma; },
                                  "Global constraint on the maximum psf size in pixels along any dimension.");
    model.def_property_readonly("num_pixels",[](Model &model) { return model.get_num_pixels(); },
                                         "Total number of image pixels.");
    model.def_property_readonly("estimator_names",[](Model &model) {return Model::estimator_names;},
                                         "Available MLE/MAP estimator names.");
    model.def_property_readonly("num_params",[](Model &model) {return model.get_num_params();},
                                  "Number of model parameters.");
    model.def_property_readonly("num_hyperparams",[](Model &model) {return model.get_num_hyperparams();},
                                  "Number of prior distribution parameters." );
    model.def_property_readonly("bounds_epsilon",[](Model &model) { return Model::bounds_epsilon; },
                                         "Minimum distance away from the boundary for feasible theta.");
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
 
    model.def("set_rng_seed",[](Model &model, SeedT seed) { model.set_rng_seed(seed); }, py::arg("seed"),
                "Re-seed the internal rng manager.");
    model.def("find_hyperparam",&ModelWrapper<Model>::find_hyperparam,py::arg("name"),py::arg("default_val")=-1,
                "Find a hyperparameter value by name or return a default_val.");
    model.def("bounded_theta", &ModelWrapper<Model>::bounded_theta,py::arg("theta"),
                "Bound parameter theta to be within the box-constraints by truncation.");
    model.def("reflected_theta", &ModelWrapper<Model>::reflected_theta,py::arg("theta"),
                "Bound parameter theta to be within the box-constraints by reflection.");
    model.def("theta_in_bounds",&ModelWrapper<Model>::theta_in_bounds,py::arg("theta"),
                "True if parameter theta is within the box-constraints.");
    
    model.def("_get_stats",&Model::get_stats,
                "Get Model description and settings.");
    
    model.def("sample_prior",&ModelWrapper<Model>::sample_prior,
                "Sample a single parameter value from the prior.");
    model.def("sample_prior",&ModelWrapper<Model>::sample_prior_stack, py::arg("count"),
                "Sample a vector of parameter values from the prior. [OpenMP]");
    
    model.def("model_image",&ModelWrapper<Model>::model_image_stack, py::arg("theta"),
                "Generate a stack of model images (expected photon count) at a stack of thetas. [OpenMP]");
    model.def("simulate_image",&ModelWrapper<Model>::simulate_image_stack, py::arg("thetas"), py::arg("count")=1,
                "Simulate a stack of images with noise using one or more parameters thetas. [OpenMP]");
    
    model.def("objective_llh",&ModelWrapper<Model>::objective_llh_stack, py::arg("images"), py::arg("thetas"), R"DOC(
        Calculate the full log-likelihood under the model objective.
        Operates on one or more images at one or more thetas in parallel using OpenMP.)DOC");
    model.def("objective_rllh",&ModelWrapper<Model>::objective_rllh_stack, py::arg("images"), py::arg("thetas"), R"DOC(
        Calculate the relative log-likelihood under the model objective.
        Operates on one or more images at one or more thetas in parallel using OpenMP.)DOC");
    model.def("objective_grad",&ModelWrapper<Model>::objective_grad_stack, py::arg("images"), py::arg("thetas"), R"DOC(
        Calculate the gradient of the log-likelihood under the model objective.
        Operates on one or more images at one or more thetas in parallel using OpenMP.)DOC");
    model.def("objective_hessian",&ModelWrapper<Model>::objective_hessian_stack, py::arg("images"), py::arg("thetas"), R"DOC(
        Calculate the Hessian of the log-likelihood under the model objective.
        Operates on one or more images at one or more thetas in parallel using OpenMP.)DOC");
    model.def("objective_negative_definite_hessian",&ModelWrapper<Model>::objective_negative_definite_hessian_stack, py::arg("images"), py::arg("thetas"), R"DOC(
        Calculate the best negative-definite approximation to the hessian of the relative log-likelihood.
        This uses a modified cholesky decomposition method to adjust the negative Hessian to be positive definite.
        Operates on one or more images at one or more thetas in parallel using OpenMP.)DOC");
    
    model.def("objective",&ModelWrapper<Model>::objective, py::arg("image"), py::arg("theta"), R"DOC(
        Returns the tuple (rllh, grad, hessian) of the model objective.
        The objective depends on the Estimator type (MLE) or (MAP).  This should be called as the objective 
        function to maximize in optimization algorithms.
        Operates on a single image and theta.)DOC");

    model.def("likelihood_objective",&ModelWrapper<Model>::likelihood_objective, py::arg("image"), py::arg("theta"), R"DOC(
        Returns the tuple (rllh, grad, hessian) of the pure log-likelihood function.
        This used as the objective for the MLE models.
        Operates on a single image and theta.)DOC");

    model.def("prior_objective",&ModelWrapper<Model>::prior_objective, py::arg("theta"), R"DOC(
        Returns the tuple (rllh, grad, hessian) of the prior log-likelihood.
        Operates on  a single theta.)DOC");
    model.def("aposteriori_objective",&ModelWrapper<Model>::aposteriori_objective, py::arg("image"), py::arg("theta"), R"DOC(
        Returns the tuple (rllh, grad, hessian) of the log-aposteriori function  
        This is equivalent to log_likelihood + log_prior, and is the objective for the MAP models.
        Operates on a single image and theta.)DOC");
    
    model.def("cr_lower_bound",&ModelWrapper<Model>::cr_lower_bound, py::arg("thetas"),
                "Returns the Cramer-Rao lower-bound at one or more thetas.");
    model.def("expected_information",&ModelWrapper<Model>::expected_information, py::arg("thetas"),
                "Returns the Expected Fisher information matrix at one or more thetas.");
    model.def("observed_information",&ModelWrapper<Model>::observed_information, py::arg("image"), py::arg("theta_mode"), R"DOC(
        Returns the Observed Fisher information matrix with respect to a single image.
        This only makes logical sense if theta_mode is the etimatated maximum (mode).  If the returned observed information
        matrix is not positive definite, the reported point is not a true local maxima.)DOC");

    model.def("_estimate_max",&ModelWrapper<Model>::estimate_max,  
              py::arg("images"),  py::arg("method")="Newton",  py::arg("theta_init")=ArrayDoubleT(), py::arg("return_stats")=false, R"DOC(
        Returns (theta_max_stack,rllh_stack,observedI_stack). Estimates the maximum of the model objective.  
        
        This is Maximum likelihood estimation (MLE) or maximum-aposeteriori (MAP) estimation depending on the model.  
        fixed_theta is a vector of fixed values free values are indicated by inf or nan. [OpenMP])DOC");

//     model.deg("estimate_profile_max",&ModelWrapper<Model>::estimate_profile_max,  
//               py::arg("image"),  py::arg("fixed_theta_stack"), py::arg("method"), py::arg("theta_mle"), py::arg("return_stats")=false,
//               "Returns (theta_profile_max_stack,rllh_stack,observedI_stack) estimating the maximum of the model objective for each image using given method and theta_init. [OpenMP]");
              
    
    model.def("estimate_mcmc_posterior",&ModelWrapper<Model>::estimate_mcmc_posterior, 
              py::arg("images"),  py::arg("Nsample")=1000, py::arg("theta_init")=ArrayDoubleT(),  
              py::arg("Nburnin")=100, py::arg("thin")=0, R"DOC(
        Returns the summarized MCMC posterior mean and covariance: (theta_mean_stack, theta_cov_stack) 
        Operates on a single image or a stack of images in parallel using OpenMP.)DOC");
    
    model.def("estimate_mcmc_sample",&ModelWrapper<Model>::estimate_mcmc_sample, 
              py::arg("images"), py::arg("Nsample")=1000, py::arg("theta_init")=ArrayDoubleT(), 
              py::arg("Nburnin")=100, py::arg("thin")=0,  R"DOC(
        Returns the full MCMC sample and relative log-likelihood: (sample_stack, sample_rllh_stack).  
        The sample can be used to estimate posterior credible intervals. 
        Operates on a single image or a stack of images in parallel using OpenMP.)DOC");

    model.def("error_bounds_expected",&ModelWrapper<Model>::error_bounds_expected,  
              py::arg("theta_est"), py::arg("confidence")=0.95, R"DOC(
        Returns error bounds for each parameter (theta_lb, theta_ub), using the Expected Fisher Information.
        Operates on one or more estimated theta values in parallel using OpenMP.

        These bounds are only valid if the estimator errors are normally distributed (i.e., the objective 
        function is regular near the maximum).  Additionally, the estimator must be unbiased and approach the 
        Cramer-Rao lower bound in accuracy.)DOC");
    
        
    model.def("error_bounds_observed",&ModelWrapper<Model>::error_bounds_observed,  
              py::arg("theta_est"), py::arg("obsI"), py::arg("confidence")=0.95, R"DOC(
        Returns error bounds for each parameter (theta_lb, theta_ub) using the Observed Fisher Information.
        Operates on one or more estimated theta values in parallel using OpenMP.
        
        The observed Fisher Information is the same as the negative hessian at the estimated maximum.  This should
        be positive definte if theta_est is a true maximum.
        
        These bounds are only valid if the estimator errors are normally distributed (i.e., the objective 
        function is regular near the maximum.)DOC");
                
//     model.deg("error_bounds_profile",&ModelWrapper<Model>::error_bounds_profile,  
//               py::arg("images"),py::arg("theta_est"), py::arg("confidence")=0.95,
//               " Returns error bounds for each parameter (theta_lb, theta_ub). Make no assumptions about the Normality of the errors or regularity of the objective and use a pure-likelihood based approach to find the estimated error bounds. [OpenMP]");
    model.def("error_bounds_posterior_credible",&ModelWrapper<Model>::error_bounds_posterior_credible,  
              py::arg("samples"), py::arg("confidence")=0.95, R"DOC(
        Returns error bounds for each parameter (theta_mean, theta_lb, theta_ub).

        Operates on one or more images, using an MCMC sample.  Assuming sufficient sample size and mcmc mixing, 
        these bounds are valid even for non-regular posterior distributions. [OpenMP])DOC");
        
    /* Debugging methods (single threaded) */
    model.def("objective_llh_components",&ModelWrapper<Model>::objective_llh_components, 
              py::arg("image"), py::arg("theta"), R"DOC(
        [Debugging Usage] Calculate for each component (each pixel and each prior parameter) the full log-likelihood contribution.
        Operates on a single image and theta.)DOC");
    model.def("objective_rllh_components",&ModelWrapper<Model>::objective_rllh_components, 
              py::arg("image"), py::arg("theta"), R"DOC(
        [Debugging Usage] Calculate for each component (each pixel and each prior parameter) the relative log-likelihood contribution.
        Operates on a single image and theta.)DOC");
    model.def("objective_grad_components",&ModelWrapper<Model>::objective_grad_components, 
              py::arg("image"), py::arg("theta"), R"DOC(
        [Debugging Usage] Calculate for each component (each pixel and each prior parameter) the contribution to the gradient of the log-likelihood.
        Operates on a single image and theta.)DOC");
    model.def("objective_hessian_components",&ModelWrapper<Model>::objective_hessian_components, 
              py::arg("image"), py::arg("theta"), R"DOC(
        [Debugging Usage] Calculate for each component (each pixel and each prior parameter) the contribution to the hessian of the log-likelihood.
        Operates on a single image and theta.)DOC");
    
    model.def("_estimate_max_debug",&ModelWrapper<Model>::estimate_max_debug,
              py::arg("image"),  py::arg("method"),  py::arg("theta_init")=ArrayDoubleT(), R"DOC(
         [Debugging Usage] Returns (theta_max, rllh, observedI, stats, sequence, sequence_rllh) For a single image.  
         The returned sequence is all evaluated points in sequence.)DOC");    
//     model.def("estimate_profile_max_debug",&ModelWrapper<Model>::estimate_max_debug,
//               py::arg("image"),  py::arg("fixed_theta"), py::arg("method"), py::arg("theta_mle"), py::arg("return_stats")=false,
//               "[DEBUGGING] Returns (theta_profile_max, rllh_stack, stats, sequence, sequence_rllh) For a single image.  The returned sequence is all evaluated points in sequence.");    
    model.def("estimate_mcmc_debug",&ModelWrapper<Model>::estimate_mcmc_debug, 
              py::arg("image"), py::arg("Nsample")=100, py::arg("theta_init")=ArrayDoubleT(), R"DOC(
        [Debugging Usage] Returns (sample, sample_rllh, candidates, candidates_rllh).  
        Running MCMC sampling for a single image.  No thinning or burnin is performed. Candidates are the 
        proposed theta values at each iteration.)DOC");
    model.def("initial_theta_estimate",&ModelWrapper<Model>::initial_theta_estimate,
              py::arg("image"), py::arg("theta_init")=ArrayDoubleT(), R"DOC(
        [Debugging Usage] Heuristic estimate of the image, with optional theta_init partially specified parameter vector.
        This is to help debug interface issues, and is internally called to initialize estimation routines when theta_init is
        not fully specified.)DOC");
}


template<class Model>
Model 
ModelWrapper<Model>::construct_fixed2D(ArrayT<ImageCoordT> &size_arr, ArrayDoubleT &psf_sigma_arr)
{
    auto size = pyarma::asVec<ImageCoordT>(size_arr);
    auto psf_sigma = pyarma::asVec(psf_sigma_arr);
    return {size,psf_sigma};
}

template<class Model>
Model 
ModelWrapper<Model>::construct_variable2D(ArrayT<ImageCoordT> &size_arr, ArrayDoubleT &min_sigma_arr, ArrayDoubleT &max_sigma_arr)
{
    auto size = pyarma::asVec<ImageCoordT>(size_arr);
    auto min_sigma = pyarma::asVec(min_sigma_arr);
    auto max_sigma = pyarma::asVec(max_sigma_arr);
    return {size,min_sigma,max_sigma};
}

template<class Model>
ArrayT<typename Model::ImageCoordT> 
ModelWrapper<Model>::get_size(Model &model)
{
    auto out = pyarma::makeArray<ImageCoordT>(Model::num_dim);
    auto size = pyarma::asVec<ImageCoordT>(out);
    size = model.get_size();
    return out;
}

template<class Model>
void 
ModelWrapper<Model>::set_size(Model &model, ArrayT<ImageCoordT> &size_arr)
{
    auto size = pyarma::asVec<ImageCoordT>(size_arr);
    model.set_size(size);
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::get_psf_sigma(Model &model)
{
    auto out = pyarma::makeArray(Model::num_dim);
    auto psf_sigma = pyarma::asVec(out);
    psf_sigma = model.get_psf_sigma();
    return out;
}

template<class Model>
void 
ModelWrapper<Model>::set_psf_sigma(Model &model, ArrayDoubleT &psf_sigma_arr)
{
    auto psf_sigma = pyarma::asVec(psf_sigma_arr);
    model.set_psf_sigma(psf_sigma);
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::get_min_sigma(Model &model)
{
    auto out = pyarma::makeArray(Model::num_dim);
    auto min_sigma = pyarma::asVec(out);
    min_sigma = model.get_min_sigma();
    return out;
}

template<class Model>
void 
ModelWrapper<Model>::set_min_sigma(Model &model, ArrayDoubleT &min_sigma_arr)
{
    auto min_sigma = pyarma::asVec(min_sigma_arr);
    model.set_min_sigma(min_sigma);
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::get_max_sigma(Model &model)
{
    auto out = pyarma::makeArray(Model::num_dim);
    auto max_sigma = pyarma::asVec(out);
    max_sigma = model.get_max_sigma();
    return out;
}

template<class Model>
void 
ModelWrapper<Model>::set_max_sigma(Model &model, ArrayDoubleT &max_sigma_arr)
{
    auto max_sigma = pyarma::asVec(max_sigma_arr);
    model.set_max_sigma(max_sigma);
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
    auto params = pyarma::asVec(arr);
    model.set_hyperparams(params);
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
ModelWrapper<Model>::bounded_theta(Model &model, ArrayDoubleT &thetas_arr)
{
    auto thetas = thetaStackAsArma(thetas_arr);
    auto out = pyarma::makeSqueezedArray(model.get_num_params(), thetas.n_cols);
    auto bd_theta = pyarma::asMat(out);
    bd_theta = model.bounded_theta_stack(thetas);    
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::reflected_theta(Model &model, ArrayDoubleT &thetas_arr)
{
    auto thetas =thetaStackAsArma(thetas_arr);
    auto out = pyarma::makeSqueezedArray(model.get_num_params(), thetas.n_cols);
    auto bd_theta = pyarma::asMat(out);
    bd_theta = model.reflected_theta_stack(thetas);    
    return out;
}

template<class Model>    
ArrayT<BoolT>
ModelWrapper<Model>::theta_in_bounds(Model &model, ArrayDoubleT &thetas_arr)
{
    auto thetas = thetaStackAsArma(thetas_arr);
    auto out = pyarma::makeArray<BoolT>(thetas.n_cols);
    auto im_bounds = pyarma::asVec<BoolT>(out);
    im_bounds = model.theta_stack_in_bounds(thetas);
    return out;
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
    auto out = pyarma::makeSqueezedArray(model.get_num_params(),count);
    auto theta = pyarma::asMat(out);
    methods::sample_prior_stack(model,theta);
    return out;
}


template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::model_image_stack(Model &model, ArrayDoubleT &thetas_arr)
{
    auto thetas = thetaStackAsArma(thetas_arr);
    auto out = makeImageStackArray(model.get_size(), thetas.n_cols);
    auto ims = imageStackAsArma<Model>(out);
    methods::model_image_stack(model,thetas, ims);
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::simulate_image_stack(Model &model, ArrayDoubleT &thetas_arr, IdxT count)
{
    auto thetas = thetaStackAsArma(thetas_arr);
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
    auto thetas = thetaStackAsArma(thetas_arr);
    auto ims = imageStackAsArma<Model>(ims_arr);
    IdxT count = std::max(thetas.n_cols, static_cast<IdxT>(model.get_size_image_stack(ims))); 
    auto out = pyarma::makeSqueezedArray(count);
    auto llh = pyarma::asVec(out);
    methods::objective::llh_stack(model,ims, thetas, llh);
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::objective_rllh_stack(Model &model, ArrayDoubleT &ims_arr, ArrayDoubleT &thetas_arr)
{
    auto thetas = thetaStackAsArma(thetas_arr);
    auto ims = imageStackAsArma<Model>(ims_arr);
    IdxT count = std::max(thetas.n_cols, static_cast<IdxT>(model.get_size_image_stack(ims))); 
    auto out = pyarma::makeSqueezedArray(count);
    auto rllh = pyarma::asVec(out);
    methods::objective::rllh_stack(model,ims, thetas, rllh);
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::objective_grad_stack(Model &model, ArrayDoubleT &ims_arr, ArrayDoubleT &thetas_arr)
{
    auto thetas = thetaStackAsArma(thetas_arr);
    auto ims = imageStackAsArma<Model>(ims_arr);
    IdxT count = std::max(thetas.n_cols, static_cast<IdxT>(model.get_size_image_stack(ims))); 
    auto out = pyarma::makeSqueezedArray(model.get_num_params(), count);
    auto grad = pyarma::asMat(out);
    methods::objective::grad_stack(model,ims, thetas, grad);
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::objective_hessian_stack(Model &model, ArrayDoubleT &ims_arr, ArrayDoubleT &thetas_arr)
{
    auto thetas = thetaStackAsArma(thetas_arr);
    auto ims = imageStackAsArma<Model>(ims_arr);
    IdxT count = std::max(thetas.n_cols, static_cast<IdxT>(model.get_size_image_stack(ims))); 
    auto out = pyarma::makeSqueezedArray(model.get_num_params(), model.get_num_params(), count);
    auto hess = pyarma::asCube(out);
    methods::objective::hessian_stack(model,ims, thetas, hess);
    copy_Usym_mat_stack(hess); // Convert upper triangular symmetric representation to full-matrix for python
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::objective_negative_definite_hessian_stack(Model &model, ArrayDoubleT &ims_arr, ArrayDoubleT &thetas_arr)
{
    auto thetas = thetaStackAsArma(thetas_arr);
    auto ims = imageStackAsArma<Model>(ims_arr);
    IdxT count = std::max(thetas.n_cols, static_cast<IdxT>(model.get_size_image_stack(ims))); 
    auto out = pyarma::makeSqueezedArray(model.get_num_params(), model.get_num_params(), count);
    auto hess = pyarma::asCube(out);
    methods::objective::negative_definite_hessian_stack(model,ims, thetas, hess);
    //copy_Usym_mat_stack(hess);  Not needed here because modified cholesky provides full symmetric matrix for same price as upper triangular
    return out;
}

template<class Model>
py::tuple 
ModelWrapper<Model>::objective(Model &model, ArrayDoubleT &im_arr, ArrayDoubleT &theta_arr)
{
    auto theta = thetaAsArma(theta_arr);
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
    copy_Usym_mat(hess); // Convert upper triangular symmetric representation to full-matrix for python

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
    auto theta = thetaAsArma(theta_arr);
    auto im = imageAsArma<Model>(im_arr);
    double rllh;
    auto N = model.get_num_params();
    auto grad_arr = pyarma::makeArray(N);
    auto grad = pyarma::asVec(grad_arr);
    auto hess_arr = pyarma::makeArray(N,N);
    auto hess = pyarma::asMat(hess_arr);
    methods::likelihood_objective(model,im,theta,rllh,grad,hess);
    copy_Usym_mat(hess); // Convert upper triangular symmetric representation to full-matrix for python

    py::tuple out(3);
    out[0] = rllh;
    out[1] = grad_arr;
    out[2] = hess_arr;
    return out; 
}

template<class Model>
py::tuple 
ModelWrapper<Model>::prior_objective(Model &model,  ArrayDoubleT &theta_arr)
{
    auto theta = thetaAsArma(theta_arr);
    double rllh;
    auto N = model.get_num_params();
    auto grad_arr = pyarma::makeArray(N);
    auto grad = pyarma::asVec(grad_arr);
    auto hess_arr = pyarma::makeArray(N,N);
    auto hess = pyarma::asMat(hess_arr);
    methods::prior_objective(model,theta,rllh,grad,hess);
    copy_Usym_mat(hess); // Convert upper triangular symmetric representation to full-matrix for python
    
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
    auto theta = thetaAsArma(theta_arr);
    auto im = imageAsArma<Model>(im_arr);
    double rllh;
    auto N = model.get_num_params();
    auto grad_arr = pyarma::makeArray(N);
    auto grad = pyarma::asVec(grad_arr);
    auto hess_arr = pyarma::makeArray(N,N);
    auto hess = pyarma::asMat(hess_arr);
    methods::aposteriori_objective(model,im,theta,rllh,grad,hess);
    copy_Usym_mat(hess); // Convert upper triangular symmetric representation to full-matrix for python
    
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
    auto thetas = thetaStackAsArma(thetas_arr);
    IdxT count = thetas.n_cols;
    auto out = pyarma::makeSqueezedArray(model.get_num_params(), count);
    auto cr_vec = pyarma::asMat(out);
    methods::cr_lower_bound_stack(model, thetas, cr_vec);
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::expected_information(Model &model, ArrayDoubleT &thetas_arr)
{
    auto thetas = thetaStackAsArma(thetas_arr);
    IdxT count = thetas.n_cols;
    auto out = pyarma::makeSqueezedArray(model.get_num_params(), model.get_num_params(), count);
    auto I_stack = pyarma::asCube(out);
    methods::expected_information_stack(model, thetas, I_stack);
    copy_Usym_mat_stack(I_stack); // Convert upper triangular symmetric representation to full-matrix for python
    return out;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::observed_information(Model &model, ArrayDoubleT &im_arr, ArrayDoubleT &theta_mode_arr)
{
    auto theta = thetaAsArma(theta_mode_arr);
    auto im = imageAsArma<Model>(im_arr);
    auto out = pyarma::makeArray(model.get_num_params(), model.get_num_params());
    auto hess = pyarma::asMat(out);
    methods::objective::hessian(model, im, theta, hess);
    copy_Usym_mat(hess); // Convert upper triangular symmetric representation to full-matrix for python
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
        theta_init_stack = thetaStackAsArma(theta_init_arr);
        model.check_param_shape(theta_init_stack);
        if(theta_init_stack.n_cols != count) {
            std::ostringstream msg;
            msg<<"Got inconsistent counts: #images="<<count<<" #theta_inits="<<theta_init_stack.n_cols;
            throw PythonError("ArrayShape",msg.str());
        }
    }
    auto theta_max_arr = pyarma::makeSqueezedArray(model.get_num_params(),count);
    auto theta_max_stack = pyarma::asMat(theta_max_arr);
    
    auto rllh_arr = pyarma::makeSqueezedArray(count);
    auto rllh_stack = pyarma::asVec(rllh_arr);
    
    auto obsI_arr = pyarma::makeSqueezedArray(model.get_num_params(),model.get_num_params(),count);
    auto obsI_stack = pyarma::asCube(obsI_arr);
    
    if(return_stats){
        StatsT stats;
        methods::estimate_max_stack(model, image_stack, method, theta_init_stack, theta_max_stack, rllh_stack, obsI_stack, stats);
        copy_Usym_mat_stack(obsI_stack); // Convert upper triangular symmetric representation to full-matrix for python

        py::tuple out(4);
        out[0] = theta_max_arr;
        out[1] = rllh_arr;
        out[2] = obsI_arr;
        out[3] = stats;
        return out; 
    } else {
        methods::estimate_max_stack(model, image_stack, method, theta_init_stack, theta_max_stack, rllh_stack, obsI_stack);
        copy_Usym_mat_stack(obsI_stack); // Convert upper triangular symmetric representation to full-matrix for python

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
ModelWrapper<Model>::estimate_mcmc_sample(Model &model, ArrayDoubleT &images_arr, IdxT Nsample, ArrayDoubleT &theta_init_arr, 
                                          IdxT Nburnin, IdxT thin)
{
    auto image_stack = imageStackAsArma<Model>(images_arr);
    IdxT count = model.get_size_image_stack(image_stack);
    MatT theta_init_stack;
    if(theta_init_arr.ndim()==0 || theta_init_arr.size()==0) {
        //Initialize empty
        theta_init_stack.set_size(model.get_num_params(),count);
        theta_init_stack.zeros();
    } else {
        theta_init_stack = thetaStackAsArma(theta_init_arr);
        model.check_param_shape(theta_init_stack);
        if(theta_init_stack.n_cols != count) {
            std::ostringstream msg;
            msg<<"Got inconsistent counts: #images="<<count<<" #theta_inits="<<theta_init_stack.n_cols;
            throw PythonError("ArrayShape",msg.str());
        }
    }
    auto sample_arr = pyarma::makeSqueezedArray(model.get_num_params(), Nsample, count);
    auto sample_stack = pyarma::asCube(sample_arr);
    auto sample_rllh_arr = pyarma::makeSqueezedArray(Nsample, count);
    auto sample_rllh_stack = pyarma::asMat(sample_rllh_arr);
    methods::estimate_mcmc_sample_stack(model,image_stack,theta_init_stack, Nsample, Nburnin, thin, sample_stack, sample_rllh_stack);
    py::tuple out(2);
    out[0] = sample_arr;
    out[1] = sample_rllh_arr;
    return out; 
}

template<class Model>
py::tuple 
ModelWrapper<Model>::estimate_mcmc_posterior(Model &model, ArrayDoubleT &images_arr, IdxT Nsample, ArrayDoubleT &theta_init_arr,  IdxT Nburnin, IdxT thin)
{
    auto image_stack = imageStackAsArma<Model>(images_arr);
    IdxT count = model.get_size_image_stack(image_stack);
    MatT theta_init_stack;
    if(theta_init_arr.ndim()==0 || theta_init_arr.size()==0) {
        //Initialize empty
        theta_init_stack.set_size(model.get_num_params(),count);
        theta_init_stack.zeros();
    } else {
        theta_init_stack = thetaStackAsArma(theta_init_arr);
        model.check_param_shape(theta_init_stack);
        if(theta_init_stack.n_cols != count) {
            std::ostringstream msg;
            msg<<"Got inconsistent counts: #images="<<count<<" #theta_inits="<<theta_init_stack.n_cols;
            throw PythonError("ArrayShape",msg.str());
        }
    }
    auto theta_mean_arr = pyarma::makeSqueezedArray(model.get_num_params(), count);
    auto theta_mean_stack = pyarma::asMat(theta_mean_arr);
    auto theta_cov_arr = pyarma::makeSqueezedArray(model.get_num_params(), model.get_num_params(), count);
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
    auto theta_stack = thetaStackAsArma(theta_arr);
    IdxT count = theta_stack.n_cols;
    auto theta_lb_arr = pyarma::makeSqueezedArray(model.get_num_params(), count);
    auto theta_lb_stack = pyarma::asMat(theta_lb_arr);
    auto theta_ub_arr = pyarma::makeSqueezedArray(model.get_num_params(), count);
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
    auto theta_stack = thetaStackAsArma(theta_arr);
    IdxT count = theta_stack.n_cols;
    auto obsI_stack = pyarma::asCube(obsI_arr);
    if(obsI_stack.n_slices != count) {
        std::ostringstream msg;
        msg<<"Got inconsistent counts: #theta="<<count<<" #obsI="<<obsI_stack.n_slices;
        throw PythonError("ArrayShape",msg.str());
    }
    auto theta_lb_arr = pyarma::makeSqueezedArray(model.get_num_params(), count);
    auto theta_lb_stack = pyarma::asMat(theta_lb_arr);
    auto theta_ub_arr = pyarma::makeSqueezedArray(model.get_num_params(), count);
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
    auto theta_mean_arr = pyarma::makeSqueezedArray(model.get_num_params(), count);
    auto theta_mean_stack = pyarma::asMat(theta_mean_arr);
    auto theta_lb_arr = pyarma::makeSqueezedArray(model.get_num_params(), count);
    auto theta_lb_stack = pyarma::asMat(theta_lb_arr);
    auto theta_ub_arr = pyarma::makeSqueezedArray(model.get_num_params(), count);
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
    auto theta = thetaAsArma(theta_arr);
    auto llh_comps = methods::objective::llh_components(model,im, theta);
    IdxT count = llh_comps.n_elem;
    auto llh_arr = pyarma::makeSqueezedArray(count);
    auto llh_stack = pyarma::asVec(llh_arr);
    llh_stack = llh_comps;
    return llh_arr;    
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::objective_rllh_components(Model &model, ArrayDoubleT &im_arr, ArrayDoubleT &theta_arr)
{
    auto im = imageAsArma<Model>(im_arr);
    auto theta = thetaAsArma(theta_arr);
    auto rllh_comps = methods::objective::rllh_components(model,im, theta);
    IdxT count = rllh_comps.n_elem;
    auto rllh_arr = pyarma::makeSqueezedArray(count);
    auto rllh_stack = pyarma::asVec(rllh_arr);
    rllh_stack = rllh_comps;
    return rllh_arr;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::objective_grad_components(Model &model, ArrayDoubleT &im_arr, ArrayDoubleT &theta_arr)
{
    auto im = imageAsArma<Model>(im_arr);
    auto theta = thetaAsArma(theta_arr);
    auto grad_comps = methods::objective::grad_components(model,im, theta);
    IdxT count = grad_comps.n_cols;
    auto grad_arr = pyarma::makeSqueezedArray(model.get_num_params(),count);
    auto grad_stack = pyarma::asMat(grad_arr);
    grad_stack = grad_comps;
    return grad_arr;
}

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::objective_hessian_components(Model &model, ArrayDoubleT &im_arr, ArrayDoubleT &theta_arr)
{
    auto im = imageAsArma<Model>(im_arr);
    auto theta = thetaAsArma(theta_arr);
    auto hess_comps = methods::objective::hessian_components(model,im, theta);
    IdxT count = hess_comps.n_slices;
    auto hess_arr = pyarma::makeSqueezedArray(model.get_num_params(),model.get_num_params(),count);
    auto hess_stack = pyarma::asCube(hess_arr);
    hess_stack = hess_comps;
    return hess_arr;
}

template<class Model>
py::tuple 
ModelWrapper<Model>::estimate_max_debug(Model &model, ArrayDoubleT &image_arr, std::string method, ArrayDoubleT &theta_init_arr)
{
    auto image = imageAsArma<Model>(image_arr);
    auto theta_init = thetaAsArma(theta_init_arr);
    auto theta_est_arr = pyarma::makeArray(model.get_num_params());
    auto theta_est = pyarma::asVec(theta_est_arr);
    auto obsI_arr = pyarma::makeArray(model.get_num_params(),model.get_num_params());
    auto obsI = pyarma::asMat(obsI_arr);
    StatsT stats;
    MatT sequence;
    VecT sequence_rllh;
    double rllh;
    methods::estimate_max_debug(model, image, method, theta_init, theta_est, rllh, obsI, sequence, sequence_rllh, stats);
    IdxT Nseq = sequence.n_cols;
    auto sequence_arr = pyarma::makeArray(model.get_num_params(), Nseq);
    auto sequence_out = pyarma::asMat(sequence_arr);
    sequence_out = sequence;
    auto sequence_rllh_arr = pyarma::makeArray(Nseq);
    auto sequence_rllh_out = pyarma::asVec(sequence_rllh_arr);
    sequence_rllh_out = sequence_rllh;
    copy_Usym_mat(obsI); // Convert upper triangular symmetric representation to full-matrix for python

    py::tuple out(6);
    out[0] = theta_est_arr;
    out[1] = rllh;
    out[2] = obsI_arr;
    out[3] = stats;
    out[4] = sequence_arr;
    out[5] = sequence_rllh_arr;    
    return out;    
}

template<class Model>
py::tuple 
ModelWrapper<Model>::estimate_mcmc_debug(Model &model, ArrayDoubleT &image_arr, IdxT Nsample, ArrayDoubleT &theta_init_arr)
{
    auto image = imageAsArma<Model>(image_arr);
    auto theta_init = thetaAsArma(theta_init_arr);
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

template<class Model>
ArrayDoubleT 
ModelWrapper<Model>::initial_theta_estimate(Model &model, ArrayDoubleT &image_arr, ArrayDoubleT &theta_init_arr)
{
    auto image = imageAsArma<Model>(image_arr);
    auto theta_est_arr = pyarma::makeArray(model.get_num_params());
    auto theta_est = pyarma::asVec(theta_est_arr);
    VecT theta_init;
    if(theta_init_arr.size() == static_cast<ssize_t>(model.get_num_params())) {
        theta_init = thetaAsArma(theta_init_arr);
    }
    auto s = model.initial_theta_estimate(image,theta_init);
    theta_est = s.theta;
    return theta_est_arr;
}

    

} /* namespace mappel::python */
} /* namespace mappel */

#endif /*_PY11_ARMADILLO_IFACE */
