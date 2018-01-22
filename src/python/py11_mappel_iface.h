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

    static ArrayDoubleT simulate_image(Model &model, ArrayDoubleT &theta); 
    static ArrayDoubleT simulate_image_stack(Model &model, ArrayDoubleT &thetas, IdxT count);
// 
//     static ArrayDoubleT objective_llh(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta); 
//     static ArrayDoubleT objective_llh_stack(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
// 
//     static ArrayDoubleT objective_rllh(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta); 
//     static ArrayDoubleT objective_rllh_stack(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
// 
//     static ArrayDoubleT objective_grad(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta); 
//     static ArrayDoubleT objective_grad_stack(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
// 
//     static ArrayDoubleT objective_grad2(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta); 
//     static ArrayDoubleT objective_grad2_stack(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
// 
//     static ArrayDoubleT objective_hessian(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta); 
//     static ArrayDoubleT objective_hessian_stack(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
// 
//     static ArrayDoubleT model_objective(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta); 
//     static ArrayDoubleT likelihood_objective(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
//     static ArrayDoubleT prior_objective(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
//     static ArrayDoubleT aposteriori_objective(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
};

template<class Model>
typename std::enable_if<std::is_base_of<ImageFormat1DBase,Model>::value, typename Model::ImageT>::type
imageAsArma(ArrayDoubleT &im)
{
    return {static_cast<double*>(im.mutable_data(0)), static_cast<IdxT>(im.size()), false, true};
}

template<class Model>
typename std::enable_if<std::is_base_of<ImageFormat1DBase,Model>::value, typename Model::ImageStackT>::type
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


// template<class Model>
// typename std::enable_if<std::is_base_of<ImageFormat2DBase,Model>::value, typename Model::ImageT>::type
// imageAsArma(Model &model, ArrayDoubleT &im)
// {
//     return {static_cast<double*>(im.mutable_data(0)), static_cast<IdxT>(im.shape(0)), static_cast<IdxT>(im.shape(1)), false, true};
// }

template<class Model>
void bindMappelModel(py::module &M)
{
    py::class_<Model> model(M, Model::name().c_str(), py::multiple_inheritance());
    if(std::is_base_of<Gauss1DModel,Model>::value) {
        model.def(py::init<typename Model::ImageCoordT,double>());
        model.def_property("size",[](Model &model) {return model.get_size();},[](Model &model,IdxT size) { model.set_size(size); },"1D-Image size in pixels." );
        model.def_property("psf_sigma",[](Model &model) {return model.get_psf_sigma();},[](Model &model,double sigma) { model.set_psf_sigma(sigma); },"Sigma of emitter (PSF) Gaussian approximation [pixels]." );
    }
    model.def_property_readonly_static("name",[](py::object /* self*/) {return Model::name().c_str();},
                                         "Model name.");
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
// 
//     model.def("objective_llh",&ModelWrapper<Model>::objective_llh,py::arg("image"), py::arg("theta"), 
//                 "Calculate the full log-likelihood for the image at theta under the model objective.");
//     model.def("objective_llh",&ModelWrapper<Model>::objective_llh,py::arg("images"), py::arg("thetas"),
//                 "Calculate the full log-likelihood for one or more images at one or more thetas under the  model objective. [OpenMP]");
}

// template<class Model>
// Model 
// ModelWrapper<Model>::init_fixed_sigma(ArrayT<ImageCoordT> &size, ArrayDoubleT &psf_sigma)
// {
//     return Model(pyarma::asVec<ImageCoordT>(size), pyarma::asVec(psf_sigma));
// }

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

// template<class Model>
// ArrayDoubleT 
// ModelWrapper<Model>::simulate_image(Model &model, ArrayDoubleT &theta_arr)
// {
//     auto theta = pyarma::asVec(theta_arr);
//     auto out = makeImageArray(model.get_size());
//     auto im = imageAsArma<Model>(out);
//     im = methods::simulate_image(model,theta);
//     return out;
// }

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
    std::cout<<"ims: ["<<ims.n_rows<<","<<ims.n_cols<<"]\n";
    methods::simulate_image_stack(model,thetas, ims);
    return out;
}


// 
// template<class Model>
//  ArrayDoubleT ModelWrapper<Model>::simulate_image(Model &model, ArrayDoubleT &theta); 
// template<class Model>
//  ArrayDoubleT ModelWrapper<Model>::simulate_image_stack(Model &model, ArrayDoubleT &thetas, IdxT count);
// 
// template<class Model>
//  ArrayDoubleT ModelWrapper<Model>::objective_llh(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta); 
// template<class Model>
//  ArrayDoubleT ModelWrapper<Model>::objective_llh_stack(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
// 
// template<class Model>
//  ArrayDoubleT ModelWrapper<Model>::objective_rllh(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta); 
// template<class Model>
//  ArrayDoubleT ModelWrapper<Model>::objective_rllh_stack(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
// 
// template<class Model>
//  ArrayDoubleT ModelWrapper<Model>::objective_grad(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta); 
// template<class Model>
//  ArrayDoubleT ModelWrapper<Model>::objective_grad_stack(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
// 
// template<class Model>
//  ArrayDoubleT ModelWrapper<Model>::objective_grad2(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta); 
// template<class Model>
//  ArrayDoubleT ModelWrapper<Model>::objective_grad2_stack(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);
// 
// template<class Model>
//  ArrayDoubleT ModelWrapper<Model>::objective_hessian(Model &model, ArrayDoubleT &image, ArrayDoubleT &theta); 
// template<class Model>
//  ArrayDoubleT ModelWrapper<Model>::objective_hessian_stack(Model &model, ArrayDoubleT &images, ArrayDoubleT &thetas);

} /* namespace mappel::python */
} /* namespace mappel */

#endif /*_PY11_ARMADILLO_IFACE */
