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

template<class Model>
class ModelWrapper
{
    using ImageCoordT = typename Model::ImageCoordT;
public:
    static Model* init_fixed_sigma(ArrayT<ImageCoordT> &size, ArrayDoubleT &psf_sigma);
    static ArrayDoubleT sample_prior(Model &model, IdxT count);
};

template<class Model>
void bindMappelModel(py::module &M)
{
    py::class_<Model> model(M, Model::name().c_str(), py::multiple_inheritance());
    if(std::is_base_of<Gauss1DModel,Model>::value) {
//         model.def(py::init(&ModelWrapper<Model>::init_fixed_sigma));
        model.def(py::init<typename Model::ImageCoordT,double>());
//         model.def("get_size",&Model::get_size);
        model.def("sample_prior",&ModelWrapper<Model>::sample_prior,py::return_value_policy::take_ownership );
//         model.def("set_size",&Model::set_size);
    }
    
}


template<class Model>
Model* 
ModelWrapper<Model>::init_fixed_sigma(ArrayT<ImageCoordT> &size, ArrayDoubleT &psf_sigma)
{
    return new Model(pyarma::asVec<ImageCoordT>(size), pyarma::asVec(psf_sigma));
}

template<class Model>
ArrayDoubleT
ModelWrapper<Model>::sample_prior(Model &model, IdxT count)
{
    ArrayDoubleT arr({model.get_num_params(), count});
    auto theta = pyarma::asMat(arr);
    methods::sample_prior_stack(model,theta);
    return arr;
}


} /* namespace mappel::python */
} /* namespace mappel */

#endif /*_PY11_ARMADILLO_IFACE */
