/** @file MappelPythonModule.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2018
 * @brief The instantiation of the mappel python module
 *
 * This is the core for the mappel::python interface
 */
#include "Gauss1DMLE.h"
#include <boost/python.hpp>

using namespace boost::python;

template<class Model>
void bind_MappelFixedSigmaModel()
{
    class_<Model, boost::noncopyable>(Model::name().c_str(), init<arma::Col<typename Model::ImageCoordT>, mappel::VecT>())
        .def_readonly("psf_sigma",&Model::psf_sigma)
        ;
}  

BOOST_PYTHON_MODULE(mappel)
{
    bind_MappelFixedSigmaModel<mappel::Gauss1DMLE>();
}
