/** @file Gauss1DsMLE_module.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The instantiation of the mappel python module for Gauss1DsMLE
 */
#include "Python.h"
#include "py11_mappel_iface.h"
#include "Mappel/Gauss1DsMLE.h"

PYBIND11_MODULE(_Gauss1DsMLE, M)
{
    M.doc()="1D Gaussian PSF model with variable sigma and Poisson image noise, under a maximum likelihood objective";
    mappel::python::bindMappelModel<mappel::Gauss1DsMLE>(M);
}

