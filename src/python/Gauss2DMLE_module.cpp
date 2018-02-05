/** @file Gauss2DMLE_module.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2018
 * @brief The instantiation of the mappel python module for Gauss2DMLE
 */
#include "Python.h"
#include "py11_mappel_iface.h"
#include "Gauss2DMLE.h"

PYBIND11_MODULE(_Gauss2DMLE, M)
{
    M.doc()="2D Gaussian PSF model with fixed sigma and Poisson image noise, under a maximum likelihood objective";
    mappel::python::bindMappelModel<mappel::Gauss2DMLE>(M);
}

