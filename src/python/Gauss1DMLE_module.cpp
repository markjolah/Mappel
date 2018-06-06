/** @file Gauss1DMLE_module.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The instantiation of the mappel python module for Gauss1DMLE
 */
#include "Python.h"
#include "py11_mappel_iface.h"
#include "Mappel/Gauss1DMLE.h"

PYBIND11_MODULE(_Gauss1DMLE, M)
{
    M.doc()="1D Gaussian PSF model with fixed sigma and Poisson image noise, under a maximum likelihood objective";
    mappel::python::bindMappelModel<mappel::Gauss1DMLE>(M);
}

