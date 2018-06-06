/** @file Gauss1DsMAP_module.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The instantiation of the mappel python module for Gauss1DsMAP
 */
#include "Python.h"
#include "py11_mappel_iface.h"
#include "Mappel/Gauss1DsMAP.h"

PYBIND11_MODULE(_Gauss1DsMAP, M)
{
    M.doc()="1D Gaussian PSF model variable fixed sigma and Poisson image noise, under a maximum a-poseteriori objective.";
    mappel::python::bindMappelModel<mappel::Gauss1DsMAP>(M);
}

