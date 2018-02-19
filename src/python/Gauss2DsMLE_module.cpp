/** @file Gauss2DsMLE_module.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The instantiation of the mappel python module for Gauss2DsMLE
 */
#include "Python.h"
#include "py11_mappel_iface.h"
#include "Gauss2DsMLE.h"

PYBIND11_MODULE(_Gauss2DsMLE, M)
{
    M.doc()="2D Gaussian PSF model with variable scalar PSF sigma under a Poisson noise assumption using a maximum-likelihood objective.";
    mappel::python::bindMappelModel<mappel::Gauss2DsMLE>(M);
}

