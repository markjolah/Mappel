/** @file Gauss2DsMAP_module.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The instantiation of the mappel python module for Gauss2DsMAP
 */
#include "Python.h"
#include "py11_mappel_iface.h"
#include "Mappel/Gauss2DsMAP.h"

PYBIND11_MODULE(_Gauss2DsMAP, M)
{
    M.doc()="2D Gaussian PSF model with variable scalar PSF sigma under a Poisson noise assumption using a maximum a-posteriori objective.";
    mappel::python::bindMappelModel<mappel::Gauss2DsMAP>(M);
}

