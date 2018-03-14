/** @file Gauss2DMAP_module.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief The instantiation of the mappel python module for Gauss2DMAP
 */
#include "Python.h"
#include "py11_mappel_iface.h"
#include "Gauss2DMAP.h"

PYBIND11_MODULE(_Gauss2DMAP, M)
{
    M.doc()="2D Gaussian PSF model with fixed sigma and Poisson image noise, under a maximum a-poseteriori objective.";
    mappel::python::bindMappelModel<mappel::Gauss2DMAP>(M);
}

