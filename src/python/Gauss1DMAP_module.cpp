/** @file Gauss1DMAP_module.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2018
 * @brief The instantiation of the mappel python module for Gauss1DMAP
 */
#include "Python.h"
#include "py11_mappel_iface.h"
#include "Gauss1DMAP.h"

PYBIND11_MODULE(_Gauss1DMAP, M)
{
    M.doc()="1D Gaussian PSF model with fixed sigma and Poisson image noise, under a maximum a-poseteriori objective.";
    mappel::python::bindMappelModel<mappel::Gauss1DMAP>(M);
}

