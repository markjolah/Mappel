/** @file Gauss2DMAP_IFace.cpp
 *  @brief The entry point for Gauss2DMAP_IFace mex module.
 * 
 * Just calls the MappleMexIface.mexFunction which is the IFace entry point.
 * 
 */
#include "Mappel2D_IFace.h"
#include "Gauss2DMAP.h"

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    mappel::Mappel2D_Iface<mappel::Gauss2DMAP> iface("Gauss2DMAP");
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
