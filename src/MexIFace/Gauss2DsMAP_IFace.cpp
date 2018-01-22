/** @file Gauss2DsMAP_IFace.cpp
 *  @brief The entry point for Gauss2DMAP_IFace mex module.
 * 
 * Just calls the MappleMexIface.mexFunction which is the IFace entry point.
 * 
 */
#include "Mappel2D_IFace.h"
#include "Gauss2DsMAP.h"

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    Mappel2D_Iface<Gauss2DsMAP> iface("Gauss2DsMAP");
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
