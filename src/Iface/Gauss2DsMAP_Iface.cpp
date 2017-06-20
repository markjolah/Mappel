/** @file Gauss2DsMAP_Iface.cpp
 *  @brief The entry point for Gauss2DMAP_Iface mex module.
 * 
 * Just calls the MappleMexIface.mexFunction which is the Iface entry point.
 * 
 */
#include "Mappel2D_Iface.h"
#include "Gauss2DsMAP.h"

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    Mappel2D_Iface<Gauss2DsMAP> iface("Gauss2DsMAP");
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
