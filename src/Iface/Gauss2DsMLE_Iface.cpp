/** @file Gauss2DsMLE_Iface.cpp
 *  @brief The entry point for Gauss2DMLE_Iface mex module.
 * 
 * Just calls the MappleMexIface.mexFunction which is the Iface entry point.
 * 
 */
#include "Mappel2D_Iface.h"
#include "Gauss2DsMLE.h"

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    Mappel2D_Iface<Gauss2DsMLE> iface("Gauss2DsMLE");
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
