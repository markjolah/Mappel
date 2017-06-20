/** @file Gauss2DMLE_Iface.cpp
 *  @brief The entry point for Gauss2DMLE_Iface mex module.
 * 
 * Just calls the MappleMexIface.mexFunction which is the Iface entry point.
 * 
 */
#include "Mappel2D_Iface.h"
#include "Gauss2DMLE.h"

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    mappel::Mappel2D_Iface<mappel::Gauss2DMLE> iface("Gauss2DMLE");
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
