/** @file Gauss2DsMLE_IFace.cpp
 *  @brief The entry point for Gauss2DMLE_IFace mex module.
 * 
 * Just calls the MappleMexIface.mexFunction which is the IFace entry point.
 * 
 */
#include "Mappel2D_IFace.h"
#include "Gauss2DsMLE.h"

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    Mappel2D_Iface<Gauss2DsMLE> iface("Gauss2DsMLE");
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
