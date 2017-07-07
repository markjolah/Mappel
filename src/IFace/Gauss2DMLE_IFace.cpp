/** @file Gauss2DMLE_IFace.cpp
 *  @brief The entry point for Gauss2DMLE_IFace mex module.
 * 
 * Just calls the MappleMexIFace.mexFunction which is the IFace entry point.
 * 
 */
#include "Mappel2D_IFace.h"
#include "Gauss2DMLE.h"

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    mappel::Mappel2D_Iface<mappel::Gauss2DMLE> iface("Gauss2DMLE");
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
