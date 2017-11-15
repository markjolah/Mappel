/** @file Gauss1DMLE_IFace.cpp
 *  @brief The entry point for Gauss1DMLE_IFace mex module.
 * 
 * Just calls the MappleMexIface.mexFunction which is the IFace entry point.
 * 
 */
#include "Mappel_IFace.h"
#include "Gauss1DMLE.h"

mappel::MappelFixedSigma_IFace<mappel::Gauss1DMLE> iface; /**< Global iface object provides a iface.mexFunction */

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
