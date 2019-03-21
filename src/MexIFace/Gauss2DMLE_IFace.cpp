/** @file Gauss2DMLE_IFace.cpp
 *  @brief The entry point for Gauss2DMLE_IFace mex module.
 *  @date 2015-2019
 */

#include "Mappel/Gauss2DMLE.h"
#include "Mappel_IFace.h"

MappelFixedSigma_IFace<mappel::Gauss2DMLE> iface;  /**< Global iface object provides a iface.mexFunction */

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
