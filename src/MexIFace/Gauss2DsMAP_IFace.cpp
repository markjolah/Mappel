/** @file Gauss2DsMAP_IFace.cpp
 *  @brief The entry point for Gauss2DsMAP_IFace mex module.
 *  @date 2015-2019
 */

#include "Mappel/Gauss2DsMAP.h"
#include "Mappel_IFace.h"

MappelVarSigma_IFace<mappel::Gauss2DsMAP> iface;  /**< Global iface object provides a iface.mexFunction */

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
