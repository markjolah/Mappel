/** @file Gauss2DMAP_IFace.cpp
 *  @brief The entry point for Gauss2DMAP_IFace mex module.
 *  @date 2015-2019
 *  @brief The entry point for LAPTrack_Iface mex module.
 */

#include "Mappel/Gauss2DMAP.h"
#include "Mappel_IFace.h"

MappelFixedSigma_IFace<mappel::Gauss2DMAP> iface;  /**< Global iface object provides a iface.mexFunction */

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
