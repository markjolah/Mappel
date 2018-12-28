/** @file Gauss1DMAP_IFace.cpp
 *  @author Mark J. Olah (mjo at cs.unm.edu)
 *  @date 2015-2018
 *  @brief The entry point for Gauss1DMAP_IFace mex module.
 * 
 */
#include "Mappel_IFace.h"
#include "Gauss1DMAP.h"

mappel::MappelFixedSigma_IFace<mappel::Gauss1DMAP> iface; /**< Global iface object provides a iface.mexFunction */

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
