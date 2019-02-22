/** @file Gauss1DMAP_IFace.cpp
 *  @author Mark J. Olah (mjo at cs.unm.edu)
 *  @date 2015-2019
 *  @brief The entry point for Gauss1DMAP MexIFace MEX module.
 * 
 */
#include "Mappel/Gauss1DMAP.h"
#include "Mappel_IFace.h"

MappelFixedSigma_IFace<mappel::Gauss1DMAP> iface; /**< Global IFace object provides a iface.mexFunction */

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
