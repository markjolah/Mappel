/** @file GaussHSMAP_Iface.cpp
 *  @brief The entry point for GaussHSMAP_Iface mex module.
 * 
 * Just calls the MappleMexIface.mexFunction which is the Iface entry point.
 * 
 */
#include "MappelHS_Iface.h"
#include "GaussHSMAP.h"

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    MappelHS_Iface<GaussHSMAP> iface("GaussHSMAP");
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
