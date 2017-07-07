/** @file GaussHSsMAP_Iface.cpp
 *  @brief The entry point for GaussHSsMAP_Iface mex module.
 * 
 * Just calls the MappleMexIface.mexFunction which is the Iface entry point.
 * 
 */
#include "MappelHS_Iface.h"
#include "GaussHSsMAP.h"

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    MappelHS_Iface<GaussHSsMAP> iface("GaussHSsMAP");
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
