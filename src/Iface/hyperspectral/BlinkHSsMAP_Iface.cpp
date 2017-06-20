/** @file BlinkHSsMAP_Iface.cpp
 *  @brief The entry point for BlinkHSsMAP_Iface mex module.
 * 
 * Just calls the MappleMexIface.mexFunction which is the Iface entry point.
 * 
 */
#include "MappelHS_Iface.h"
#include "BlinkHSsMAP.h"

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    MappelHS_Iface<BlinkHSsMAP> iface("BlinkHSsMAP");
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
