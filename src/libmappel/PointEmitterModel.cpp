/** @file PointEmitterModel.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-13-2014
 * @brief The class definition and template Specializations for PointEmitterModel
 */

#include "PointEmitterModel.h"
// #include "util.h"
// #include <omp.h>

namespace mappel {

PointEmitterModel::PointEmitterModel(int num_params_)
    : num_params(num_params_)
{
    //Save these booleans internally to speed up bounds checking which happens very often
    VecT lb(num_params), ub(num_params);
    lb.fill(0);
    ub.fill(INFINITY);
    set_bounds(lb,ub);
}

void PointEmitterModel::set_bounds(const ParamT &lbound_, const ParamT &ubound_)
{
    if(static_cast<int>(lbound_.n_elem) != num_params) throw std::logic_error("Invalid lower bound size");
    if(static_cast<int>(ubound_.n_elem) != num_params) throw std::logic_error("Invalid upper bound size");
    lbound = lbound_;
    ubound = ubound_;
    lbound_valid.set_size(num_params);
    ubound_valid.set_size(num_params);
    for(int n=0;n<num_params;n++){
        lbound_valid(n) = std::isfinite(lbound(n));
        ubound_valid(n) = std::isfinite(ubound(n));
    }
}

std::ostream& operator<<(std::ostream &out, PointEmitterModel &model)
{
    auto stats=model.get_stats();
    out<<"["<<model.name()<<":";
    for(auto it=stats.cbegin(); it!=stats.cend(); ++it) out<<" "<<it->first<<"="<<it->second;
    out<<"]";
    return out;
}

} /* namespace mappel */
