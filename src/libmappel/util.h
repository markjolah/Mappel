
#ifndef _MAPPEL_UTIL_H
#define _MAPPEL_UTIL_H

#include <cstdint>
#include <cmath>
#include <memory>
#include <utility>
#include <string>
#include <map>
#include <sstream>
#include <armadillo>
#include <BacktraceException/BacktraceException.h>

namespace mappel {
    
// Do we need this still
// extern const std::vector<std::string> model_names;

    
using BoolT = uint16_t;
using IdxT = arma::uword;
using IdxVecT = arma::Col<IdxT>; /**< A type to represent integer data arrays */
using IdxMatT = arma::Mat<IdxT>; /**< A type to represent integer data arrays */
using VecT = arma::vec; /**< A type to represent floating-point data arrays */
using MatT = arma::mat; /**< A type to represent floating-point data matricies */
using CubeT = arma::cube; /**< A type to represent floating-point data cubes */
using VecFieldT = arma::field<VecT>;
using StatsT = std::map<std::string,double>;  /**< A convenient form for reporting dictionaries of named FP data to matlab */
void enable_all_cpus();

bool istarts_with(const char* s, const char* pattern);
const char * icontains(const char* s, const char* pattern);
int maxidx(const VecT &v);



using MappelError = backtrace_exception::BacktraceException;

struct BadSizeError : public MappelError 
{
    BadSizeError(std::string message) : MappelError("BadSize",message) {}
};

struct BadShapeError : public MappelError 
{
    BadShapeError(std::string message) : MappelError("BadShape",message) {}
};

struct BoundsError : public MappelError 
{
    BoundsError(std::string message) : MappelError("BoundsError",message) {}
};

struct NumericalError : public MappelError 
{
    NumericalError(std::string message) : MappelError("NumericalError",message) {}
};

struct NotImplementedError : public MappelError 
{
    NotImplementedError(std::string message) : MappelError("NotImplemented",message) {}
};



/** @brief sign (signum) function: -1/0/1
 * 
 */
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

inline double restrict_value_range(double val, double minval, double maxval)
{
    if(!std::isfinite(val)) throw NumericalError("Non-finite value in restrict_value_range.");
    return (val<minval) ? minval : ((val>maxval) ? maxval : val);
}

template<typename T, typename ...Args>
std::unique_ptr<T> make_unique( Args&& ...args )
{
    return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
}


} /* namespace mappel */

/* Statistics Functions */



#endif /* _MAPPEL_UTIL_H */
