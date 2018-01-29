
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
using StringVecT = std::vector<std::string>;

/* Allow easier enabale_if compilation for subclasses */
template<class ModelT,class ModelBaseT> using IsSubclassT = typename std::enable_if<std::is_base_of<ModelBaseT,ModelT>::value,int>::type;
template<class ReturnT, class ModelT,class ModelBaseT> using ReturnIfSubclassT = typename std::enable_if<std::is_base_of<ModelBaseT,ModelT>::value,ReturnT>::type;

template<class Model> using ImageCoordT = typename Model::ImageCoordT; /* Model's image coordinate type */
template<class Model> using ImagePixelT = typename Model::ImagePixelT; /* Image's pixel data type */

template<class Model> using ParamT = typename Model::ParamT; /* The Model's paramter type (e.g., theta) */
template<class Model> using ParamVecT = typename Model::ParamVecT; /* The Model's paramter type (e.g., theta) */
template<class Model> using ImageT = typename Model::ImageT; /* The Model's image type  */
template<class Model> using ModelDataT = typename Model::ModelDataT; /* Model's data type (for EMCCD same as Model::ImageT) */
template<class Model> using StencilT = typename Model::Stencil;  /* The Model's theta stencil  */

template<class Model> using ImageStackT = typename Model::ImageStackT; /* Model's image type  */
template<class Model> using ModelDataStackT = typename Model::ModelDataStackT; /* Model's data stack type (for EMCCD same as Model::ImageStackT) */
template<class Model> using StencilVecT = typename Model::StencilVecT;  /* The Model's Vector of stencils type */


void enable_all_cpus();

bool istarts_with(const char* s, const char* pattern);
const char * icontains(const char* s, const char* pattern);
int maxidx(const VecT &v);



using MappelError = backtrace_exception::BacktraceException;

struct BadSizeError : public MappelError 
{
    BadSizeError(std::string message) : MappelError("BadSize",message) {}
};

struct BadValueError : public MappelError 
{
    BadValueError(std::string message) : MappelError("BadValue",message) {}
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

std::ostream& operator<<(std::ostream &out,const StatsT &stats);


// MatT sliceView(const CubeT &cube, IdxT slice)
// {
//     if(slice >= cube.n_slices){
//         std::ostringstream msg;
//         msg<<"Got bad slice idx:"<<slice;
//         throw BoundsError(msg.str());
//     }
//     IdxT slice_size = cube.n_rows * cube.n_cols;
//     return { static_cast<const double *>(cube.memptr())+slice_size*slice, cube.n_rows, cube.n_cols, false, true };
// }

} /* namespace mappel */

/* Statistics Functions */



#endif /* _MAPPEL_UTIL_H */
