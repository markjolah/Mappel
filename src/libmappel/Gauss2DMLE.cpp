/** @file Gauss2DMLE.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2014-2018
 * @brief The class definition and template Specializations for Gauss2DMLE
 */
#include "Gauss2DMLE.h"

namespace mappel {
const std::string Gauss2DMLE::name("Gauss2DMLE");

Gauss2DMLE::Gauss2DMLE(ImageCoordT size, double psf_sigma) :
    Gauss2DModel(ImageSizeT(2,arma::fill::ones)*size, VecT(2,arma::fill::ones)*size)
{ }
    
Gauss2DMLE::Gauss2DMLE(const ImageSizeT &size, double psf_sigma) :
    Gauss2DModel(size, VecT(2,arma::fill::ones)*size)
{ }

Gauss2DMLE::Gauss2DMLE(const ImageSizeT &size, const VecT &psf_sigma) : 
            PointEmitterModel(make_default_prior(size)), 
            ImageFormat2DBase(size),
            Gauss2DModel(size, psf_sigma)
{ }

template<class PriorDistT>
Gauss2DMLE::Gauss2DMLE(const ImageSizeT &size, const VecT &psf_sigma, PriorDistT&& prior) : 
            PointEmitterModel(std::forward<PriorDistT>(prior)), 
            ImageFormat2DBase(size),
            Gauss2DModel(size, psf_sigma)
{ }

} /* namespace mappel */
