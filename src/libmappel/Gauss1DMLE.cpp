/** @file Gauss1DMLE.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2013-2017
 * @brief The class definition and template Specializations for Gauss1DMLE
 */

#include "Gauss1DMLE.h"

namespace mappel {

Gauss1DMLE::Gauss1DMLE(arma::Col<ImageCoordT> size, VecT psf_sigma) : 
            PointEmitterModel(make_default_prior(size(0))), 
            ImageFormat1DBase(size(0)),
            Gauss1DModel(size(0), psf_sigma(0))
{ }

Gauss1DMLE::Gauss1DMLE(ImageSizeT size, double psf_sigma) : 
            PointEmitterModel(make_default_prior(size)), 
            ImageFormat1DBase(size),
            Gauss1DModel(size, psf_sigma)
{ }

template<class PriorDistT>
Gauss1DMLE::Gauss1DMLE(ImageSizeT size, double psf_sigma, PriorDistT&& prior) : 
            PointEmitterModel(std::forward<PriorDistT>(prior)), 
            ImageFormat1DBase(size),
            Gauss1DModel(size, psf_sigma)
{ }


} /* namespace mappel */
