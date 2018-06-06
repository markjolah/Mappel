/** @file Gauss1DMLE.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2018
 * @brief The class definition and template Specializations for Gauss1DMLE
 */

#include "Mappel/Gauss1DMLE.h"

namespace mappel {
const std::string Gauss1DMLE::name("Gauss1DMLE");

Gauss1DMLE::Gauss1DMLE(arma::Col<ImageCoordT> size, VecT psf_sigma) :
    Gauss1DMLE(size(0),psf_sigma(0))
{ }

Gauss1DMLE::Gauss1DMLE(ImageSizeT size, double psf_sigma) : 
            PointEmitterModel(make_default_prior(size)), 
            ImageFormat1DBase(size),
            Gauss1DModel(size, psf_sigma)
{ }

Gauss1DMLE::Gauss1DMLE(ImageSizeT size, double psf_sigma, CompositeDist&& prior) : 
            PointEmitterModel(std::move(prior)), 
            ImageFormat1DBase(size),
            Gauss1DModel(size, psf_sigma)
{ }


} /* namespace mappel */
