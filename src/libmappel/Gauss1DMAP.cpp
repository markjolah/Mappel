/** @file Gauss1DMAP.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief The class definition and template Specializations for Gauss1DMAP
 */

#include "Mappel/Gauss1DMAP.h"

namespace mappel {
const std::string Gauss1DMAP::name("Gauss1DMAP");

Gauss1DMAP::Gauss1DMAP(arma::Col<ImageCoordT> size, VecT psf_sigma) :
    Gauss1DMAP(size(0), psf_sigma(0))
{ }

Gauss1DMAP::Gauss1DMAP(ImageSizeT size, double psf_sigma) : 
            PointEmitterModel(make_default_prior(size)), 
            ImageFormat1DBase(size),
            Gauss1DModel(size, psf_sigma)
{ }

Gauss1DMAP::Gauss1DMAP(ImageSizeT size, double psf_sigma, CompositeDist&& prior) : 
            PointEmitterModel(std::move(prior)), 
            ImageFormat1DBase(size),
            Gauss1DModel(size, psf_sigma)
{ }

// Gauss1DMAP::Gauss1DMAP(const Gauss1DMAP &o) : 
//             PointEmitterModel(o), 
//             ImageFormat1DBase(o),
//             Gauss1DModel(o)
// { }

// Gauss1DMAP::Gauss1DMAP(Gauss1DMAP &&o) : 
//             PointEmitterModel(std::move(o)), 
//             ImageFormat1DBase(std::move(o)),
//             Gauss1DModel(std::move(o))
// { }

// Gauss1DMAP(Gauss1DMAP &&o) : PointEmitterModel(o), ImageFormat1DBase(o), Gauss1DModel(o) {}

// Gauss1DMAP::Gauss1DMAP(Gauss1DMAP &o) : 
//             PointEmitterModel(make_default_prior(size)), 
//             ImageFormat1DBase(size),
//             Gauss1DModel(size, psf_sigma)
// { }

} /* namespace mappel */
