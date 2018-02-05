/** @file Gauss1DMAP.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief The class definition and template Specializations for Gauss1DMAP
 */

#include "Gauss1DMAP.h"

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

template<class PriorDistT>
Gauss1DMAP::Gauss1DMAP(ImageSizeT size, double psf_sigma, PriorDistT&& prior) : 
            PointEmitterModel(std::forward<PriorDistT>(prior)), 
            ImageFormat1DBase(size),
            Gauss1DModel(size, psf_sigma)
{ }


} /* namespace mappel */
