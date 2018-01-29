/** @file Gauss1DsMLE.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief The class definition and template Specializations for Gauss1DsMLE
 */

#include "Gauss1DsMLE.h"

namespace mappel {
const std::string Gauss1DsMLE::name("Gauss1DsMLE");

Gauss1DsMLE::Gauss1DsMLE(arma::Col<ImageCoordT> size, VecT min_sigma, VecT max_sigma) : 
            PointEmitterModel(make_default_prior(size(0),min_sigma(0),max_sigma(0))), 
            ImageFormat1DBase(size(0)),
            Gauss1DsModel(size(0))
{ }

Gauss1DsMLE::Gauss1DsMLE(ImageSizeT size, double min_sigma, double max_sigma) : 
            PointEmitterModel(make_default_prior(size,min_sigma,max_sigma)), 
            ImageFormat1DBase(size),
            Gauss1DsModel(size)
{ }

template<class PriorDistT>
Gauss1DsMLE::Gauss1DsMLE(ImageSizeT size, PriorDistT&& prior) : 
            PointEmitterModel(std::forward<PriorDistT>(prior)), 
            ImageFormat1DBase(size),
            Gauss1DsModel(size)
{ }


} /* namespace mappel */
