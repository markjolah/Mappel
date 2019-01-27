/** @file Gauss1DMLE.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class definition and template Specializations for Gauss1DMLE
 */
#include "Mappel/Gauss1DMLE.h"

namespace mappel {
const std::string Gauss1DMLE::name("Gauss1DMLE");

Gauss1DMLE::Gauss1DMLE(arma::Col<ImageCoordT> size, VecT psf_sigma, const std::string &prior_type)
    : Gauss1DMLE(size(0), psf_sigma(0), prior_type)
{ }

Gauss1DMLE::Gauss1DMLE(ImageSizeT size, double psf_sigma, const std::string &prior_type)
    : Gauss1DMLE(size,psf_sigma,make_default_prior(size,prior_type))
{ }

Gauss1DMLE::Gauss1DMLE(ImageSizeT _size, double psf_sigma, CompositeDist&& _prior)
    : PointEmitterModel(std::move(_prior)), 
      ImageFormat1DBase(_size),
      Gauss1DModel(size, psf_sigma),
      PoissonNoise1DObjective(),
      MLEstimator()
{ }

Gauss1DMLE::Gauss1DMLE(ImageSizeT _size, double psf_sigma, const CompositeDist& _prior)
    : PointEmitterModel(_prior), 
      ImageFormat1DBase(_size),
      Gauss1DModel(size, psf_sigma),
      PoissonNoise1DObjective(),
      MLEstimator()
{ }

Gauss1DMLE::Gauss1DMLE(const Gauss1DMLE &o) 
    : PointEmitterModel(o), 
      ImageFormat1DBase(o),
      Gauss1DModel(o),
      PoissonNoise1DObjective(o),
      MLEstimator(o)
{ }

Gauss1DMLE::Gauss1DMLE(Gauss1DMLE &&o) 
    : PointEmitterModel(std::move(o)), 
      ImageFormat1DBase(std::move(o)),
      Gauss1DModel(std::move(o)),
      PoissonNoise1DObjective(std::move(o)),
      MLEstimator(std::move(o))
{ }

Gauss1DMLE& Gauss1DMLE::operator=(const Gauss1DMLE &o)
{
    if(&o == this) return *this; //self assignment guard
    PointEmitterModel::operator=(o);
    ImageFormat1DBase::operator=(o);
    Gauss1DModel::operator=(o);
    PoissonNoise1DObjective::operator=(o);
    MLEstimator::operator=(o);
    return *this;
}

Gauss1DMLE& Gauss1DMLE::operator=(Gauss1DMLE &&o)
{
    if(&o == this) return *this; //self assignment guard
    PointEmitterModel::operator=(std::move(o));
    ImageFormat1DBase::operator=(std::move(o));
    Gauss1DModel::operator=(std::move(o));
    PoissonNoise1DObjective::operator=(std::move(o));
    MLEstimator::operator=(std::move(o));
    return *this;
}

} /* namespace mappel */
