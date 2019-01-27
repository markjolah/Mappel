/** @file Gauss1DMAP.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class definition and template Specializations for Gauss1DMAP
 */
#include "Mappel/Gauss1DMAP.h"

namespace mappel {
const std::string Gauss1DMAP::name("Gauss1DMAP");

Gauss1DMAP::Gauss1DMAP(arma::Col<ImageCoordT> size, VecT psf_sigma, const std::string &prior_type)
    : Gauss1DMAP(size(0), psf_sigma(0), prior_type)
{ }

Gauss1DMAP::Gauss1DMAP(ImageSizeT size, double psf_sigma, const std::string &prior_type)
    : Gauss1DMAP(size,psf_sigma,make_default_prior(size,prior_type))
{ }

Gauss1DMAP::Gauss1DMAP(ImageSizeT _size, double psf_sigma, CompositeDist&& _prior) 
    : PointEmitterModel(std::move(_prior)), 
      ImageFormat1DBase(_size),
      Gauss1DModel(size, psf_sigma),
      PoissonNoise1DObjective(),
      MAPEstimator()
{ }

Gauss1DMAP::Gauss1DMAP(ImageSizeT _size, double psf_sigma, const CompositeDist& _prior) 
    : PointEmitterModel(_prior), 
      ImageFormat1DBase(_size),
      Gauss1DModel(size, psf_sigma),
      PoissonNoise1DObjective(),
      MAPEstimator()
{ }

Gauss1DMAP::Gauss1DMAP(const Gauss1DMAP &o) 
    : PointEmitterModel(o), 
      ImageFormat1DBase(o),
      Gauss1DModel(o),
      PoissonNoise1DObjective(o),
      MAPEstimator(o)
{ }

Gauss1DMAP::Gauss1DMAP(Gauss1DMAP &&o) 
    : PointEmitterModel(std::move(o)), 
      ImageFormat1DBase(std::move(o)),
      Gauss1DModel(std::move(o)),
      PoissonNoise1DObjective(std::move(o)),
      MAPEstimator(std::move(o))
{ }

Gauss1DMAP& Gauss1DMAP::operator=(const Gauss1DMAP &o)
{
    if(&o == this) return *this; //self assignment guard
    PointEmitterModel::operator=(o);
    ImageFormat1DBase::operator=(o);
    Gauss1DModel::operator=(o);
    PoissonNoise1DObjective::operator=(o);
    MAPEstimator::operator=(o);
    return *this;
}

Gauss1DMAP& Gauss1DMAP::operator=(Gauss1DMAP &&o)
{
    if(&o == this) return *this; //self assignment guard
    PointEmitterModel::operator=(std::move(o));
    ImageFormat1DBase::operator=(std::move(o));
    Gauss1DModel::operator=(std::move(o));
    PoissonNoise1DObjective::operator=(std::move(o));
    MAPEstimator::operator=(std::move(o));
    return *this;
}

} /* namespace mappel */
