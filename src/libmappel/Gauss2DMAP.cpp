/** @file Gauss2DMAP.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2018
 * @brief The class definition and template Specializations for Gauss2DMAP
 */
#include "Mappel/Gauss2DMAP.h"

namespace mappel {
const std::string Gauss2DMAP::name("Gauss2DMAP");

Gauss2DMAP::Gauss2DMAP(ImageCoordT size, double psf_sigma) 
    : Gauss2DMAP(ImageSizeT(2,arma::fill::ones)*size, VecT(2,arma::fill::ones)*size)
{ }
    
Gauss2DMAP::Gauss2DMAP(const ImageSizeT &size, double psf_sigma) 
    : Gauss2DMAP(size, VecT(2,arma::fill::ones)*size)
{ }

Gauss2DMAP::Gauss2DMAP(const ImageSizeT &size, const VecT &psf_sigma) 
    : Gauss2DMAP(size,psf_sigma,make_default_prior(size))
{ }

Gauss2DMAP::Gauss2DMAP(const ImageSizeT &_size, const VecT &psf_sigma, CompositeDist&& _prior) 
    : PointEmitterModel(std::move(_prior)), 
      ImageFormat2DBase(_size),
      Gauss2DModel(size, psf_sigma),
      PoissonNoise2DObjective(),
      MAPEstimator()
{ }

Gauss2DMAP::Gauss2DMAP(const ImageSizeT &_size, const VecT &psf_sigma, const CompositeDist& _prior)
    : PointEmitterModel(std::move(_prior)), 
      ImageFormat2DBase(_size),
      Gauss2DModel(size, psf_sigma),
      PoissonNoise2DObjective(),
      MAPEstimator()
{ }

Gauss2DMAP::Gauss2DMAP(const Gauss2DMAP &o) 
    : PointEmitterModel(o), 
      ImageFormat2DBase(o),
      Gauss2DModel(o),
      PoissonNoise2DObjective(o),
      MAPEstimator(o)
{ }

Gauss2DMAP::Gauss2DMAP(Gauss2DMAP &&o) 
    : PointEmitterModel(std::move(o)), 
      ImageFormat2DBase(std::move(o)),
      Gauss2DModel(std::move(o)),
      PoissonNoise2DObjective(std::move(o)),
      MAPEstimator(std::move(o))
{ }

Gauss2DMAP& Gauss2DMAP::operator=(const Gauss2DMAP &o)
{
    if(&o == this) return *this; //self assignment guard
    PointEmitterModel::operator=(o);
    ImageFormat2DBase::operator=(o);
    Gauss2DModel::operator=(o);
    PoissonNoise2DObjective::operator=(o);
    MAPEstimator::operator=(o);
    return *this;
}

Gauss2DMAP& Gauss2DMAP::operator=(Gauss2DMAP &&o)
{
    if(&o == this) return *this; //self assignment guard
    PointEmitterModel::operator=(std::move(o));
    ImageFormat2DBase::operator=(std::move(o));
    Gauss2DModel::operator=(std::move(o));
    PoissonNoise2DObjective::operator=(std::move(o));
    MAPEstimator::operator=(std::move(o));
    return *this;
}

} /* namespace mappel */
