/** @file Gauss2DMAP.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class definition and template Specializations for Gauss2DMAP
 */
#include "Mappel/Gauss2DMAP.h"

namespace mappel {
const std::string Gauss2DMAP::name("Gauss2DMAP");

Gauss2DMAP::Gauss2DMAP(ImageCoordT size_, double psf_sigma_, const std::string &prior_type)
    : Gauss2DMAP(  ImageSizeT{size_,size_}, VecT{psf_sigma_, psf_sigma_} , prior_type)
{ }
    
Gauss2DMAP::Gauss2DMAP(const ImageSizeT &size_, double psf_sigma_, const std::string &prior_type)
    : Gauss2DMAP(size_, VecT{psf_sigma_, psf_sigma_} , prior_type)
{ }

Gauss2DMAP::Gauss2DMAP(ImageSizeT &&size_, VecT &&psf_sigma_, CompositeDist&& prior_)
    : PointEmitterModel(std::move(prior_)),
      ImageFormat2DBase(std::move(size_)),
      Gauss2DModel(size, std::move(psf_sigma_)),
      PoissonNoise2DObjective(),
      MAPEstimator()
{ }

Gauss2DMAP::Gauss2DMAP(const ImageSizeT &_size, const VecT &psf_sigma, CompositeDist&& _prior) 
    : PointEmitterModel(std::move(_prior)), 
      ImageFormat2DBase(_size),
      Gauss2DModel(size, psf_sigma),
      PoissonNoise2DObjective(),
      MAPEstimator()
{ }

Gauss2DMAP::Gauss2DMAP(const ImageSizeT &_size, const VecT &psf_sigma, const CompositeDist& _prior)
    : PointEmitterModel(_prior),
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
