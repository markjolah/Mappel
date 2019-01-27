/** @file Gauss2DMLE.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class definition and template Specializations for Gauss2DMLE
 */
#include "Mappel/Gauss2DMLE.h"

namespace mappel {
const std::string Gauss2DMLE::name("Gauss2DMLE");

Gauss2DMLE::Gauss2DMLE(ImageCoordT size, double psf_sigma, const std::string &prior_type)
    : Gauss2DMLE(ImageSizeT(2,arma::fill::ones)*size, VecT(2,arma::fill::ones)*size, prior_type)
{ }
    
Gauss2DMLE::Gauss2DMLE(const ImageSizeT &size, double psf_sigma, const std::string &prior_type)
    : Gauss2DMLE(size, VecT(2,arma::fill::ones)%size, make_default_prior(size,prior_type))
{ }

Gauss2DMLE::Gauss2DMLE(const ImageSizeT &size, const VecT &psf_sigma, const std::string &prior_type)
    : Gauss2DMLE(size, psf_sigma, make_default_prior(size,prior_type))
{ }

Gauss2DMLE::Gauss2DMLE(const ImageSizeT &_size, const VecT &psf_sigma, CompositeDist&& _prior) 
    : PointEmitterModel(std::move(_prior)), 
      ImageFormat2DBase(_size),
      Gauss2DModel(size, psf_sigma),
      PoissonNoise2DObjective(),
      MLEstimator()
{ }

Gauss2DMLE::Gauss2DMLE(const ImageSizeT &_size, const VecT &psf_sigma, const CompositeDist& _prior) 
    : PointEmitterModel(_prior), 
      ImageFormat2DBase(_size),
      Gauss2DModel(size, psf_sigma),
      PoissonNoise2DObjective(),
      MLEstimator()
{ }

Gauss2DMLE::Gauss2DMLE(const Gauss2DMLE &o) 
    : PointEmitterModel(o), 
      ImageFormat2DBase(o),
      Gauss2DModel(o),
      PoissonNoise2DObjective(o),
      MLEstimator(o)
{ }

Gauss2DMLE::Gauss2DMLE(Gauss2DMLE &&o) 
    : PointEmitterModel(std::move(o)), 
      ImageFormat2DBase(std::move(o)),
      Gauss2DModel(std::move(o)),
      PoissonNoise2DObjective(std::move(o)),
      MLEstimator(std::move(o))
{ }

Gauss2DMLE& Gauss2DMLE::operator=(const Gauss2DMLE &o)
{
    if(&o == this) return *this; //self assignment guard
    PointEmitterModel::operator=(o);
    ImageFormat2DBase::operator=(o);
    Gauss2DModel::operator=(o);
    PoissonNoise2DObjective::operator=(o);
    MLEstimator::operator=(o);
    return *this;
}

Gauss2DMLE& Gauss2DMLE::operator=(Gauss2DMLE &&o)
{
    if(&o == this) return *this; //self assignment guard
    PointEmitterModel::operator=(std::move(o));
    ImageFormat2DBase::operator=(std::move(o));
    Gauss2DModel::operator=(std::move(o));
    PoissonNoise2DObjective::operator=(std::move(o));
    MLEstimator::operator=(std::move(o));
    return *this;
}

} /* namespace mappel */
