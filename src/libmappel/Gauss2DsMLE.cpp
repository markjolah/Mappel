/** @file Gauss2DsMLE.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2018
 * @brief The class definition and template Specializations for Gauss2DsMLE
 */
#include "Mappel/Gauss2DsMLE.h"

namespace mappel {
const std::string Gauss2DsMLE::name("Gauss2DsMLE");

Gauss2DsMLE::Gauss2DsMLE(const ImageSizeT &size, const VecT &min_sigma, double max_sigma_ratio) 
    : Gauss2DsMLE{size, min_sigma, VecT{max_sigma_ratio*min_sigma}}
{ }

Gauss2DsMLE::Gauss2DsMLE(const ImageSizeT &size, const VecT &min_sigma, const VecT &max_sigma) 
    : Gauss2DsMLE{size, min_sigma,make_default_prior(size,compute_max_sigma_ratio(min_sigma,max_sigma))}
{ }

Gauss2DsMLE::Gauss2DsMLE(const ImageSizeT &_size, const VecT &min_sigma, CompositeDist&& _prior) 
    : PointEmitterModel(std::move(_prior)), 
      ImageFormat2DBase(_size),
      Gauss2DsModel(size, min_sigma, prior.ubound()(4) * min_sigma),
      PoissonNoise2DObjective(),
      MLEstimator()
{ }

Gauss2DsMLE::Gauss2DsMLE(const ImageSizeT &_size, const VecT &min_sigma, const CompositeDist& _prior) 
    : PointEmitterModel(_prior), 
      ImageFormat2DBase(_size),
      Gauss2DsModel(size, min_sigma, prior.ubound()(4) * min_sigma),
      PoissonNoise2DObjective(),
      MLEstimator()
{ }

Gauss2DsMLE::Gauss2DsMLE(const Gauss2DsMLE &o) 
    : PointEmitterModel(o), 
      ImageFormat2DBase(o),
      Gauss2DsModel(o),
      PoissonNoise2DObjective(o),
      MLEstimator(o)
{ }

Gauss2DsMLE::Gauss2DsMLE(Gauss2DsMLE &&o) 
    : PointEmitterModel(std::move(o)), 
      ImageFormat2DBase(std::move(o)),
      Gauss2DsModel(std::move(o)),
      PoissonNoise2DObjective(std::move(o)),
      MLEstimator(std::move(o))
{ }

Gauss2DsMLE& Gauss2DsMLE::operator=(const Gauss2DsMLE &o)
{
    if(&o == this) return *this; //self assignment guard
    PointEmitterModel::operator=(o);
    ImageFormat2DBase::operator=(o);
    Gauss2DsModel::operator=(o);
    PoissonNoise2DObjective::operator=(o);
    MLEstimator::operator=(o);
    return *this;
}

Gauss2DsMLE& Gauss2DsMLE::operator=(Gauss2DsMLE &&o)
{
    if(&o == this) return *this; //self assignment guard
    PointEmitterModel::operator=(std::move(o));
    ImageFormat2DBase::operator=(std::move(o));
    Gauss2DsModel::operator=(std::move(o));
    PoissonNoise2DObjective::operator=(std::move(o));
    MLEstimator::operator=(std::move(o));
    return *this;
}

} /* namespace mappel */
