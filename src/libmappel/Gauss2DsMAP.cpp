/** @file Gauss2DsMAP.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2018
 * @brief The class definition and template Specializations for Gauss2DsMAP
 */
#include "Mappel/Gauss2DsMAP.h"

namespace mappel {
const std::string Gauss2DsMAP::name("Gauss2DsMAP");

Gauss2DsMAP::Gauss2DsMAP(const ImageSizeT &size, const VecT &min_sigma, double max_sigma_ratio) 
    : Gauss2DsMAP{size,min_sigma, VecT{max_sigma_ratio*min_sigma}}
{ }

Gauss2DsMAP::Gauss2DsMAP(const ImageSizeT &size, const VecT &min_sigma, const VecT &max_sigma) 
    : Gauss2DsMAP{size,min_sigma,make_default_prior(size, compute_max_sigma_ratio(min_sigma,max_sigma))}
{ }

Gauss2DsMAP::Gauss2DsMAP(const ImageSizeT &_size, const VecT &min_sigma, CompositeDist&& _prior) 
    : PointEmitterModel(std::move(_prior)), 
      ImageFormat2DBase(_size),
      Gauss2DsModel(size, min_sigma, prior.ubound()(4) * min_sigma),
      PoissonNoise2DObjective(),
      MAPEstimator()
{ }

Gauss2DsMAP::Gauss2DsMAP(const ImageSizeT &_size, const VecT &min_sigma, const CompositeDist& _prior) 
    : PointEmitterModel(_prior), 
      ImageFormat2DBase(_size),
      Gauss2DsModel(size, min_sigma, prior.ubound()(4) * min_sigma),
      PoissonNoise2DObjective(),
      MAPEstimator()
{ }

Gauss2DsMAP::Gauss2DsMAP(const Gauss2DsMAP &o) 
    : PointEmitterModel(o), 
      ImageFormat2DBase(o),
      Gauss2DsModel(o),
      PoissonNoise2DObjective(o),
      MAPEstimator(o)
{ }

Gauss2DsMAP::Gauss2DsMAP(Gauss2DsMAP &&o) 
    : PointEmitterModel(std::move(o)), 
      ImageFormat2DBase(std::move(o)),
      Gauss2DsModel(std::move(o)),
      PoissonNoise2DObjective(std::move(o)),
      MAPEstimator(std::move(o))
{ }

Gauss2DsMAP& Gauss2DsMAP::operator=(const Gauss2DsMAP &o)
{
    if(&o == this) return *this; //self assignment guard
    PointEmitterModel::operator=(o);
    ImageFormat2DBase::operator=(o);
    Gauss2DsModel::operator=(o);
    PoissonNoise2DObjective::operator=(o);
    MAPEstimator::operator=(o);
    return *this;
}

Gauss2DsMAP& Gauss2DsMAP::operator=(Gauss2DsMAP &&o)
{
    if(&o == this) return *this; //self assignment guard
    PointEmitterModel::operator=(std::move(o));
    ImageFormat2DBase::operator=(std::move(o));
    Gauss2DsModel::operator=(std::move(o));
    PoissonNoise2DObjective::operator=(std::move(o));
    MAPEstimator::operator=(std::move(o));
    return *this;
}

} /* namespace mappel */
