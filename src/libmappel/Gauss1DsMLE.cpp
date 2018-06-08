/** @file Gauss1DsMLE.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief The class definition and template Specializations for Gauss1DsMLE
 */

#include "Mappel/Gauss1DsMLE.h"

namespace mappel {
const std::string Gauss1DsMLE::name("Gauss1DsMLE");

Gauss1DsMLE::Gauss1DsMLE(arma::Col<ImageCoordT> size, VecT min_sigma, VecT max_sigma)
    : Gauss1DsMLE(size(0),make_default_prior(size(0),min_sigma(0),max_sigma(0)))
{ }

Gauss1DsMLE::Gauss1DsMLE(ImageSizeT size, double min_sigma, double max_sigma) 
    : Gauss1DsMLE(size,make_default_prior(size,min_sigma,max_sigma))
{ }

Gauss1DsMLE::Gauss1DsMLE(ImageSizeT _size, CompositeDist&& _prior) : 
            PointEmitterModel(std::move(_prior)), 
            ImageFormat1DBase(_size),
            Gauss1DsModel(size),
            PoissonNoise1DObjective(),
            MLEstimator()
{ }

Gauss1DsMLE::Gauss1DsMLE(ImageSizeT _size, const CompositeDist& _prior) : 
            PointEmitterModel(_prior), 
            ImageFormat1DBase(_size),
            Gauss1DsModel(size),
            PoissonNoise1DObjective(),
            MLEstimator()
{ }

Gauss1DsMLE::Gauss1DsMLE(const Gauss1DsMLE &o) :
            PointEmitterModel(o), 
            ImageFormat1DBase(o),
            Gauss1DsModel(o),
            PoissonNoise1DObjective(o),
            MLEstimator(o)
{ }

Gauss1DsMLE::Gauss1DsMLE(Gauss1DsMLE &&o) :
            PointEmitterModel(std::move(o)), 
            ImageFormat1DBase(std::move(o)),
            Gauss1DsModel(std::move(o)),
            PoissonNoise1DObjective(std::move(o)),
            MLEstimator(std::move(o))
{ }

Gauss1DsMLE& Gauss1DsMLE::operator=(const Gauss1DsMLE &o)
{
    if(&o == this) return *this; //self assignment guard
    PointEmitterModel::operator=(o);
    ImageFormat1DBase::operator=(o);
    Gauss1DsModel::operator=(o);
    PoissonNoise1DObjective::operator=(o);
    MLEstimator::operator=(o);
    return *this;
}

Gauss1DsMLE& Gauss1DsMLE::operator=(Gauss1DsMLE &&o)
{
    if(&o == this) return *this; //self assignment guard
    PointEmitterModel::operator=(std::move(o));
    ImageFormat1DBase::operator=(std::move(o));
    Gauss1DsModel::operator=(std::move(o));
    PoissonNoise1DObjective::operator=(std::move(o));
    MLEstimator::operator=(std::move(o));
    return *this;
}

} /* namespace mappel */
