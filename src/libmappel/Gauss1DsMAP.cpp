/** @file Gauss1DsMAP.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief The class definition and template Specializations for Gauss1DsMAP
 */

#include "Mappel/Gauss1DsMAP.h"

namespace mappel {
const std::string Gauss1DsMAP::name("Gauss1DsMAP");

Gauss1DsMAP::Gauss1DsMAP(arma::Col<ImageCoordT> size, VecT min_sigma, VecT max_sigma, const std::string &prior_type)
    : Gauss1DsMAP(size(0),make_default_prior(size(0),min_sigma(0),max_sigma(0),prior_type))
{ }

Gauss1DsMAP::Gauss1DsMAP(ImageSizeT size, double min_sigma, double max_sigma, const std::string &prior_type)
    : Gauss1DsMAP(size,make_default_prior(size,min_sigma,max_sigma,prior_type))
{ }

Gauss1DsMAP::Gauss1DsMAP(ImageSizeT _size, CompositeDist&& _prior)
          : PointEmitterModel(std::move(_prior)),
            ImageFormat1DBase(_size),
            Gauss1DsModel(size),
            PoissonNoise1DObjective(),
            MAPEstimator()
{ }

Gauss1DsMAP::Gauss1DsMAP(ImageSizeT _size, const CompositeDist& _prior)
          : PointEmitterModel(_prior),
            ImageFormat1DBase(_size),
            Gauss1DsModel(size),
            PoissonNoise1DObjective(),
            MAPEstimator()
{ }

Gauss1DsMAP::Gauss1DsMAP(const Gauss1DsMAP &o)
          : PointEmitterModel(o),
            ImageFormat1DBase(o),
            Gauss1DsModel(o),
            PoissonNoise1DObjective(o),
            MAPEstimator(o)
{ }

Gauss1DsMAP::Gauss1DsMAP(Gauss1DsMAP &&o)
          : PointEmitterModel(std::move(o)),
            ImageFormat1DBase(std::move(o)),
            Gauss1DsModel(std::move(o)),
            PoissonNoise1DObjective(std::move(o)),
            MAPEstimator(std::move(o))
{ }

Gauss1DsMAP& Gauss1DsMAP::operator=(const Gauss1DsMAP &o)
{
    if(&o == this) return *this; //self assignment guard
    PointEmitterModel::operator=(o);
    ImageFormat1DBase::operator=(o);
    Gauss1DsModel::operator=(o);
    PoissonNoise1DObjective::operator=(o);
    MAPEstimator::operator=(o);
    return *this;
}

Gauss1DsMAP& Gauss1DsMAP::operator=(Gauss1DsMAP &&o)
{
    if(&o == this) return *this; //self assignment guard
    PointEmitterModel::operator=(std::move(o));
    ImageFormat1DBase::operator=(std::move(o));
    Gauss1DsModel::operator=(std::move(o));
    PoissonNoise1DObjective::operator=(std::move(o));
    MAPEstimator::operator=(std::move(o));
    return *this;
}


} /* namespace mappel */
