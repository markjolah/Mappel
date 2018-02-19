/** @file Gauss2DsMAP.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2018
 * @brief The class definition and template Specializations for Gauss2DsMAP
 */
#include "Gauss2DsMAP.h"

namespace mappel {

const std::string Gauss2DsMAP::name("Gauss2DsMAP");

Gauss2DsMAP::Gauss2DsMAP(const ImageSizeT &size, const VecT &min_sigma, double max_sigma_ratio) : 
    Gauss2DsMAP(size,min_sigma, VecT{max_sigma_ratio*min_sigma})
{ }

Gauss2DsMAP::Gauss2DsMAP(const ImageSizeT &size, const VecT &min_sigma, const VecT &max_sigma) :
    PointEmitterModel(make_default_prior(size, compute_max_sigma_ratio(min_sigma,max_sigma))), 
    ImageFormat2DBase(size),
    Gauss2DsModel(size, min_sigma, max_sigma)
{ }

Gauss2DsMAP::Gauss2DsMAP(const ImageSizeT &size, const VecT &min_sigma, CompositeDist&& prior) :
    PointEmitterModel(std::move(prior)), 
    ImageFormat2DBase(size),
    Gauss2DsModel(size, min_sigma, prior.ubound()(4) * min_sigma)
{ }

} /* namespace mappel */
