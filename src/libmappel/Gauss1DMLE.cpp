/** @file Gauss1DMLE.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2013-2017
 * @brief The class definition and template Specializations for Gauss1DMLE
 */

#include "Gauss1DMLE.h"

namespace mappel {
Gauss1DMLE::Gauss1DMLE(int size, double psf_sigma); 
        : PointEmitterModel(make_prior(size,psf_sigma)), 
          ImageFormat1DBase(size),
          Gauss1DModel(psf_sigma)
    {}
    
constexpr CompositeDist Gauss1DMLE::make_prior(int size, double psf_sigma)
{
    return CompositeDist(SymmetricBetaDist(beta_x,0,size,"x"),
                         GammaDist(mean_I,kappa_I,"I"),
                         GammaDist(mean_bg*size,kappa_bg,"bg"));
    
}

} /* namespace mappel */
