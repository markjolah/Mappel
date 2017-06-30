/** @file ImageFormat1DBase.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-26-2014
 * @brief The class definition and template Specializations for ImageFormat1DBase
 */
#include "ImageFormat1DBase.h"

namespace mappel {

ImageFormat1DBase::ImageFormat1DBase(int size_)
    : size(size_)
{
    if(size <= min_size) {
        std::ostringstream os;
        os<<"Bad problem Size= "<<size_<<"< Min size="<<min_size;
        throw MappelException("ImageFormat1DBase",os.str());
    }
}

StatsT
ImageFormat1DBase::get_stats() const
{
    StatsT stats;
    stats["imageDimensions"] = 1;
    stats["imageSize.X"]=size;
    stats["imageLimits.Xmin"]=0;
    stats["imageLimits.Xmax"]=size;
    return stats;
}

} /* namespace mappel */
