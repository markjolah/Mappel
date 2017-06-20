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
    assert(size > 1);
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
