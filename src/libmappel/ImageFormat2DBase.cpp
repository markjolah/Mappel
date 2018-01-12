/** @file ImageFormat2DBase.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-26-2014
 * @brief The class definition and template Specializations for ImageFormat2DBase
 */
#include "ImageFormat2DBase.h"

namespace mappel {

ImageFormat2DBase::ImageFormat2DBase(ImageSizeT size_)
    : size(size_), num_pixels(arma::prod(size))
{
    check_size(size);
}

StatsT
ImageFormat2DBase::get_stats() const
{
    StatsT stats;
    stats["imageDimensions"] = 2;
    stats["imageSize.X"]=size(0);
    stats["imageSize.Y"]=size(1);
    stats["imageLimits.Xmin"]=0;
    stats["imageLimits.Xmax"]=size(0);
    stats["imageLimits.Ymin"]=0;
    stats["imageLimits.Ymax"]=size(1);
    return stats;
}

} /* namespace mappel */
