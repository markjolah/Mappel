/** @file ImageFormat1DBase.cpp
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 03-26-2014
 * @brief The class definition and template Specializations for ImageFormat1DBase
 */
#include "ImageFormat1DBase.h"
#include <sstream>

namespace mappel {

ImageFormat1DBase::ImageFormat1DBase(ImageSizeT size_)
    : size(size_)
{
    check_size(size);
}

ImageFormat1DBase::ImageFormat1DBase(const arma::Col<ImageSizeT> &size_)
    : size(size_[0])
{
    check_size(size);
}

void ImageFormat1DBase::check_size(ImageSizeT size_)
{
    if(size_ <= min_size) {
        std::ostringstream msg;
        msg<<"ImageFormat1DBase: Got Size= "<<size_<<"< Min size="<<min_size;
        throw BadSizeError(msg.str());
    }
}

StatsT ImageFormat1DBase::get_stats() const
{
    StatsT stats;
    stats["imageDimensions"] = 1;
    stats["imageSize.X"]=size;
    stats["imageLimits.Xmin"]=0;
    stats["imageLimits.Xmax"]=size;
    return stats;
}

} /* namespace mappel */
