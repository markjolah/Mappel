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

void ImageFormat1DBase::set_size(const ImageSizeT &size_)
{
    check_size(size_);
    size = size_;
}

void ImageFormat1DBase::check_size(const ImageSizeT &size_)
{
    if(size_ <= min_size) {
        std::ostringstream msg;
        msg<<"ImageFormat1DBase: Got Size= "<<size_<<"< Min size="<<min_size;
        throw BadSizeError(msg.str());
    }
}

void ImageFormat1DBase::check_image_shape(const ImageT &im) const
{
    if(im.n_elem != size) {
        std::ostringstream msg;
        msg<<"ImageFormat1DBase: Got bad image Size= "<<im.n_elem<<" Expected size="<<size;
        throw BadShapeError(msg.str());
    }
}

void ImageFormat1DBase::check_image_shape(const ImageStackT &ims) const
{
    if(ims.n_rows != size) {
        std::ostringstream msg;
        msg<<"ImageFormat1DBase: Got bad image stack #rows= "<<ims.n_rows<<" Expected #rows="<<size;
        throw BadShapeError(msg.str());
    }
}


StatsT ImageFormat1DBase::get_stats() const
{
    StatsT stats;
    stats["imageDimensions"] = 1;
    stats["imageSize.X"]=get_size();
    stats["imageNumPixels"]=get_num_pixels();
    stats["imageLimits.Xmin"]=0;
    stats["imageLimits.Xmax"]=get_size();
    return stats;
}

} /* namespace mappel */
