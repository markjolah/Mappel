/** @file ImageFormat1DBase.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2018
 * @brief The class definition and template Specializations for ImageFormat1DBase
 */
#include "Mappel/ImageFormat1DBase.h"

namespace mappel {
const ImageFormat1DBase::ImageCoordT ImageFormat1DBase::num_dim = 1;  /**< Number of image dimensions. */
const ImageFormat1DBase::ImageCoordT ImageFormat1DBase::global_min_size = 3; /**< Minimum size along any dimension of the image. */
const ImageFormat1DBase::ImageCoordT ImageFormat1DBase::global_max_size = 512; /**< Maximum size along any dimension of the image.  This is insanely big to catch obvious errors */

ImageFormat1DBase::ImageFormat1DBase(ImageSizeT size)
    : size(size)
{
    check_size(size);
}

ImageFormat1DBase::ImageCoordT  
ImageFormat1DBase::get_size(IdxT idx) const
{ 
    if(idx > 0) {
        std::ostringstream msg;
        msg<<"ImageFormat1DBase::get_size() idx="<<idx<<" is invalid.";
        throw ParameterValueError(msg.str());
    }
    return size; 
}

void ImageFormat1DBase::set_size(const ImageSizeT &size_)
{
    check_size(size_);
    size = size_;
}

/** @brief Check the size argument for the model
 * 
 */
void ImageFormat1DBase::check_size(const ImageSizeT &size_)
{
    if(size_ < global_min_size) {
        std::ostringstream msg;
        msg<<"ImageFormat1DBase::check_size: Got Size= "<<size_<<"< Min size="<<global_min_size;
        throw ParameterValueError(msg.str());
    } else if(size_ > global_max_size) {
        std::ostringstream msg;
        msg<<"ImageFormat1DBase::check_size: Got Size= "<<size_<<"> Max size="<<global_max_size;
        throw ParameterValueError(msg.str());
    } else if(!std::isfinite(size_)) {
        std::ostringstream msg;
        msg<<"ImageFormat1DBase::check_size: Got non-finite Size= "<<size_;
        throw ParameterValueError(msg.str());
    }
}

/** @brief Check the shape of a single images is correct for model size
 * 
 */
void ImageFormat1DBase::check_image_shape(const ImageT &im) const
{
    if(im.n_elem != size) {
        std::ostringstream msg;
        msg<<"ImageFormat1DBase::check_image_shape: Got bad image Size= "<<im.n_elem<<" Expected size="<<size;
        throw ArrayShapeError(msg.str());
    }
}

/** @brief Check the shape of a stack of images is correct for model size
 * 
 */
void ImageFormat1DBase::check_image_shape(const ImageStackT &ims) const
{
    if(ims.n_rows != size) {
        std::ostringstream msg;
        msg<<"ImageFormat1DBase::check_image_shape: Got bad image stack #rows= "<<ims.n_rows<<" Expected #rows="<<size;
        throw ArrayShapeError(msg.str());
    }
}


StatsT ImageFormat1DBase::get_stats() const
{
    StatsT stats;
    stats["imageDimensions"] = num_dim;
    stats["imageSize.X"] = get_size();
    stats["imageNumPixels"] = get_num_pixels();
    stats["imageLimits.Xmin"] = 0;
    stats["imageLimits.Xmax"] = get_size();
    return stats;
}

} /* namespace mappel */
